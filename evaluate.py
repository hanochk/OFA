#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import os
import sys

import numpy as np
import torch
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results
import cv2
from inference_pipeline import inference_preprocess, construct_sample
from PIL import Image
import tqdm
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def main(cfg: DictConfig, **kwargs):
    utils.import_user_module(cfg.common)

    print("cfg", cfg.common.cpu)

    reset_logging()
    logger.info(cfg)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    print("cfg.common_eval.path", cfg.dataset.gen_subset)
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # print("Load model",         utils.split_paths(cfg.common_eval.path),
    #     overrides,
    #     cfg.checkpoint.checkpoint_suffix,
    #     cfg.checkpoint.checkpoint_shard_count, cfg.checkpoint.checkpoint_shard_count)

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
 # OFA visual grounding over LSMDC + likelihood over BB
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    inference_pipeline = kwargs.pop("inference_pipeline", False)
    if inference_pipeline:  # HK added switch for inference per loaded image rather than COCO packed dataset
        patch_resize_transform = inference_preprocess(cfg, task)
    # Download an image from COCO or you can use other images with wget
    #     os.system("! wget     http: // farm4.staticflickr.com / 3539 / 3836680545_2     ccb331621_z.jpg")
    #     os.system("! mv     3836680545_2     ccb331621_z.jpg     test.jpg")
    #     image = Image.open('./test.jpg')
        path_lsmdc = '../../lsmdc_s1_gt_mdf'
        df_lsmdc = pd.read_csv(os.path.join(path_lsmdc, "lsmdc_meta.csv"), index_col=False)
        # example from colab
        # image = Image.open('../../dataset/caption_data/COCO_train2014_000000292160.jpg')
        # # Construct input sample & preprocess for GPU if cuda available
        # text = "a blue turtle-like pokemon with round head"

        patch_image_size = cfg.task.patch_image_size
    else:
        task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    if not inference_pipeline:
        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(cfg.dataset.gen_subset),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), *[m.max_positions() for m in models]
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=cfg.distributed_training.distributed_world_size,
            shard_id=cfg.distributed_training.distributed_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    print(" #################################### HK ################################################")
    results = []
    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()

    if inference_pipeline:
        path = os.path.join('/opt/share/hanoch/ofa_inference', cfg.dataset.gen_subset)
        if not os.path.exists(path):
            os.makedirs(path)
        # for idx, file in enumerate(tqdm.tqdm(filenames)):
        for idx, row in tqdm.tqdm(df_lsmdc.iterrows()):
            # row = df_lsmdc.loc[(df_lsmdc.movie == 'Movies/114207550')].iloc[0]
            print(type(row)) #row = df_lsmdc.loc[(df_lsmdc.movie == 'Movies/114207550')][df_lsmdc.mdf==85].iloc[0]  ;
            print(row['file'])
            file = str(row['file']) + '.png'
            text = row['groundtruth']
            print(os.path.join(path_lsmdc, file))
            image = Image.open(os.path.join(path_lsmdc, file))
            movie_id = file.split('_')[0]
            mdf = int(file.split('_')[2])
            scene = int(file.split('_')[4].split('.png')[0])

            sample = construct_sample(image, task=task, text1=text,
                                      patch_resize_transform=patch_resize_transform,
                                      patch_image_size=patch_image_size)

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
            with torch.no_grad():
                result, scores, lprob = eval_step(task, generator, models, sample)
                # display(image)
                scores_lin_prob = np.exp(lprob)
                print("SoftMax score of the decoder", lprob, lprob.sum())
                # caption = result[0]['caption']
                # print('Caption: {}'.format(caption))
                caption = text
                window_name = 'Image'
                # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                image = np.array(image)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                normalizedImg = np.zeros_like(img)
                normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
                img = normalizedImg.astype('uint8')

                image = cv2.rectangle(
                    img,
                    (int(result[0]["box"][0]), int(result[0]["box"][1])),
                    (int(result[0]["box"][2]), int(result[0]["box"][3])),
                    (0, 255, 0),
                    3
                )
                # print(caption)
                cv2.imshow(window_name, img)

                cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption + '_ prob_' + str(scores_lin_prob[0].__format__('.3f')))
                cv2.putText(image, row['file'] + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                            lineType=cv2.LINE_AA, org=(10, 40))
                fname = str(row['file']) + '_' + str(caption) + '.png'
                cv2.imwrite(os.path.join(path, fname),
                            image)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

    else:
        path = '/opt/share/hanoch/ofa_vg'
        path = os.path.join(path, cfg.dataset.gen_subset)
        if not os.path.exists(path):
            os.makedirs(path)

        for sample in progress:
            if "net_input" not in sample:
                continue
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
            print("HK: sample", sample.keys())
            # print("sample", sample)
            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample, **kwargs)
                print("result", result[0].keys())
                print("scores", scores)
                print("uniq_id", result[0]["uniq_id"])

                src_tokens = sample['net_input']['src_tokens'][0]
                print("src_tokens", src_tokens)
                src_tokens = utils.strip_pad(src_tokens, task.tgt_dict.pad())
                caption = task.bpe.decode(task.tgt_dict.string(src_tokens))
                print(type(caption))
                image = sample['net_input']['patch_images'][0].permute(1,2,0).cpu().numpy()
                image = image.astype('float32')

                window_name = 'Image'
                # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                normalizedImg = np.zeros_like(img)
                normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
                img = normalizedImg.astype('uint8')

                image = cv2.rectangle(
                    img,
                    (int(result[0]["box"][0]), int(result[0]["box"][1])),
                    (int(result[0]["box"][2]), int(result[0]["box"][3])),
                    (0, 255, 0),
                    3
                )
                print(caption)
                start_theme_caption = caption.find('"')
                end_theme_caption = caption.find('describe?')
                print(start_theme_caption, end_theme_caption)
                caption_trim = caption[start_theme_caption+1:end_theme_caption-2].strip()
                print(caption_trim)

                cv2.imshow(window_name, image)
                cv2.setWindowTitle(window_name, caption_trim)
                cv2.putText(image, caption_trim, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, org=(10, 40))
                cv2.imwrite(os.path.join(path, str(caption_trim) + '.png'), image )#(image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

            results += result
            score_sum += sum(scores) if scores is not None else 0
            score_cnt += len(scores) if scores is not None else 0
            progress.log({"sentences": sample["nsentences"]})

        merge_results(task, cfg, logger, score_cnt, score_sum, results)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    parser.add_argument("--inference-pipeline", action='store_true', help='HK added pipeline inference processing one file by another')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(
        cfg, main, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval, zero_shot=args.zero_shot, inference_pipeline=args.inference_pipeline
    )

if __name__ == "__main__":
    cli_main()
