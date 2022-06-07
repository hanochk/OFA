# -*- coding: utf-8 -*-
import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import transforms

# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False


class OfaMultiModalVisualGrounding():

    def __init__(self):
        # Register caption task
        # tasks.register_task('caption',CaptionTask)

        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        use_fp16 = False

        bpe_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./../utils/BPE")
        # Load pretrained ckpt & config
        overrides = {"bpe_dir": "../../utils/BPE"}
        self.models, cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('../../checkpoints/refcoco_large_best.pt'),
            arg_overrides=overrides
        )

        cfg.common.seed = 7
        cfg.generation.beam = 5
        cfg.generation.min_len = 4
        cfg.generation.max_len_a = 0
        cfg.generation.max_len_b = 4
        cfg.generation.no_repeat_ngram_size = 3

        # Fix seed for stochastic decoding
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        self.cfg = cfg
        # Move models to GPU
        for model in self.models:
            model.eval()
            if use_fp16:
                model.half()
            if use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()

    def _encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

        # Construct input for refcoco task

    def construct_sample(self, image: Image, text: str):
        patch_image_size = self.cfg.task.patch_image_size
        w, h = image.size
        w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
        h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self._encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True,
                               append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": torch.randn(1, 4)
        }
        return sample

        # Function to turn FP32 to FP16
    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def find_visual_grounding(self, image, text):
        # Run eval step for caption
        # Run eval step for refcoco
        # Construct input sample & preprocess for GPU if cuda available
        sample = self.construct_sample(image, text)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if use_fp16 else sample

        with torch.no_grad():
            result, scores, lprob = eval_step(self.task, self.generator, self.models, sample)
            return result, scores, lprob


if __name__ == '__main__':
    ofa_vg = OfaMultiModalVisualGrounding()

    image = Image.open('../../pokemon.jpg')
    text = "a blue turtle-like pokemon with round head"


    result, scores, lprob = ofa_vg.find_visual_grounding(image, text)



    scores_lin_prob = np.exp(lprob)
    print("SoftMax score of the decoder", lprob, lprob.sum())
    caption = text
    print('Caption: {}'.format(caption))
    window_name = 'Image'
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
    movie_id = '111'
    mdf = '-1'
    path = '../../results'
    file = 'pokemon'
    cv2.imshow(window_name, img)

    cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption + '_ prob_' + str(
        scores_lin_prob[0].__format__('.3f')))
    cv2.putText(image, file + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA, org=(10, 40))
    fname = str(file) + '_' + str(caption) + '.png'
    cv2.imwrite(os.path.join(path, fname),
                image)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

