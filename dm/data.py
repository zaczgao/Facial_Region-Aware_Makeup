#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import math
import random
import numpy as np
import cv2
import PIL.Image
import albumentations as A
import imgaug.augmenters as iaa
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import GaussianNoise

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from dm.const import imagenet_templates_small
from style_clip.augment import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from utils.vis_utils import show_result


class DMTransformations(object):
    def __init__(self, resolution, interpolation):
        self.transforms_resize = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=interpolation),  # Use dynamic interpolation method
                # transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                # transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            ]
        )

        self.transforms_post = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.transforms_conditioning = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=interpolation),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, img):
        img_resize = self.transforms_resize(img)
        img_post = self.transforms_post(img_resize)

        return img_resize, img_post


def center_crop_arr(pil_image, image_size, seg_mask=None, cond_image=None):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    img_roi = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if seg_mask is not None:
        if seg_mask.ndim == 2:
            seg_mask = seg_mask.unsqueeze(0)

        seg_mask = TF.resize(seg_mask, arr.shape[:2], interpolation=transforms.InterpolationMode.NEAREST)
        seg_mask = seg_mask[:, crop_y: crop_y + image_size, crop_x: crop_x + image_size].squeeze(0)

    cond_image_roi = []
    if cond_image is not None:
        for img in cond_image:
            img = np.array(img.resize(arr.shape[:2][::-1], resample=PIL.Image.BICUBIC))
            cond_image_roi.append(img[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    return img_roi, seg_mask, cond_image_roi


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, seg_mask=None, cond_image=None):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    img_roi = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if seg_mask is not None:
        if seg_mask.ndim == 2:
            seg_mask = seg_mask.unsqueeze(0)

        seg_mask = TF.resize(seg_mask, arr.shape[:2], interpolation=transforms.InterpolationMode.NEAREST)
        seg_mask = seg_mask[:, crop_y: crop_y + image_size, crop_x: crop_x + image_size].squeeze(0)

    cond_image_roi = []
    if cond_image is not None:
        for img in cond_image:
            img = np.array(img.resize(arr.shape[:2][::-1], resample=PIL.Image.BICUBIC))
            cond_image_roi.append(img[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    return img_roi, seg_mask, cond_image_roi


class MakeupDataset(Dataset):
    def __init__(self, data_root, resolution, center_crop, random_flip,
                 tokenizer, base_prompt, use_templates, vector_shuffle, drop_tokens, drop_tokens_rate,
                 swap_pair_rate, drop_cond_rate, skip_background, num_parts, geo_mode):
        self.data_root = data_root
        self.resolution = resolution
        self.random_crop = not center_crop
        self.random_flip = random_flip

        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.use_templates = use_templates
        self.vector_shuffle = vector_shuffle
        self.drop_tokens = drop_tokens
        self.drop_tokens_rate = drop_tokens_rate
        self.swap_pair_rate = swap_pair_rate
        self.drop_text_rate = drop_cond_rate[0]
        self.drop_style_rate = drop_cond_rate[1]
        self.drop_all_rate = drop_cond_rate[2]
        self.skip_background = skip_background
        self.num_parts = num_parts
        self.geo_mode = geo_mode

        self.label_names = ['background', 'face', 'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip', 'hair']
        self.label_group = [
            ['background', 'hair', 'imouth'],
            ['face'],
            ['rb', 'lb', 're', 'le'],
            ['nose'],
            ['ulip', 'llip']
        ]

        additional_targets = {'image_id': 'image', 'image_pose': 'masks', 'seg_mask': 'masks',
                              'face_mask': 'mask', 'exp_mask': 'mask'}
        flip_prob = 0.5 if random_flip else 0.0
        self.transforms_random = A.Compose([
            A.SmallestMaxSize(max_size=resolution, interpolation=cv2.INTER_LANCZOS4),
            A.CenterCrop(resolution, resolution) if center_crop else A.RandomCrop(height=resolution, width=resolution),
            A.HorizontalFlip(p=flip_prob),
            A.Affine(scale=(1.0, 1.0), translate_percent=(-0.1, 0.1), rotate=(-10, 10), interpolation=cv2.INTER_LANCZOS4, p=0.5),
        ], additional_targets=additional_targets
        )

        self.transforms_post = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transforms_post_cond = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # X-NEMO
        self.deform = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.02, 0.04), nb_rows=(3, 4), nb_cols=(3, 4))
        ])

        self.transforms_style = transforms.Compose([
            # transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.7, 1.3),
                                                            interpolation=transforms.InterpolationMode.BICUBIC)],
                                   p=1.0),
            transforms.ElasticTransform(alpha=100.),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomChoice([
                    GaussianNoise(mean=0.0, sigma=0.03),
                    transforms.GaussianBlur(kernel_size=(3, 3)),
                ])
            ], p=1.0),
            transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])

        self.samples = self.make_dataset(data_root)

    def make_dataset(self, data_root):
        root_sub_folder = os.listdir(data_root)
        if "id" in root_sub_folder:
            data_root = [data_root]
        else:
            data_root = [os.path.join(data_root, folder) for folder in root_sub_folder]

        instances = []
        for directory in data_root:
            img_id_root = os.path.join(directory, "id")
            img_id_path_list = glob.glob(img_id_root + "/*.png") + glob.glob(img_id_root + "/*/*.png")
            img_id_path_list = sorted(img_id_path_list)

            for img_id_path in img_id_path_list:
                id_name = os.path.splitext(os.path.basename(img_id_path))[0]

                makeup_dir = os.path.join(directory, "makeup_mix", id_name)
                if not os.path.isdir(makeup_dir):
                    continue

                img_makeup_file_list = sorted(os.listdir(makeup_dir))
                for img_makeup_file in img_makeup_file_list:
                    img_makeup_path = os.path.join(makeup_dir, img_makeup_file)
                    instances.append([img_id_path, img_makeup_path])

        return instances

    def preprocess_caption(self, token_idx, is_drop_text):
        if self.use_templates and random.random() < 0.5 and self.base_prompt != "":
            base_prompt = self.base_prompt
            if base_prompt.lower().startswith("a "):
                base_prompt = base_prompt[2:]

            caption = random.choice(imagenet_templates_small).format(base_prompt)
        else:
            caption = f"{self.base_prompt}"

        if is_drop_text:
            caption = ""

        text_input_ids = self.tokenizer(
            caption,
            vector_shuffle=self.vector_shuffle,
            replace_token=True,
            token_idx=token_idx,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return text_input_ids

    def preprocess_image(self, img_makeup: np.ndarray, img_id: np.ndarray, img_pose: np.ndarray, seg_mask, face_mask, exp_mask):
        additional_input = {'image_id': img_id,
                            'image_pose': img_pose.transpose(2, 0, 1),
                            'seg_mask': seg_mask.cpu().numpy().astype(np.uint8),
                            'face_mask': face_mask.cpu().numpy().astype(np.uint8),
                            'exp_mask': exp_mask.cpu().numpy().astype(np.uint8)
                            }

        augmented = self.transforms_random(image=img_makeup, **additional_input)

        img_makeup_aug = augmented["image"]
        img_id_aug = augmented["image_id"]
        img_pose_aug = augmented['image_pose'].transpose(1, 2, 0)
        seg_mask_aug = torch.from_numpy(augmented['seg_mask']).to(torch.float32)
        face_mask_aug = torch.from_numpy(augmented['face_mask']).to(torch.float32)
        exp_mask_aug = torch.from_numpy(augmented['exp_mask']).to(torch.float32)

        return img_makeup_aug, img_id_aug, img_pose_aug, seg_mask_aug, face_mask_aug, exp_mask_aug

    def prep_mask(self, seg_pred, label_names, label_group):
        H, W = seg_pred.shape

        seg_mask_group = torch.zeros((len(label_group), H, W), device=seg_pred.device)
        for group_idx, group in enumerate(label_group):
            for label in group:
                idx = np.where(np.array(label_names) == label)[0][0]
                seg_mask_group[group_idx] += (seg_pred == idx).float()

        bg_label = ['background', 'hair']
        bg_mask = torch.zeros((H, W), device=seg_pred.device)
        for label in bg_label:
            idx = np.where(np.array(label_names) == label)[0][0]
            bg_mask += (seg_pred == idx).float()
        face_mask = 1. - bg_mask

        exp_label = ['rb', 'lb', 're', 'le', 'ulip', 'imouth', 'llip']
        exp_mask = torch.zeros((H, W), device=seg_pred.device)
        for label in exp_label:
            idx = np.where(np.array(label_names) == label)[0][0]
            exp_mask += (seg_pred == idx).float()

        return seg_mask_group, face_mask, exp_mask

    def __getitem__(self, index):
        id_path, makeup_path = self.samples[index]
        img_id_pil = PIL.Image.open(id_path).convert("RGB")
        img_makeup_pil = PIL.Image.open(makeup_path).convert("RGB")
        assert img_id_pil.size == img_makeup_pil.size

        if random.random() < self.swap_pair_rate:
            img_id_pil_tmp = img_id_pil
            img_id_pil = img_makeup_pil
            img_makeup_pil = img_id_pil_tmp

        data_root = os.path.join(os.path.dirname(makeup_path), "../../")
        data_file_name = os.path.splitext(os.path.basename(id_path))[0]
        data_folder = id_path.split(os.sep)[-2]
        if data_folder == "id":
            data_folder = ""

        if self.geo_mode == "3d":
            pose_path = os.path.join(data_root, "3d", data_folder, "{}.png".format(data_file_name))
        elif self.geo_mode == "keypoint":
            pose_path = os.path.join(data_root, "pose", data_folder, "{}.png".format(data_file_name))
        img_pose_pil = PIL.Image.open(pose_path).convert("RGB")
        assert img_id_pil.size == img_pose_pil.size

        # lms68 = np.load(os.path.join(data_root, "lms68", data_folder, "{}.npy".format(data_file_name)))

        mask_path = os.path.join(data_root, "mask", data_folder, "{}.pt".format(data_file_name))
        mask_dict = torch.load(mask_path, map_location="cpu")
        seg_pred = mask_dict["seg_pred"].to(torch.float32)
        seg_mask, face_mask, exp_mask = self.prep_mask(seg_pred, self.label_names, self.label_group)
        if self.skip_background:
            seg_mask = seg_mask[1:]
        token_idx = (seg_mask.sum(dim=(1, 2)) >= 1).nonzero(as_tuple=True)[0]
        assert seg_mask.shape[0] == self.num_parts
        assert img_id_pil.size[1] == seg_mask.shape[1] and img_id_pil.size[0] == seg_mask.shape[2], f"{id_path}, \t, {makeup_path}"

        # face_emb = np.load(os.path.join(data_root, "face_emb", data_folder, "{}.npy".format(data_file_name)))
        # face_emb = torch.from_numpy(face_emb).reshape([1, -1])

        # # https://github.com/sail-sg/MDT
        # if self.random_crop:
        # 	img_makeup_np, seg_mask, img_cond_np = random_crop_arr(img_makeup_pil, self.resolution, seg_mask=seg_mask,
        # 	                                                cond_image=[img_id_pil, img_pose_pil])
        # else:
        # 	img_makeup_np, seg_mask, img_cond_np = center_crop_arr(img_makeup_pil, self.resolution, seg_mask=seg_mask,
        # 	                                                cond_image=[img_id_pil, img_pose_pil])
        #
        # img_id_np, img_pose_np = img_cond_np
        #
        # if self.random_flip and random.random() < 0.5:
        # 	img_makeup_np = img_makeup_np[:, ::-1]
        # 	seg_mask = TF.hflip(seg_mask)
        # 	img_id_np = img_id_np[:, ::-1]
        # 	img_pose_np = img_pose_np[:, ::-1]

        img_makeup, img_id, img_pose, seg_mask, face_mask, exp_mask = self.preprocess_image(
            np.array(img_makeup_pil),
            np.array(img_id_pil),
            np.array(img_pose_pil),
            seg_mask,
            face_mask,
            exp_mask)

        img_makeup = self.transforms_post(img_makeup)
        img_id = self.transforms_post_cond(img_id)
        img_pose = self.transforms_post_cond(img_pose)

        # img_style = self.deform(image=np.array(img_makeup_pil))
        # img_style = self.transforms_style(PIL.Image.fromarray(img_style))
        img_style = self.transforms_style(img_makeup_pil)

        # set cfg drop rate
        is_drop_text = False
        is_drop_style = False
        rand_num = random.random()
        if rand_num < self.drop_text_rate:
            is_drop_text = True
        elif rand_num < (self.drop_text_rate + self.drop_style_rate):
            is_drop_style = True
        elif rand_num < (self.drop_text_rate + self.drop_style_rate + self.drop_all_rate):
            is_drop_text = True
            is_drop_style = True

        text_input_ids = self.preprocess_caption(token_idx.tolist(), is_drop_text)

        return {
            "pixel_values": img_makeup,
            "style_pixel_values": img_style,
            "id_pixel_values": img_id,
            "pose_pixel_values": img_pose,
            "seg_mask": seg_mask,
            "face_mask": face_mask,
            "exp_mask": exp_mask,
            "input_ids": text_input_ids,
            # "face_emb": face_emb,
            "is_drop_style": torch.tensor(is_drop_style)
        }

    def __len__(self):
        return len(self.samples)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    style_pixel_values = torch.stack([example["style_pixel_values"] for example in examples])
    style_pixel_values = style_pixel_values.to(memory_format=torch.contiguous_format).float()
    id_pixel_values = torch.stack([example["id_pixel_values"] for example in examples])
    id_pixel_values = id_pixel_values.to(memory_format=torch.contiguous_format).float()
    pose_pixel_values = torch.stack([example["pose_pixel_values"] for example in examples])
    pose_pixel_values = pose_pixel_values.to(memory_format=torch.contiguous_format).float()
    seg_mask = torch.stack([example["seg_mask"] for example in examples])
    face_mask = torch.stack([example["face_mask"] for example in examples])
    exp_mask = torch.stack([example["exp_mask"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    # face_emb = torch.cat([example["face_emb"] for example in examples])
    is_drop_style = torch.stack([example["is_drop_style"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "style_pixel_values": style_pixel_values,
        "id_pixel_values": id_pixel_values,
        "pose_pixel_values": pose_pixel_values,
        "seg_mask": seg_mask,
        "face_mask": face_mask,
        "exp_mask": exp_mask,
        "input_ids": input_ids,
        # "face_emb": face_emb,
        "is_drop_style": is_drop_style
    }


class SyntheticDataset(MakeupDataset):
    def __init__(self, data_root, resolution, center_crop, random_flip,
                 tokenizer, base_prompt, use_templates, vector_shuffle, drop_tokens, drop_tokens_rate,
                 swap_pair_rate, drop_cond_rate, skip_background, num_parts, geo_mode):
        super().__init__(data_root, data_root, resolution, center_crop, random_flip,
                 tokenizer, base_prompt, use_templates, vector_shuffle, drop_tokens, drop_tokens_rate,
                 swap_pair_rate, drop_cond_rate, skip_background, num_parts, geo_mode)

        self.image = PIL.Image.new('RGB', (self.resolution, self.resolution))
        self.dataset_size = 100

    def __getitem__(self, idx):
        img_id_pil = self.image
        img_makeup_pil = self.image
        img_pose_pil = self.image
        img_warp_pil = self.image

        seg_mask = (torch.randn(self.num_parts, self.resolution, self.resolution) > 0).float()
        face_mask = (torch.randn(self.resolution, self.resolution) > 0).float()
        exp_mask = (torch.randn(self.resolution, self.resolution) > 0).float()
        face_emb = torch.randn(1, 512)

        img_makeup_np, img_id_np, img_pose_np, seg_mask, face_mask, exp_mask = self.preprocess_image(
            np.array(img_makeup_pil),
            np.array(img_id_pil),
            np.array(img_pose_pil),
            seg_mask,
            face_mask,
            exp_mask)

        img_makeup = self.transforms_post(img_makeup_np)
        img_id = self.transforms_post_cond(img_id_np)
        img_pose = self.transforms_post_cond(img_pose_np)

        img_style = self.transforms_style(img_warp_pil)

        token_idx = torch.arange(self.num_parts)
        text_input_ids = self.preprocess_caption(token_idx.tolist(), False)

        return {
            "pixel_values": img_makeup,
            "style_pixel_values": img_style,
            "id_pixel_values": img_id,
            "pose_pixel_values": img_pose,
            "seg_mask": seg_mask,
            "face_mask": face_mask,
            "exp_mask": exp_mask,
            "input_ids": text_input_ids,
            # "face_emb": face_emb,
            "is_drop_style": torch.tensor(False)
        }

    def __len__(self):
        return self.dataset_size


def get_dataset_cls(dataset_type):
    if dataset_type == "makeup":
        return MakeupDataset
    elif dataset_type == "syn":
        return SyntheticDataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


if __name__ == '__main__':
    from dm.multi_token_clip import MultiTokenCLIPTokenizer
    from utils.face_analysis import FaceAnalyser, show_face_result
    from utils.misc import denormalize_batch

    data_dir = "./output/makeup_pair_ffhq"
    # data_dir = "./output/makeup_pair_qwen"
    model_id = "../../pretrain/stable-diffusion-2-1-base"
    num_parts = 4
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer",
    )
    placeholder_token = "<part>"
    # base_prompt = f"person with {placeholder_token} makeup"
    # base_prompt = f"{placeholder_token} person"
    base_prompt = "person with makeup"

    dataset = MakeupDataset(data_dir, 512, center_crop=False, random_flip=True,
                            tokenizer=tokenizer, base_prompt=base_prompt, use_templates=True, vector_shuffle=True, drop_tokens=False, drop_tokens_rate=0.,
                            swap_pair_rate=0.1, drop_cond_rate=[0.05, 0.05, 0.05],
                            skip_background=True, num_parts=num_parts, geo_mode="3d")

    img = PIL.Image.open("./assets/images/00128-img_swap.png")
    img_style = dataset.deform(image=np.array(img))
    img_style = dataset.transforms_style(PIL.Image.fromarray(img_style))
    img_style = denormalize_batch(img_style.unsqueeze(dim=0), OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    img_style = transforms.functional.to_pil_image(img_style[0])
    img_style.save("img_style.png")
    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150, exp_ratio=-1, align=False)
    face_info, is_small_face = face_analyser.get_face_info(img_bgr=np.array(img)[:, :, ::-1], verbose=False)
    face_info[0]["face_3d"].save("3d.png")


    print("dataset size", len(dataset))
    print(dataset[0])

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                                         drop_last=False, collate_fn=collate_fn)

    for i, batch in enumerate(loader):
        img_makeup = denormalize_batch(batch["pixel_values"], 0.5, 0.5)
        img_style = denormalize_batch(batch["style_pixel_values"], OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        img_id = batch["id_pixel_values"]
        img_pose = batch["pose_pixel_values"]
        exp_mask = batch["exp_mask"]

        img_exp = transforms.functional.to_pil_image(img_makeup[0] * exp_mask[0])
        img_makeup = transforms.functional.to_pil_image(img_makeup[0])
        img_style = transforms.functional.to_pil_image(img_style[0])
        img_id = transforms.functional.to_pil_image(img_id[0])
        img_pose = transforms.functional.to_pil_image(img_pose[0])

        img_makeup.save("img_makeup.png")
        img_style.save("img_style.png")
        img_id.save("img_id.png")
        img_pose.save("img_pose.png")
