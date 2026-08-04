#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

__author__ = "GZ"

import os
import sys
import glob
import random
import numpy as np
import gc
import functools
import PIL.Image
import cv2

import torch
import torch.nn as nn

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.vis_utils import show_result


IMG_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"
}


def is_image_file(file):
    return os.path.splitext(file)[1].lower() in IMG_EXTENSIONS


def find_all_image(data_dir):
    img_path_list = []

    for file in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True):
        if os.path.isfile(file) and is_image_file(file):
            img_path_list.append(file)

    return img_path_list


def load_image(image_path, height=None, width=None, interpolate=PIL.Image.Resampling.LANCZOS, mode="RGB"):
    if isinstance(image_path, str):
        image = PIL.Image.open(image_path).convert(mode)
    elif isinstance(image_path, PIL.Image.Image):
        image = image_path.convert(mode)
    else:
        raise TypeError(f"Unsupported image input: {type(image_path)!r}")

    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be provided together")
        image = image.resize((width, height), resample=interpolate)

    return image


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_batch(batch, mean, std):
    """denormalize for visualization"""
    dtype = batch.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
    std = torch.as_tensor(std, dtype=dtype, device=batch.device)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    batch = batch * std + mean
    return batch


def imgtensor2numpy(image):
    image_np = np.clip(image.cpu().numpy(), 0, 1)  # [0, 1]
    image_np = image_np.transpose(0, 2, 3, 1)
    image_np = (image_np * 255).astype(np.uint8)

    return image_np


# def set_trainable_modules_to_train(model):
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             set_trainable_modules_to_train(module)
#         else:
#             params = list(module.parameters())
#             has_trainable_params = (len(params) > 0) and all([p.requires_grad for p in params])
#             is_behavior_dependent = isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
#
#             if has_trainable_params or is_behavior_dependent:
#                 module.train()
#
#     return model

def set_trainable_modules_to_train(model):
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue

        params = list(module.parameters(recurse=False))
        has_trainable_params = (len(params) > 0) and all([p.requires_grad for p in params])
        is_behavior_dependent = isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

        if has_trainable_params or is_behavior_dependent:
            module.train()


def save_trainable_param(model, out_path):
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    buffer_names = [n for n, _ in model.named_buffers()]

    save_names = trainable_param_names + buffer_names

    save_state = {
        k: v for k, v in model.state_dict().items()
        if k in save_names
    }
    assert len(save_names) == len(save_state)

    if len(save_state) > 0:
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(save_state, out_path)


def compare_model_param(model_1, model_2, rtol=1e-05, atol=1e-04):
    assert id(model_1) != id(model_2)

    state_1 = model_1.state_dict()
    state_2 = model_2.state_dict()

    mismatches = []

    for key1 in state_1.keys():
        key2 = key1

        t1 = state_1[key1]
        t2 = state_2[key2]

        if t1.shape != t2.shape:
            mismatches.append(f"Mismatch shape: {key1} {t1.shape} vs {key2} {t2.shape}")
            continue

        try:
            torch.testing.assert_close(t1, t2, rtol=rtol, atol=atol)
        except AssertionError:
            max_diff = (t1 - t2).abs().max().item()
            mismatches.append(f"Mismatch: {key1}/{key2} (max abs diff={max_diff:.2e})")

    if len(mismatches) > 0:
        print("Found mismatches:")
        for m in mismatches:
            print("  ", m)
        # raise AssertionError(f"{len(mismatches)} parameters mismatched.")
    else:
        print("All parameters match.")


def gpu_mem_profile(func):
    """Decorator to log GPU memory before/after a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        out = func(*args, **kwargs)  # run your function

        gc.collect()
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()

        print(
            f"[{func.__name__}] "
            f"Allocated: {allocated_before/1024**2:.2f}MB -> {allocated_after/1024**2:.2f}MB | "
            f"Reserved: {reserved_before/1024**2:.2f}MB -> {reserved_after/1024**2:.2f}MB"
        )
        return out
    return wrapper


if __name__ == '__main__':
    pass
