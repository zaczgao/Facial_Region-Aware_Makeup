#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

__author__ = "GZ"

import os
import sys
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


def load_image(image_path, height=None, width=None, interpolate=PIL.Image.Resampling.LANCZOS):
    if type(image_path) is str:
        image = PIL.Image.open(image_path).convert("RGB")
        if height is not None:
            image = image.resize((width, height), resample=interpolate)
    else:
        image = image_path

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


def set_trainable_modules_to_train(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            set_trainable_modules_to_train(module)
        else:
            params = list(module.parameters())
            has_trainable_params = (len(params) > 0) and all([p.requires_grad for p in params])
            is_behavior_dependent = isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

            if has_trainable_params or is_behavior_dependent:
                module.train()

    return model


def compare_model_params(model_a, model_b, tol=0.0):
    assert id(model_a) != id(model_b)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    for k in sd_a.keys():
        kb = k.replace("default.weight", "default_0.weight")
        if not torch.equal(sd_a[k], sd_b[kb]):
            diff = torch.max(torch.abs(sd_a[k] - sd_b[kb])).item()
            if diff > tol:
                print(f"Parameter '{k}' differs (max abs diff = {diff:.2e})")
    print("All parameters are checked.")


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
