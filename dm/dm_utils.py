#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import random

import torch
import torch.nn.functional as F

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


def downsample_corrupt(model_input, downsample_min_corrupt_ratio, downsample_max_corrupt_ratio):
    corrupt_ratio = random.uniform(downsample_min_corrupt_ratio, downsample_max_corrupt_ratio)

    is_5d = model_input.ndim == 5

    if is_5d:
        B, C, T, H, W = model_input.shape
        model_input = model_input.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = model_input.shape

    h0, w0 = model_input.shape[-2:]

    h1 = max(1, int(round(h0 * corrupt_ratio)))
    w1 = max(1, int(round(w0 * corrupt_ratio)))

    model_input = F.interpolate(model_input, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)

    model_input = F.interpolate(model_input, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    if is_5d:
        model_input = model_input.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return model_input


def noise_corrupt(model_input, corrupt_ratio=1 / 3, num_frames=None, is_frame_independent=False):
    batch_size = model_input.shape[0]

    if is_frame_independent:
        noise_sigma_shape = (batch_size, 1, num_frames)
    else:
        noise_sigma_shape = (batch_size,)
    noise_sigma = (
        torch.rand(size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype) * corrupt_ratio
    )
    while len(noise_sigma.shape) < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)

    result = noise_sigma * torch.randn_like(model_input) + (1 - noise_sigma) * model_input

    return result


def corrupt_latent(model_input, corrupt_prob):
    if model_input.ndim < 3:
        raise ValueError("model_input must have at least 3 dimensions")

    if not (sum(corrupt_prob) < 1):
        raise ValueError("corrupt_prob must be in [0, 1)")

    batch_size = model_input.shape[0]
    corrupt_p_down, corrupt_p_noise = corrupt_prob

    rand = torch.rand(batch_size, device=model_input.device)
    mask_noise = (rand >= corrupt_p_down) & (rand < corrupt_p_down + corrupt_p_noise)

    result_noise = noise_corrupt(model_input, corrupt_ratio=0.1)

    view_shape = [batch_size] + [1] * (model_input.ndim - 1)
    mask_noise = mask_noise.view(*view_shape)

    result = torch.where(mask_noise, result_noise, model_input)

    return result


if __name__ == '__main__':
    main()
