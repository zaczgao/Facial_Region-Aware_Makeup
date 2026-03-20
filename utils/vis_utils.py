#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
# if sys.platform == 'win32':
#     matplotlib.use('Qt5Agg')

import torch

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


def show_result(x):
    if isinstance(x, torch.Tensor):
        while x.ndim < 4:
            x = x[None]

        x = x.permute(0, 2, 3, 1)[0]
        if x.shape[-1] == 1:
            x = x.expand(-1, -1, 3)
        x = x.detach().cpu().numpy()
    elif isinstance(x, PIL.Image.Image):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        if x.shape[-1] == 1:
            x = np.tile(x, (1, 1, 3))

    plt.imshow(x / 255)
    plt.axis("off")
    plt.show(block=True)


def show_face_result(img_bgr, bbox=None, lms=None, lms68=None, thickness=3):
    color = (0, 0, 255)
    img_vis = np.copy(img_bgr)

    if bbox is not None:
        cv2.rectangle(img_vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)

    if lms is not None:
        for pt in lms:
            cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 1, color, thickness)

    if lms68 is not None:
        for pt in lms68:
            cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), thickness)

    show_result(img_vis[:, :, ::-1])


def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = PIL.Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)


if __name__ == '__main__':
    pass
