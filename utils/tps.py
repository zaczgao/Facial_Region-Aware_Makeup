#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://github.com/raphaelreme/torch-tps/blob/main/example/image_warping.py
"""

__author__ = "GZ"

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import PIL.Image

import torch
import torchvision
from torch_tps import ThinPlateSpline

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.vis_utils import show_result


def random_ctrl_points(xy_min, xy_max, scale, n_points, height, width):
    """Generate random ctrl points

    (In proportion of the desired shapes)
    """
    # input_ctrl_y = xy_min[1] + (xy_max[1] - xy_min[1]) * torch.rand(n_points, 1)
    # input_ctrl_x = xy_min[0] + (xy_max[0] - xy_min[0]) * torch.rand(n_points, 1)
    # input_ctrl = torch.cat([input_ctrl_y, input_ctrl_x], dim=1)

    roi_width, roi_height = xy_max - xy_min

    input_ctrl_y = torch.linspace(xy_min[1], xy_max[1], steps=n_points)
    input_ctrl_x = torch.linspace(xy_min[0], xy_max[0], steps=n_points)
    XX, YY = torch.meshgrid(input_ctrl_x, input_ctrl_y, indexing='xy')
    input_ctrl = torch.cat((YY[..., None], XX[..., None]), dim=-1)
    input_ctrl = input_ctrl.reshape(-1, 2)

    output_ctrl = input_ctrl + torch.randn(n_points**2, 2) * scale * torch.tensor((roi_height, roi_width))

    corners = torch.tensor(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    corners *= torch.tensor((height, width))

    return torch.cat((input_ctrl, corners)), torch.cat((output_ctrl, corners))


class TPSDeform:
    def __init__(self, scale=0.05, n_points=4, device='cuda'):
        self.scale = scale
        self.n_points = n_points
        self.device = device
        self.size = None
        self.input_indices = None

        self.tps = ThinPlateSpline(0.5, device=device)

    def fit_tps(self, height, width, roi: np.ndarray):
        self.size = torch.tensor((height, width), device=self.device)

        # x, y
        if roi.size > 4:
            xy_min = np.min(roi, axis=0)
            xy_max = np.max(roi, axis=0)
        else:
            xy_min, xy_max = roi[:2], roi[2:]
        input_ctrl, output_ctrl = random_ctrl_points(xy_min, xy_max, scale=self.scale, n_points=self.n_points,
                                                     height=height, width=width)

        # Fit the thin plate spline from output to input
        self.tps.fit(output_ctrl, input_ctrl)

        # Create the 2d meshgrid of indices for output image
        i = torch.arange(height, dtype=torch.float32)
        j = torch.arange(width, dtype=torch.float32)

        ii, jj = torch.meshgrid(i, j, indexing="ij")
        output_indices = torch.cat((ii[..., None], jj[..., None]), dim=-1)  # Shape (H, W, 2)

        # Transform it into the input indices
        self.input_indices = self.tps.transform(output_indices.reshape(-1, 2)).reshape(height, width, 2)

    def __call__(self, image: PIL.Image.Image, roi: np.ndarray, skip_fit=False, verbose=False):
        width, height = image.size

        if not skip_fit:
            self.fit_tps(height, width, roi)

        assert torch.equal(self.size, torch.tensor((height, width), device=self.device))

        # Interpolate the resulting image
        grid = 2 * self.input_indices / self.size - 1  # Into [-1, 1]
        grid = torch.flip(grid, (-1,))  # Grid sample works with x,y coordinates, not i, j
        torch_image = torch.tensor(np.array(image), dtype=torch.float32, device=self.device).permute(2, 0, 1)[None, ...]
        img_warped = torch.nn.functional.grid_sample(torch_image, grid[None, ...], align_corners=False)[0]
        img_warped = torchvision.transforms.functional.to_pil_image(img_warped / 255.)

        if verbose:
            show_result(np.hstack((np.array(image), np.array(img_warped))))

        return img_warped


if __name__ == '__main__':
    import cv2
    from utils.face_analysis import FaceAnalyser

    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150, align=False)

    tps_deform = TPSDeform()

    img_bgr = cv2.imread("./assets/images/tps_test.png", cv2.IMREAD_COLOR)
    face_info, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr, verbose=True)
    img_warped = tps_deform(PIL.Image.fromarray(img_bgr[:, :, ::-1]),
                                    face_info[0]["landmark_3d_68"][:, :2], verbose=True)
