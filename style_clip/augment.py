#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import numpy as np
import PIL.Image
import imgaug.augmenters as iaa

import torch
import torchvision.transforms as transforms
from torchvision.transforms.v2 import GaussianNoise

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.tps import TPSDeform

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def get_clip_transform(size=224, mean=None, std=None):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    normalize = transforms.Normalize(mean=mean, std=std)

    preprocess = transforms.Compose([
        transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocess


class ContrastiveTransformations(object):

    def __init__(self, size=224, mean=None, std=None, lambda_c=0):
        mean = mean or OPENAI_DATASET_MEAN
        std = std or OPENAI_DATASET_STD
        normalize = transforms.Normalize(mean=mean, std=std)

        transforms_branch0 = transforms.Compose([
            transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

        # csd
        transforms_branch1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(p=0.3),
            # transforms.RandomRotation(degrees=np.random.choice([0, 90, 180, 270])),
            transforms.RandomApply([transforms.RandomAffine(30, translate=(0.2, 0.2), scale=(0.7, 1.3))],
                                   p=1.),
            transforms.ElasticTransform(alpha=50.),
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomChoice([
                    GaussianNoise(mean=0.0, sigma=0.1),
                    transforms.GaussianBlur(kernel_size=(5, 5)),
                ])
            ], p=0.5),
            normalize,
        ])

        transforms_branch2 = transforms.Compose([
            # transforms.RandomResizedCrop(size=size, scale=(0.5, 1.), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                           saturation=0.5, hue=0.1)
                                    ], p=0.6),
            transforms.RandomApply([transforms.RandomInvert(), transforms.RandomGrayscale(),
                                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 4))], p=0.8),
            # GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            normalize
        ])

        self.transforms_b0 = transforms_branch1 if lambda_c < 1e-3 else transforms_branch0
        self.transforms_b1 = transforms_branch1
        self.transforms_b2 = transforms_branch2

        # self.deform = TPSDeform(device='cpu')
        # self.deform.fit_tps(1024, 1024, np.array([100, 100, 900, 900]))
        # X-NEMO
        self.deform = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.02, 0.04), nb_rows=(3, 4), nb_cols=(3, 4))
        ])

    def __call__(self, x):
        # sample = self.deform(sample, None, skip_fit=True)
        x1 = self.transforms_b0(PIL.Image.fromarray(self.deform(image=np.array(x))))
        x2 = self.transforms_b1(x)
        x3 = self.transforms_b2(x)
        return [x1, x2, x3]


if __name__ == '__main__':
    pass
