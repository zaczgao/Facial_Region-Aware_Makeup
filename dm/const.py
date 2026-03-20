#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


# https://github.com/huggingface/diffusers/blob/main/examples/research_projects/multi_token_textual_inversion/textual_inversion.py
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    # "a photo of a clean {}",
    # "a photo of a dirty {}",
    "a dark photo of the {}",
    # "a photo of my {}",
    # "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    # "a photo of the clean {}",
    "a rendition of a {}",
    # "a photo of a nice {}",
    "a good photo of a {}",
    # "a photo of the nice {}",
    # "a photo of the small {}",
    # "a photo of the weird {}",
    # "a photo of the large {}",
    # "a photo of a cool {}",
    # "a photo of a small {}",
]


if __name__ == '__main__':
	pass
