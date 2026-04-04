#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://www.reddit.com/r/StableDiffusion/comments/16gxwzf/how_do_we_get_different_faces/

flash-attention: build on server sbg node with gpu
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
export CUDAHOSTCXX=$CXX
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.6.3
python setup.py install
"""

__author__ = "GZ"

import os
import sys
import re
import random
import argparse
import glob
import csv
import PIL.Image
import json

import torch

from transformers import CLIPTokenizer

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.data_utils import get_makeup_list, get_img_context, init_sd, init_flux, \
    sample_sd, sample_flux, merge_anno, create_data_split


def create_makeup_style(args):
    makeup_info = get_makeup_list(args.data_path)
    print("{} makeup styles".format(len(makeup_info)))

    print("makeup list start {}, end {}".format(args.start_idx, args.end_idx))
    if args.end_idx == -1:
        args.end_idx = len(makeup_info)

    if args.method == "sd":
        pipe = init_sd()
    elif args.method == "flux":
        pipe = init_flux()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    anno_info = []
    with open("./assets/race_name.json", "r", encoding="utf-8") as f:
        race_info = json.load(f)

    print("negative prompt:", args.prompt_neg)
    for makeup_idx, (category, style, desp) in enumerate(makeup_info):
        if makeup_idx < args.start_idx or makeup_idx >= args.end_idx:
            continue

        print("makeup:", makeup_idx, style)

        for race_idx, race in enumerate(race_info.keys()):
            for img_idx in range(args.num_img_race):
                region = random.choice(list(race_info[race].keys()))
                random_name = random.choice(race_info[race][region])

                context, age = get_img_context()

                prompt_base = "Realistic close-up photography of {}, a {} {} woman with makeup. The makeup is {} style, {}".format(random_name, age, region, style, desp)
                prompt = f"{prompt_base} {context}."
                print("prompt:", prompt)

                tokens = tokenizer(prompt)["input_ids"]
                if len(tokens) > 77:
                    print("exceeds 77 tokens and will be truncated!")
                assert len(tokens) <= 128, "{} exceeds 128 tokens and will be truncated!".format(prompt)

                if args.method == "sd":
                    image = sample_sd(pipe, prompt, args.prompt_neg)
                elif args.method == "flux":
                    image = sample_flux(pipe, prompt)

                class_str = "{:03d}".format(makeup_idx)
                out_dir_img = os.path.join(args.out_dir, "image", class_str)
                os.makedirs(out_dir_img, exist_ok=True)
                out_file = "{}-{}-{}-{}".format(class_str, style.replace(" ", "_"), "_".join(race.split(" ")),
                                                "{:03d}.png".format(img_idx))
                image = image.resize((args.img_size, args.img_size), resample=PIL.Image.Resampling.LANCZOS)
                image.save(os.path.join(out_dir_img, out_file))

                anno_info.append([out_file, makeup_idx, prompt])

    anno_path = os.path.join(args.out_dir, "{}-{:02d}_{:02d}.csv".format(os.path.basename(args.out_dir),
                                                                                    args.start_idx, args.end_idx))
    with open(anno_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "caption"])
        for info in anno_info:
            buf = [info[0], str(info[1]), info[2]]
            writer.writerow(buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument("--data_path", type=str, default="./assets/makeup_gpto3.json", help="path to dataset")
    parser.add_argument("--out_dir", type=str, default="./output/makeup_style", help="output folder")
    parser.add_argument("--method", type=str, default="sd", help="sd model")
    parser.add_argument("--prompt_neg", type=str, default="", help="negative prompt")
    parser.add_argument("--start_idx", type=int, default=0, help="")
    parser.add_argument("--end_idx", type=int, default=-1, help="")
    parser.add_argument("--num_img_race", type=int, default=100, help="")
    parser.add_argument("--img_size", type=int, default=512, help="")
    args = parser.parse_args()

    if not args.prompt_neg:
        # same face, repetition, duplicate identity, low diversity,
        args.prompt_neg = """multiple people, multiple faces, blurry, out of focus, low quality, low resolution, 
		side view, profile view, turned head, half-face, cropped face, 
		asymmetrical face, distorted face"""

        args.prompt_neg = re.sub(r"[\t\n]", "", args.prompt_neg)

    print(args)

    # create_makeup_style(args)

    merge_anno_path = merge_anno(args.out_dir)
    create_data_split(merge_anno_path)
