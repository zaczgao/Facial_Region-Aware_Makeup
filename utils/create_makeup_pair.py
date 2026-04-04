#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LCM-Lookahead for Encoder-based Text-to-Image Personalization
"""

__author__ = "GZ"

import os
import sys
import re
import argparse
import numpy as np
import json
import cv2
import PIL.Image
import glob
import shutil
import csv
import time

import torch

try:
    from openai import OpenAI
except ImportError:
    print("openai not found")

try:
    from google import genai
except ImportError:
    print("gemini not found")

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.face_analysis import FaceAnalyser, FaceParser
from utils.data_utils import OPENAI_API_KEY, get_makeup_list, get_img_context, \
    init_flux, init_flux2, init_qwen_t2i, init_kontext, init_qwen_edit, \
    sample_flux, sample_flux2, sample_qwen_t2i, sample_kontext, sample_qwen_edit, \
    sample_gpt_t2i, sample_gpt_edit, add_alpha, encode_pil_image, sample_gpt_text, \
    sample_gemini_text, \
    merge_anno, create_data_split
from utils.face_swap import FaceSwap_Wrapper, blend_image_mask
from utils.vis_utils import show_result


def get_bbox_mask(bbox, height, width, thickness):
    mask_out = np.zeros((height, width, 1), dtype=np.float32)
    mask_in = np.zeros((height, width, 1), dtype=np.float32)

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)
    mask_out[y1:y2, x1:x2] = 1

    x1_in = x1 + thickness
    y1_in = y1 + thickness
    x2_in = x2 - thickness
    y2_in = y2 - thickness
    mask_in[y1_in:y2_in, x1_in:x2_in] = 1

    mask = mask_out - mask_in

    return mask


def add_bbox(img: np.ndarray, bbox, thickness):
    height, width = img.shape[:2]

    img_color = np.full(img.shape, (255, 0, 0), dtype=np.uint8)

    mask = get_bbox_mask(bbox, height, width, thickness)

    img_new = img * (1. - mask) + img_color * mask
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)

    return img_new


def remove_bbox(img_ori: np.ndarray, img_edit: np.ndarray, bbox, thickness):
    height, width = img_ori.shape[:2]

    mask = get_bbox_mask(bbox, height, width, thickness)

    img_new = img_edit * (1. - mask) + img_ori * mask
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)

    return img_new


def create_makeup_pair(args, img_id_path_list):
    verbose = True if sys.platform == 'win32' else False

    print("id start {}, end {}".format(args.start_idx, args.end_idx))
    if args.end_idx == -1:
        args.end_idx = len(img_id_path_list)

    makeup_info = get_makeup_list(args.data_makeup_path)
    makeup_filter_offset = 0
    if args.filter_makeup:
        makeup_info_subset = [info for info in makeup_info if info[0] not in ["Natural"]]
        makeup_filter_offset = len(makeup_info) - len(makeup_info_subset)
        makeup_info = makeup_info_subset

    # face_analyser = FaceAnalyser(det_thresh=0.5, min_h=500, min_w=500, align=False, td_mode="")
    # face_parser = FaceParser()

    if args.edit_method == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif args.edit_method == "kontext":
        pipe = init_kontext()
    elif args.edit_method == "flux2":
        pipe = init_flux2()
    elif args.edit_method == "qwenedit":
        pipe = init_qwen_edit()

    makeup_dir = os.path.join(args.out_dir, "makeup")
    os.makedirs(makeup_dir, exist_ok=True)
    mask_raw_dir = os.path.join(args.out_dir, "mask")
    os.makedirs(mask_raw_dir, exist_ok=True)

    prompt_base = "Add makeup to the person while keeping the original identity, facial features, expression and hairstyle. The makeup is {} Maintain the original photographic quality, lighting, position, background, camera angle, framing, and perspective."

    anno_info = []

    for img_id_idx, img_id_path in enumerate(img_id_path_list):
        if img_id_idx < args.start_idx or img_id_idx >= args.end_idx:
            continue

        img_id = PIL.Image.open(img_id_path).convert('RGB')
        img_id = img_id.resize((args.img_size, args.img_size), resample=PIL.Image.Resampling.LANCZOS)
        img_id_name = os.path.splitext(os.path.basename(img_id_path))[0]

        # face_info_id, pick_idx_id = face_analyser.get_face_info(img_bgr=np.array(img_id)[:, :, ::-1], verbose=verbose)
        # thickness = 5
        # img_id = add_bbox(img_id, face_info_id[pick_idx_id]["bbox"], thickness)

        out_dir_img = os.path.join(makeup_dir, img_id_name)
        os.makedirs(out_dir_img, exist_ok=True)
        if args.skip_exist_id and len(os.listdir(out_dir_img)) > 0:
            print("skip existed id:", img_id_idx)
            continue
        print("process id:", img_id_idx, img_id_path)

        if args.num_makeup > len(makeup_info):
            args.num_makeup = len(makeup_info)
        subset_idx = np.random.choice(len(makeup_info), size=args.num_makeup, replace=False)
        makeup_info_subset = [makeup_info[i] for i in subset_idx]
        for makeup_idx, (category, style, desp) in enumerate(makeup_info_subset):
            prompt = prompt_base.format(desp)
            print(prompt)

            for img_idx in range(args.num_img_makeup):
                out_file = "{}-{:03d}-{}-{:02d}.png".format(img_id_name,
                                                            subset_idx[makeup_idx]+makeup_filter_offset,
                                                            style.replace(" ", "_"), img_idx)
                out_path = os.path.join(out_dir_img, out_file)

                if args.edit_method == "openai":
                    mask_alpha_path = os.path.join(mask_raw_dir, "{}_alpha.png".format(img_id_name))
                    if not os.path.exists(mask_alpha_path):
                        _, _, face_mask, pick_idx = face_parser.get_face_seg(np.array(img_id), verbose=verbose)
                        add_alpha(PIL.Image.fromarray(255. * face_mask[pick_idx].cpu().numpy()), mask_alpha_path)

                    img_makeup = sample_gpt_edit(client, img_id_path, mask_alpha_path, prompt, gpt_model="gpt-4.1")
                elif args.edit_method == "kontext":
                    img_makeup = sample_kontext(pipe, img_id, prompt)
                elif args.edit_method == "flux2":
                    img_makeup = sample_flux2(pipe, prompt, img_id)
                elif args.edit_method == "qwenedit":
                    img_makeup = sample_qwen_edit(pipe, img_id, prompt)

                img_makeup.save(out_path)

                anno_info.append([f"{img_id_name}/{out_file}", subset_idx[makeup_idx]+makeup_filter_offset, prompt])

    anno_path = os.path.join(args.out_dir, "{}-{:05d}_{:05d}.csv".format(os.path.basename(args.out_dir),
                                                                         args.start_idx, args.end_idx))
    with open(anno_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "caption"])
        for info in anno_info:
            buf = [info[0], str(info[1]), info[2]]
            writer.writerow(buf)


def get_celeb_list(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        celeb_info = json.load(f)

    celeb_info_list = []
    for category in celeb_info.keys():
        for name in celeb_info[category]:
            celeb_info_list.append([category, name])

    return celeb_info_list


def sample_celeb_id(args):
    print("sampling celeb id images...")

    celeb_info = get_celeb_list(args.data_id_path)

    if args.t2i_method == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif args.t2i_method == "flux2":
        pipe = init_flux2()
    elif args.t2i_method == "qwen":
        pipe = init_qwen_t2i()


    id_dir = os.path.join(args.out_dir, "id")
    os.makedirs(id_dir, exist_ok=True)

    for celeb_idx, (profession, celeb) in enumerate(celeb_info):
        print("celeb:", celeb_idx, celeb)

        for img_idx in range(args.num_img_no_makeup):
            context, _ = get_img_context(use_hair=False, use_face=False, use_expression=True)
            prompt_base = "Realistic close-up photography of {}, no makeup, clean skin, natural skin texture".format(celeb)
            prompt_no_makeup = f"{prompt_base}, {context}."
            print("no makeup prompt:", prompt_no_makeup)

            out_dir_img = os.path.join(id_dir, celeb.replace(" ", "_"))
            os.makedirs(out_dir_img, exist_ok=True)
            out_path = os.path.join(out_dir_img, "{}-{:02d}.png".format(celeb.replace(" ", "_"), img_idx))
            if args.t2i_method == "openai":
                img_id = sample_gpt_t2i(client, prompt_no_makeup, gpt_model="gpt-4.1")
            elif args.t2i_method == "flux2":
                img_id = sample_flux2(pipe, prompt_no_makeup)
            elif args.t2i_method == "qwen":
                img_id = sample_qwen_t2i(pipe, prompt_no_makeup)

            img_id = img_id.resize((args.img_size, args.img_size), resample=PIL.Image.Resampling.LANCZOS)
            img_id.save(out_path)


def read_ffhq_id_list(anno_path):
    img_file_list = []
    if os.path.exists(anno_path):
        with open(anno_path, "r", encoding="utf-8") as f:
            for img_file in f:
                img_file = img_file.strip()
                img_file_list.append(img_file)

    return img_file_list


def create_ffhq_subset(anno_path, num_sample):
    img_file_list = read_ffhq_id_list(anno_path)

    img_file_list = np.random.permutation(img_file_list).tolist()
    img_file_list_subset = img_file_list[:num_sample]
    img_file_list_subset = sorted(img_file_list_subset)

    anno_dir = os.path.dirname(anno_path)
    with open(os.path.join(anno_dir, "ffhq_id-subset.txt"), "w", encoding="utf-8") as f:
        for img_file in img_file_list_subset:
            f.write(img_file + "\n")


def create_ffhq_id_list(data_id_path, data_face_dir, num_id, use_prev=True):
    img_file_list_prev = read_ffhq_id_list(data_id_path)
    img_file_list_prev = sorted(img_file_list_prev)

    if use_prev:
        print(f"Using exsited ffhq id list at {data_id_path}")
        return img_file_list_prev

    print("Creating ffhq id list...")

    # 1024x1024
    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=500, min_w=500, align=False, td_mode="")

    img_path_list = glob.glob(data_face_dir + "/*/*/*.png")
    img_path_list = sorted(img_path_list)

    # with open(os.path.join(data_face_dir, "ffhq-dataset-v2.json"), "r", encoding="utf-8") as f:
    #     anno_info = json.load(f)

    img_file_list_filter = []
    for img_path in img_path_list:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        face_info, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr, verbose=False)

        if len(face_info) == 1:
            pick_idx = face_analyser.find_largest_face(face_info)
            if face_info[pick_idx]["age"] >= 18:
                match = re.search((r'ffhq/(.+)'), img_path)
                img_file = match.group(1)
                img_file_list_filter.append(img_file)

    if num_id > len(img_file_list_filter):
        num_id = len(img_file_list_filter)

    img_file_list_new = []
    if len(img_file_list_prev) < num_id:
        img_file_list_filter = np.random.permutation(img_file_list_filter).tolist()
        count = len(img_file_list_prev)
        for img_file in img_file_list_filter:
            if count >= num_id:
                break

            if img_file not in img_file_list_prev:
                img_file_list_new.append(img_file)
                count += 1

    img_file_list_subset = img_file_list_prev + img_file_list_new
    img_file_list_subset = sorted(img_file_list_subset)

    print("original/filter/prev/subset {}/{}/{}/{} id images".format(len(img_path_list), len(img_file_list_filter),
                                                             len(img_file_list_prev), len(img_file_list_subset)))

    with open(data_id_path, "w", encoding="utf-8") as f:
        for img_file in img_file_list_subset:
            f.write(img_file + "\n")

    return img_file_list_subset


def create_id_list(args):
    id_dir = os.path.join(args.out_dir, "id")
    os.makedirs(id_dir, exist_ok=True)

    if "ffhq" in args.data_id_path:
        img_id_file_list = create_ffhq_id_list(args.data_id_path, args.data_face_dir, args.num_id)

        img_id_path_list = []
        for img_id_file in img_id_file_list:
            img_id_path = os.path.join(id_dir, os.path.basename(img_id_file))
            if not os.path.exists(img_id_path):
                # shutil.copy2(os.path.join(args.data_face_dir, img_id_file), id_dir)
                image = PIL.Image.open(os.path.join(args.data_face_dir, img_id_file))
                image = image.resize((args.img_size, args.img_size), resample=PIL.Image.Resampling.LANCZOS)
                image.save(img_id_path)
            img_id_path_list.append(img_id_path)
    else:
        img_id_path_list = glob.glob(id_dir + "/*/*.png")
        if len(img_id_path_list) == 0:
            sample_celeb_id(args)
            img_id_path_list = glob.glob(id_dir + "/*/*.png")

    assert len(img_id_path_list) > 0
    img_id_path_list = sorted(img_id_path_list)

    print("Found {} id images from {} at {}".format(len(img_id_path_list), args.data_id_path, id_dir))

    return img_id_path_list


class FaceIDFilter():
    def __init__(self, mode="gpt"):
        self.mode = mode
        # https://arxiv.org/pdf/2508.11624
        # OMNIEDIT
        self.prompt = """
        You are an expert in analyzing face images and assessing image quality. Two face images will be provided: 
        the first with little or no makeup, and the second being an edited version of the first with makeup applied. 
        Complete the following tasks.
        
        # Task 1. Image quality
        - Assess whether the second image is a realistic photograph. Consider AI artifacts, anatomy, lighting and texture. Ignore makeup.
            - 0 = not a realistic photograph.
            - 1 = realistic photograph.
        
        # Task 2. Similarity of facial identity and expression
        - Evaluate the similarity of facial identity and facial expression between the two face images. 
        - Consider facial identity and expression, ignore makeup, hairstyle and background. 
        - Provide two separate scores on a scale from 0 to 10:
        - score1: Identity similarity
            - Focus on facial structure (jawline, cheekbones, nose shape), eye shape and eye color, and relative facial proportions
            - Scale
                - 0 = Completely different identities
                - 5 = Some shared features but likely different identities
                - 10 = Same facial features and eye color
        - score2: Expression similarity
            - Focus on overall facial muscle positioning, mouth openness, and eye openness
            - Scale
                - 0 = Completely different facial expressions
                - 5 = Same general emotion but noticeable intensity difference
                - 10 = Same facial expression
        - Output the two scores in a list: [score1, score2].

        # Return your answer strictly in this JSON format. Provide brief but specific reasoning (1 sentence per criterion).
        {
        "image_quality": 0 or 1,
        "similarity_score": [score1, score2],
        "reasoning": "Reasoning for each scoring criterion"
        }
        """

        json_schema = {
                        "type": "object",
                        "properties": {
                            "image_quality": {"type": ["number", "null"],
                                              "description": "image quality score"},
                            "similarity_score": {
                                "type": "array",
                                "items": {"type": ["number", "null"]},
                                "description": "An array of 2 numbers: [identity_similarity, expression_similarity]."
                            },
                            "reasoning": {"type": "string",
                                          "description": "final reasoning answer"},
                        },
                        "required": ["image_quality", "similarity_score", "reasoning"],
                        "additionalProperties": False,
                    }

        if mode == "gpt":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.response_format = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        elif mode == "gemini":
            self.client = None
            self.response_format = json_schema

    @torch.no_grad()
    def __call__(self, img_id, img_makeup, img_size=256):
        img_id = encode_pil_image(PIL.Image.fromarray(img_id).resize((img_size, img_size), resample=PIL.Image.Resampling.LANCZOS))
        img_makeup = encode_pil_image(PIL.Image.fromarray(img_makeup).resize((img_size, img_size), resample=PIL.Image.Resampling.LANCZOS))

        success = False
        while not success:
            try:
                if self.mode == "gpt":
                    result = sample_gpt_text(self.client, self.prompt, [img_id, img_makeup],
                                          response_format=self.response_format)
                elif self.mode == "gemini":
                    result = sample_gemini_text(self.client, self.prompt, [img_id, img_makeup],
                                          response_format=self.response_format)


                if isinstance(result, str):
                    result = result.replace('```json', '')
                    result = result.replace('```', '')
                    result = json.loads(result)
                else:
                    result = json.loads(result)

                quality = result.get("image_quality", 0)
                score = result.get("similarity_score", [0, 0])
                if not isinstance(score, list) or len(score) != 2:
                    score = [0, 0]

                quality = int(quality or 0)
                score = [int(score[0] or 0), int(score[1] or 0)]

                success = True
            except Exception as e:
                print(f"[Warning] Error during API call: {e}\nRetry in an hour.")
                time.sleep(3600)  # 3600 seconds = 1 hour

        if quality == 1 and score[0] >= 9 and score[1] >= 8:
            return False
        else:
            print("id filter response:", result)
            return True


def process_id_makeup(img_id_path_list, data_root, img_size, min_h, min_w, mix_mode="affine", start_idx=0, end_idx=-1, use_exist=False):
    verbose = True if sys.platform == 'win32' else False

    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=min_h, min_w=min_w, align=False, td_mode="")
    face_parser = FaceParser()

    faceid_filter = FaceIDFilter(mode="gemini")

    face_swap = FaceSwap_Wrapper(face_analyser, face_parser, mix_mode)

    mix_dir = os.path.join(data_root, "makeup_mix")
    os.makedirs(mix_dir, exist_ok=True)
    if use_exist:
        mix_dir_new = os.path.join(data_root, "makeup_mix_new")
        os.makedirs(mix_dir_new, exist_ok=True)

    filter_list_id = []
    filter_list_align = []

    print("id start {}, end {}".format(start_idx, end_idx))
    if end_idx == -1:
        end_idx = len(img_id_path_list)

    for img_id_idx, img_id_path in enumerate(img_id_path_list):
        if img_id_idx < start_idx or img_id_idx >= end_idx:
            continue

        img_id = PIL.Image.open(img_id_path).convert('RGB')
        img_id = img_id.resize((img_size, img_size), resample=PIL.Image.Resampling.LANCZOS)
        img_id = np.array(img_id)
        img_id_name = os.path.splitext(os.path.basename(img_id_path))[0]

        makeup_dir = os.path.join(data_root, "makeup", img_id_name)
        img_makeup_file_list = sorted(os.listdir(makeup_dir))
        for img_makeup_file in img_makeup_file_list:
            img_makeup_path = os.path.join(makeup_dir, img_makeup_file)
            img_makeup = PIL.Image.open(img_makeup_path).convert('RGB')
            img_makeup = np.array(img_makeup)
            assert img_id.shape == img_makeup.shape

            out_dir_img = os.path.join(mix_dir, img_id_name)
            mix_path = os.path.join(out_dir_img, img_makeup_file)

            if use_exist and not os.path.exists(mix_path):
                print("skip:", img_id_idx, img_makeup_path)
                continue
            print("process:", img_id_idx, img_makeup_path)

            if mix_mode in ["swap", "affine"]:
                result = face_swap(img_makeup, img_id, verbose)

                if result["skip"]:
                    print("align filter: {} and {}. Due to {}".format(img_id_path, img_makeup_path, result))
                    filter_list_align.append([img_id_path, img_makeup_path])
                    continue

                img_mix = result["image"]

                is_skip_id = faceid_filter(img_id, img_mix)
                if is_skip_id:
                    print("id filter: {} and {}".format(img_id_path, img_makeup_path))
                    filter_list_id.append([img_id_path, img_makeup_path])
                    continue

            elif mix_mode == "blend":
                _, _, face_mask, pick_idx = face_parser.get_face_seg(img_makeup, verbose=verbose)
                img_mix = blend_image_mask(img_id, img_makeup, None, face_mask[pick_idx].cpu().numpy(),
                                           use_clone=False)

            if use_exist:
                out_dir_img = os.path.join(mix_dir_new, img_id_name)
                mix_path = os.path.join(out_dir_img, img_makeup_file)

            os.makedirs(out_dir_img, exist_ok=True)
            img_mix = PIL.Image.fromarray(img_mix)
            img_mix.save(mix_path)

    print("id filter {} images".format(len(filter_list_id)))
    print("align filter {} images".format(len(filter_list_align)))


def check_label(makeup_dir):
    import matplotlib.pyplot as plt

    img_makeup_path_list = glob.glob(makeup_dir + "/*/*.png")
    img_makeup_path_list = sorted(img_makeup_path_list)
    num_samples = len(img_makeup_path_list)

    label_info = {}
    for img_makeup_path in img_makeup_path_list:
        img_makeup_file = os.path.basename(img_makeup_path)

        match = re.search(r"-([0-9]+)-(.+)-[0-9]+\.png$", img_makeup_file)
        label = match.group(1)
        style = match.group(2)

        if label not in label_info:
            label_info[label] = [0, style]
        label_info[label][0] += 1

    print("total classes:", len(label_info))
    label_cnt = []
    for label in label_info.keys():
        label_cnt.append(label_info[label][0])
        print("label:", label, label_info[label][1], label_info[label][0] / num_samples)

    fig, ax = plt.subplots()
    ax.bar(list(range(len(label_info))), np.array(label_cnt) / num_samples)
    ax.set_xticks(list(range(len(label_info))))
    # plt.show()
    plt.savefig("distribution_class.png", dpi=300, bbox_inches="tight")


def resize_all_images(input_dir, output_dir, img_size, extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"}):
    for cur_dir, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(cur_dir, input_dir)
        output_file_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_file_dir, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in extensions:
                continue

            input_path = os.path.join(cur_dir, file)
            output_path = os.path.join(output_file_dir, file)

            img = PIL.Image.open(input_path)
            img = img.resize((img_size, img_size), resample=PIL.Image.Resampling.LANCZOS)
            img.save(output_path)
            print(f"Resized: {input_path} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create makeup image pairs")
    parser.add_argument("--edit_method", type=str, default="kontext", help="edit model")
    parser.add_argument("--t2i_method", type=str, default="qwen", help="t2i model")
    parser.add_argument("--data_makeup_path", type=str, default="./assets/makeup_gpto3.json", help="path to makeup style")
    parser.add_argument("--data_id_path", type=str, default="./assets/celebrity_gpto4mini.json", help="path to id list")
    parser.add_argument("--data_face_dir", type=str, default="../../data/facial/ffhq", help="path to face dataset")
    parser.add_argument("--process", type=str, default="edit", choices=["edit", "anno", "mix"])
    parser.add_argument("--out_dir", type=str, default="./output/makeup_pair_ffhq")
    parser.add_argument("--start_idx", type=int, default=0, help="")
    parser.add_argument("--end_idx", type=int, default=-1, help="")
    parser.add_argument("--num_id", type=int, default=20000)
    parser.add_argument("--num_img_no_makeup", type=int, default=10)
    parser.add_argument("--num_makeup", type=int, default=50)
    parser.add_argument("--num_img_makeup", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=1024, help="")
    parser.add_argument('--min_h', default=500, type=int, help='min height')
    parser.add_argument('--min_w', default=500, type=int, help='min width')
    parser.add_argument("--filter_makeup", type=int, default=0, help="")
    parser.add_argument("--skip_exist_id", type=int, default=1, help="")
    args = parser.parse_args()

    # create_ffhq_id_list(args.data_id_path, args.data_face_dir, args.num_id, use_prev=False)
    # create_ffhq_subset(args.data_id_path, num_sample=20000)

    # resize_all_images(args.out_dir, os.path.join(args.out_dir, os.path.basename(args.out_dir)+"-resize"), args.img_size)

    img_id_path_list = create_id_list(args)

    if args.process == "edit":
        create_makeup_pair(args, img_id_path_list)
    elif args.process == "anno":
        merge_anno_path = merge_anno(args.out_dir)
        create_data_split(merge_anno_path)
    elif args.process == "mix":
        process_id_makeup(img_id_path_list, args.out_dir, args.img_size, args.min_h, args.min_w,
                          start_idx=args.start_idx, end_idx=args.end_idx, use_exist=False)
        check_label(os.path.join(args.out_dir, "makeup_mix"))
