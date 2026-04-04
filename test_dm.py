#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import argparse
import random
import numpy as np
import copy
import PIL.Image
import cv2

import torch
import torchvision

from diffusers import DDIMScheduler
from diffusers.models.attention_processor import Attention

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from dm.multi_token_clip import MultiTokenCLIPTokenizer
from dm.tokenizer import add_tokens
from dm.text_encoder import CustomCLIPTextModel
from dm.unet import CustomUNet2DConditionModel
from dm.attn_proc import setup_attn_processor, load_attn_processor
from dm.makeup_adapter import MakeupAdapter
from dm.pipeline import MakeupSDPipeline
from dm.losses import _compute_avg_attn_map
from style_clip.model import StyleCLIP
from style_clip import clip_utils
from utils.face_analysis import FaceAnalyser, FaceParser
from utils.misc import load_image
from utils.vis_utils import show_result, concatenate_images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--use_fp16", type=int, default=0)

    parser.add_argument("--style_clip_ckpt", type=str, default="")
    parser.add_argument("--use_clip_lora", type=int, default=0, help="Use clip hidden states")
    parser.add_argument("--clip_hidden", type=str, default="24", help="clip hidden block idx")
    parser.add_argument("--placeholder_token", type=str, default="<part>", help="A token to use as a placeholder for the concept.")
    parser.add_argument("--num_parts", type=int, default=4, help="Number of facial regions")
    parser.add_argument("--num_heads_part", type=int, default=16, help="Number of head per facial region")
    parser.add_argument("--use_lora", type=int, default=0, help="Use lora for sd backbone")
    parser.add_argument("--use_ipa", type=int, default=0, help="Use ipa for makeup style")
    parser.add_argument("--use_text_inv", type=int, default=0, help="Use text inversion for makeup style")
    parser.add_argument("--ipa_scale", type=float, default=1.)
    parser.add_argument("--geo_mode", type=str, default="3d", choices=["3d", "normal", "keypoint"])

    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--anno_path", type=str, default="")
    parser.add_argument("--data_id_path", type=str, default="")
    parser.add_argument("--data_makeup_path", type=str, default="")
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument('--guidance_scale', default=7.5, type=float, help='cfg')
    parser.add_argument("--detect_face", type=int, default=0)
    parser.add_argument('--exp_ratio', default=0.4, type=float, help='')
    parser.add_argument("--use_square", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--token_idx", type=str, default=None)
    parser.add_argument("--vis_all", type=int, default=0)
    parser.add_argument("--vis_attn", type=int, default=0)

    args = parser.parse_args()

    return args


def paste_back_crop(img_src_path, img_tgt_path, exp_ratio, use_square):
    verbose = False

    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150, exp_ratio=exp_ratio, use_square=use_square,
                                 align=False, td_mode="3ddfa")

    img_src = load_image(img_src_path)
    img_tgt = load_image(img_tgt_path)

    face_info, is_small_face = face_analyser.get_face_info(img_bgr=np.array(img_src)[:, :, ::-1], verbose=verbose)
    pick_idx = face_analyser.find_largest_face(face_info)

    bbox = face_info[pick_idx]["bbox_crop"]
    height, width = bbox[3] - bbox[1], bbox[2] - bbox[0]
    img_tgt = img_tgt.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)

    img_src = np.array(img_src)
    img_tgt = np.array(img_tgt)
    img_src[bbox[1]:bbox[3], bbox[0]:bbox[2]] = img_tgt
    img_out = PIL.Image.fromarray(img_src)
    img_out.save("test.png")


def group_mask(seg_mask, label_names, label_group, out_dir=None):
    _, H, W = seg_mask.shape

    seg_mask_group = torch.zeros((len(label_group), H, W), device=seg_mask.device)
    for group_idx, group in enumerate(label_group):
        for label in group:
            idx = np.where(np.array(label_names) == label)[0][0]
            seg_mask_group[group_idx] += seg_mask[idx]

        mask = seg_mask_group[group_idx].cpu().numpy().astype(np.uint8) * 255

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"mask-{group_idx}.png"), mask)

    return seg_mask_group


def vis_attn_map(img, unet, prompt, tokenizer, placeholder_token_ids, attn_size=[8,16,32,64], num_heads_part=4, use_ipa=True, alpha=0.5, out_dir="./attn"):
    input_ids = tokenizer(
            prompt,
            replace_token=True,
            token_idx=None,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

    cross_attn_probs = {}
    for size in attn_size:
        cross_attn_probs[size] = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if module.cross_attn_probs is not None:
                a = module.cross_attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
                size = a.size(1)
                cross_attn_probs[size][name] = a

    avg_attn_map = _compute_avg_attn_map(cross_attn_probs[64])
    # avg_attn_map = cross_attn_probs[64]["up_blocks.3.attentions.2.transformer_blocks.0.attn2"]
    B, H, W, seq_length = avg_attn_map.size()
    located_attn_map = []

    if use_ipa:
        located_attn_map = avg_attn_map.permute(0, 3, 1, 2)

        located_attn_map = located_attn_map.reshape(B, -1, num_heads_part, H, W)
        located_attn_map = torch.mean(located_attn_map, dim=2)
    else:
        # locate the attn map
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            for bi in range(B):
                learnable_idx = (input_ids[bi] == placeholder_token_id).nonzero(as_tuple=True)[0]

                if len(learnable_idx) != 0:  # only assign if found
                    if len(learnable_idx) == 1:
                        offset_learnable_idx = learnable_idx
                    else:  # if there is two and above.
                        raise NotImplementedError

                    located_map = avg_attn_map[bi, :, :, offset_learnable_idx]
                    located_attn_map.append(located_map)
                else:
                    located_attn_map.append(torch.zeros((H, W, 1), device=avg_attn_map.device, dtype=avg_attn_map.dtype))

        M = len(placeholder_token_ids)
        located_attn_map = (
            torch.stack(located_attn_map, dim=0).reshape(M, B, H, W).transpose(0, 1)
        )  # (B, M, 16, 16)

    for idx, attn_map_part in enumerate(located_attn_map[1]):
        attn_map_part = attn_map_part.cpu().numpy()
        attn_map_part = cv2.resize(attn_map_part, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        att_map_norm = (attn_map_part - attn_map_part.min()) / (attn_map_part.max() - attn_map_part.min())
        heatmap = cv2.applyColorMap((att_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = ((1 - alpha) * img + heatmap * alpha).astype(np.uint8)

        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"attn-{idx}.png"), overlay)


def init_pipeline(pretrained_model_name_or_path, revision, variant, placeholder_token, num_parts, num_heads_part, use_lora,
                  use_ipa, use_text_inv,
                  style_clip_ckpt, use_clip_lora, clip_hidden, **kargs):
    ipa_scale = kargs["ipa_scale"]
    ckpt_dir = kargs["ckpt_dir"]
    device = kargs["device"]
    weight_dtype = kargs["weight_dtype"]
    attn_size = kargs.get("attn_size", [])

    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CustomCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    unet = CustomUNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    style_clip = StyleCLIP("vit_large", use_text=False)
    if use_clip_lora:
        style_clip.prep_lora_model()
    clip_utils.load_network(style_clip, style_clip_ckpt, "state_dict", strict=False)
    style_clip = style_clip.eval()
    style_clip = style_clip.to(device, dtype=weight_dtype)

    if isinstance(clip_hidden, str):
        clip_hidden = [int(a) for a in clip_hidden.split(",")]
    makeup_adapter = MakeupAdapter(
        style_in_dim=style_clip.clip_dim,
        style_out_dim=unet.config.cross_attention_dim if use_ipa else text_encoder.text_model.embeddings.token_embedding.embedding_dim,
        style_seq_len=256 * len(clip_hidden),
        num_parts=num_parts,
        num_heads_part=num_heads_part,
        unet=unet,
        use_ipa=use_ipa,
        use_text_inv=use_text_inv
    )
    makeup_adapter.load_from_checkpoint(os.path.join(ckpt_dir, "makeup_adapter.pt"))
    makeup_adapter = makeup_adapter.eval()
    makeup_adapter = makeup_adapter.to(device, dtype=weight_dtype)

    placeholder_token_ids = add_tokens(
        tokenizer,
        text_encoder,
        placeholder_token,
        num_parts,
    )

    pipeline = MakeupSDPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        controlnet=makeup_adapter.control_id,
        # controlnet=[makeup_adapter.control_id, makeup_adapter.control_pose],
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )

    # load attention processors
    if use_lora:
        pipeline.load_lora_weights(ckpt_dir, weight_name="pytorch_lora_weights.safetensors")

    if use_ipa:
        setup_attn_processor(pipeline.unet, attn_size=attn_size, use_ipa=use_ipa)
        load_attn_processor(pipeline.unet, os.path.join(ckpt_dir, "ipa_layers.pt"))
        pipeline.set_ip_adapter_scale(ipa_scale)

    pipeline.init_extra(
        style_clip=style_clip,
        makeup_adapter=makeup_adapter,
        placeholder_token_ids=placeholder_token_ids,
        clip_hidden=clip_hidden)
    pipeline = pipeline.to(device, dtype=weight_dtype)

    return pipeline


def main():
    args = parse_args()

    args.data_root = os.path.normpath(args.data_root)

    verbose = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if args.use_fp16 else torch.float32

    if args.detect_face:
        face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150,
                                     exp_ratio=args.exp_ratio, use_square=args.use_square, align=False, td_mode="3ddfa")
        face_parser = FaceParser(mode="celebm")

        face_parser.set_bg_label(['background', 'neck', 'cloth', 'rr', 'lr', 'hair', 'eyeg', 'hat', 'earr', 'neck_l', 'imouth'])

    attn_size = []
    if args.vis_attn:
        attn_size = [64]

    pipeline = init_pipeline(args.pretrained_model_name_or_path, args.revision, args.variant,
                             args.placeholder_token, args.num_parts, args.num_heads_part, args.use_lora,
                             args.use_ipa, args.use_text_inv,
                             args.style_clip_ckpt, args.use_clip_lora, args.clip_hidden,
                             ipa_scale=args.ipa_scale, ckpt_dir=args.ckpt_dir,
                             device=device, weight_dtype=weight_dtype, attn_size=attn_size)

    seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator().manual_seed(seed)

    control_scale = 1.
    num_inference_steps = 50
    guidance_scale = args.guidance_scale

    label_group = [
        ['background', 'hair', 'imouth'],
        ['face'],
        ['rb', 'lb', 're', 'le'],
        ['nose'],
        ['ulip', 'llip']
    ]

    token_idx = args.token_idx
    if args.token_idx is not None:
        token_idx = [int(a) for a in args.token_idx.split(",")]

    img_id_path_list, img_makeup_path_list, img_makeup_path_list_2 = [], [], []
    if args.data_makeup_path:
        img_makeup_path_list_2 = args.data_makeup_path.split(";")
        if len(img_makeup_path_list_2) > 1:
            args.data_makeup_path = img_makeup_path_list_2[0]
            img_makeup_path_list_2 = img_makeup_path_list_2[1:]
        else:
            img_makeup_path_list_2 = []

        if os.path.isfile(args.data_makeup_path):
            img_id_path_list = [args.data_id_path]
            img_makeup_path_list = [args.data_makeup_path]
        else:
            img_id_file_list = sorted(os.listdir(args.data_id_path))
            img_makeup_file_list = sorted(os.listdir(args.data_makeup_path))
            for img_id_file in img_id_file_list:
                for img_makeup_file in img_makeup_file_list:
                    img_id_path = os.path.join(args.data_id_path, img_id_file)
                    img_makeup_path = os.path.join(args.data_makeup_path, img_makeup_file)
                    if os.path.isfile(img_id_path) and os.path.isfile(img_makeup_path):
                        img_id_path_list.append(img_id_path)
                        img_makeup_path_list.append(img_makeup_path)
    else:
        with open(args.anno_path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split()
                img_id_path_list.append(os.path.join(args.data_root, "id", items[0]))
                img_makeup_path_list.append(os.path.join(args.data_root, "makeup", items[1]))

    img_makeup_list_2 = []
    for img_makeup_path in img_makeup_path_list_2:
        img_makeup = load_image(img_makeup_path)

        if args.detect_face:
            face_info, is_small_face = face_analyser.get_face_info(img_bgr=np.array(img_makeup)[:, :, ::-1],
                                                                   verbose=verbose)
            if len(face_info) > 0:
                pick_idx = face_analyser.find_largest_face(face_info)
                img_makeup = PIL.Image.fromarray(face_info[pick_idx]["face"][:, :, ::-1])

        img_makeup_list_2.append(img_makeup)

    if not args.data_name:
        args.data_name = args.data_root.split(os.sep)[-1]
    out_dir_img = os.path.join(args.out_dir, args.data_name)
    os.makedirs(out_dir_img, exist_ok=True)

    for img_id_path, img_makeup_path in zip(img_id_path_list, img_makeup_path_list):
        img_id = load_image(img_id_path)
        img_makeup = load_image(img_makeup_path)

        if args.detect_face:
            face_info, is_small_face = face_analyser.get_face_info(img_bgr=np.array(img_id)[:, :, ::-1], verbose=verbose)
            pick_idx = face_analyser.find_largest_face(face_info)

            if args.geo_mode == "3d":
                img_pose = face_info[pick_idx]["face_3d"]
            elif args.geo_mode == "keypoint":
                img_pose = face_analyser.get_lms_image(face_info[pick_idx]["landmark_3d_68"][:, :2],
                                                       img_id.size[1], img_id.size[0],
                                                       bbox=face_info[pick_idx]["bbox_crop"],
                                                       image=np.array(img_id), verbose=verbose)
            img_id = PIL.Image.fromarray(face_info[pick_idx]["face"][:, :, ::-1])

            img_face_rgb = cv2.cvtColor(face_info[pick_idx]["face"], cv2.COLOR_BGR2RGB)
            face_info_face = copy.deepcopy(face_info)
            for info, info_face in zip(face_info, face_info_face):
                info_face["bbox"] = info["bbox_face"]
                info_face["kps"] = info["kps_face"]
            seg_masks, seg_masks_dilate, face_mask, pick_idx = face_parser.get_face_seg(img_face_rgb, face_info_face, verbose=verbose)
            face_mask = face_mask[pick_idx]

            # if token_idx is not None:
            #     seg_masks_group = group_mask(seg_masks_dilate[pick_idx], face_parser.label_names, label_group)
            #     seg_masks_group = seg_masks_group[1:]
            #     face_mask = torch.zeros_like(face_mask)
            #     for idx in token_idx:
            #         face_mask += seg_masks_group[idx]

            # makeup
            face_info, is_small_face = face_analyser.get_face_info(img_bgr=np.array(img_makeup)[:, :, ::-1], verbose=verbose)
            if len(face_info) > 0:
                pick_idx = face_analyser.find_largest_face(face_info)
                img_makeup = PIL.Image.fromarray(face_info[pick_idx]["face"][:, :, ::-1])
        else:
            img_id_name = os.path.splitext(os.path.basename(img_id_path))[0]

            if args.geo_mode == "3d":
                img_pose_path = os.path.join(args.data_root, "3d", "{}.png".format(img_id_name))
            elif args.geo_mode == "keypoint":
                img_pose_path = os.path.join(args.data_root, "pose", "{}.png".format(img_id_name))
            img_pose = load_image(img_pose_path)

            face_mask = torch.load(os.path.join(args.data_root, "mask", "{}.pt".format(img_id_name)), map_location="cpu")
            face_mask = face_mask.to(torch.float32)

        face_mask = torchvision.transforms.functional.to_pil_image(face_mask).convert("L")

        assert img_id.size == img_pose.size and img_id.size == face_mask.size
        img_id = img_id.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.LANCZOS)
        img_pose = img_pose.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.NEAREST)
        face_mask = face_mask.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.NEAREST)
        # face_mask = PIL.Image.new("L", (args.resolution, args.resolution), color=255)

        face_mask_blurred = pipeline.mask_processor.blur(face_mask, blur_factor=10)

        if len(img_makeup_list_2) > 0:
            img_makeup = [img_makeup] + img_makeup_list_2

        image = pipeline(
            prompt=args.validation_prompt,
            image=img_id,
            mask_image=face_mask_blurred,
            control_image=[img_id, img_pose],
            control_mode=[0, 1],
            image_makeup=img_makeup,
            num_inference_steps=num_inference_steps,
            negative_prompt="",
            controlnet_conditioning_scale=control_scale,
            guidance_scale=guidance_scale,
            generator=generator,
            token_idx=token_idx
        ).images[0]

        img_id_name = os.path.splitext(os.path.basename(img_id_path))[0]
        img_makeup_name = os.path.splitext(os.path.basename(img_makeup_path))[0]
        image.save(os.path.join(out_dir_img, "{}-{}.png".format(img_id_name, img_makeup_name)))

        if args.vis_all:
            image_all = []
            if img_id is not None:
                image_all += [img_id]

            if isinstance(img_makeup, list):
                image_all += img_makeup
            else:
                image_all += [img_makeup]

            image_all += [image]

            concatenate_images(image_all, os.path.join(out_dir_img, "{}-{}-cat.jpg".format(img_id_name, img_makeup_name)))

        if args.vis_attn:
            vis_attn_map(np.array(img_id)[:, :, ::-1], pipeline.unet, args.validation_prompt, pipeline.tokenizer, pipeline.placeholder_token_ids,
                         num_heads_part=args.num_heads_part, use_ipa=args.use_ipa, alpha=0.5, out_dir=os.path.join(args.out_dir, "attn"))

            group_mask(seg_masks[0], face_parser.label_names, label_group, out_dir=os.path.join(args.out_dir, "attn"))


if __name__ == '__main__':
    main()
