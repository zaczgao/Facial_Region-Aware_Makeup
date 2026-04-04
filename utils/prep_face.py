#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

__author__ = "GZ"

import os
import sys
import argparse
import numpy as np
import cv2
import glob
import shutil
import copy
import PIL.Image

import torch

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.face_analysis import FaceAnalyser, FaceParser, crop_bbox
from utils.tps import TPSDeform
from utils.vis_utils import show_result


def prep_face(args):
    verbose = True if sys.platform == 'win32' else False

    face_analyser = FaceAnalyser(det_thresh=args.det_thresh, min_h=args.min_h, min_w=args.min_w,
                                 exp_ratio=args.exp_ratio, use_square=args.use_square,
                                 align=args.align, image_size=args.image_size, td_mode="3ddfa")
    face_parser = FaceParser()
    tps_deform = TPSDeform()
    
    img_path_list = glob.glob(args.data_dir + "/*.png") + glob.glob(args.data_dir + "/*/*.png")
    img_path_list = sorted(img_path_list)
    print("Found {} face images at {}".format(len(img_path_list), args.data_dir))

    no_face_list = []
    small_face_list = []

    img_folder = os.path.basename(args.data_dir)

    for _, img_path in enumerate(img_path_list):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.dirname(img_path)
        relative_path = os.path.relpath(img_dir, args.data_dir)
        out_dir_img = os.path.join(args.out_dir, img_folder, relative_path)
        os.makedirs(out_dir_img, exist_ok=True)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        face_info, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr, verbose=verbose)

        if len(face_info) == 0:
            if is_small_face:
                print("small faces at {}".format(img_path))
                small_face_list.append(img_path)
            else:
                print("no faces at {}".format(img_path))
                no_face_list.append(img_path)
        else:
            pick_idx = face_analyser.find_largest_face(face_info)
            img_face_rgb = cv2.cvtColor(face_info[pick_idx]["face"], cv2.COLOR_BGR2RGB)

            out_path_file = os.path.join(out_dir_img, "{}.png".format(img_name))
            cv2.imwrite(out_path_file, face_info[pick_idx]["face"])

            if args.data_dir2:
                sub_data_dir = os.path.join(args.data_dir2, img_name)
                sub_part = os.path.basename(args.data_dir2)
                if os.path.isdir(sub_data_dir):
                    sub_img_file_list = sorted(os.listdir(sub_data_dir))
                    for sub_img_file in sub_img_file_list:
                        sub_img_path = os.path.join(args.data_dir2, img_name, sub_img_file)
                        sub_img_bgr = cv2.imread(sub_img_path, cv2.IMREAD_COLOR)
                        sub_img_face, _ = crop_bbox(sub_img_bgr, face_info[pick_idx]["bbox_crop"], exp_ratio=0)

                        sub_out_dir_face = os.path.join(args.out_dir, sub_part, img_name)
                        os.makedirs(sub_out_dir_face, exist_ok=True)
                        sub_out_file = os.path.join(sub_out_dir_face, sub_img_file)
                        cv2.imwrite(sub_out_file, sub_img_face)

            # img_warped = tps_deform(PIL.Image.fromarray(img_bgr[:, :, ::-1]),
            #                                 face_info[pick_idx]["landmark_3d_68"][:, :2], verbose=verbose)
            # img_face_warped = tps_deform(PIL.Image.fromarray(face_info[pick_idx]["face"][:, :, ::-1]),
            #                                      face_info[pick_idx]["landmark_68_face"][:, :2], verbose=verbose)

            if args.use_face_data:
                # lms68
                out_dir_lms68 = os.path.join(args.out_dir, "lms68", relative_path)
                os.makedirs(out_dir_lms68, exist_ok=True)
                out_path_file = os.path.join(out_dir_lms68, "{}.npy".format(img_name))
                np.save(out_path_file, face_info[pick_idx]["landmark_68_face"])

                # face recognition embedding
                # out_dir_emb = os.path.join(args.out_dir, "face_emb", relative_path)
                # os.makedirs(out_dir_emb, exist_ok=True)
                # out_file = os.path.join(out_dir_emb, "{}.npy".format(img_name))
                # np.save(out_file, face_info[pick_idx]["face_emb"])

                # landmark image
                img_face_lms = face_analyser.get_lms_image(face_info[pick_idx]["landmark_3d_68"][:, :2],
                                                           img_bgr.shape[0], img_bgr.shape[1],
                                                           bbox=face_info[pick_idx]["bbox_crop"],
                                                           image=img_bgr, verbose=verbose)
                out_dir_pose = os.path.join(args.out_dir, "pose", relative_path)
                os.makedirs(out_dir_pose, exist_ok=True)
                out_path_file = os.path.join(out_dir_pose, "{}.png".format(img_name))
                img_face_lms.save(out_path_file)

                # 3d image
                out_dir_3d = os.path.join(args.out_dir, "3d", relative_path)
                os.makedirs(out_dir_3d, exist_ok=True)
                out_file = os.path.join(out_dir_3d, "{}.png".format(img_name))
                face_info[pick_idx]["face_3d"].save(out_file)

                # segmentation mask
                face_info_face = copy.deepcopy(face_info)
                for info, info_face in zip(face_info, face_info_face):
                    info_face["bbox"] = info["bbox_face"]
                    info_face["kps"] = info["kps_face"]
                seg_masks, seg_masks_dilate, face_mask, pick_idx = \
                    face_parser.get_face_seg(img_face_rgb, face_info_face, verbose=verbose)
                out_dir_mask = os.path.join(args.out_dir, "mask", relative_path)
                os.makedirs(out_dir_mask, exist_ok=True)
                out_path_file = os.path.join(out_dir_mask, "{}.pt".format(img_name))

                seg_preds_dilate = face_parser.get_pred_from_mask(seg_masks_dilate)
                mask_dict = {"seg_pred": seg_preds_dilate[pick_idx].to(torch.uint8)}
                torch.save(mask_dict, out_path_file)

    print("{} faces not found".format(len(no_face_list)))
    print("{} small faces".format(len(small_face_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument("--data_dir", type=str, default="./output/makeup_style/image", help="path to dataset")
    parser.add_argument('--data_dir2', type=str, default="", help="path to dataset")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument('--det_thresh', default=0.5, type=float, help='det_thresh')
    parser.add_argument('--min_h', default=250, type=int, help='min height')
    parser.add_argument('--min_w', default=250, type=int, help='min width')
    parser.add_argument('--exp_ratio', default=0.4, type=float, help='')
    parser.add_argument("--use_square", type=int, default=1)
    parser.add_argument('--align', default=0, type=int, help='align face image')
    parser.add_argument('--image_size', default=256, type=int, help='aligned image size')
    parser.add_argument('--use_face_data', default=0, type=int, help='')
    args = parser.parse_args()

    print(args)

    prep_face(args)
