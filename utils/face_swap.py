#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
face swap based on Delaunay triangulation

https://github.com/khoi03/Face-Swap-4-different-setups/blob/master/Code/mediapipe_delaunay.py
https://github.com/Frostday/Face-Swapping
https://github.com/guipleite/CV2-Face-Swap
https://github.com/spmallick/learnopencv/tree/master/FaceSwap
"""

__author__ = "GZ"

import os
import sys
import numpy as np
import cv2
import PIL.Image
from scipy.spatial import Delaunay

import torch

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.vis_utils import show_result


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_triangles(convexhull, landmarks_points):
    rect = cv2.boundingRect(convexhull)  # trả về (x,y,w,h)
    subdiv = cv2.Subdiv2D(rect)  # tạo 1 không gian trống Delaunay subdivision
    subdiv.insert(landmarks_points.tolist())  # bỏ những điểm landmarks vào không gian
    triangles = subdiv.getTriangleList()  # trả về 1 danh sách các điểm tam giác
    triangles = np.array(triangles, dtype=np.int32)
    # print(triangles)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])  # điểm thứ nhất
        pt2 = (t[2], t[3])  # điểm thứ 2
        pt3 = (t[4], t[5])  # điểm thứ 3

        index_pt1 = np.where((landmarks_points == pt1).all(axis=1))[0][0]
        # index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((landmarks_points == pt2).all(axis=1))[0][0]
        # index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((landmarks_points == pt3).all(axis=1))[0][0]
        # index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return indexes_triangles


def apply_triangulation(triangle_index, landmark_points, img=None):
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]
    triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect

    cropped_triangle = None
    if img is not None:
        cropped_triangle = img[y: y + h, x: x + w]

    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    # print(x,y)
    # print(tr1_pt1,tr1_pt2,tr1_pt3)
    # print(points)
    cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

    return points, cropped_triangle, cropped_triangle_mask, rect


def warp_triangle(rect, points1, points2, src_cropped_triangle, tgt_cropped_triangle_mask):
    (x, y, w, h) = rect
    matrix = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h),
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_REFLECT_101)

    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=tgt_cropped_triangle_mask)
    return warped_triangle


def add_piece_of_new_face(new_face, rect, warped_triangle):
    (x, y, w, h) = rect

    new_face_rect_area = new_face[y: y + h, x: x + w]
    new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
    new_face[y: y + h, x: x + w] = new_face_rect_area


def swap_new_face(dest_image, dest_convexHull, new_face):
    face_mask = np.zeros(dest_image.shape[:2], dtype=np.uint8)
    head_mask = cv2.fillConvexPoly(face_mask, dest_convexHull, 255)
    face_mask = cv2.bitwise_not(head_mask)
    head_without_face = cv2.bitwise_and(dest_image, dest_image, mask=face_mask)

    result = cv2.add(head_without_face, new_face)

    (x, y, w, h) = cv2.boundingRect(dest_convexHull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.NORMAL_CLONE)


# put the face of src to tgt
def swap_face_delaunay(src_image, tgt_image, src_landmark_points, tgt_landmark_points, verbose=False):
    src_landmark_points = np.around(src_landmark_points).astype(np.int32)
    tgt_landmark_points = np.around(tgt_landmark_points).astype(np.int32)

    # src_convexHull = cv2.convexHull(src_landmark_points)  # return the outermost point
    # indexes_triangles = get_triangles(convexhull=src_convexHull, landmarks_points=src_landmark_points)
    indexes_triangles = Delaunay(src_landmark_points).simplices.tolist()

    tgt_convexHull = cv2.convexHull(tgt_landmark_points)
    height, width, channels = tgt_image.shape
    new_face = np.zeros((height, width, channels), np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        points, src_cropped_triangle, cropped_triangle_mask, _ = apply_triangulation(
            triangle_index=triangle_index,
            landmark_points=src_landmark_points,
            img=src_image)

        # Triangulation of second face
        points2, _, tgt_cropped_triangle_mask, rect = apply_triangulation(triangle_index=triangle_index,
                                                                          landmark_points=tgt_landmark_points)

        # Warp triangles
        warped_triangle = warp_triangle(rect=rect, points1=points, points2=points2,
                                        src_cropped_triangle=src_cropped_triangle,
                                        tgt_cropped_triangle_mask=tgt_cropped_triangle_mask)

        # Reconstructing destination face
        add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

        if verbose:
            tr1_pt1 = src_landmark_points[triangle_index[0]]
            tr1_pt2 = src_landmark_points[triangle_index[1]]
            tr1_pt3 = src_landmark_points[triangle_index[2]]
            src_image_tri = src_image.copy()
            cv2.line(src_image_tri, tr1_pt1, tr1_pt2, (255, 0, 0), 1)
            cv2.line(src_image_tri, tr1_pt2, tr1_pt3, (255, 0, 0), 1)
            cv2.line(src_image_tri, tr1_pt1, tr1_pt3, (255, 0, 0), 1)
            tr2_pt1 = tgt_landmark_points[triangle_index[0]]
            tr2_pt2 = tgt_landmark_points[triangle_index[1]]
            tr2_pt3 = tgt_landmark_points[triangle_index[2]]
            tgt_image_tri = tgt_image.copy()
            cv2.line(tgt_image_tri, tr2_pt1, tr2_pt2, (255, 0, 0), 1)
            cv2.line(tgt_image_tri, tr2_pt2, tr2_pt3, (255, 0, 0), 1)
            cv2.line(tgt_image_tri, tr2_pt1, tr2_pt3, (255, 0, 0), 1)

            image_horizontal = np.hstack((src_image, src_image_tri, tgt_image_tri, new_face))
            cv2.imshow('image', image_horizontal)
            cv2.waitKey(0)

    # Face swapped (putting 1st face into 2nd face)
    result = swap_new_face(dest_image=tgt_image, dest_convexHull=tgt_convexHull, new_face=new_face)

    if verbose:
        cv2.imshow('image', result)
        cv2.waitKey(0)

    return result


def smooth_mask(mask: np.ndarray, ksize):
    mask = cv2.GaussianBlur(mask * 255., ksize, 0) / 255.
    mask = np.clip(mask, 0, 1)

    return mask


def blend_image_mask(img_ori: np.ndarray, img_edit: np.ndarray, mask_ori: np.ndarray, mask_edit: np.ndarray,
                     use_clone=True, ksize=(7, 7), verbose=False):

    if mask_ori is not None and mask_edit is not None:
        mask = np.clip(mask_ori + mask_edit, 0, 1)
    elif mask_ori is not None:
        mask = mask_ori
    elif mask_edit is not None:
        mask = mask_edit

    if use_clone:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # extend by 3 pixels
        mask = cv2.dilate(mask * 255., kernel, iterations=4) / 255.

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # shrink by 3 pixels
        # mask = cv2.erode(mask * 255., kernel, iterations=1) / 255.

        # seamless clone to have smooth border
        mask_points = cv2.findNonZero((mask * 255.).astype(np.uint8))
        x, y, w, h = cv2.boundingRect(mask_points)
        center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
        img_seam = cv2.seamlessClone(img_edit, img_ori, (mask * 255.).astype(np.uint8), center_face, cv2.NORMAL_CLONE)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # shrink by 3 pixels
        mask = cv2.erode(mask * 255., kernel, iterations=1) / 255.

        mask = smooth_mask(mask, ksize)
        mask = np.expand_dims(mask, axis=-1)
        img_mix = mask * img_edit.astype(np.float32) + (1. - mask) * img_seam.astype(np.float32)
        img_mix = np.clip(img_mix, 0, 255).astype(np.uint8)
    else:
        mask = smooth_mask(mask, ksize)
        mask = np.expand_dims(mask, axis=-1)
        img_mix = mask * img_edit.astype(np.float32) + (1. - mask) * img_ori.astype(np.float32)
        img_mix = np.clip(img_mix, 0, 255).astype(np.uint8)

    if verbose:
        img_viz = mask * img_edit.astype(np.float32)
        show_result(255. * np.hstack((mask_ori, mask_edit, mask[:, :, 0])))
        show_result(np.hstack((img_ori, img_edit, img_seam, img_mix)))

    return img_mix


class FaceSwap_Wrapper():
    def __init__(self, face_analyser, face_parser, mode):
        self.face_analyser = face_analyser
        self.face_parser = face_parser
        self.mode = mode
        assert mode in ["swap", "affine"]

    def filter_face_align(self, seg_masks_ori, seg_masks_edit):
        label_group = [
            {"labels": ["re", "le"], "iou_thresh": 0.75, "abs_thresh": 4000},
            {"labels": ["imouth"], "iou_thresh": 0.7, "abs_thresh": 0}
        ]

        for group in label_group:
            label_pick = group["labels"]
            iou_thresh = group["iou_thresh"]
            abs_thresh = group["abs_thresh"]
            abs_thresh = (seg_masks_ori.shape[-1] / 1024)**2 * abs_thresh

            masks_ori = self.face_parser.pick_mask_from_labels(None, self.face_parser.label_names, label_pick,
                                                                     seg_masks_ori.unsqueeze(0))
            masks_edit = self.face_parser.pick_mask_from_labels(None, self.face_parser.label_names, label_pick,
                                                                seg_masks_edit.unsqueeze(0))

            masks_ori = masks_ori.squeeze(0)
            masks_edit = masks_edit.squeeze(0)

            mask_union = torch.clamp(masks_ori + masks_edit, 0, 1)
            mask_inter = masks_ori * masks_edit
            iou = mask_inter.sum() / (mask_union.sum() + 1e-6)

            if (masks_ori.sum() < abs_thresh) or (masks_edit.sum() < abs_thresh) or (iou < iou_thresh and mask_union.sum() > 3000):
                return True

        return False

    @torch.no_grad()
    def __call__(self, src_image, tgt_image, verbose=False):
        result = {"skip": False,
                  "det": None,
                  "det_warp": None,
                  "align": None,
                  "image": None
                  }

        face_info_src, _ = self.face_analyser.get_face_info(img_bgr=src_image[:, :, ::-1].copy(), verbose=verbose)
        face_info_tgt, _ = self.face_analyser.get_face_info(img_bgr=tgt_image[:, :, ::-1].copy(), verbose=verbose)

        if len(face_info_src) == 0 or len(face_info_tgt) == 0:
            result["det"] = False
            result["skip"] = True
            return result

        pick_idx_src = self.face_analyser.find_largest_face(face_info_src)
        pick_idx_tgt = self.face_analyser.find_largest_face(face_info_tgt)

        if self.mode == "swap":
            lms68_src = face_info_src[pick_idx_src]["landmark_3d_68"]
            lms68_tgt = face_info_tgt[pick_idx_tgt]["landmark_3d_68"]

            img_mix = swap_face_delaunay(src_image, tgt_image, lms68_src[:, :2], lms68_tgt[:, :2], verbose=verbose)
        else:
            lms_src = face_info_src[pick_idx_src]["landmark_98"]
            lms_tgt = face_info_tgt[pick_idx_tgt]["landmark_98"]

            lms_src = lms_src[:, :2].reshape(-1, 1, 2)
            lms_tgt = lms_tgt[:, :2].reshape(-1, 1, 2)

            # M, _ = cv2.estimateAffinePartial2D(lms_src, lms_tgt, method=cv2.LMEDS)
            M, _ = cv2.estimateAffine2D(lms_src, lms_tgt, method=cv2.LMEDS)
            src_image_warp = cv2.warpAffine(src_image, M, (tgt_image.shape[1], tgt_image.shape[0]),
                                              flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

            # failing to detect lms68 causes distorted face
            face_info_warp, _ = self.face_analyser.get_face_info(img_bgr=src_image_warp[:, :, ::-1].copy(), verbose=verbose)
            if len(face_info_warp) == 0:
                result["det_warp"] = False
                result["skip"] = True
                return result

            seg_masks_warp, _, face_mask_warp, pick_idx_warp = self.face_parser.get_face_seg(src_image_warp, face_info_warp, verbose=verbose)
            seg_masks_tgt, _, face_mask_tgt, pick_idx_tgt = self.face_parser.get_face_seg(tgt_image, face_info_tgt, verbose=verbose)

            # misaligned eyes and teeth
            is_skip = self.filter_face_align(seg_masks_tgt[pick_idx_tgt], seg_masks_warp[pick_idx_warp])
            if is_skip:
                result["align"] = False
                result["skip"] = True
                return result

            mask_edit = face_mask_warp[pick_idx_warp].cpu().numpy()
            mask_tgt = face_mask_tgt[pick_idx_tgt].cpu().numpy()

            img_mix = blend_image_mask(tgt_image, src_image_warp, mask_tgt, mask_edit)

        result["image"] = img_mix
        return result


if __name__ == '__main__':
    from utils.face_analysis import FaceAnalyser, FaceParser, show_face_result

    verbose = True
    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150, align=False)
    face_parser = FaceParser()

    mode = "affine"
    face_swap = FaceSwap_Wrapper(face_analyser, face_parser, mode)

    img_makeup = PIL.Image.open("./assets/images/00128-000-Dewy_Minimalist-00.png")
    img_id = PIL.Image.open("./assets/images/00128.png")
    # img_makeup = PIL.Image.open("./assets/images/test-swap2.png").resize((1024, 1024))
    # img_id = PIL.Image.open("./assets/images/test-swap1.png").resize((1024, 1024))
    img_makeup = np.array(img_makeup)
    img_id = np.array(img_id)

    result = face_swap(img_makeup, img_id, verbose)
    print(result)
    img_mix = result["image"]

    img_mix = PIL.Image.fromarray(img_mix)
    img_mix.save("img_swap.png")

    face_parser = FaceParser()
    _, _, face_mask, pick_idx = face_parser.get_face_seg(img_makeup, verbose=verbose)
    img_mix = blend_image_mask(img_id, img_makeup, None, face_mask[pick_idx].cpu().numpy(), use_clone=False)
    img_mix = PIL.Image.fromarray(img_mix)
    img_mix.save("img_blend.png")
