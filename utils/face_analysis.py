#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://github.com/deepinsight/insightface/tree/master/model_zoo

https://github.com/Yusepp/YOLOv8-Face
pip install ultralytics

facer Bug fix:
\envs\torch-2\Lib\site-packages\facer\face_parsing\farl.py
/envs/torch-2/lib/python3.10/site-packages/facer/face_parsing/farl.py
line148: w_images, grid, inv_grid = self.warp_images(images, data)
add: w_images = torch.clamp(w_images, max=1.)

https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
https://onnxruntime.ai/docs/install/#cuda-11x

https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace
"""

__author__ = "GZ"

import os
import sys
import numpy as np
import math
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.app.common import Face
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not found")

import facer

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.model_3d import TDDFAV3, SMIRK
from utils.vis_utils import show_result, show_face_result


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def calc_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-7)


def get_clean_mask(mask, min_size):
    """
    :param mask: 0, 1 uint8
    :param min_size: min size of the mask
    """
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_size:
            clean_mask[labels == i] = 1

    outlier_mask = (mask.astype(np.bool_) & ~clean_mask.astype(np.bool_)).astype(np.uint8)

    return clean_mask, outlier_mask


def get_patch(landmarks, color='lime', closed=False):
    contour = landmarks
    ops = [Path.MOVETO] + [Path.LINETO] * (len(contour) - 1)
    facecolor = (0, 0, 0, 0)  # Transparent fill color, if open
    if closed:
        contour.append(contour[0])
        ops.append(Path.CLOSEPOLY)
        facecolor = color
    path = Path(contour, ops)
    return patches.PathPatch(path, facecolor=facecolor, edgecolor=color, lw=4)


# https://huggingface.co/spaces/pcuenq/uncanny-faces/blob/main/app.py
def conditioning_from_landmarks(landmarks, height, width):
    # Precisely control output image size
    dpi = 72
    # fig, ax = plt.subplots(1, figsize=[size/dpi, size/dpi], tight_layout={'pad': 0})
    # fig.set_dpi(dpi)

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])

    black = np.zeros((height, width, 3))
    ax.imshow(black)

    face_patch = get_patch(landmarks[0:17])
    l_eyebrow = get_patch(landmarks[17:22], color='yellow')
    r_eyebrow = get_patch(landmarks[22:27], color='yellow')
    nose_v = get_patch(landmarks[27:31], color='orange')
    nose_h = get_patch(landmarks[31:36], color='orange')
    l_eye = get_patch(landmarks[36:42], color='magenta', closed=True)
    r_eye = get_patch(landmarks[42:48], color='magenta', closed=True)
    outer_lips = get_patch(landmarks[48:60], color='cyan', closed=True)
    inner_lips = get_patch(landmarks[60:68], color='blue', closed=True)

    ax.add_patch(face_patch)
    ax.add_patch(l_eyebrow)
    ax.add_patch(r_eyebrow)
    ax.add_patch(nose_v)
    ax.add_patch(nose_h)
    ax.add_patch(l_eye)
    ax.add_patch(r_eye)
    ax.add_patch(outer_lips)
    ax.add_patch(inner_lips)

    plt.axis('off')
    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    # assert width == height
    # assert width == size
    buffer = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    buffer = buffer[:, :, 0:3]
    plt.close(fig)
    return PIL.Image.fromarray(buffer)


def apply_image_padding(image, pad_percent=0.2):
    height, width = image.shape[:2]
    pad_height = int(height * pad_percent)
    pad_width = int(width * pad_percent)

    image_pad = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    offset = [pad_width, pad_height]
    return image_pad, offset


def add_bbox_margin(roi, h_img, w_img, exp_ratio):
    if roi.size > 4:
        x1, y1 = np.min(roi, axis=0)
        x2, y2 = np.max(roi, axis=0)
    else:
        x1, y1, x2, y2 = roi

    w_box = x2 - x1
    h_box = y2 - y1

    if exp_ratio >= 0:
        x1 = max(0, int(math.floor(x1 - w_box * exp_ratio)))
        x2 = min(w_img, int(math.ceil(x2 + w_box * exp_ratio)))
        y1 = max(0, int(math.floor(y1 - h_box * exp_ratio)))
        y2 = min(h_img, int(math.ceil(y2 + h_box * exp_ratio)))
    else:
        x1 = 0
        x2 = w_img
        y1 = 0
        y2 = h_img

    return [x1, y1, x2, y2]


def make_bbox_square(bbox, h_img, w_img, bbox_src=None, mode="large"):
    x1, y1, x2, y2 = bbox
    w_box = x2 - x1
    h_box = y2 - y1

    center_h = int((y1 + y2) / 2)
    center_w = int((x1 + x2) / 2)
    if bbox_src is not None:
        center_h_src = int((bbox_src[1] + bbox_src[3]) / 2)
        center_w_src = int((bbox_src[0] + bbox_src[2]) / 2)

    if mode == "large":
        size = max(w_box, h_box)
    elif mode == "small":
        size = min(w_box, h_box)
    half_size = int((size - size % 2) // 2)
    half_size_h = int((h_box - h_box % 2) // 2)
    half_size_w = int((w_box - w_box % 2) // 2)

    if bbox_src is not None:
        offset_h = center_h_src - center_h
        offset_w = center_w_src - center_w

        center_h = center_h + offset_h
        center_w = center_w + offset_w

        center_h = min(max(half_size_h, center_h), h_img - half_size_h)
        center_w = min(max(half_size_w, center_w), w_img - half_size_w)

    dx1 = center_w
    dx2 = w_img - center_w
    dy1 = center_h
    dy2 = h_img - center_h
    half_size = min(dx1, dx2, dy1, dy2, half_size)

    x1 = max(center_w - half_size, 0)
    x2 = min(center_w + half_size, w_img)
    y1 = max(center_h - half_size, 0)
    y2 = min(center_h + half_size, h_img)

    assert (x2 - x1) == (y2 - y1), "BBox is not square!"

    return [x1, y1, x2, y2]


def crop_bbox(image, bbox, exp_ratio, use_square=False):
    new_box = add_bbox_margin(bbox, image.shape[0], image.shape[1], exp_ratio)

    if exp_ratio >=0 and use_square:
        new_box = make_bbox_square(new_box, image.shape[0], image.shape[1], bbox)

    new_box = np.array(new_box).astype(np.int32)
    left, top, right, bottom = new_box

    img_face = image[top: bottom, left: right].copy()

    return img_face, new_box


def get_area(box):
    height = box[3] - box[1]
    width = box[2] - box[0]

    area = height * width
    return area


def convert_insightface2facer(all_face_info):
    rects = []
    points = []
    scores = []
    image_ids = []
    for info in all_face_info:
        rects.append(torch.from_numpy(info["bbox"].astype(np.float32)))
        points.append(torch.from_numpy(info["kps"].astype(np.float32)))
        scores.append(info["det_score"])
        image_ids.append(0)

    if len(rects) == 0:
        return {
            'rects': torch.Tensor(),
            'points': torch.Tensor(),
            'scores': torch.Tensor(),
            'image_ids': torch.Tensor(),
        }

    faces = {
        'rects': torch.stack(rects, dim=0),
        'points': torch.stack(points, dim=0),
        'scores': torch.tensor(scores),
        'image_ids': torch.tensor(image_ids),
    }

    return faces


class FaceAnalyser():
    def __init__(self, det_thresh=0.5, min_h=150, min_w=150, exp_ratio=0.25, use_square=True,
                 align=False, image_size=256, det_mode="insightface", td_mode="", device_id=0):

        if torch.cuda.is_available():
            device = f"cuda:{device_id}"
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": device_id},
                )
            ]
            ctx_id = device_id
        else:
            device = "cpu"
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        self.device = device

        self.det_thresh = det_thresh
        self.min_h = min_h
        self.min_w = min_w
        self.exp_ratio = exp_ratio
        self.use_square = use_square
        self.align = align
        self.image_size = image_size
        self.det_mode = det_mode
        self.td_mode = td_mode

        model = FaceAnalysis(name='buffalo_l', root="./checkpoints", providers=providers,
                             allowed_modules=['detection', 'landmark_3d_68', 'genderage'])
        model.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for GPU
        self.model = model

        for task_name, component in model.models.items():
            session = getattr(component, "session", None)
            active_providers = session.get_providers()
            assert active_providers[0] == "CUDAExecutionProvider", f"active providers: {active_providers} for {task_name}"

        if det_mode == "yolo":
            model_yolo = YOLO('./checkpoints/yolov8l_100e.pt')
            self.model_yolo = model_yolo.to(device)

        # use facer to get better lms
        self.face_aligner = facer.face_aligner('farl/wflw/448', device=device)  # optional: "farl/ibug300w/448", "farl/wflw/448", "farl/aflw19/448"

        if td_mode == "flame":
            self.model_3d = SMIRK(device=device)
        elif td_mode == "3ddfa":
            self.model_3d = TDDFAV3(device=device)

    @torch.no_grad()
    def get_facer_lms(self, img, all_face_info):
        faces = convert_insightface2facer(all_face_info)

        if faces["rects"].shape[0] > 0:
            faces = {k: v.to(self.device) for k, v in faces.items()}
            img_tensor = facer.hwc2bchw(torch.from_numpy(img)).to(device=self.device)  # image: 1 x 3 x h x w
            faces = self.face_aligner(img_tensor, faces)

            for idx, info in enumerate(all_face_info):
                info["landmark_98"] = faces["alignment"][idx].cpu().numpy()

        torch.cuda.empty_cache()

        return all_face_info

    @torch.no_grad()
    def detect_face(self, img_bgr, verbose=False):
        if self.det_mode == "insightface":
            bboxes, kpss = self.model.det_model.detect(img_bgr,
                                                 max_num=0,
                                                 metric='default')
            if bboxes.shape[0] == 0:
                return []
            all_face_info = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[i]
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                all_face_info.append(face)
        elif self.det_mode == "yolo":
            results = self.model_yolo.predict(img_bgr, verbose=verbose)
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()

            all_face_info = []
            for j in range(len(bboxes)):
                (x1, y1, x2, y2), score, c = bboxes[j], scores[j], cls[j]

                face = Face(bbox=bboxes[j], kps=None, det_score=score)
                all_face_info.append(face)

        return all_face_info

    @torch.no_grad()
    def get_face_info_insightface(self, img_bgr: np.ndarray, verbose=False):
        all_face_info = self.detect_face(img_bgr, verbose)

        if len(all_face_info) == 0:
            img_bgr_pad, offset = apply_image_padding(img_bgr)
            all_face_info = self.detect_face(img_bgr_pad, verbose)

            for info in all_face_info:
                info.bbox = info.bbox - np.array(offset + offset)
                info.kps = info.kps - np.expand_dims(np.array(offset), axis=0)
                # info.landmark_3d_68[:, :2] = info.landmark_3d_68[:, :2] - np.expand_dims(np.array(offset), axis=0)
                # info.landmark_2d_106 = info.landmark_2d_106 - np.expand_dims(np.array(offset), axis=0)

        for face in all_face_info:
            for taskname, model in self.model.models.items():
                if taskname=='detection' or (taskname=='recognition' and self.det_mode == "yolo"):
                    continue
                model.get(img_bgr, face)

        try:
            self.get_facer_lms(img_bgr[:, :, ::-1].copy(), all_face_info)
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
            msg = str(e).lower()

            if (
                    "out of memory" in msg
                    or "cudaerrormemoryallocation" in msg
                    or "memory allocation" in msg
            ):
                print("CUDA OOM caught:", e)
                all_face_info = []

                import gc
                gc.collect()
                torch.cuda.empty_cache()

        face_info = []
        is_small_face = False
        for info in all_face_info:
            if info.det_score < self.det_thresh:
                continue

            x1, y1, x2, y2 = info.bbox  # x1, y1, x2, y2
            center_h = (y1 + y2) / 2
            center_w = (x1 + x2) / 2
            x1 = max(0, x1)
            x2 = min(img_bgr.shape[1], x2)
            y1 = max(0, y1)
            y2 = min(img_bgr.shape[0], y2)
            if (x2 - x1) < self.min_w or (y2 - y1) < self.min_h or \
                    center_w < 0 or center_w >= img_bgr.shape[1] or \
                    center_h < 0 or center_h >= img_bgr.shape[0]:
                is_small_face = True
                continue
            is_small_face = False

            face_info.append({})
            face_info[-1]["det_score"] = info.det_score
            face_info[-1]["bbox"] = info.bbox
            face_info[-1]["bbox_crop"] = info.bbox
            face_info[-1]["bbox_face"] = info.bbox
            face_info[-1]["kps"] = info.kps
            face_info[-1]["landmark_3d_68"] = info.landmark_3d_68
            face_info[-1]["landmark_98"] = info.landmark_98
            # face_info[-1]["face_emb"] = info.embedding  # for tight face
            face_info[-1]["age"] = info.age
            face_info[-1]["gender"] = info.gender  # 0-female

            if self.align:
                img_face = face_align.norm_crop(img_bgr, info.kps,
                                                image_size=self.image_size)  # Align face using keypoints
            else:
                img_face, new_box = crop_bbox(img_bgr, info.bbox, exp_ratio=self.exp_ratio, use_square=self.use_square)
                face_info[-1]["bbox_crop"] = new_box

            offset = face_info[-1]["bbox_crop"][:2].astype(np.float32)
            face_info[-1]["face"] = img_face
            face_info[-1]["bbox_face"] = face_info[-1]["bbox_face"] - np.concatenate([offset, offset])
            face_info[-1]["landmark_68_face"] = face_info[-1]["landmark_3d_68"][:, :2] - offset
            face_info[-1]["landmark_98_face"] = face_info[-1]["landmark_98"] - offset
            face_info[-1]["kps_face"] = face_info[-1]["kps"] - offset

            if verbose:
                show_face_result(img_bgr, info.bbox, info.kps, info.landmark_3d_68[:, :2])
                show_face_result(img_face, bbox=face_info[-1]["bbox_face"], lms=face_info[-1]["kps_face"],
                                 lms68=face_info[-1]["landmark_68_face"])
                show_face_result(img_face, bbox=face_info[-1]["bbox_face"], lms68=face_info[-1]["landmark_98_face"])

        return face_info, is_small_face

    def get_3d(self, face_info, verbose=False):
        for i, info in enumerate(face_info):
            img_face = info["face"][:, :, ::-1].copy()

            if self.td_mode == "flame":
                face_3d, param_3d, _ = self.model_3d(img_face, info["bbox_face"])
            elif self.td_mode == "3ddfa":
                face_3d, param_3d, _ = self.model_3d(img_face, info["kps_face"])

            face_info[i]["face_3d"] = PIL.Image.fromarray(face_3d)
            face_info[i]["param_3d"] = param_3d

        if verbose:
            show_face_result(face_info[0]["face"]*0.5 + np.array(face_info[0]["face_3d"])*0.5)

    def get_face_info(self, img_bgr, verbose=False):
        face_info, is_small_face = self.get_face_info_insightface(img_bgr=img_bgr, verbose=verbose)

        if self.td_mode:
            self.get_3d(face_info, verbose=verbose)

        return face_info, is_small_face

    def get_lms_image(self, lms68: np.ndarray, height, width):
        img_lms = conditioning_from_landmarks(lms68.tolist(), height, width)

        return img_lms

    def find_largest_face(self, face_info):
        score_buf = []
        for info in face_info:
            score_buf.append(get_area(info["bbox"]))
        pick_idx = np.argmax(score_buf)
        return pick_idx


# follow-your-emoji: dilate eye and eyebrow mask
def dilate_mask(mask, kernel_size=3, num_iter=4):
    assert kernel_size % 2 == 1

    # mask: [B, 1, H, W], [B, H, W] or [H, W]
    original_ndim = mask.ndim
    if original_ndim == 2:
        # [H, W] -> [1, 1, H, W]
        x = mask.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        # [B, H, W] -> [B, 1, H, W]
        x = mask.unsqueeze(1)
    else:
        # [B, C, H, W] -> unchanged
        x = mask

    padding = (kernel_size - 1) // 2
    x = x.float()

    for _ in range(num_iter):
        # mask_points = cv2.findNonZero(mask[0].cpu().numpy().astype(np.uint8))
        # x, y, w, h = cv2.boundingRect(mask_points)

        x = torch.nn.functional.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    out = (x > 0).float()

    if original_ndim == 2:
        # [1, 1, H, W] -> [H, W]
        out = out.squeeze(0).squeeze(0)
    elif original_ndim == 3:
        # [B, 1, H, W] -> [B, H, W]
        out = out.squeeze(1)

    return out


class FaceParser():
    def __init__(self, mode="lapa", device_id=0):
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.mode = mode

        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)

        if mode == "lapa":
            self.face_parser = facer.face_parser('farl/lapa/448', device=device)  # optional "farl/celebm/448"
            self.label_names = ['background', 'face', 'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip', 'hair']
            self.bg_label = ['background', 'hair']
        elif mode == "celebm":
            self.face_parser = facer.face_parser('farl/celebm/448', device=device)
            self.label_names = ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're', 'le', 'nose',
                                'imouth', 'llip', 'ulip', 'hair', 'eyeg', 'hat', 'earr', 'neck_l']
            self.bg_label = ['background', 'neck', 'cloth', 'rr', 'lr', 'hair', 'eyeg', 'hat', 'earr', 'neck_l']

    def set_bg_label(self, bg_label):
        self.bg_label = bg_label

    @torch.no_grad()
    def get_face_seg(self, img: np.ndarray, face_info=None, verbose=False):
        img_tensor = facer.hwc2bchw(torch.from_numpy(img)).to(device=self.device)  # image: 1 x 3 x h x w

        if face_info is not None:
            faces = convert_insightface2facer(face_info)
            faces = {k: v.to(self.device) for k, v in faces.items()}
        else:
            faces = self.face_detector(img_tensor)
            if faces["rects"].shape[0] == 0:
                img_pad, offset = apply_image_padding(img)
                img_tensor_pad = facer.hwc2bchw(torch.from_numpy(img_pad)).to(device=self.device)  # image: 1 x 3 x h x w
                faces = self.face_detector(img_tensor_pad)

                if faces["rects"].shape[0] == 0:
                    return None, None, None, None

                faces["rects"] = faces["rects"] - torch.tensor(offset + offset).unsqueeze(dim=0).to(device=self.device)
                faces["points"] = faces["points"] - torch.tensor(offset).unsqueeze(dim=0).unsqueeze(dim=0).to(device=self.device)

        pick_idx = self.find_largest_face(faces)

        faces = self.face_parser(img_tensor, faces)

        label_names = faces['seg']['label_names']
        for n1, n2 in zip(self.label_names, label_names):
            assert n1 == n2
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
        vis_img = vis_seg_probs.sum(0, keepdim=True)

        seg_preds = seg_probs.argmax(dim=1)
        seg_masks, seg_masks_dilate = self.get_seg_mask(seg_preds, label_names)

        bg_mask = self.pick_mask_from_labels(seg_preds, label_names, label_pick=self.bg_label)
        face_mask = 1. - bg_mask

        if verbose:
            alpha = 0.5
            img_mix = alpha * img + (1. - alpha) * seg_masks[pick_idx, 2].unsqueeze(dim=-1).cpu().numpy() * 255
            show_result(img_mix)
            img_mix = alpha * img + (1. - alpha) * seg_masks_dilate[pick_idx, 4].unsqueeze(dim=-1).cpu().numpy() * 255
            show_result(img_mix)
            img_mix = alpha * img + (1. - alpha) * face_mask[pick_idx].unsqueeze(dim=-1).cpu().numpy() * 255
            show_result(img_mix)
            facer.show_bhw(vis_img)
            facer.show_bchw(facer.draw_bchw(img_tensor, faces))

        return seg_masks, seg_masks_dilate, face_mask, pick_idx

    def get_mask_from_pred(self, seg_preds, label_all):
        N, H, W = seg_preds.shape
        seg_masks = torch.zeros((N, len(label_all), H, W), device=seg_preds.device)  # nfaces x nclasses x h x w

        for idx, label in enumerate(label_all):
            seg_masks[:, idx] = (seg_preds == idx).float()

        return seg_masks

    def get_pred_from_mask(self, seg_masks):
        BS, _, H, W = seg_masks.shape
        seg_preds = torch.zeros((BS, H, W), device=seg_masks.device)

        for idx in range(seg_masks.shape[1]):
            seg_preds[seg_masks[:, idx] == 1] = idx

        return seg_preds

    def get_seg_mask(self, seg_preds, label_all):
        dilate_label_list = ['rb', 'lb', 're', 'le', 'ulip', 'llip']
        dilate_iter_list = [4, 4, 24, 24, 4, 4]
        dilate_factor = max(seg_preds.shape[1:]) / 1024

        seg_masks = self.get_mask_from_pred(seg_preds, label_all)
        # show_result(seg_masks[0, idx].unsqueeze(dim=0) * 255)

        # get dilate masks
        seg_preds_dilate = seg_preds.clone()
        for i, dilate_label in enumerate(dilate_label_list):
            dilate_idx = np.where(np.array(label_all) == dilate_label)[0][0]
            dilate_iter = max(math.ceil(dilate_iter_list[i]*dilate_factor), 1)
            mask_dilate = dilate_mask(seg_masks[:, dilate_idx].clone(), num_iter=dilate_iter)
            seg_preds_dilate[mask_dilate == 1] = dilate_idx

        seg_masks_dilate = self.get_mask_from_pred(seg_preds_dilate, label_all)
        assert torch.equal(seg_masks_dilate.sum(dim=(1, 2, 3)), seg_masks.sum(dim=(1, 2, 3)))
        assert torch.equal(self.get_pred_from_mask(seg_masks_dilate), seg_preds_dilate)

        return seg_masks, seg_masks_dilate

    def pick_mask_from_labels(self, seg_preds, label_all, label_pick, seg_masks=None):
        if seg_masks is None:
            mask = torch.zeros_like(seg_preds)

            for label in label_pick:
                assert label in label_all
                idx = np.where(np.array(label_all) == label)[0][0]
                mask[seg_preds == idx] = 1
        else:
            BS, _, H, W = seg_masks.shape
            mask = torch.zeros((BS, H, W), device=seg_masks.device)

            for label in label_pick:
                assert label in label_all
                idx = np.where(np.array(label_all) == label)[0][0]
                mask += seg_masks[:, idx]

        return mask

    def find_largest_face(self, face_info):
        score_buf = []
        for idx in range(face_info["rects"].shape[0]):
            score_buf.append(get_area(face_info["rects"][idx]))
        pick_idx = torch.argmax(torch.tensor(score_buf))
        return pick_idx


if __name__ == '__main__':
    import copy

    verbose = True

    face_analyser = FaceAnalyser(det_thresh=0.5, min_h=150, min_w=150, exp_ratio=0.25, align=False, td_mode="3ddfa")
    face_parser = FaceParser()

    img_path = "./assets/images/test-swap2.png"

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

    face_info, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr, verbose=verbose)
    pick_idx = face_analyser.find_largest_face(face_info)
    img_face = face_info[pick_idx]["face"]
    img_face_rgb = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)

    img_face_lms = face_analyser.get_lms_image(face_info[pick_idx]["landmark_68_face"], img_face.shape[0], img_face.shape[1])

    face_info_face = copy.deepcopy(face_info)
    for info, info_face in zip(face_info, face_info_face):
        info_face["bbox"] = info["bbox_face"]
        info_face["kps"] = info["kps_face"]
    seg_masks, seg_masks_dilate, face_mask, pick_idx = \
        face_parser.get_face_seg(img_face_rgb, face_info_face, verbose=verbose)


    img_path1 = "./assets/images/306.png"
    img_path2 = "./assets/images/aaa.jpg"
    # img_path2 = "./assets/images/2855.png"
    img_bgr1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img_bgr2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    face_info1, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr1, verbose=verbose)
    face_info2, is_small_face = face_analyser.get_face_info(img_bgr=img_bgr2, verbose=verbose)
    R_dist = face_analyser.model_3d.calc_rel_rotation(face_info1[0]["param_3d"], face_info2[0]["param_3d"])
    print(R_dist)
