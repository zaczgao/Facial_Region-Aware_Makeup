#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3DDFA:
# Some results in the paper are rendered by pytorch3d and nvdiffrast
# This repository only uses nvdiffrast for convenience.
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
git checkout v0.4.0
pip install --no-build-isolation .

in /envs/torch-2/lib/python3.10/site-packages/nvdiffrast/nvdiffrast/torch/ops.py
_cached_plugin[gl] = torch.utils.cpp_extension.load(
# Import, cache, and return the compiled module.
#_cached_plugin[gl] = importlib.import_module(plugin_name)

# In some scenarios, nvdiffrast may not be usable. Therefore, we additionally provide a fast CPU renderer based on face3d.
# The results produced by the two renderers may have slight differences, but we consider these differences to be negligible.
# Please note that we still highly recommend using nvdiffrast.
cd util/cython_renderer/
python setup.py build_ext -i

pytorch3d: build on server sbg node with gpu
module load cudnn/8.9.7.29-11-cuda-11.8.0-gcc-12.2.0
git clone https://github.com/facebookresearch/pytorch3d
cd pytorch3d
git checkout v0.7.9
pip install --no-build-isolation .

win:
x64 Native Tools Command Prompt
set DISTUTILS_USE_SDK=1

pip install chumpy --no-build-isolation

"""

__author__ = "GZ"

import os
import sys
import PIL.Image
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
from scipy.spatial.transform import Rotation as R_scipy

import torch

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

try:
    from TDDFA.model.recon import face_model
    from TDDFA.util.preprocess import load_lm3d, align_img
except ImportError:
    print("3DDFA not found")

try:
    from smirk.src.smirk_encoder import SmirkEncoder
    from smirk.src.FLAME.FLAME import FLAME
    from smirk.src.renderer.renderer import Renderer
    from smirk.src.FLAME.lbs import batch_rodrigues
except ImportError:
    print("smirk not found")

from utils.vis_utils import show_result, show_face_result


def back_resize_crop_img(img, trans_params, ori_img, resample_method=PIL.Image.BICUBIC):

    w0, h0, s, t, target_size = trans_params[0], trans_params[1], trans_params[2], [trans_params[3],
                                                                                    trans_params[4]], 224

    img = PIL.Image.fromarray(img)
    ori_img = PIL.Image.fromarray(ori_img)
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size

    old_img = ori_img
    old_img = old_img.resize((w, h), resample=resample_method)

    old_img.paste(img, (left, up, right, below))
    old_img = old_img.resize((int(w0), int(h0)), resample=resample_method)

    old_img = np.array(old_img)
    return old_img


class TDDFAV3():
    def __init__(self, device="cuda"):
        self.device = device

        self.lm3d_std = load_lm3d()
        self.model = face_model(backbone='resnet50', device=device, use_ldm68=True, use_ldm106=False, use_ldm134=False)

    @torch.no_grad()
    def __call__(self, img, lms5, alpha_dict_tgt=None):
        H = img.shape[0]

        landmarks = lms5.copy()
        landmarks[:, -1] = H - 1 - landmarks[:, -1]

        trans_params, img_face, lm_face, _ = align_img(PIL.Image.fromarray(img), landmarks, self.lm3d_std)
        img_face = torch.tensor(np.array(img_face) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        results, param_3d = self.model.forward(img_face.to(self.model.device), alpha_dict_tgt=alpha_dict_tgt)

        img_render = self.render(trans_params, img, results)

        return img_render, param_3d

    def render(self, trans_params, img, result_dict):
        render_shape = (result_dict['render_shape'][0]*255).astype(np.uint8)
        render_face = (result_dict['render_face'][0]*255).astype(np.uint8)
        render_mask  = (np.stack((result_dict['render_mask'][0][:,:,0],)*3, axis=-1)*255).astype(np.uint8)

        if trans_params is not None:
            render_shape = back_resize_crop_img(render_shape, trans_params, np.zeros_like(img), resample_method = PIL.Image.BICUBIC)
            render_face = back_resize_crop_img(render_face, trans_params, np.zeros_like(img), resample_method = PIL.Image.BICUBIC)
            render_mask = back_resize_crop_img(render_mask, trans_params, np.zeros_like(img), resample_method = PIL.Image.NEAREST)

        # render_shape = ((render_shape/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8)[:,:,::-1]

        return render_shape

    # def rotmat_to_ypr(self, R):
    #     """
    #     R: (B, 3, 3) relative rotation
    #     Returns yaw, pitch, roll in radians
    #     """
    #     yaw = torch.atan2(R[:, 0, 2], R[:, 2, 2])
    #     pitch = torch.asin(torch.clamp(-R[:, 1, 2], -1.0, 1.0))
    #     roll = torch.atan2(R[:, 1, 0], R[:, 1, 1])
    #     return yaw, pitch, roll

    def rotmat_to_ypr(self, R):
        """
        R: (B,3,3) returned from compute_rotation()
        Returns:
            yaw, pitch, roll (radians)
        """

        # because matrix is transposed,
        # we read elements accordingly

        sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 0, 1] ** 2)
        sy = torch.clamp(sy, min=1e-6)

        pitch = torch.atan2(R[:, 1, 2], R[:, 2, 2])  # X
        yaw = torch.atan2(-R[:, 0, 2], sy)  # Y
        roll = torch.atan2(R[:, 0, 1], R[:, 0, 0])  # Z

        return yaw, pitch, roll

    def calc_rel_rotation(self, param_src, param_tgt):
        if not isinstance(param_src, list):
            param_src = [param_src]
            param_tgt = [param_tgt]

        if isinstance(param_src[0], dict):
            angle_src = [param["angle"] for param in param_src]
            angle_tgt = [param["angle"] for param in param_tgt]
        else:
            angle_src = param_src
            angle_tgt = param_tgt

        angle_src = torch.cat(angle_src).to(self.device)
        angle_tgt = torch.cat(angle_tgt).to(self.device)

        R_src = self.model.compute_rotation(angle_src)  # bs,3,3
        R_tgt = self.model.compute_rotation(angle_tgt)

        R_rel = torch.matmul(R_tgt, R_src.transpose(1, 2))  # relative rotation

        # Yaw / Pitch / Roll from relative rotation
        yaw, pitch, roll = self.rotmat_to_ypr(R_rel)
        yaw_deg = torch.rad2deg(yaw)
        pitch_deg = torch.rad2deg(pitch)
        roll_deg = torch.rad2deg(roll)

        # roll, yaw, pitch = R_scipy.from_matrix(R_rel[0].cpu().numpy()).as_euler('zyx', degrees=True)

        # Geodesic rotation distance (SO(3))
        trace = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle_rad)

        # R_rel = R_rel.cpu().numpy()
        # R_dist = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
        # R_dist = np.clip(R_dist, -1, 1)
        # R_dist = np.rad2deg(np.arccos(R_dist))

        return angle_deg, yaw_deg, pitch_deg, roll_deg


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    if landmarks.ndim == 2:
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])
    else:
        left = landmarks[0]
        right = landmarks[2]
        top = landmarks[1]
        bottom = landmarks[3]

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


class SMIRK():
    def __init__(self, device="cuda"):
        self.device = device
        self.image_size = 224

        smirk_encoder = SmirkEncoder().to(device)
        checkpoint = torch.load(os.path.join(SCRIPT_DIR, "../smirk/assets/SMIRK_em1.pt"))
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if
                              'smirk_encoder' in k}  # checkpoint includes both smirk_encoder and smirk_generator

        smirk_encoder.load_state_dict(checkpoint_encoder)
        smirk_encoder.eval()
        self.smirk_encoder = smirk_encoder

        self.flame = FLAME(flame_model_path=os.path.join(SCRIPT_DIR, "../smirk/assets/FLAME2020/generic_model.pkl"),
                           flame_lmk_embedding_path=os.path.join(SCRIPT_DIR, "../smirk/assets/landmark_embedding.npy")).to(device)
        self.renderer = Renderer(obj_filename=os.path.join(SCRIPT_DIR, "../smirk/assets/head_template.obj"),
                                 mask_path=os.path.join(SCRIPT_DIR, "../smirk/assets/FLAME_masks/FLAME_masks.pkl")).to(device)

    @torch.no_grad()
    def __call__(self, image, roi, param_3d_tgt=None):
        orig_image_height, orig_image_width, _ = image.shape

        tform = crop_face(image, roi, scale=1.4, image_size=self.image_size)
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_image = cropped_image.to(self.device)

        outputs = self.smirk_encoder(cropped_image)

        if param_3d_tgt is not None:
            outputs['expression_params'] = param_3d_tgt['expression_params'].clone()
            outputs['pose_params'] = param_3d_tgt['pose_params'].clone()
            outputs['jaw_params'] = param_3d_tgt['jaw_params'].clone()

        flame_output = self.flame.forward(outputs)
        renderer_output = self.renderer.forward(flame_output['vertices'], outputs['cam'],
                                           landmarks_fan=flame_output['landmarks_fan'],
                                           landmarks_mp=flame_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']

        rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(
            np.uint8)
        rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width),
                                 preserve_range=True).astype(np.uint8)

        return rendered_img_orig, outputs

    # https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html
    def calc_rel_rotation(self, param_src, param_tgt):
        if not isinstance(param_src, list):
            param_src = [param_src]
            param_tgt = [param_tgt]

        if isinstance(param_src[0], dict):
            pose_src = [param['pose_params'] for param in param_src]
            pose_tgt = [param['pose_params'] for param in param_tgt]
        else:
            pose_src = param_src
            pose_tgt = param_tgt

        pose_src = torch.cat(pose_src)
        pose_tgt = torch.cat(pose_tgt)

        batch_size = pose_src.shape[0]

        R_src = batch_rodrigues(pose_src.view(-1, 3)).view([batch_size, 3, 3])
        R_tgt = batch_rodrigues(pose_tgt.view(-1, 3)).view([batch_size, 3, 3])

        R_rel = torch.matmul(R_tgt, R_src.transpose(1, 2))  # relative rotation

        # Geodesic rotation distance (SO(3))
        trace = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle_rad)

        # R_rel = R_rel.cpu().numpy()
        # R_dist = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
        # R_dist = np.clip(R_dist, -1, 1)
        # R_dist = np.rad2deg(np.arccos(R_dist))

        return angle_deg


if __name__ == '__main__':
    pass
