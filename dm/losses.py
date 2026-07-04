#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft

from diffusers.models.attention_processor import Attention

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.vis_utils import show_result


# https://github.com/facebookresearch/detr/blob/main/models/detr.py
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def _compute_avg_attn_map(attn_probs):
    avg_attn_map = []
    for name in attn_probs:
        avg_attn_map.append(attn_probs[name])

    # average over layers
    avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(dim=0)  # (L,B,H,W,77) -> (B,H,W,77)
    return avg_attn_map


def _compute_attn_loss_sub(
        seg_mask,
        located_attn_map,
        attn_scale=0.1,
        gs_blur_attn=False,
        mode="pixel",
        eps=1e-7,
):
    B, M, H, W = located_attn_map.size()
    seg_mask = seg_mask.float()

    if gs_blur_attn:
        seg_mask = Ft.gaussian_blur(seg_mask, [3, 3], [1, 1])
        seg_mask = F.interpolate(seg_mask, (H, W), mode="bilinear")
    else:
        seg_mask = F.interpolate(seg_mask, (H, W), mode="nearest")

    seg_mask = seg_mask.to(located_attn_map.dtype)
    assert located_attn_map.shape == seg_mask.shape
    assert seg_mask.max() <= 1

    if mode == "seg":
        src_masks = located_attn_map.reshape(-1, H, W).flatten(1)
        target_masks = seg_mask.reshape(-1, H, W).flatten(1)
        loss_focal = sigmoid_focal_loss(src_masks, target_masks, B * M)
        loss_dice = dice_loss(src_masks, target_masks, B * M)
        loss = loss_focal + loss_dice
    elif mode == "pixel":
        log_probs = F.log_softmax(located_attn_map / attn_scale, dim=1)
        loss_pixel = torch.sum(-seg_mask * log_probs, dim=1, keepdim=True)   # B, 1, H, W

        loss_mask_sum = (loss_pixel * seg_mask).sum(dim=(2, 3))  # B, M
        pixel_per_mask = seg_mask.sum(dim=(2, 3))  # B, M
        loss_mask = loss_mask_sum / (pixel_per_mask + eps)
        active = (pixel_per_mask > 0).to(located_attn_map.dtype)  # B, M

        loss = torch.mean((loss_mask * active).sum(dim=1) / (active.sum(dim=1) + eps))
    elif mode == "part":
        target = seg_mask.flatten(2)  # B, M, HW
        active = (target.sum(dim=-1) > 0).to(located_attn_map.dtype)  # B, M
        target = target / (target.sum(dim=-1, keepdim=True) + eps)

        # Spatial distribution per part/token
        logits = located_attn_map.flatten(2) / attn_scale  # B, M, HW
        log_probs = F.log_softmax(logits, dim=-1)

        loss_per_part = -(target * log_probs).sum(dim=-1)  # B, M
        loss = (loss_per_part * active).sum() / (active.sum() + eps)

    return loss


def _compute_attn_loss(
        batch,
        cross_attn_probs,
        placeholder_token_ids,
        use_ipa,
        is_drop,
        attn_scale=0.1,
        gs_blur_attn=False,
        mode="pixel"
):
    seg_mask = batch["seg_mask"]
    avg_attn_map = _compute_avg_attn_map(cross_attn_probs)
    ori_B, H, W, seq_length = avg_attn_map.size()

    keep = ~is_drop.bool()
    if not keep.any():
        zero = avg_attn_map.new_zeros(())
        empty_map = avg_attn_map.new_zeros((ori_B, seg_mask.shape[1], H, W))
        return zero, zero.detach(), empty_map

    seg_mask = seg_mask[keep]
    avg_attn_map = avg_attn_map[keep]
    B = avg_attn_map.shape[0]

    # locate the attn map
    if use_ipa:
        located_attn_map = avg_attn_map.permute(0, 3, 1, 2)

        num_heads_part = located_attn_map.shape[1] // seg_mask.shape[1]
        located_attn_map = located_attn_map.reshape(B, -1, num_heads_part, H, W)
        located_attn_map = torch.mean(located_attn_map, dim=2)
    else:
        located_attn_map = []
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            for bi in range(B):
                if "input_ids" in batch:
                    learnable_idx = (batch["input_ids"][keep][bi] == placeholder_token_id).nonzero(as_tuple=True)[0]
                else:
                    learnable_idx = (
                            batch["input_ids_one"][keep][bi] == placeholder_token_id
                    ).nonzero(as_tuple=True)[0]

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

    attn_loss = _compute_attn_loss_sub(
        seg_mask,
        located_attn_map,
        attn_scale,
        gs_blur_attn,
        mode
    )

    return attn_loss, located_attn_map.detach().max(), located_attn_map


def calc_attn_loss(batch,
                   unet,
                   placeholder_token_ids,
                   use_ipa,
                   is_drop,
                   attn_size=16,
                   attn_scale=0.1,
                   gs_blur_attn=False,
                   mode="pixel"
                   ):
    if isinstance(attn_size, int):
        attn_size = [attn_size]

    cross_attn_probs = {}
    self_attn_probs = {}
    embeddings = {}
    for size in attn_size:
        cross_attn_probs[size] = {}
        self_attn_probs[size] = {}
        embeddings[size] = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and hasattr(module, "cross_attn_probs"):
            if module.cross_attn_probs is not None:
                a = module.cross_attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
                size = a.size(1)
                cross_attn_probs[size][name] = a
            elif module.self_attn_probs is not None:
                a = module.self_attn_probs.mean(
                    dim=1
                )  # (B,Head,H*W,H*W) -> (B,H*W,H*W)
                size = int(a.size(1) ** 0.5)
                self_attn_probs[size][name] = a

            if module.embeddings is not None:
                e = module.embeddings  # (B, HW, C)
                size = int(e.size(1) ** 0.5)
                embeddings[size][name] = e

    total_loss = 0
    avg_max = 0
    batch["located_attn_map"] = {}

    for size in attn_size:
        attn_loss, max_attn_val, located_attn_map = _compute_attn_loss(
            batch,
            cross_attn_probs[size],
            placeholder_token_ids,
            use_ipa,
            is_drop,
            attn_scale=attn_scale,
            gs_blur_attn=gs_blur_attn,
            mode=mode
        )
        total_loss += attn_loss
        avg_max += max_attn_val

        batch["located_attn_map"][size] = located_attn_map

    total_loss /= len(attn_size)
    avg_max /= len(attn_size)

    return total_loss, avg_max


if __name__ == '__main__':
    pass
