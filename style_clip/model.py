#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple, Union
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

# import clip
from transformers import CLIPVisionConfig, CLIPTextConfig, CLIPVisionModelWithProjection, AutoProcessor, \
    CLIPTextModelWithProjection, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from style_clip.losses import SupConLoss
from style_clip.distributed import is_dist_avail_and_initialized

# id, backbone dim, proj dim
CLIP_ARCH = {
    "vit_base": ["openai/clip-vit-base-patch16", 768, 512],
    "vit_large": ["openai/clip-vit-large-patch14", 1024, 768],
    "vit_huge": ["laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1280, 1024],
    "vit_giant": ["laion/CLIP-ViT-g-14-laion2B-s12B-b42K", 1408, 1024],
}


class StyleLoss(nn.Module):
    def __init__(self, temperature=0.07, args=None):
        super().__init__()
        self.temperature = temperature
        self.args = args

        self.loss_content = SupConLoss(temperature=temperature)
        self.loss_style = SupConLoss(temperature=temperature)
        self.loss_text = SupConLoss(temperature=temperature)

    def forward(self, content_emb, style_emb, text_features=None, targets=None, text_labels=None):
        # style_emb = utils.split_reshape(style_emb, self.batch_size_per_gpu, [0, 1])
        # content_emb = utils.split_reshape(content_emb, self.batch_size_per_gpu, [0, -1])
        style_emb = torch.stack([style_emb[:, 0], style_emb[:, 1]], dim=1)
        content_emb = torch.stack([content_emb[:, 0], content_emb[:, -1]], dim=1)

        # Gather tensors from all GPUs
        if is_dist_avail_and_initialized():
            content_emb = torch.cat(torch.distributed.nn.all_gather(content_emb), dim=0)
            style_emb = torch.cat(torch.distributed.nn.all_gather(style_emb), dim=0)
            text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0).detach()
            targets = torch.cat(torch.distributed.nn.all_gather(targets), dim=0).detach()
            text_labels = torch.cat(torch.distributed.nn.all_gather(text_labels), dim=0).detach()

        # Compute content loss (SimCLR loss, doesn't use labels)
        loss_c = self.loss_content(content_emb)
        # loss_c = loss_c.clamp(max=self.args.clamp_content_loss)
        # loss_c = -1 * loss_c

        loss_s_sup = self.loss_style(style_emb, labels=targets)
        loss_s_ssl = self.loss_style(style_emb)

        text_features = torch.cat([text_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1)
        loss_text = self.loss_text(style_emb, pos_features=text_features, labels=text_labels)

        loss_s = self.args.lambda_sup * loss_s_sup + self.args.lambda_ssl * loss_s_ssl + self.args.lambda_text * loss_text
        loss = self.args.lambda_c * loss_c + self.args.lambda_s * loss_s

        loss_dict = {}
        loss_dict["l_s_ssl"] = loss_s_ssl
        loss_dict["l_s_sup"] = loss_s_sup
        loss_dict["l_text"] = loss_text
        loss_dict["l_c"] = loss_c
        loss_dict["loss"] = loss

        return loss_dict


def _tie_weights(src_model, tgt_model):
    for param_src, param_tgt in zip(src_model.parameters(), tgt_model.parameters()):
        param_tgt.data.copy_(param_src.data)


# def set_freeze_to_eval(model):
#     for name, module in model.named_modules():
#         # Check if the module has parameters
#         params = list(module.parameters(recurse=False))
#         if not params:
#             continue  # Skip modules without parameters
#
#         # If all parameters are frozen, set this module to eval
#         if all(not p.requires_grad for p in params):
#             module.eval()


# def set_freeze_to_eval(model):
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             set_freeze_to_eval(module)
#
#         params = list(module.parameters())
#         if (len(params) > 0) and all([not p.requires_grad for p in params]):
#             module.eval()
#
#     return model


class CustomTokenizer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_ARCH[model_name][0])

    def forward(self, text):
        text_input = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # text_input = self.tokenizer(text, padding=True, return_tensors="pt")

        return text_input


class StyleCLIP(nn.Module):
    def __init__(self, model_name, text_model_name="vit_giant", use_text=True):
        super().__init__()

        self.use_text = use_text

        model_id = CLIP_ARCH[model_name][0]
        self.clip_dim = CLIP_ARCH[model_name][1]
        projection_dim = CLIP_ARCH[model_name][2]

        text_model_id = CLIP_ARCH[text_model_name][0]
        text_projection_dim = CLIP_ARCH[text_model_name][2]

        if self.use_text:
            text_config = CLIPTextConfig.from_pretrained(text_model_id)
            text_config.projection_dim = text_projection_dim
            text_model = CLIPTextModelWithProjection.from_pretrained(text_model_id, config=text_config)
            text_model.requires_grad_(False)
            text_model.eval()
            self.text = text_model

        vision_config = CLIPVisionConfig.from_pretrained(model_id)
        vision_config.projection_dim = projection_dim
        assert self.clip_dim == vision_config.hidden_size
        vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id, config=vision_config)
        vision_model.requires_grad_(True)

        self.style_proj = nn.Linear(vision_model.config.hidden_size, text_projection_dim, bias=False)
        self.content_proj = nn.Linear(vision_model.config.hidden_size, text_projection_dim, bias=False)
        if text_projection_dim == vision_model.config.projection_dim:
            _tie_weights(vision_model.visual_projection, self.style_proj)
            _tie_weights(vision_model.visual_projection, self.content_proj)
        self.visual = vision_model
        self.visual.visual_projection = nn.Identity()

        self.processor = AutoProcessor.from_pretrained(model_id)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            if isinstance(self.visual, PeftModel):
                self.visual.eval()
                for name, module in self.visual.named_modules():
                    if isinstance(module, LoraLayer):
                        module.train()
            else:
                self._set_freeze_to_eval(self._get_vision_groups())

        if self.use_text:
            self.text.eval()

        return self

    def _set_freeze_to_eval(self, x):
        if isinstance(x, Sequence):
            for g in x:
                self._set_freeze_to_eval(g)
        else:
            params = list(x.parameters())
            if (len(params) > 0) and all([not p.requires_grad for p in params]):
                x.eval()

    @property
    def dtype(self):
        return self.visual.dtype

    def _get_vision_groups(self):
        groups = [
            [
                self.visual.vision_model.embeddings,
                self.visual.vision_model.pre_layrnorm,
            ],
            *self.visual.vision_model.encoder.layers[:-1],
            [
                self.visual.vision_model.encoder.layers[-1],
                self.visual.vision_model.post_layernorm,
            ]
        ]

        return groups

    # openclip
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.visual.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = self._get_vision_groups()

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    # https://github.com/ssundaram21/dreamsim/training/train.py
    def prep_lora_model(self, lora_r=8, lora_alpha=8, lora_dropout=0.2):
        # target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        target_modules = []
        for idx in range(20, 24):
            target_modules.append(f"layers.{idx}.self_attn.q_proj")
            target_modules.append(f"layers.{idx}.self_attn.k_proj")
            target_modules.append(f"layers.{idx}.self_attn.v_proj")
            target_modules.append(f"layers.{idx}.self_attn.out_proj")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=target_modules
        )
        self.visual = get_peft_model(self.visual, config)

    def encode_image(self, image, normalize=True, output_hidden_states=False):
        outputs = self.visual(image, output_hidden_states=output_hidden_states)
        # feature = outputs.pooler_output  # pooled CLS states
        emb = outputs.image_embeds  # with proj
        if output_hidden_states:
            featmap = outputs.hidden_states
        else:
            featmap = [outputs.last_hidden_state]

        content_emb = self.content_proj(emb)
        style_emb = self.style_proj(emb)

        if normalize:
            content_emb = F.normalize(content_emb, dim=-1, p=2)
            style_emb = F.normalize(style_emb, dim=-1, p=2)

        return emb, featmap, content_emb, style_emb

    def encode_text(self, text, normalize=True):
        outputs = self.text(**text)
        features = outputs.text_embeds

        if normalize:
            features = F.normalize(features, dim=-1, p=2)

        return features

    @torch.no_grad()
    def get_image_feat(self, img_input, hidden_layer_idx=[24]):
        if isinstance(img_input, PIL.Image.Image):
            img_tensor = self.processor(images=img_input, return_tensors="pt").pixel_values
        elif isinstance(img_input, torch.Tensor):
            img_tensor = img_input
        assert img_tensor.ndim == 4

        img_tensor = img_tensor.to(device=self.visual.device, dtype=self.visual.dtype)
        emb, featmap, content_emb, style_emb = self.encode_image(img_tensor, normalize=False, output_hidden_states=True)

        if isinstance(hidden_layer_idx, str):
            hidden_layer_idx = [int(a) for a in hidden_layer_idx.split(",")]
        featmap_select = []
        for idx in hidden_layer_idx:
            featmap_select.append(featmap[idx][:, 1:])
        featmap = torch.cat(featmap_select, dim=1)

        return emb, featmap, content_emb, style_emb

    def forward(self, image, text=None, output_dict=True):
        if image.ndim > 4:
            batch_size, n_view, _, _, _ = image.shape
            image = torch.cat(torch.unbind(image, dim=1), dim=0)
        else:
            batch_size, _, _, _ = image.shape

        image_emb, image_featmap, content_emb, style_emb = self.encode_image(image)

        if image_emb.shape[0] > batch_size:
            image_emb = torch.stack(torch.chunk(image_emb, n_view, dim=0), dim=1)
            image_featmap = [torch.stack(torch.chunk(featmap, n_view, dim=0), dim=1) for featmap in image_featmap]
            content_emb = torch.stack(torch.chunk(content_emb, n_view, dim=0), dim=1)
            style_emb = torch.stack(torch.chunk(style_emb, n_view, dim=0), dim=1)

        text_features = None
        if text is not None:
            with torch.no_grad():
                text_features = self.encode_text(text)

        if output_dict:
            out_dict = {
                "image_emb": image_emb,
                "content_emb": content_emb,
                "style_emb": style_emb,
                "text_features": text_features,
            }
            return out_dict

        return image_emb, content_emb, style_emb, text_features


if __name__ == '__main__':
    import numpy as np
    from style_clip import clip_utils
    import open_clip

    batch_size = 4
    n_view = 2
    x = torch.randn(batch_size, n_view, 3, 224, 224)
    targets = torch.randint(100, (batch_size,))
    text_labels = torch.randint(200, (batch_size,))

    lock_image = False

    tokenizer = CustomTokenizer("vit_large")
    text = tokenizer(["a photo of a cat"] * batch_size)

    open_tokenizer = open_clip.get_tokenizer('ViT-H-14')
    open_text = open_tokenizer(["a photo of a cat", "a dog", "a cat"])


    model = StyleCLIP("vit_large")

    if lock_image:
        model.lock_image_tower(unlocked_groups=1, freeze_bn_stats=True)
    else:
        model.prep_lora_model()

    for p in model.parameters():
        assert p.dtype == torch.float32

    if lock_image:
        params = clip_utils.collect_params(model)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    params_name = []
    for pn, p in model.named_parameters():
        if p.requires_grad:
            params_name.append(pn)
            print(pn)

    model.train()

    # import clip
    # clipmodel, _ = clip.load("ViT-B/16")
    # print(list(clipmodel.visual.named_parameters()))

    out = model(x, text)
    content_emb, style_emb, text_features = out["content_emb"], out["style_emb"], out["text_features"]

    random_pixels = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    random_image = PIL.Image.fromarray(random_pixels, 'RGB')
    model.get_image_feat(random_image, hidden_layer_idx="21,24")

    criterion = StyleLoss(0.07)
    loss = criterion(content_emb, style_emb, text_features, targets=targets, text_labels=text_labels)