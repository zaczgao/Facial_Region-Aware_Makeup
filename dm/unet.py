#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torch

from diffusers import UNet2DConditionModel

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from utils.misc import set_trainable_modules_to_train


# - down blocks (3x down blocks) * (2x attention blocks) * (1x transformer layers)  = 6
# - mid blocks (1x mid blocks) * (1x attention blocks) * (1x transformer layers) = 1
# - up blocks (3x up blocks) * (3x attention blocks) * (1x transformer layers) = 9
# => 16*2 layers including self and cross attention
class CustomUNet2DConditionModel(UNet2DConditionModel):
    def train(self, mode: bool = True):
        super().train(False)

        if mode:
            set_trainable_modules_to_train(self)

        return self


if __name__ == '__main__':
    from peft import LoraConfig
    from dm.attn_proc import setup_attn_processor

    unet = CustomUNet2DConditionModel.from_pretrained(
        "../../pretrain/stable-diffusion-2-1-base", subfolder="unet",
    )

    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        # lora_dropout=0.1,
        init_lora_weights="gaussian",
        target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
    )

    # Add adapter and make sure the trainable params are in float32.
    # unet.add_adapter(unet_lora_config)

    ipa_attn_params, ipa_attn_layers = setup_attn_processor(unet, attn_size=[32, 64],
                                                            use_ipa=True)

    params = []
    for pn, p in unet.named_parameters():
        if p.requires_grad:
            params.append(p)
            print(pn)

    unet.train()
