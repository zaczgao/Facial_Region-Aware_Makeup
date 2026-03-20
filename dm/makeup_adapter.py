#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torch
import torch.nn as nn

from transformers import Blip2QFormerConfig, Blip2QFormerModel

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

from dm.resampler import Resampler
from dm.transformer.transformer_predictor import TransformerPredictor
from dm.pos_embed import get_1d_sincos_pos_embed
from dm.controlnet_union import ControlNetUnionModel


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels, bias=True, num_mlp=2, use_bn=False):
        super().__init__()

        self.mlp = nn.Module()
        for i in range(num_mlp-1):
            self.mlp.add_module(f'linear{i}', nn.Linear(in_channels, hid_channels, bias=bias))
            if use_bn:
                self.mlp.add_module(f'bn{i}', nn.BatchNorm1d(hid_channels))
            self.mlp.add_module(f'relu{i}', nn.ReLU(inplace=True))
            in_channels = hid_channels
        self.mlp.add_module(f'linear{num_mlp-1}', nn.Linear(hid_channels, out_channels, bias=bias))

    def forward(self, x):
        for name, block in self.mlp._modules.items():
            x = block(x)
        return x


class QformerProjector(nn.Module):
    def __init__(self, num_queries, in_dim, output_dim, seq_len=None, use_pos_emb=False):
        super().__init__()

        # qformer_config = {
        #     "attention_probs_dropout_prob": 0.1,
        #     "classifier_dropout": 0.,
        #     "cross_attention_frequency": 2,
        #     "encoder_hidden_size": 1408,
        #     "hidden_act": "gelu",
        #     "hidden_dropout_prob": 0.1,
        #     "hidden_size": 768,
        #     "initializer_range": 0.02,
        #     "intermediate_size": 3072,
        #     "layer_norm_eps": 1e-12,
        #     "max_position_embeddings": 512,
        #     "model_type": "blip_2_qformer",
        #     "num_attention_heads": 12,
        #     "num_hidden_layers": 12,
        #     "pad_token_id": 0,
        #     "position_embedding_type": "absolute",
        #     "torch_dtype": "float16",
        #     "transformers_version": "4.53.3",
        #     "use_qformer_text_input": False,
        #     "vocab_size": 30522
        # }

        qformer_config = Blip2QFormerConfig.from_pretrained("Salesforce/blip2-opt-2.7b")
        qformer_config.cross_attention_frequency = 1
        qformer_config.num_hidden_layers = 4
        qformer_config.encoder_hidden_size = in_dim
        qformer_config.hidden_size = 1280
        qformer_config.num_attention_heads = 20
        qformer_config.attention_probs_dropout_prob = 0.
        qformer_config.hidden_dropout_prob = 0.
        qformer_config.torch_dtype = "float32"

        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, qformer_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        self.qformer = Blip2QFormerModel._from_config(qformer_config)

        self.proj_out = nn.Linear(qformer_config.hidden_size, output_dim)

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            pos_emb_query = get_1d_sincos_pos_embed(qformer_config.hidden_size, num_queries)
            self.register_buffer("pos_emb_query", pos_emb_query.unsqueeze(0))
            pos_emb_x = get_1d_sincos_pos_embed(in_dim, seq_len)
            self.register_buffer("pos_emb_x", pos_emb_x.unsqueeze(0))

    def forward(self, image_embeds):
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.use_pos_emb:
            query_tokens = query_tokens + self.pos_emb_query.to(dtype=image_embeds.dtype)
            image_embeds = image_embeds + self.pos_emb_x.to(dtype=image_embeds.dtype)

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )

        out = self.proj_out(query_outputs[0])

        return out


class MformerProjector(nn.Module):
    def __init__(self, style_in_dim, style_out_dim, num_parts):
        super().__init__()

        self.model = TransformerPredictor(in_channels=style_in_dim, mask_classification=False,
                                          num_classes=0, hidden_dim=1024, num_queries=num_parts,
                                          nheads=8, dropout=0.1, dim_feedforward=2048,
                                          enc_layers=0, dec_layers=4, pre_norm=False, deep_supervision=False,
                                          enforce_input_project=True, mask_dim=style_out_dim)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        size = int(seq_len ** 0.5)
        assert seq_len % size == 0

        x = x.reshape(bs, size, size, -1).permute(0, 3, 1, 2)
        out = self.model(x, None)

        return out["mask_embed"]


class MakeupAdapter(nn.Module):
    def __init__(self, style_mode="resampler", style_in_dim=1024, style_out_dim=1024, style_seq_len=None, num_parts=4, num_heads_part=4,
                 unet=None, use_ipa=False, use_text_inv=False, controlnet_model_name_or_path=None):
        super().__init__()

        self.style_mode = style_mode
        self.num_heads_part = num_heads_part
        self.use_ipa = use_ipa
        self.use_text_inv = use_text_inv

        num_queries = num_parts * num_heads_part

        if self.use_ipa or self.use_text_inv:
            # project CLIP to token embedding space (vocab)
            if style_mode == "resampler":
                self.style_proj = Resampler(
                    dim=1280,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=num_queries,
                    embedding_dim=style_in_dim,
                    output_dim=style_out_dim,
                    ff_mult=4,
                    seq_len=style_seq_len,
                    use_pos_emb=True,
                    is_attn_state=False
                )
            elif style_mode == "mformer":
                self.style_proj = MformerProjector(style_in_dim, style_out_dim, num_parts)
            elif style_mode == "qformer":
                self.style_proj = QformerProjector(num_queries, style_in_dim, style_out_dim, style_seq_len, use_pos_emb=True)

        if controlnet_model_name_or_path is not None:
            self.control_id = ControlNetUnionModel.from_pretrained(controlnet_model_name_or_path)
        else:
            self.control_id = ControlNetUnionModel.from_unet(unet)

    def load_from_checkpoint(self, ckpt_path):
        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path)

            if self.use_ipa or self.use_text_inv:
                self.style_proj.load_state_dict(state_dict['style_proj'])
            self.control_id.load_state_dict(state_dict['control_id'])
            # self.control_pose.load_state_dict(state_dict['control_pose'])
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))

    def save_checkpoint(self, output_path):
        state_dict = {
            'control_id': self.control_id.state_dict(),
            # 'control_pose': self.control_pose.state_dict(),
        }

        if self.use_ipa or self.use_text_inv:
            state_dict['style_proj'] = self.style_proj.state_dict()

        torch.save(state_dict, output_path)

    def forward(self, noisy_latents, timesteps, input_ids, text_encoder, placeholder_token_ids, style_feat, is_drop_style, face_cond):
        bsz = noisy_latents.shape[0]
        face_id, face_pose, null_text_embeds = face_cond

        if self.use_ipa:
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            style_feat_uc = torch.zeros_like(style_feat)
            mask = is_drop_style.view(-1, 1, 1).to(style_feat.device)
            style_feat = style_feat * (~mask).float() + style_feat_uc * mask.float()

            style_emb = self.style_proj(style_feat)

            # encoder_hidden_states = torch.cat([encoder_hidden_states, style_emb], dim=1)
            encoder_hidden_states = [encoder_hidden_states, style_emb]
        elif self.use_text_inv:
            style_emb = self.style_proj(style_feat)

            # Get the text embedding for conditioning
            modified_hs = text_encoder.text_model.forward_embeddings_with_mapper(
                input_ids, None, style_emb, placeholder_token_ids
            )
            encoder_hidden_states = text_encoder(input_ids, hidden_states=modified_hs)[0]
        else:
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]


        down_block_res_samples, mid_block_res_sample = self.control_id(
            noisy_latents,
            timesteps,
            encoder_hidden_states=null_text_embeds.repeat(bsz, 1, 1),
            controlnet_cond=[face_id, face_pose],
            control_type=torch.ones([face_id.shape[0], 2], device=face_id.device, dtype=torch.int32),
            control_type_idx=[0, 1],
            return_dict=False,
        )
        # down_block_res_samples_id, mid_block_res_sample_id = self.control_id(
        #     noisy_latents,
        #     timesteps,
        #     encoder_hidden_states=null_text_embeds.repeat(bsz, 1, 1),
        #     controlnet_cond=face_id,
        #     return_dict=False,
        # )
        # down_block_res_samples_pose, mid_block_res_sample_pose = self.control_pose(
        #     noisy_latents,
        #     timesteps,
        #     encoder_hidden_states=null_text_embeds.repeat(bsz, 1, 1),
        #     controlnet_cond=face_pose,
        #     return_dict=False,
        # )
        #
        # down_block_res_samples = [
        #     samples_prev + samples_curr
        #     for samples_prev, samples_curr in zip(down_block_res_samples_id, down_block_res_samples_pose)
        # ]
        # mid_block_res_sample = mid_block_res_sample_id + mid_block_res_sample_pose

        # drop condition
        # for bi in range(input_ids.shape[0]):
        #     if is_drop_id[bi]:
        #         for sample in down_block_res_samples:
        #             sample[bi] = torch.zeros_like(sample[bi])
        #
        #         mid_block_res_sample[bi] = torch.zeros_like(mid_block_res_sample[bi])

        return encoder_hidden_states, down_block_res_samples, mid_block_res_sample


if __name__ == '__main__':
    from diffusers import UNet2DConditionModel

    batch_size = 2
    clip_dim = 1024
    num_parts = 4
    num_heads_part = 16
    style_feat = torch.randn(batch_size, 256, clip_dim)

    unet = UNet2DConditionModel.from_pretrained("../../pretrain/stable-diffusion-2-1-base", subfolder="unet")
    model = MakeupAdapter(style_in_dim=clip_dim,
                          style_out_dim=1024,
                          style_seq_len=256,
                          num_parts=num_parts,
                          num_heads_part=num_heads_part,
                          unet=unet, use_ipa=True, use_text_inv=False)

    style_emb = model.style_proj(style_feat)
    print(style_emb.shape)
