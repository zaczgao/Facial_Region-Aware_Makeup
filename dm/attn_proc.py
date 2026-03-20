#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import deprecate

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


# https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py
# https://github.com/huggingface/diffusers/blob/157c9011d87e52632113024c1dc5125426971556/examples/dreambooth/train_dreambooth_lora.py
def setup_attn_processor(unet, **kwargs):
    attn_procs = {}
    attn_procs_params = []
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attention_class = (
                CustomAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomAttnProcessor
            )
            attn_procs[name] = attention_class()
        else:
            layer_name = name.split(".processor")[0]
            try:
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
            except KeyError:
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.base_layer.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.base_layer.weight"],
                }
            attn_procs[name] = CustomCrossAttnProcessor(attn_size=kwargs["attn_size"],
                                                        use_ipa=kwargs["use_ipa"], hidden_size=hidden_size,
                                                        cross_attention_dim=cross_attention_dim)
            if kwargs["use_ipa"]:
                attn_procs[name].load_state_dict(weights)
            attn_procs_params.extend(attn_procs[name].parameters())

    del unet_sd
    unet.set_attn_processor(attn_procs)
    attn_procs_layers = AttnProcsLayers(unet.attn_processors)

    return attn_procs_params, attn_procs_layers


# chirpy3d
def load_attn_processor(unet, filename):
    print(f"Load attn processors from {filename}")
    attn_layers = AttnProcsLayers(unet.attn_processors)
    if "safetensors" in filename:
        from safetensors.torch import load_file

        attn_layers.load_state_dict(load_file(filename))
    else:
        attn_layers.load_state_dict(torch.load(filename, map_location="cpu"))


# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# https://github.com/huggingface/diffusers/blob/157c9011d87e52632113024c1dc5125426971556/src/diffusers/models/attention_processor.py
class CustomCrossAttnProcessor(nn.Module):
    def __init__(self, attn_size, use_self_attn=False, use_hidden_state=False,
                 use_ipa=True, hidden_size=None, cross_attention_dim=None, scale=1.0, norm_ipa=False):
        super().__init__()

        if isinstance(attn_size, int):
            attn_size = [attn_size]

        self.attn_size = attn_size
        self.use_self_attn = use_self_attn
        self.use_hidden_state = use_hidden_state
        self.use_ipa = use_ipa
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.norm_ipa = norm_ipa

        if self.use_ipa:
            self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
            scale=None, upcast_attention: bool = False, upcast_softmax: bool = False,
    ):
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )
        del baddbmm_input

        if upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        attention_probs = attention_probs.to(dtype)
        attention_scores = attention_scores.to(dtype)

        return attention_probs, attention_scores

    def forward(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored. Please remove it, "
                                   "as passing it will raise an error in the future. `scale` should directly be "
                                   "passed while calling the underlying pipeline component i.e., "
                                   "via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

        is_self_attention = True if encoder_hidden_states is None else False

        if encoder_hidden_states is not None:
            if self.use_ipa:
                # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                # encoder_hidden_states, ip_hidden_states = (
                #     encoder_hidden_states[:, :end_pos, :],
                #     encoder_hidden_states[:, end_pos:, :],
                # )

                encoder_hidden_states, ip_hidden_states = encoder_hidden_states

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if (not is_self_attention) and self.use_ipa:
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # IP-Adapter
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_query = query.reshape(batch_size*attn.heads, -1, head_dim)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            attention_probs_ip, attention_scores_ip = self.get_attention_scores(ip_query, ip_key, None,
                                                                                attn.scale, attn.upcast_attention,
                                                                                attn.upcast_softmax)
            ip_hidden_states = torch.bmm(attention_probs_ip, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            # norm
            if self.norm_ipa:
                mean_latents = torch.mean(hidden_states, dim=-1, keepdim=True)
                std_latents = torch.std(hidden_states, dim=-1, keepdim=True, unbiased=False)
                mean_ip = torch.mean(ip_hidden_states, dim=-1, keepdim=True)
                std_ip = torch.std(ip_hidden_states, dim=-1, keepdim=True, unbiased=False)
                ip_hidden_states = std_latents * (ip_hidden_states - mean_ip) / (std_ip + 1e-7) + mean_latents

            hidden_states = hidden_states + self.scale * ip_hidden_states
        else:
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs, attention_scores = self.get_attention_scores(query, key, attention_mask,
                                                                          attn.scale, attn.upcast_attention,
                                                                          attn.upcast_softmax)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # store attention map
        total_qk = hidden_states.size(1)  # (B*Head,HW,L)
        sqrt_total = int(total_qk ** 0.5)
        attn.cross_attn_probs = None
        attn.self_attn_probs = None
        attn.embeddings = None

        if self.use_ipa:
            attn_probs_cache = attention_scores_ip
        else:
            # attn_probs_cache = attention_probs
            attn_probs_cache = attention_scores

        if is_self_attention and sqrt_total in self.attn_size and self.use_self_attn:
            attn.self_attn_probs = attn_probs_cache.reshape(batch_size, -1, sqrt_total * sqrt_total, sqrt_total * sqrt_total)
            attn.cross_attn_probs = None
            if self.use_hidden_state:
                attn.embeddings = hidden_states.clone()
        elif (not is_self_attention) and sqrt_total in self.attn_size:
            attn.cross_attn_probs = attn_probs_cache.reshape(batch_size, -1, sqrt_total, sqrt_total, attn_probs_cache.shape[2])
            attn.self_attn_probs = None
            if self.use_hidden_state:
                attn.embeddings = hidden_states.clone()  # (B,C,H,W)

        return hidden_states


class CustomAttnProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored. Please remove it, "
                                   "as passing it will raise an error in the future. `scale` should directly be "
                                   "passed while calling the underlying pipeline component i.e., "
                                   "via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomAttnProcessor2_0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = ("The `scale` argument is deprecated and will be ignored. Please remove it, "
                                   "as passing it will raise an error in the future. `scale` should directly be "
                                   "passed while calling the underlying pipeline component i.e., "
                                   "via `cross_attention_kwargs`.")
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


if __name__ == '__main__':
    pass
