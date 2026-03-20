#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://github.com/kamwoh/chirpy3d
"""

__author__ = "GZ"

import os
import sys
from collections import defaultdict
from typing import Any, Optional, Tuple, Union

import torch

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
    CLIPTextModel,
    _prepare_4d_attention_mask,
    _create_4d_causal_attention_mask
)
from transformers.models.clip.configuration_clip import CLIPTextConfig

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)


class CustomCLIPTextTransformer(CLIPTextTransformer):
    def forward_embeddings(self, input_ids, position_ids, inputs_embeds):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def forward_embeddings_with_mapper(
            self, input_ids, position_ids, mapper_outputs, placeholder_token_ids
    ):
        inputs_embeds = self.embeddings.token_embedding(input_ids)
        dtype = inputs_embeds.dtype

        offset = defaultdict(int)
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            # Overwrite the index of the placeholder token with the mapper output for each entry in the batch
            # learnable_idxs = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[1]
            # inputs_embeds[torch.arange(input_ids.shape[0]), learnable_idxs] = mapper_outputs[:, i].to(dtype)

            for bi in range(input_ids.shape[0]):
                learnable_idx = (input_ids[bi] == placeholder_token_id).nonzero(
                    as_tuple=True
                )[0]

                if len(learnable_idx) != 0:  # only assign if found
                    if len(learnable_idx) == 1:
                        offset_learnable_idx = learnable_idx
                    else:  # if there is two and above.
                        start = offset[(bi, placeholder_token_id)]
                        offset_learnable_idx = learnable_idx[start: start + 1]
                        offset[(bi, placeholder_token_id)] += 1

                    # print(i, offset_learnable_idx)

                    # before = inputs_embeds[bi, learnable_idx]
                    inputs_embeds[bi, offset_learnable_idx] = mapper_outputs[bi, i].to(
                        dtype
                    )
                    # after = inputs_embeds[bi, learnable_idx]

        return self.forward_embeddings(input_ids, position_ids, inputs_embeds)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,

        hidden_states: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if hidden_states is None:
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomCLIPTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CustomCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,

        hidden_states: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            hidden_states=hidden_states,
        )

if __name__ == '__main__':
    pass
