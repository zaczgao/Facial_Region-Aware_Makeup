#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import itertools
from tqdm.auto import tqdm
import PIL.Image

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torchvision import transforms

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
# from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.34.0.dev0")

from dm.multi_token_clip import MultiTokenCLIPTokenizer
from dm.text_encoder import CustomCLIPTextModel
from dm.unet import CustomUNet2DConditionModel
from dm.attn_proc import setup_attn_processor
from dm.makeup_adapter import MakeupAdapter
from dm.tokenizer import add_tokens
from dm.losses import calc_attn_loss
from dm.data import MakeupDataset, collate_fn, get_dataset_cls
from dm.pipeline import MakeupSDPipeline
from style_clip.model import StyleCLIP
from style_clip import clip_utils
from test_dm import init_pipeline
from utils.misc import load_image, compare_model_params
from utils.vis_utils import show_result, concatenate_images

logger = get_logger(__name__, log_level="INFO")


def load_pretrain(
        unet,
        makeup_adapter,
        pretrain_dir,
        map_location=None
):
    if os.path.isdir(pretrain_dir):
        # unet.load_state_dict(torch.load(os.path.join(pretrain_dir, 'unet.pt'), map_location=map_location))

        state_dict = torch.load(os.path.join(pretrain_dir, 'makeup_adapter.pt'), map_location=map_location)
        makeup_adapter.control_id.load_state_dict(state_dict['control_id'])

        logger.info(f"Loaded adapter pretrained models from {pretrain_dir}")
    else:
        logger.info(f"No pretrained models found at {pretrain_dir}")


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
    verbose=False
):
    logger.info(
        f"Running validation epoch {epoch}... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    control_scale = 1.
    num_inference_steps = 50
    guidance_scale = 7.5
    num_pair = 100

    img_id_path_list, img_makeup_path_list = [], []
    with open(args.val_anno_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            img_id_path_list.append(os.path.join(args.val_data_root, "id", items[0]))
            img_makeup_path_list.append(os.path.join(args.val_data_root, "makeup", items[1]))

    img_id_path_list = img_id_path_list[:num_pair]
    img_makeup_path_list = img_makeup_path_list[:num_pair]

    if is_final_validation:
        val_out_dir = os.path.join(args.output_dir, "validation", f"epoch-{epoch:03d}-final")
    else:
        val_out_dir = os.path.join(args.output_dir, "validation", f"epoch-{epoch:03d}")
    os.makedirs(val_out_dir, exist_ok=True)

    with autocast_ctx:
        for idx, (img_id_path, img_makeup_path) in enumerate(zip(img_id_path_list, img_makeup_path_list)):
            img_id = load_image(img_id_path)
            img_makeup = load_image(img_makeup_path)

            img_id_name = os.path.splitext(os.path.basename(img_id_path))[0]
            if args.geo_mode == "3d":
                img_pose_path = os.path.join(args.val_data_root, "3d", "{}.png".format(img_id_name))
            elif args.geo_mode == "keypoint":
                img_pose_path = os.path.join(args.val_data_root, "pose", "{}.png".format(img_id_name))
            img_pose = load_image(img_pose_path)

            face_mask = torch.load(os.path.join(args.val_data_root, "mask", "{}.pt".format(img_id_name)), map_location="cpu")
            face_mask = face_mask.to(torch.float32)
            face_mask = torchvision.transforms.functional.to_pil_image(face_mask).convert("L")

            assert img_id.size == img_pose.size and img_id.size == face_mask.size
            img_id = img_id.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.LANCZOS)
            img_pose = img_pose.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.NEAREST)
            face_mask = face_mask.resize((args.resolution, args.resolution), resample=PIL.Image.Resampling.NEAREST)

            face_mask_blurred = pipeline.mask_processor.blur(face_mask, blur_factor=10)

            for _ in range(args.num_validation_images):
                img = pipeline(
                    prompt=args.validation_prompt,
                    image=img_id,
                    mask_image=face_mask_blurred,
                    control_image=[img_id, img_pose],
                    control_mode=[0, 1],
                    image_makeup=img_makeup,
                    num_inference_steps=num_inference_steps,
                    negative_prompt="",
                    controlnet_conditioning_scale=control_scale,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

                images.append(img)

            concatenate_images([img_id, img_pose, img_makeup, img], os.path.join(val_out_dir, f"{epoch:03d}-{idx:03d}.jpg"))

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    parser.add_argument("--log_frequency", type=int, default=100, help="")

    parser.add_argument("--stage1_pretrain_dir", type=str, default="")
    parser.add_argument("--style_clip_ckpt", type=str, default="")
    parser.add_argument("--use_clip_lora", type=int, default=0, help="Use clip lora layers")
    parser.add_argument("--clip_hidden", type=str, default="24", help="clip hidden block idx")
    parser.add_argument("--placeholder_token", type=str, default="<part>", help="A token to use as a placeholder for the concept.")
    parser.add_argument("--attn_size", type=str, default="8,16")
    parser.add_argument("--num_parts", type=int, default=4, help="Number of facial regions")
    parser.add_argument("--num_heads_part", type=int, default=16, help="Number of head per facial region")
    parser.add_argument("--skip_background", default=False, action="store_true")
    parser.add_argument("--use_lora", type=int, default=0, help="Use lora for sd backbone")
    parser.add_argument("--use_ipa", type=int, default=0, help="Use ipa for makeup style")
    parser.add_argument("--use_text_inv", type=int, default=0, help="Use text inversion for makeup style")
    parser.add_argument("--geo_mode", type=str, default="3d", choices=["3d", "normal", "keypoint"])
    # parser.add_argument("--num_tokens", type=int, default=16, help="Number of tokens to query from the CLIP image encoding.")
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--weight_attn", type=float, default=0.1)
    parser.add_argument("--use_templates", default=False, action="store_true")
    parser.add_argument("--vector_shuffle", default=False, action="store_true", help="shuffle ph tokens")
    parser.add_argument("--drop_tokens", default=False, action="store_true")
    parser.add_argument("--drop_tokens_rate", type=float, default=0.5)
    parser.add_argument("--swap_pair_rate", type=float, default=0.1)
    parser.add_argument("--drop_text_rate", type=float, default=0.1)
    parser.add_argument("--drop_style_rate", type=float, default=0.05)
    parser.add_argument("--drop_all_rate", type=float, default=0.05)

    parser.add_argument("--val_data_root", type=str, default="")
    parser.add_argument("--val_anno_path", type=str, default="")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    print(args)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CustomCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = CustomUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    style_clip = StyleCLIP("vit_large", use_text=False)
    if args.use_clip_lora:
        style_clip.prep_lora_model()
    clip_utils.load_network(style_clip, args.style_clip_ckpt, "state_dict", strict=False)

    if not args.skip_background:
        args.num_parts = args.num_parts + 1

    clip_hidden = [int(a) for a in args.clip_hidden.split(",")]
    makeup_adapter = MakeupAdapter(
        style_in_dim=style_clip.clip_dim,
        style_out_dim=unet.config.cross_attention_dim if args.use_ipa else text_encoder.text_model.embeddings.token_embedding.embedding_dim,
        style_seq_len=256 * len(clip_hidden),
        num_parts=args.num_parts,
        num_heads_part=args.num_heads_part,
        unet=unet,
        use_ipa=args.use_ipa,
        use_text_inv=args.use_text_inv
    )

    # initialize placeholder token
    placeholder_token = args.placeholder_token
    placeholder_token_ids = add_tokens(
        tokenizer,
        text_encoder,
        placeholder_token,
        args.num_parts,
    )

    base_prompt = args.validation_prompt
    logger.info(f"base prompt: {base_prompt}")
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    style_clip.requires_grad_(False)
    style_clip.eval()

    if args.use_lora:
        # peft/tuners/lora/model.py _replace_module
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
            # target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v"],
        )

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)

    attn_size = [int(a) for a in args.attn_size.split(",")]
    ipa_attn_params, ipa_attn_layers = setup_attn_processor(unet, attn_size=attn_size, use_ipa=args.use_ipa)
    # unet.down_blocks[1].attentions[0].transformer_blocks
    if accelerator.is_main_process:
        print(unet)
        print(makeup_adapter)

    if args.stage1_pretrain_dir:
        load_pretrain(unet, makeup_adapter, args.stage1_pretrain_dir, map_location='cpu')

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    style_clip.to(accelerator.device, dtype=weight_dtype)
    makeup_adapter.to(accelerator.device)

    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            makeup_adapter.control_id.enable_xformers_memory_efficient_attention()
            # makeup_adapter.control_pose.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(model, CustomUNet2DConditionModel):
                    torch.save(model.state_dict(), os.path.join(output_dir, 'unet.pt'))

                    if args.use_lora:
                        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(model)))
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=output_dir,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                    if args.use_ipa:
                        torch.save(ipa_attn_layers.state_dict(), os.path.join(output_dir, "ipa_layers.pt"))
                elif isinstance(model, MakeupAdapter):
                    model.save_checkpoint(os.path.join(output_dir, 'makeup_adapter.pt'))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, CustomUNet2DConditionModel):
                model.load_state_dict(torch.load(os.path.join(input_dir, 'unet.pt'), map_location='cpu'))
            elif isinstance(model, MakeupAdapter):
                model.load_from_checkpoint(os.path.join(input_dir, 'makeup_adapter.pt'))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        makeup_adapter.control_id.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.lr_adapter = (
                args.lr_adapter * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    unet_params = [p for p in unet.parameters() if p.requires_grad]
    adapter_params = [p for p in makeup_adapter.parameters() if p.requires_grad]
    opt_params = [
        {"params": unet_params},
        {"params": adapter_params, "lr": args.lr_adapter},
    ]

    optimizer = optimizer_cls(
        opt_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Get the specified interpolation method from the args
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")

    dataset_cls = get_dataset_cls(args.dataset_name)
    train_dataset = dataset_cls(args.train_data_dir, args.resolution, args.center_crop, args.random_flip,
                                tokenizer, base_prompt, args.use_templates, args.vector_shuffle, args.drop_tokens, args.drop_tokens_rate,
                                args.swap_pair_rate, [args.drop_text_rate, args.drop_style_rate, args.drop_all_rate],
                                args.skip_background, args.num_parts, args.geo_mode)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, makeup_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, makeup_adapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("makeup", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    null_text_input_ids = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    null_text_embeds = text_encoder(null_text_input_ids.to(accelerator.device))[0]

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        makeup_adapter.train()

        train_loss = 0.0
        train_loss_diff = 0.0
        train_loss_diff_mask = 0.0
        train_loss_attn = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, makeup_adapter):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                _, style_feat, _, _ = style_clip.get_image_feat(batch["style_pixel_values"],
                                                                hidden_layer_idx=args.clip_hidden)

                face_cond = [batch["id_pixel_values"].to(dtype=weight_dtype),
                             batch["pose_pixel_values"].to(dtype=weight_dtype),
                             null_text_embeds]

                # # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states, down_block_res_samples, mid_block_res_sample = \
                    makeup_adapter(noisy_latents, timesteps, batch["input_ids"], text_encoder,
                                   placeholder_token_ids, style_feat, batch["is_drop_style"], face_cond)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents,
                                  timesteps,
                                  encoder_hidden_states=encoder_hidden_states,
                                  down_block_additional_residuals=[
                                      sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                  ],
                                  mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                                  return_dict=False)[0]

                if args.snr_gamma is None:
                    loss_diff = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Scale-Robust and Fine-Controllable Identity Customization via Local and Global Complementation
                    # follow-your-emoji
                    face_mask = batch["face_mask"].unsqueeze(1).to(latents.device)
                    face_mask = F.interpolate(face_mask.float(), target.shape[-2:], mode="bilinear")
                    loss_diff_face = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss_diff_face = torch.mean(torch.sum(loss_diff_face * face_mask, dim=(1,2,3)) / (face_mask.sum(dim=(1,2,3)) + 1e-6))

                    exp_mask = batch["exp_mask"].unsqueeze(1).to(latents.device)
                    exp_mask = F.interpolate(exp_mask.float(), target.shape[-2:], mode="bilinear")
                    loss_diff_exp = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss_diff_exp = torch.mean(torch.sum(loss_diff_exp * exp_mask, dim=(1, 2, 3)) / (exp_mask.sum(dim=(1, 2, 3)) + 1e-6))

                    loss_diff_mask = 0.5 * (loss_diff_face + loss_diff_exp)

                    loss = 0.5 * (loss_diff + loss_diff_mask)

                    loss_attn = torch.tensor(0.).to(loss.device)
                    if args.use_ipa or args.use_text_inv:
                        loss_attn, max_attn = calc_attn_loss(batch, unet, placeholder_token_ids, args.use_ipa,
                                                             is_drop=batch["is_drop_style"], attn_size=attn_size)
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                loss = loss + args.weight_attn * loss_attn

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_diff = accelerator.gather(loss_diff.repeat(args.train_batch_size)).mean()
                train_loss_diff += avg_loss_diff.item() / args.gradient_accumulation_steps

                avg_loss_diff_mask = accelerator.gather(loss_diff_mask.repeat(args.train_batch_size)).mean()
                train_loss_diff_mask += avg_loss_diff_mask.item() / args.gradient_accumulation_steps

                avg_loss_attn = accelerator.gather(loss_attn.repeat(args.train_batch_size)).mean()
                train_loss_attn += avg_loss_attn.item() / args.gradient_accumulation_steps

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_params + adapter_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                 "loss_diff": train_loss_diff,
                                 "loss_diff_mask": train_loss_diff_mask,
                                 "loss_attn": train_loss_attn,
                                 }, step=global_step)
                train_loss = 0.0
                train_loss_diff = 0.0
                train_loss_diff_mask = 0.0
                train_loss_attn = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # unwrapped_unet = unwrap_model(unet)
                        # unet_lora_state_dict = convert_state_dict_to_diffusers(
                        #     get_peft_model_state_dict(unwrapped_unet)
                        # )
                        #
                        # StableDiffusionPipeline.save_lora_weights(
                        #     save_directory=save_path,
                        #     unet_lora_layers=unet_lora_state_dict,
                        #     safe_serialization=True,
                        # )

                        logger.info(f"Saved state to {save_path}")

                if (global_step - 1) % args.log_frequency == 0 or global_step == 1 or global_step == args.max_train_steps:
                    if global_step == 1:
                        steps_to_update = 1
                    elif global_step == args.max_train_steps:
                        steps_to_update = (global_step - 1) % args.log_frequency or args.log_frequency
                    else:
                        steps_to_update = args.log_frequency

                    logs = {"step_loss": loss.detach().item(),
                            "loss_diff": loss_diff.detach().item(),
                            "loss_diff_mask": loss_diff_mask.detach().item(),
                            "loss_attn": loss_attn.detach().item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "lr_adapter": lr_scheduler.get_last_lr()[1]}
                    progress_bar.set_postfix(**logs, refresh=False)
                    progress_bar.update(steps_to_update)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                pipeline = MakeupSDPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    controlnet=unwrap_model(makeup_adapter).control_id,
                    scheduler=scheduler,
                    text_encoder=unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline.init_extra(
                    style_clip=unwrap_model(style_clip),
                    makeup_adapter=unwrap_model(makeup_adapter),
                    placeholder_token_ids=placeholder_token_ids,
                    clip_hidden=args.clip_hidden)

                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

    logger.info("Training finished. Run final testing.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            unet = unet.to(torch.float32)
            unwrapped_unet = unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        if args.use_ipa:
            torch.save(ipa_attn_layers.to(torch.float32).state_dict(), os.path.join(args.output_dir, "ipa_layers.pt"))

        makeup_adapter = makeup_adapter.to(torch.float32)
        makeup_adapter = unwrap_model(makeup_adapter)
        makeup_adapter.save_checkpoint(os.path.join(args.output_dir, "makeup_adapter.pt"))

        # del unet, makeup_adapter, optimizer, scheduler, train_dataloader
        # torch.cuda.empty_cache()

        # Final inference
        # Load previous pipeline
        if args.validation_prompt is not None:
            pipeline = init_pipeline(args.pretrained_model_name_or_path, args.revision, args.variant,
                                     args.placeholder_token, args.num_parts, args.num_heads_part, args.use_lora,
                                     args.use_ipa, args.use_text_inv,
                                     args.style_clip_ckpt, args.use_clip_lora, args.clip_hidden,
                                     ipa_scale=1., ckpt_dir=args.output_dir,
                                     device=accelerator.device, weight_dtype=torch.float32)

            compare_model_params(unwrap_model(unet.to(torch.float32)), pipeline.unet)
            compare_model_params(makeup_adapter, pipeline.makeup_adapter)

            # run inference
            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
