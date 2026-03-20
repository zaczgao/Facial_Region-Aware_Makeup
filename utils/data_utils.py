#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
https://huggingface.co/docs/hub/security-tokens

"""

__author__ = "GZ"

import os
import sys
import random
import numpy as np
import json
import glob
import pandas as pd
import PIL.Image
import io
import base64
import requests

import torch

from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline, FluxPipeline, FluxKontextPipeline, Flux2Pipeline, Flux2KleinPipeline, \
    QwenImagePipeline, QwenImageEditPlusPipeline

try:
    from openai import OpenAI
except ImportError:
    print("openai not found")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("gemini not found")

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

HF_TOKEN=f""
OPENAI_API_KEY=f""
OPENROUTER_API_KEY=f""


def init_sd():
    login(token=HF_TOKEN)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
        safety_checker=None,
    )
    # pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    return pipe


def init_flux():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    return pipe


def init_flux2():
    login(token=HF_TOKEN)

    # pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16)
    # pipe.enable_model_cpu_offload()
    pipe = pipe.to("cuda")

    return pipe


def init_qwen_t2i():
    pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    return pipe


def init_kontext():
    login(token=HF_TOKEN)

    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    return pipe


def init_qwen_edit():
    pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    return pipe


def sample_sd(pipe, prompt, prompt_neg):
    seed = random.randint(0, np.iinfo(np.int32).max)

    guidance_scale = 4.5
    image = pipe(
        prompt=prompt,
        negative_prompt=prompt_neg,
        num_inference_steps=40,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]

    return image


def sample_flux(pipe, prompt):
    seed = random.randint(0, np.iinfo(np.int32).max)

    image = pipe(
        prompt,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]

    return image


def sample_qwen_t2i(pipe, prompt):
    seed = random.randint(0, np.iinfo(np.int32).max)

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    # Generate with different aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["1:1"]

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]

    return image


# def sample_flux2(pipe, prompt, image=None):
#     seed = random.randint(0, np.iinfo(np.int32).max)
#
#     guidance_scale_min = 1.5
#     guidance_scale_max = 2.5
#     guidance_scale = random.uniform(guidance_scale_min, guidance_scale_max)
#
#     steps_min = 40
#     steps_max = 50
#     steps = random.randint(steps_min, steps_max)
#
#     image = pipe(
#         prompt=prompt,
#         image=image,
#         num_inference_steps=steps,  # 28 steps can be a good trade-off
#         guidance_scale=guidance_scale,
#         generator=torch.Generator().manual_seed(seed)
#     ).images[0]
#
#     return image

def sample_flux2(pipe, prompt, image=None):
    seed = random.randint(0, np.iinfo(np.int32).max)

    guidance_scale_min = 3.0
    guidance_scale_max = 4.0
    guidance_scale = random.uniform(guidance_scale_min, guidance_scale_max)

    steps_min = 40
    steps_max = 50
    steps = random.randint(steps_min, steps_max)

    image = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]

    return image


# https://huggingface.co/spaces/black-forest-labs/FLUX.1-Kontext-Dev/blob/main/app.py
def sample_kontext(pipe, input_image, prompt):
    seed = random.randint(0, np.iinfo(np.int32).max)

    guidance_scale_min = 1.5
    guidance_scale_max = 2.5
    guidance_scale = random.uniform(guidance_scale_min, guidance_scale_max)

    steps_min = 26
    steps_max = 30
    steps = random.randint(steps_min, steps_max)

    image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    return image


def sample_qwen_edit(pipe, input_image, prompt):
    seed = random.randint(0, np.iinfo(np.int32).max)

    true_cfg_scale_min = 3.0
    true_cfg_scale_max = 4.0
    true_cfg_scale = random.uniform(true_cfg_scale_min, true_cfg_scale_max)

    steps_min = 36
    steps_max = 40
    steps = random.randint(steps_min, steps_max)

    inputs = {
        "image": input_image,
        "prompt": prompt,
        "generator": torch.Generator().manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    image = pipe(**inputs).images[0]

    return image


def sample_gpt_t2i(client, prompt, model="gpt-4.1"):
    response = client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "image_generation", "quality": "high"}],
    )

    # Save the image to a file
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]

        # Decode base64 into bytes
        img_bytes = base64.b64decode(image_base64)

        # Convert bytes into PIL Image
        image = PIL.Image.open(io.BytesIO(img_bytes))

    return image


def add_alpha(mask: PIL.Image.Image, out_path):
    # 1. Load your black & white mask as a grayscale image
    mask = mask.convert("L")

    # 2. Convert it to RGBA so it has space for an alpha channel
    mask_rgba = mask.convert("RGBA")

    # 3. Then use the mask itself to fill that alpha channel
    mask_rgba.putalpha(mask)

    # 4. Convert the mask into bytes
    buf = io.BytesIO()
    mask_rgba.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    # 5. Save the resulting file
    with open(out_path, "wb") as f:
        f.write(mask_bytes)


def create_file(client, file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def encode_pil_image(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def sample_gpt_edit(client, img_path, mask_path, prompt, model="gpt-4.1"):
    fileId = create_file(client, img_path)
    maskId = create_file(client, mask_path)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "file_id": fileId,
                    }
                ],
            },
        ],
        tools=[
            {
                "type": "image_generation",
                "quality": "high",
                "input_fidelity": "high",
                "input_image_mask": {
                    "file_id": maskId,
                },
            },
        ],
    )

    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]

        # Decode base64 into bytes
        img_bytes = base64.b64decode(image_base64)

        # Convert bytes into PIL Image
        image = PIL.Image.open(io.BytesIO(img_bytes))

    return image


def sample_gpt_text(client, prompt, img_list=None, model="gpt-5", **kargs):
    """
    https://platform.openai.com/docs/guides/structured-outputs
    https://platform.openai.com/docs/guides/images-vision
    """
    assert model in ["gpt-5", "o3"], f"Not implemented model {model}"
    sys_promot = "You are a helpful assistant."
    input = [
        {"role": "system", "content": sys_promot},
        {"role": "user", "content": []}]
    input[1]["content"].append({"type": "input_text", "text": f"{prompt}"})

    if img_list is not None:
        for img in img_list:
            input[1]["content"].append(
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{img}",
                 "detail": "high"
                 })

    response_format = kargs.get('response_format', None)

    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=input,
        text={"format": {"type": "json_object"}} if response_format is None else response_format
    )

    if response.status == "completed":
        # In this case the model has either successfully finished generating the JSON object according to your schema, or the model generated one of the tokens you provided as a "stop token"
        return response.output_text
    elif response.status == "incomplete":
        print(response.incomplete_details.reason)
        return response.output_text
    else:
        raise Exception(f'Failed to post: {response}')


# def sample_gemini_text(client, prompt, img_list=None, model="gemini-3-flash-preview", **kargs):
#     contents = [f"{prompt}"]
#
#     if img_list is not None:
#         for img in img_list:
#             contents.append(types.Part.from_bytes(
#                 data=base64.b64decode(img),  # Decoding back to bytes for the SDK
#                 mime_type="image/png"
#             ))
#
#     response_format = kargs.get('response_format', None)
#
#     response = client.models.generate_content(
#         model=model,
#         contents=contents,
#         config={
#             "response_mime_type": "application/json",
#             "response_json_schema": response_format,
#         },
#     )
#
#     return response.text


def sample_gemini_text(client, prompt, img_list=None, model="gemini-3-pro-preview", **kargs):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }

    # Read and encode the image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"
                },
            ]
        }
    ]

    if img_list is not None:
        for img in img_list:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}"
                }
            }
            )

    response_format = kargs.get('response_format', None)

    payload = {
        "model": f"google/{model}",
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_format,
            },
        },
        "plugins": [
            {"id": "response-healing"}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    elif "error" in result:
        print(result)
        return result
    else:
        raise Exception(
            f"OpenRouter API error "
            f"(status={response.status_code}): {result}"
        )


def get_img_context(use_hair=True, use_face=True, use_expression=True):
    hairstyle_options = ["long wavy hair", "ponytail", "messy bun hairstyle", "shoulder-length hair", "blunt bob haircut",
                         "low chignon bun hairstyle", "textured pixie haircut"]
    haircolor_options = ["black hair", "blonde hair", "brown hair"]
    lighting_options = ["outdoor light", "warm indoor lighting", "soft studio lighting", "golden hour sunlight",
                        "cinematic lighting", "overcast natural light", "neon lighting"]
    age_options = ["young", "middle-aged", "old"]
    face_options = ["oval face", "round face", "square face", "heart face shape", "oblong face"]
    expression_options = ["neutral expression", "smiling"]
    attribute_options = ["wearing earrings", "wearing necklace", "wearing hat"]


    hairstyle = random.choice(hairstyle_options)
    haircolor = random.choice(haircolor_options)
    lighting = random.choice(lighting_options)
    age = random.choice(age_options)

    context = ""
    if use_hair:
        context = f"{hairstyle}, {haircolor}"

    face = random.choice(face_options)
    if use_face:
        context = f"{context}, {face}" if context else f"{face}"

    expression = random.choice(expression_options)
    if use_expression:
        context = f"{context}, {expression}" if context else f"{expression}"

    num_attri = random.randint(0, 2)
    if num_attri > 0:
        attri = np.random.permutation(attribute_options).tolist()[:num_attri]
        context = f"{context}, {', '.join(attri)}" if context else f"{', '.join(attri)}"

    context = f"{context}, {lighting}" if context else f"{lighting}"

    return context, age


def get_makeup_list(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        makeup_info = json.load(f)

    makeup_info_list = []
    for category in makeup_info.keys():
        for item in makeup_info[category]:
            style = item["name"]
            desp = item["description"]
            makeup_info_list.append([category, style, desp])

    return makeup_info_list


def merge_anno(data_dir, split=None):
    anno_path_list = glob.glob(data_dir + "/*.csv")
    anno_path_list = sorted(anno_path_list)
    name = os.path.basename(anno_path_list[0]).split("-")[0]

    df_list = []
    for anno_path in anno_path_list:
        if split is not None and split != anno_path.split("-")[1]:
            continue

        print("process", anno_path)

        df = pd.read_csv(anno_path, encoding="utf-8")
        df_list.append(df)

    if split is not None:
        merge_anno_path = os.path.join(data_dir, "{}-{}.csv".format(name, split))
    else:
        merge_anno_path = os.path.join(data_dir, "{}.csv".format(name))
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values(by=df_all.columns[0])
    df_all.to_csv(merge_anno_path, index=False)

    hist = df_all['class'].value_counts(normalize=True)
    print(hist)

    return merge_anno_path


def create_split(anno_path):
    df = pd.read_csv(anno_path, encoding="utf-8")

    df_shuffle = df.sample(frac=1, random_state=42, ignore_index=False)

    num_sample = len(df)
    train_idx = int(0.95 * num_sample)

    df_shuffle_train = df_shuffle.iloc[:train_idx]
    df_shuffle_val = df_shuffle.iloc[train_idx:]

    df_shuffle_train = df_shuffle_train.sort_index()
    df_shuffle_val = df_shuffle_val.sort_index()

    out_dir = os.path.dirname(anno_path)
    out_file_name = os.path.splitext(os.path.basename(anno_path))[0]

    df_shuffle_train.to_csv(os.path.join(out_dir, "{}-train.csv".format(out_file_name)), index=False)
    df_shuffle_val.to_csv(os.path.join(out_dir, "{}-val.csv".format(out_file_name)), index=False)

    print(df_shuffle_val['class'].value_counts(normalize=False))


def sample_style_desp():
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = """
    You are a makeup artist. Please provide 50 distinct makeup styles, covering categories like natural, dramatic, 
    cultural, seasonal, celebrity-inspired, fantasy, artistic, Historical, Theatrical, Gothic, Festival, Bridal, 
    Cyberpunk, Kawaii, etc. Please break these styles down into different categories and make sure each has a unique 
    and detailed description about how the makeup looks like on the face. Each style description should be 
    more than 25 words, but no more than 30 words.
    
    Return the makeup categories, style names and descriptions strictly in this JSON format:
    {
        "...": [
            {
              "name": "...",
              "description": "..."
            },
        ]
    }
    """

    result = sample_gpt_text(client, prompt, model="o3")

    print(result)


if __name__ == '__main__':
    sample_style_desp()

    # get_makeup_list("./assets/makeup_gpto3.json")