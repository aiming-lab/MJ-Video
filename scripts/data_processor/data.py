import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import requests
from urllib.parse import urlparse
import cv2
from PIL import Image
import torch
import requests
import numpy as np
import av
import regex as re
import shutil
from pathlib import Path
video_cache_dir = Path(__file__).parent / "video_cache"
if not video_cache_dir.exists():
    video_cache_dir.mkdir(exist_ok=True)
from huggingface_hub import hf_hub_download

def load_template(task, template):
    with open(Path(__file__).parent / "templates" / task / f"{template}.txt") as f:
        return f.read()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_video(url, file_path):
    if "huggingface.co/datasets" in url:
        # https://huggingface.co/datasets/tianleliphoebe/genai-arena-video-mp4/blob/main/dir/79e4e57959b4411e8d7532d655cbd6c6.mp4
        # repo_id = tianleliphoebe/genai-arena-video-mp4
        # revision = main
        # file_path = 79e4e57959b4411e8d7532d655cbd6c6.mp4
        # hf_hub_download(repo_id="tianleliphoebe/genai-arena-video-mp4", filename="79e4e57959b4411e8d7532d655cbd6c6.mp4", repo_type="dataset", local_dir="./video")
        _url = url.split("datasets/")[1]
        repo_id = _url.split("/blob")[0]
        revision_filename = _url.split("/blob/")[1]
        revision = revision_filename[:revision_filename.find("/")]
        filename = revision_filename[revision_filename.find("/")+1:]
        hub_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
        shutil.copy(hub_file_path, file_path)
        return file_path
    else:
        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return file_path

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound is None:
        # Sample uniformly spaced frames across the video
        frame_indices = np.linspace(first_idx, max_frame, num_segments, endpoint=False, dtype=int)
    else:
        # Sample frames within the specified time bound
        start_time, end_time = bound  # bound is in seconds
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_segments, endpoint=False, dtype=int)
    return frame_indices


# def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     max_frame = len(vr) - 1
#     fps = float(vr.get_avg_fps())

#     pixel_values_list, num_patches_list = [], []
#     transform = build_transform(input_size=input_size)
#     frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
#         img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(tile) for tile in img]
#         pixel_values = torch.stack(pixel_values)
#         num_patches_list.append(pixel_values.shape[0])
#         pixel_values_list.append(pixel_values)
#     pixel_values = torch.cat(pixel_values_list)
#     return pixel_values, num_patches_list

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    if video_path.startswith("http"):
        video_path = download_video(video_path, video_cache_dir / Path(video_path).name)
    else:
        video_path = Path(video_path)

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# path = 'OpenGVLab/InternVL2-2B'
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True).eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# # set the max number of tiles in `max_num`
# pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# generation_config = dict(max_new_tokens=1024, do_sample=True)

# # pure-text conversation (纯文本对话)
# question = 'Hello, who are you?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# video_path = './examples/red-panda.mp4'
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'What is the red panda doing?'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')