# load the GenAI-Bench(GenAI-Bench-1600) benchmark
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer
from dataset import VideoDataset, VideoDataCollator
from internvl2 import InternVLChatModel, InternVLChatConfig, prepare_chat_input
from moe_reward import InternVLChatRewardModeling, InternVLChatRewardModelingConfig
import os
import pandas as pd
from data import load_video
from torch import distributed as dist
import os
import av
import numpy as np
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification
from data import download_video
import shutil
from pathlib import Path
video_cache_dir = Path(__file__).parent / "video_cache"
if not video_cache_dir.exists():
    video_cache_dir.mkdir(exist_ok=True)


os.environ['WORLD_SIZE'] = str(1)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(12345)
os.environ['LOCAL_RANK'] = str(0)
os.environ['RANK'] = str(0)

dist.init_process_group(backend='nccl', world_size=1, rank=0)


model_name="TIGER-Lab/VideoScore"
processor = AutoProcessor.from_pretrained(model_name,torch_dtype=torch.bfloat16)
model = Idefics2ForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


MAX_NUM_FRAMES=16
ROUND_DIGIT=3
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output_sora a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output_sora example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""
def _read_video_pyav(
    container,
    indices
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    
def video_score_eval(video_path, video_prompt):
    if video_path.startswith("http"):
        video_path = download_video(video_path, video_cache_dir / Path(video_path).name)
    else:
        video_path = Path(video_path)
    # sample uniformly 8 frames from the video
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames > MAX_NUM_FRAMES:
        indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
    else:
        indices = np.arange(total_frames)

    frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_soras = model(**inputs)

    logits = output_soras.logits
    num_aspects = logits.shape[-1]

    aspect_scores = []
    total_score = 0
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
        total_score += round(logits[0, i].item(),ROUND_DIGIT)
    return total_score


def parse_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model_name', type=str, default="OpenGVLab/InternVL2-2B", help="Model name or path")
    parser.add_argument('--checkpoint_path', type=str, default="overall_output_mse_three_epoch/checkpoint-39", help="Path to the checkpoint file (optional)")
    parser.add_argument('--num_segments', type=int, default=8, help="Frames of Videos")

    # InternVLChatRewardModelingConfig Arguments
    parser.add_argument('--num_objectives', type=int, default=28, help="Number of objectives")
    parser.add_argument('--num_aspects', type=int, default=5, help="Number of aspects")
    parser.add_argument('--aspect2criteria', type=dict, default={
        0: [0, 1, 2, 3, 4],
        1: [5, 6, 7, 8, 9, 10],
        2: [11, 12, 13, 14, 15],
        3: [16, 17, 18, 19, 20, 21, 22],
        4: [23, 24, 25, 26, 27]
    }, help="Mapping of aspects to criteria")
    parser.add_argument('--gating_temperature', type=float, default=1.0, help="Gating temperature")
    parser.add_argument('--gating_hidden_dim', type=int, default=1024, help="Dimensionality of gating hidden layer")
    parser.add_argument('--gating_n_hidden', type=int, default=3, help="Number of hidden layers in gating")

    # Generation Config Arguments
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Maximum number of new tokens for generation")
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether to sample during generation")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    dataset = load_dataset("TIGER-Lab/GenAI-Bench", "video_generation", split='test_v1')
    
    prefer_truth = 0
    prefer_total = 0
    truth = 0
    total = 0

    for i, example in tqdm(enumerate(dataset), desc="Running Inference", total=len(dataset)):
        vote_type = example["vote_type"]
        left_video = example["left_video"]
        right_video = example["right_video"]
        prompt = example["prompt"]
        score_left = video_score_eval(left_video, prompt)
        score_right = video_score_eval(right_video, prompt)
        left_judge = "good" if score_left > 2 else "bad"
        right_judge = "good" if score_right > 2 else "bad"
        if vote_type == "rightvote":
            prefer_total += 1
            total += 1
            if score_right > score_left:
                prefer_truth += 1
                truth += 1
        if vote_type == "leftvote":
            prefer_total += 1
            total += 1
            if score_right < score_left:
                prefer_truth += 1
                truth += 1
        if vote_type == "bothbad_vote":
            total += 1
            if left_judge == "bad" and right_judge == "bad":
                truth += 1
        if vote_type == "tievote":
            total += 1
            if left_judge == "good" and right_judge == "good":
                truth += 1
    print(f"prefer_Acc: {prefer_truth / prefer_total}")
    print(f"Acc: {truth / total}")


if __name__ == "__main__":
    main()