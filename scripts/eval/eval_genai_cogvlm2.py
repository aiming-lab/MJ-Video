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

import json
import os
import re
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch
from fuzzywuzzy import process

fuzzy_list = [
    'RATING: Extremely Poor',
    'RATING: Very Poor',
    'RATING: Poor',
    'RATING: Below Average',
    'RATING: Average',
    'RATING: Above Average',
    'RATING: Good',
    'RATING: Very Good',
    'RATING: Excellent',
    'RATING: Outstanding'
]


def get_local_path(video_path):
    if video_path.startswith("http"):
        video_path = download_video(video_path, video_cache_dir / Path(video_path).name)
    else:
        video_path = Path(video_path)
    return video_path

def evaluate_videos(video0_path, video1_path, prompt, model, template):
    video_0_path = get_local_path(video_0_path)
    video_1_path = get_local_path(video_1_path)
    start_time = time.time()  # 记录开始时间
    response0, _ = inference(model, template, prompt, videos=[video0_path])
    response1, _ = inference(model, template, prompt, videos=[video1_path])
    score0 = response0
    score1 = response1  
    end_time = time.time()  # 记录结束时间

    latency = (end_time - start_time) / 2  # 每个视频的平均延迟

    def extract_ratings(response):
        best_match, score = process.extractOne(response, fuzzy_list)
        match = re.search(r'RATING:\s*(.+)', best_match)
        return match.group(1) if match else None

    video_0_rating = extract_ratings(score0)
    video_1_rating = extract_ratings(score1)

    # 定义文字到数字的映射
    rating_scale = {
        "Extremely Poor": 1,
        "Very Poor": 2,
        "Poor": 3,
        "Below Average": 4,
        "Average": 5,
        "Above Average": 6,
        "Good": 7,
        "Very Good": 8,
        "Excellent": 9,
        "Outstanding": 10
    }

    video_0_score = rating_scale.get(video_0_rating, 0)
    video_1_score = rating_scale.get(video_1_rating, 0)
    return video_0_score, video_1_score, latency, response0, response1, prompt


prompt_overall = """
As a professional "Text-to-Video" quality assessor, your task is to determine whether the generated video will be preferred by humans.
Please analyze step by step and provide a rating from the scale: ["Extremely Poor", "Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good", "Excellent", "Outstanding"], where "Extremely Poor" is the worst and "Outstanding" is the best.

Do not analyze, and must give a rating. You cannot refuse to answer.

Now, proceed with evaluating the video based on the prompt description provided. The prompt is:
{caption}

Directly output your rating in the following format:
```
{{RATING: YOUR RATING}}
```
"""



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
    model_type = ModelType.cogvlm2_video_13b_chat
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                           model_kwargs={'device_map': 'auto'})

    model.generation_config.max_new_tokens = 1024
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    dataset = load_dataset("TIGER-Lab/GenAI-Bench", "video_generation", split='test_v1')
    
    prefer_truth = 0
    prefer_total = 0
    prefer_same_truth = 0
    truth = 0
    total = 0

    for i, example in tqdm(enumerate(dataset), desc="Running Inference", total=len(dataset)):
        vote_type = example["vote_type"]
        left_video = example["left_video"]
        right_video = example["right_video"]
        prompt = example["prompt"]
        prompt = prompt_overall.format(caption=prompt)
        score_left, score_right, _, _, _, prompt = evaluate_videos(left_video, right_video, prompt, model, template)
        left_judge = "good" if score_left > 5 else "bad"
        right_judge = "good" if score_right > 5 else "bad"
        if vote_type == "rightvote":
            prefer_total += 1
            total += 1
            if score_right > score_left:
                prefer_truth += 1
                prefer_same_truth += 1
                truth += 1
            if score_right == score_left:
                prefer_same_truth += 0.5
        if vote_type == "leftvote":
            prefer_total += 1
            total += 1
            if score_right < score_left:
                prefer_truth += 1
                prefer_same_truth += 1
                truth += 1
            if score_right == score_left:
                prefer_same_truth += 0.5
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
    print(f"prefer_same_Acc: {prefer_same_truth / prefer_total}")


if __name__ == "__main__":
    main()