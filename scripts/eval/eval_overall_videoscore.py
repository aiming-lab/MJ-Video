import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import argparse

from swift.llm import (
    get_model_tokenizer, get_template, inference,ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

import av
import numpy as np
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

model_name="TIGER-Lab/VideoScore"
processor = AutoProcessor.from_pretrained(model_name,torch_dtype=torch.bfloat16)
model = Idefics2ForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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

def video_score_eval(video_path, video_prompt):
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

def evaluate_videos(caption, video0_path, video1_path):
    video_0_score = video_score_eval(video0_path, caption)
    video_1_score = video_score_eval(video1_path, caption)
    print(f"Score: {video_0_score}")
    print(f"Score: {video_1_score}")
    return video_0_score, video_1_score

def process_overall_file(json_file_path, result_dir, videos_dir, output_file_name, model):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    all_results = []

    for item in data:
        caption = item['caption']
        video0_path_relative = item['video_0_path']
        video1_path_relative = item['video_1_path']
        video0_path = os.path.join(videos_dir, video0_path_relative)
        video1_path = os.path.join(videos_dir, video1_path_relative)
        preference = item["overall_preference"]
        if preference == "Same" or preference == "Hard to judge":
            continue
        video_0_rating, video_1_rating = evaluate_videos(caption, video0_path, video1_path)
        result = {
            "caption": caption,
            "video_0_uid": video0_path,
            "video_1_uid": video1_path,
            "video_score_0": video_0_rating,
            "video_score_1": video_1_rating,
            "ground_truth": preference
        }
        print(result)
        all_results.append(result)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        output_file = os.path.join(result_dir ,output_file_name)
        with open(output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)

    output_file = os.path.join(result_dir, output_file_name)
    with open(output_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)



def main(args):

    # Run the processing function
    process_overall_file(args.json_file_path, args.output_dir, args.videos_dir, args.output_file_name, model)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate video preferences using a text-to-video quality assessment model.")
    
    # Arguments for paths
    parser.add_argument('--json_file_path', type=str, default="datas/test.json", help="Path to the JSON file containing video data.")
    parser.add_argument('--videos_dir', type=str, default="datas/videos", help="Directory containing the videos.")
    parser.add_argument('--output_dir', type=str, default="output_overall", help="Directory to save the output results.")
    parser.add_argument('--output_file_name', type=str, default="eval_result_overall_videoscore.json", help="Name of the output file.")
    
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(args)