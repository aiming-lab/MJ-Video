import json
import os
import re
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter

# huggingface-cli download --repo-type dataset --local-dir-use-symlinks False --resume-download PKU-Alignment/SafeSora --local-dir ./SafeSora 下载数据集

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

def evaluate_videos(video0_path, video1_path, prompt, model, template):
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

def process_overall_file(json_file_path, result_dir, videos_dir, output_file_name, model, template):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    all_results = []

    for item in data:
        caption = item['prompt_text']
        video0_path_relative = item['video_0']["video_path"]
        video1_path_relative = item['video_1']["video_path"]
        video0_path = os.path.join(videos_dir, video0_path_relative)
        video1_path = os.path.join(videos_dir, video1_path_relative)
        sub_preference = item["sub_preferences"]
        helpfulness = item["helpfulness"]
        harmlessness = item["harmlessness"]
        preference = helpfulness
        if harmlessness != preference:
            continue
        same = True
        for key in sub_preference.keys():
            prefer = sub_preference[key]
            if prefer != preference:
                same = False
                break
        if not same:
            continue
        prompt = prompt_overall.format(caption=caption)
        video_0_rating, video_1_rating, _, response0, response1, prompt = evaluate_videos(video0_path, video1_path, prompt, model, template)
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
    model_type = "internvl2-2b"
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                           model_kwargs={'device_map': 'auto'})

    model.generation_config.max_new_tokens = 1024
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    # Run the processing function
    process_overall_file(args.json_file_path, args.output_dir, args.videos_dir, args.output_file_name, model, template)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate video preferences using a text-to-video quality assessment model.")
    
    # Arguments for paths
    parser.add_argument('--json_file_path', type=str, default="Safesora/config-test.json", help="Path to the JSON file containing video data.")
    parser.add_argument('--videos_dir', type=str, default="Safesora/", help="Directory containing the videos.")
    parser.add_argument('--output_dir', type=str, default="output_safesora", help="Directory to save the output results.")
    parser.add_argument('--output_file_name', type=str, default="eval_result_safesora_internvl2_2b.json", help="Name of the output file.")
    
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(args)