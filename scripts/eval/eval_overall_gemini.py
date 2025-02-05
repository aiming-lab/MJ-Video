import json
import os
import re
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import torch
from fuzzywuzzy import process
import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import torch
import cv2
import base64
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


openai_api_key = ""
gemini_api_key = ""


class VideoModerator:
    def __init__(self, model_id, device, openai_api_key=None, gemini_api_key=None, ckpt_dir=None):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None

        if "gpt-4o" in self.model_id:
            self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        elif "gemini" in self.model_id:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            self.genai = genai
            self.genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = self.genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

    def generate_response(self, question, video_path):
        if "gpt-4o" in self.model_id:
            # 处理视频并调用 gpt-4o API
            video = cv2.VideoCapture(video_path)
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            video.release()

            sampled_frames = base64Frames[0::50][:10]
            video_slice = map(lambda x: {"image": x, "resize": 768}, sampled_frames)
            PROMPT_MESSAGES = [{"role": "user", "content": [question, *video_slice]}]

            params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200,
            }
            result = self.client.chat.completions.create(**params)
            return result.choices[0].message.content
        elif "gemini" in self.model_id:
            max_try = 1
            retries = 0
            video = cv2.VideoCapture(video_path)
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            video.release()

            sampled_frames = base64Frames[0::50][:10]
            inputs = [question] + sampled_frames
            chat_session = self.model.start_chat(
                history=[]
            )
            while retries < max_try:
                try:
                    response = chat_session.send_message(inputs)
                    return response.text
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    retries += 1
            return None
            # response = chat_session.send_message(inputs)
            # return response.text

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

def evaluate_videos(video0_path, video1_path, prompt):
    start_time = time.time()  # 记录开始时间
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "gemini-1.5-flash"  # 或 'gemini'
    moderator = VideoModerator(model_id, device, openai_api_key, gemini_api_key)
    response0 = moderator.generate_response(prompt, video0_path)
    response1 = moderator.generate_response(prompt, video1_path)
    score0 = response0
    score1 = response1  
    end_time = time.time()  # 记录结束时间

    latency = (end_time - start_time) / 2  # 每个视频的平均延迟

    def extract_ratings(response):
        best_match, score = process.extractOne(response, fuzzy_list)
        match = re.search(r'RATING:\s*(.+)', best_match)
        return match.group(1) if match else None
    if response0 is None:
        video_0_score = -1
    if response1 is None:
        video_1_score = -1
    if response0 is not None:
        video_0_rating = extract_ratings(score0)
    if response1 is not None:
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
    if response0 is not None:
        video_0_score = rating_scale.get(video_0_rating, 0)
    if response1 is not None:
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

def process_overall_file(json_file_path, result_dir, videos_dir, output_file_name):
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
        prompt = prompt_overall.format(caption=caption)
        video_0_rating, video_1_rating, _, response0, response1, prompt = evaluate_videos(video0_path, video1_path, prompt)
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
    process_overall_file(args.json_file_path, args.output_dir, args.videos_dir, args.output_file_name)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate video preferences using a text-to-video quality assessment model.")
    
    # Arguments for paths
    parser.add_argument('--json_file_path', type=str, default="datas/test.json", help="Path to the JSON file containing video data.")
    parser.add_argument('--videos_dir', type=str, default="datas/videos", help="Directory containing the videos.")
    parser.add_argument('--output_dir', type=str, default="output_overall", help="Directory to save the output results.")
    parser.add_argument('--output_file_name', type=str, default="eval_result_overall_gemini.json", help="Name of the output file.")
    
    args = parser.parse_args()
    
    # Call main with parsed arguments
    main(args)