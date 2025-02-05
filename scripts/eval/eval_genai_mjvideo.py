# load the GenAI-Bench(GenAI-Bench-1600) benchmark
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer
from dataset import VideoDataset, VideoDataCollator
from ..model import InternVLChatRewardModeling, InternVLChatRewardModelingConfig, prepare_chat_input
import os
import pandas as pd
from ..data_processor import load_video
from torch import distributed as dist
import os

os.environ['WORLD_SIZE'] = str(1)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(12345)
os.environ['LOCAL_RANK'] = str(0)
os.environ['RANK'] = str(0)


dist.init_process_group(backend='nccl', world_size=1, rank=0)


def parse_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model_name', type=str, default="OpenGVLab/InternVL2-2B", help="Model name or path")
    parser.add_argument('--checkpoint_path', type=str, default="../../checkpoints/overall_output_mse_three_epoch/checkpoint-39", help="Path to the checkpoint file (optional)")
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

def find_safetensors_file(checkpoint_folder):
    # List all files in the checkpoint folder
    files = os.listdir(checkpoint_folder)
    
    # Filter for safetensors files (files ending with '.safetensors')
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_folder}")
    
    # For now, just return the first safetensors file found
    return os.path.join(checkpoint_folder, safetensors_files[0])

def main():
    # Parse arguments
    args = parse_args()

    # Initialize tokenizer, model, and configuration
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    config = InternVLChatRewardModelingConfig.from_pretrained(
        args.model_name,
        num_objectives=args.num_objectives,
        num_aspects=args.num_aspects,
        aspect2criteria=args.aspect2criteria,
        gating_temperature=args.gating_temperature,
        gating_hidden_dim=args.gating_hidden_dim,
        gating_n_hidden=args.gating_n_hidden
    )

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample
    }

    model = InternVLChatRewardModeling(name=args.model_name, config=config).cuda()

    if args.checkpoint_path is not None:
        try:
            from safetensors.torch import load_file
            # Find a safetensors file in the checkpoint folder
            safetensors_file = find_safetensors_file(args.checkpoint_path)

            # Load the safetensors checkpoint file
            checkpoint = load_file(safetensors_file)

            # Load weights into the model
            model.load_state_dict(checkpoint, strict=True)

            print(f"Model weights successfully loaded from {safetensors_file} using safetensors")

        except FileNotFoundError as e:
            print(e)
        except KeyError as e:
            print(f"Key error: {e} in the safetensors file. Ensure the file contains the correct keys.")
        except Exception as e:
            print(f"An error occurred while loading the safetensors checkpoint: {e}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(torch.bfloat16).cuda()
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.eval()

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
        pixel_values_left, num_patches_list_left = load_video(left_video, num_segments=8, max_num=1)
        pixel_values_left = pixel_values_left.to(torch.bfloat16).to(model.model.device)
        video_prefix_left = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list_left))])
        pixel_values_right, num_patches_list_right = load_video(right_video, num_segments=8, max_num=1)
        pixel_values_right = pixel_values_right.to(torch.bfloat16).to(model.model.device)
        video_prefix_right = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list_right))])
        prompt_left = video_prefix_left + prompt
        prompt_right = video_prefix_right + prompt
        input_ids_left, attention_mask_left = prepare_chat_input(config, tokenizer, pixel_values_left, prompt_left, generation_config, device=model.model.device)
        input_ids_right, attention_mask_right = prepare_chat_input(config, tokenizer, pixel_values_right, prompt_right, generation_config, device=model.model.device)
        score_left = model.forward(pixel_values_left, input_ids_left, attention_mask_left).score[0]
        score_right = model.forward(pixel_values_right, input_ids_right, attention_mask_right).score[0]
        left_judge = "good" if score_left > 0 else "bad"
        right_judge = "good" if score_right > 0 else "bad"
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