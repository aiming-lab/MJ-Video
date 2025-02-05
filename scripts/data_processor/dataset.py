import json
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
from concurrent.futures import ThreadPoolExecutor
from boto3.session import Session
import torch
import torch.nn.functional as F
from transformers import DefaultDataCollator
from internvl2.conversation import get_conv_template


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def download_file(s3, bucket, file_key, download_directory):
    # Get the relative file path from the file_key
    relative_file_path = os.path.dirname(file_key)
    file_name = os.path.basename(file_key)  # Get just the filename from the key

    # Create the full download path including the original file structure
    full_download_directory = os.path.join(download_directory, relative_file_path)
    if not os.path.exists(full_download_directory):
        os.makedirs(full_download_directory)  # Create the directories if they don't exist

    download_path = os.path.join(full_download_directory, file_name)

    # Download the file
    try:
        if not os.path.exists(download_path):
            print(download_path)
            s3.download_file(Bucket=bucket, Key=file_key, Filename=download_path)
        else:
            pass
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def process_labels(labels, mse=True, overall=False):
    score_list = []
    related_list = []
    label_list = []
    value = labels
    if overall:
        if value == 1:
            score_list.append(1)
            related_list.append(1)
        elif value == 2:
            if mse:
                score_list.append(-1)
            else:
                score_list.append(0)
            related_list.append(1)
        else:
            score_list.append(0)
            related_list.append(0)
        return score_list, related_list, label_list
    for key, value in labels.items():
        label_list.append(key)
        if value == 1:
            score_list.append(1)  # Good
            related_list.append(1)  # Relevant
        elif value == 2:
            if mse:
                score_list.append(-1)  # Bad
            else:
                score_list.append(0)  # Bad
            related_list.append(1)  # Relevant
        else:
            score_list.append(0)  # Neutral
            related_list.append(0)  # Irrelevant
    return score_list, related_list, label_list

def deal_preference(labels, overall=False):
    preference_list = []
    mask_list = []
    if not overall:
        for key, value in labels.items():
            if value == "Video 1 better":
                preference_list.append(0)
                mask_list.append(1)
            elif value == "Video 2 better":
                preference_list.append(1)
                mask_list.append(1)
            else:
                preference_list.append(1)
                mask_list.append(0)
    else:
        value = labels
        if value == "Video 1 better":
            preference_list.append(0)
            mask_list.append(1)
        elif value == "Video 2 better":
            preference_list.append(1)
            mask_list.append(1)
        else:
            preference_list.append(1)
            mask_list.append(0)
    return preference_list, mask_list

def prepare_chat_input(config, tokenizer, pixel_values, question, generation_config, history=None, return_history=False, 
                    num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                    verbose=False, device='cpu'):
    
    img_context_token_id = None
    conv_template = get_conv_template(config.template)
    system_message = conv_template.system_message
    
    if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    img_context_token_id = img_context_token_id

    template = get_conv_template(config.template)
    template.system_message = system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    image_size = config.force_image_size or config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    # print(f'L287: query={query}=')


    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(device)

    # print(f'L293: input ids={input_ids}=')

    attention_mask = model_inputs['attention_mask'].to(device)
    generation_config['eos_token_id'] = eos_token_id

    return input_ids, attention_mask

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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
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

class VideoDataset(IterableDataset):
    def __init__(self, json_path, tokenizer=None, config=None, generation_config=None, root="./datas/videos", 
            aws_credentials = {
                'aws_key': None,
                'aws_secret_key': None,
                'bucket': None,
                'region': None
            },
            num_workers=12,
            check=False, 
            dispatch_batches=False,
            num_segments=2, 
            overall=False
        ):
        # Load JSON data from the file
        with open(json_path, 'r', encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.overall = overall
        self.root = root
        self.config = config
        self.generation_config = generation_config
        session = Session(aws_access_key_id=aws_credentials["aws_key"], aws_secret_access_key=aws_credentials["aws_secret_key"], region_name=aws_credentials["region"])
        self.s3 = session.client("s3")
        self.bucket = aws_credentials["bucket"]
        self.num_segments = num_segments
        if check:
            self.__check_video__(num_workers)

    def __check_video__(self, num_workers=12):
        # 使用ThreadPoolExecutor来并发下载视频
        with ThreadPoolExecutor(max_workers=num_workers) as executor:  # 设置并发线程数
            for item in self.data:
                # 构建两个视频路径
                video_0_path = item["video_0_path"]
                video_1_path = item["video_1_path"]
                
                # 提交下载任务
                executor.submit(download_file, self.s3, self.bucket, video_0_path, self.root)
                executor.submit(download_file, self.s3, self.bucket, video_1_path, self.root)

    def __len__(self):
        # Return the number of video pairs
        if not self.overall:
            return len(self.data)
        else:
            l = 0
            for item in self.data:
                if item["overall_preference"] == "Video 1 better" or item["overall_preference"] == "Video 2 better":
                    l += 1
            return l

    def __iter__(self, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        # If the data is a list, access the correct item; otherwise, return the single dictionary
        for item in self.data:
            # Extract relevant data for videos
            video_0_path = os.path.join(self.root, item["video_0_path"])
            video_1_path = os.path.join(self.root, item["video_1_path"])

            video_0_label = item["video_0_label"]
            video_1_label = item["video_1_label"]

            video_0_overall_label = item["video_0_overall_score"]
            video_1_overall_label = item["video_1_overall_score"]

            # preference result
            aspect_preference_list, aspect_preference_mask = deal_preference(item["category_preference"])
            overall_preference_list, overall_preference_mask = deal_preference(item["overall_preference"], True)
            if self.overall and overall_preference_mask[0] == 0:
                continue

            video_0_total_score, video_0_total_related, _ = process_labels(item["video_0_total_score"], overall=True)
            video_1_total_score, video_1_total_related, _ = process_labels(item["video_1_total_score"], overall=True)

            video_0_overall_score_list, video_0_overall_related_list, video_0_overall_label  = process_labels(video_0_overall_label)
            video_1_overall_score_list, video_1_overall_related_list, video_1_overall_label  = process_labels(video_1_overall_label)
            assert video_0_overall_label == video_1_overall_label

            aspect_label = video_0_overall_label

            video_0_score_list, video_0_related_list, video_0_label = process_labels(video_0_label)
            video_1_score_list, video_1_related_list, video_1_label = process_labels(video_1_label)
            assert video_0_label == video_1_label

            criteria_label = video_0_label

            caption = item["caption"]
            pixel_values_video_0, num_patches_list_video_0 = load_video(video_0_path, num_segments=self.num_segments, max_num=1)
            pixel_values_video_1, num_patches_list_video_1 = load_video(video_1_path, num_segments=self.num_segments, max_num=1)
            pixel_values_video_0 = pixel_values_video_0.to(torch.bfloat16).cuda()
            pixel_values_video_1 = pixel_values_video_1.to(torch.bfloat16).cuda()
            video_0_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list_video_0))])
            video_1_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list_video_1))])
            caption_video_0 = video_0_prefix + caption
            caption_video_1 = video_1_prefix + caption
            input_ids_video_0, attention_mask_video_0 = prepare_chat_input(self.config, self.tokenizer, pixel_values_video_0, caption_video_0, self.generation_config)
            input_ids_video_1, attention_mask_video_1 = prepare_chat_input(self.config, self.tokenizer, pixel_values_video_1, caption_video_1, self.generation_config)

            video_0_dict = {
                "criteria_score": video_0_score_list,
                "pixel_values": pixel_values_video_0,
                "criteria_related": video_0_related_list,
                "aspect_score": video_0_overall_score_list,
                "aspect_related": video_0_overall_related_list,
                "total_score": video_0_total_score,
                "total_related": video_0_total_related,
                "input_ids": input_ids_video_0,
                "attention_mask": attention_mask_video_0
            }

            video_1_dict = {
                "criteria_score": video_1_score_list,
                "pixel_values": pixel_values_video_1,
                "criteria_related": video_1_related_list,
                "aspect_score": video_1_overall_score_list,
                "aspect_related": video_1_overall_related_list,
                "total_score": video_1_total_score,
                "total_related": video_1_total_related,
                "input_ids": input_ids_video_1,
                "attention_mask": attention_mask_video_1
            }

            preference_dict = {
                "aspect_preference": aspect_preference_list,
                "aspect_mask": aspect_preference_mask,
                "overall_preference": overall_preference_list,
                "overall_mask": overall_preference_mask
            }

            yield {
                "criteria_label": criteria_label, 
                "aspect_label": aspect_label, 
                "video_0_dict": video_0_dict, 
                "video_1_dict": video_1_dict,
                "preference_dict": preference_dict
            }

class VideoDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, max_length=3072):
        self.tokenizer = tokenizer
        self.padding_value = tokenizer.pad_token_id
        self.max_length = max_length


    def __call__(self, batch):
        # Initialize lists to hold the processed data
        video_0_input_ids = []
        video_0_attention_mask = []
        video_0_criteria_score = []
        video_0_criteria_related = []
        video_0_aspect_score = []
        video_0_aspect_related = []
        video_0_pixel_values = []
        video_0_overall_score = []
        video_0_overall_related = []

        video_1_input_ids = []
        video_1_attention_mask = []
        video_1_criteria_score = []
        video_1_criteria_related = []
        video_1_aspect_score = []
        video_1_aspect_related = []
        video_1_pixel_values = []
        video_1_overall_score = []
        video_1_overall_related = []

        aspect_preference = []
        aspect_mask = []
        overall_preference = []
        overall_mask = []

        max_len_video_0 = 0
        max_len_video_1 = 0

        # First pass to find the maximum sequence length for padding
        if self.max_length is not None:
            max_len_video_0 = self.max_length
            max_len_video_1 = self.max_length
        else:
            for item in batch:
                video_0_dict = item["video_0_dict"]
                video_1_dict = item["video_1_dict"]
                
                # Update max sequence length for video 0 and video 1
                max_len_video_0 = max(max_len_video_0, len(video_0_dict["input_ids"][0]))
                max_len_video_1 = max(max_len_video_1, len(video_1_dict["input_ids"][0]))

        # Second pass to pad sequences to the same length
        for item in batch:
            video_0_dict = item["video_0_dict"]
            video_1_dict = item["video_1_dict"]
            preference_dict = item["preference_dict"]

            # Pad video 0 data
            padded_video_0_input_ids = F.pad(video_0_dict["input_ids"][0], 
                                            (0, max_len_video_0 - len(video_0_dict["input_ids"][0])), 
                                            value=self.padding_value)
            padded_video_0_attention_mask = F.pad(video_0_dict["attention_mask"][0], 
                                                (0, max_len_video_0 - len(video_0_dict["attention_mask"][0])), 
                                                value=0)

            video_0_input_ids.append(padded_video_0_input_ids)
            video_0_attention_mask.append(padded_video_0_attention_mask)
            video_0_pixel_values.append(video_0_dict["pixel_values"])
            video_0_criteria_score.append(torch.tensor(video_0_dict["criteria_score"]))
            video_0_criteria_related.append(torch.tensor(video_0_dict["criteria_related"]))
            video_0_aspect_score.append(torch.tensor(video_0_dict["aspect_score"]))
            video_0_aspect_related.append(torch.tensor(video_0_dict["aspect_related"]))
            video_0_overall_score.append(torch.tensor(video_0_dict["total_score"]))
            video_0_overall_related.append(torch.tensor(video_0_dict["total_related"]))

            # Pad video 1 data
            padded_video_1_input_ids = F.pad(video_1_dict["input_ids"][0], 
                                            (0, max_len_video_1 - len(video_1_dict["input_ids"][0])), 
                                            value=self.padding_value)
            padded_video_1_attention_mask = F.pad(video_1_dict["attention_mask"][0], 
                                                (0, max_len_video_1 - len(video_1_dict["attention_mask"][0])), 
                                                value=0)

            video_1_input_ids.append(padded_video_1_input_ids)
            video_1_attention_mask.append(padded_video_1_attention_mask)
            video_1_pixel_values.append(video_1_dict["pixel_values"])
            video_1_criteria_score.append(torch.tensor(video_1_dict["criteria_score"]))
            video_1_criteria_related.append(torch.tensor(video_1_dict["criteria_related"]))
            video_1_aspect_score.append(torch.tensor(video_1_dict["aspect_score"]))
            video_1_aspect_related.append(torch.tensor(video_1_dict["aspect_related"]))
            video_1_overall_score.append(torch.tensor(video_1_dict["total_score"]))
            video_1_overall_related.append(torch.tensor(video_1_dict["total_related"]))

            aspect_preference.append(torch.tensor(preference_dict["aspect_preference"]))
            aspect_mask.append(torch.tensor(preference_dict["aspect_mask"]))
            overall_preference.append(torch.tensor(preference_dict["overall_preference"]))
            overall_mask.append(torch.tensor(preference_dict["overall_mask"]))

        # Convert lists into tensors
        device = "cpu"
        video_0_input_ids = torch.stack(video_0_input_ids).to(device)
        video_0_attention_mask = torch.stack(video_0_attention_mask).to(device)
        video_0_criteria_score = torch.stack(video_0_criteria_score).to(device)
        video_0_criteria_related = torch.stack(video_0_criteria_related).to(device)
        video_0_aspect_score = torch.stack(video_0_aspect_score).to(device)
        video_0_aspect_related = torch.stack(video_0_aspect_related).to(device)
        video_0_pixel_values = torch.stack(video_0_pixel_values, 0).to(device)
        video_0_overall_score = torch.stack(video_0_overall_score).to(device)
        video_0_overall_related = torch.stack(video_0_overall_related).to(device)

        video_1_input_ids = torch.stack(video_1_input_ids).to(device)
        video_1_attention_mask = torch.stack(video_1_attention_mask).to(device)
        video_1_criteria_score = torch.stack(video_1_criteria_score).to(device)
        video_1_criteria_related = torch.stack(video_1_criteria_related).to(device)
        video_1_aspect_score = torch.stack(video_1_aspect_score).to(device)
        video_1_aspect_related = torch.stack(video_1_aspect_related).to(device)
        video_1_pixel_values = torch.stack(video_1_pixel_values, 0).to(device)
        video_1_overall_score = torch.stack(video_1_overall_score).to(device)
        video_1_overall_related = torch.stack(video_1_overall_related).to(device)

        aspect_preference = torch.stack(aspect_preference).to(device)
        aspect_mask = torch.stack(aspect_mask).to(device)
        overall_preference = torch.stack(overall_preference).to(device)
        overall_mask = torch.stack(overall_mask).to(device)

        return {
            "video_0_input_ids": video_0_input_ids,
            "video_0_attention_mask": video_0_attention_mask,
            "video_0_pixel_values": video_0_pixel_values,
            "video_0_criteria_score": video_0_criteria_score,
            "video_0_criteria_related": video_0_criteria_related,
            "video_0_aspect_score": video_0_aspect_score,
            "video_0_aspect_related": video_0_aspect_related,
            "video_0_overall_score": video_0_overall_score,
            "video_0_overall_related": video_0_overall_related,
            "video_1_input_ids": video_1_input_ids,
            "video_1_attention_mask": video_1_attention_mask,
            "video_1_pixel_values": video_1_pixel_values,
            "video_1_criteria_score": video_1_criteria_score,
            "video_1_criteria_related": video_1_criteria_related,
            "video_1_aspect_score": video_1_aspect_score,
            "video_1_aspect_related": video_1_aspect_related,
            "video_1_overall_score": video_1_overall_score,
            "video_1_overall_related": video_1_overall_related,
            "aspect_preference": aspect_preference,
            "aspect_mask": aspect_mask,
            "overall_preference": overall_preference,
            "overall_mask": overall_mask
        }


if __name__ == "__main__":
    dataset = VideoDataset(json_path="./datas/data_config.json")
