# MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences in Video Generation

<a target="_blank" href="https://arxiv.org/pdf/2502.01719">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/aiming-lab/MJ-Video">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://github.com/aiming-lab/MJ-Video">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://github.com/aiming-lab/MJ-Video/MJ-BENCH-VIDEO">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://github.com/aiming-lab/MJ-Video">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>

This repository contains the implementation of the paper "MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences in Video Generation".  We create a fine-grained video preference dataset **MJ-BENCH-VIDEO** and a MoE-structured video reward model **MJ-VIDEO**. 

## :tada: News

**MJ-VIDEO-4B** coming soon !

**Aligned Video Generation Model** coming soon !

## :white_check_mark: Abstract

Recent advancements in video generation have significantly improved the ability to synthesize videos from text instructions. However, existing models still struggle with key challenges such as instruction misalignment, content hallucination, safety concerns, and bias. Addressing these limitations, we introduce MJ-BENCH-VIDEO, a large-scale video preference benchmark designed to evaluate video generation across five critical aspects: Alignment, Safety, Fineness, Coherence & Consistency, and Bias & Fairness. This benchmark incorporates 28 fine-grained criteria to provide a comprehensive evaluation of video preference. Building upon this dataset, we propose MJ-VIDEO, a Mixture-of-Experts (MoE)-based video reward model designed to deliver fine-grained reward. MJ-VIDEO can dynamically select relevant experts to accurately judge the preference based on the input text-video pair. This architecture enables more precise and adaptable preference judgments. Through extensive benchmarking on MJ-BENCH-VIDEO, we analyze the limitations of existing video reward models and demonstrate the superior performance of MJ-VIDEO in video preference assessment, achieving 17.58% and 15.87% improvements in overall and fine-grained preference judgments, respectively. Additionally, introducing MJ-VIDEO for preference tuning in video generation enhances the alignment performance.

## :fuelpump: Installation

To install the necessary dependencies, run the following command:

For the testing of models **other than InternVL2-4B and CogVLM2**, use the following commands for installation:

```bash
git clone git@github.com:aiming-lab/MJ-Video.git
conda create -n mjvideo python=3.10
cd MJ-Video
pip install -r requirements.txt
```

For **InternVL2-4B**, modify the versions of some libraries and use the following commands to create the environment:

```bash
conda create -n mjvideo_4b python=3.10
cd MJ-Video
pip install -r requirements_4b.txt
```

For **CogVLM2**, modify the versions of some libraries and use the following commands to create the environment:

```bash
conda create -n mjvideo_cog python=3.10
cd MJ-Video
pip install -r requirements_cog.txt
```

For the environment of **fine-tuning VADER on VideoCrafter2**, please refer to [VADER](https://github.com/mihirp1998/VADER).

## :car:Vide Preference Dataset

Our dataset is available at [MJ-BENCH-VIDEO](https://github.com/aiming-lab/MJ-Video/MJ-BENCH-VIDEO).

You can download our dataset from Hugging Face and use the code in [scripts/data/dataset.py](scripts/data/dataset.py) to load the dataset for training and evaluation.

## :factory: Video Reward Model

Our reward model is available at [MJ-VIDEO](https://github.com/aiming-lab/MJ-Video/MJ-BENCH-VIDEO).

If you want to use your own dataset or our dataset, use the code in [scripts/train](scripts/train) to conduct the training.

For inference, you can refer to [scritps/model/moe-playground.ipynb](scritps/model/moe-playground.ipynb), or use the following code for inference.

```python
from model import InternVLChatRewardModeling, InternVLChatRewardModelingConfig, prepare_chat_input
from data_processor import load_video
from torch import distributed as dist
import os

os.environ['WORLD_SIZE'] = str(1)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(12345)
os.environ['LOCAL_RANK'] = str(0)
os.environ['RANK'] = str(0)

dist.init_process_group(backend='nccl', world_size=1, rank=0)

### prepare model
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
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(torch.bfloat16).cuda()
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.eval()

## prepare data
caption = "Generate a video of a tiger dancing."
pixel_values, num_patches_list = load_video(video, num_segments=8, max_num=1)
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
pixel_values = pixel_values.to(torch.bfloat16).to(model.model.device)
prompt = video_prefix + caption
input_ids, attention_mask = prepare_chat_input(config, tokenizer, pixel_values, prompt, generation_config, device=model.model.device)

## inference
output = model.forward(pixel_values, input_ids, attention_mask_left)

## criteria score
output.reward

## aspect score
output.aspect_scores

## overall score
output.score
```

## :airplane: Preference Alignment

We trained our video generation model based on VADER; here are a few examples.

<img src=".\asserts\Case_study_Align.png" alt="Case_study_Align" style="zoom: 25%;" />

## :train:Leadboard

#### Testing on Aspect Annotations in MJ-BENCH-VIDEO

The **bolded** numbers in the table represent the best results, while the italic numbers indicate the second-best results. 
The "C&C" in the table refers to "Coherence and Consistency," while "B&F" refers to "Bias and Fairness." 
In cases where certain models show strong bias, causing the F1 score to be NaN, a "/" is used in place of the result in the table. 
For preference comparison, we report the results of the "strict" metric. 

| Model            | Alignment (Acc) | Alignment (F1) | Alignment (strict) | Safety (Acc) | Safety (F1) | Safety (strict) | Fineness (Acc) | Fineness (F1) | Fineness (strict) | C&C (Acc) | C&C (F1)  | C&C (strict) | B&F (Acc) | B&F (F1)  | B&F (strict) |
| ---------------- | --------------- | -------------- | ------------------ | ------------ | ----------- | --------------- | -------------- | ------------- | ----------------- | --------- | --------- | ------------ | --------- | --------- | ------------ |
| InternVL2-2B     | _70.75_         | 60.42          | 17.71              | 66.67        | 55.02       | 16.67           | 63.59          | 49.87         | 3.125             | _71.81_   | _46.04_   | 10.34        | 74.11     | **63.19** | 54.54        |
| InternVL2-4B     | 57.00           | 55.00          | 26.96              | 75.49        | 60.37       | 0.00            | 52.48          | 49.92         | 7.143             | 43.02     | 33.11     | 17.86        | 66.32     | _56.27_   | 54.55        |
| InternVL2-8B     | 44.21           | 44.21          | 33.33              | 76.72        | 72.60       | 16.67           | 47.71          | 47.27         | 18.75             | 27.76     | 24.29     | 12.07        | 15.51     | 13.88     | 50.00        |
| InternVL2-26B    | 65.47           | _62.96_        | 40.51              | _84.44_      | _78.26_     | 20.00           | **69.81**      | 51.91         | 14.29             | 59.03     | 41.51     | 16.33        | _82.05_   | 59.85     | 30.00        |
| Qwen2-VL-2B      | 54.28           | 53.03          | 19.35              | 59.82        | 56.93       | 25.00           | 56.75          | 51.86         | 3.448             | 37.90     | 31.18     | 16.39        | 20.00     | 19.31     | 38.46        |
| Qwen2-VL-7B      | 58.31           | 56.19          | 41.94              | 55.35        | 52.81       | 25.00           | 47.56          | 46.33         | 31.03             | 32.58     | 27.68     | 19.67        | 14.61     | 13.13     | 23.08        |
| MiniCPM-8B       | 65.53           | 61.38          | 48.72              | 72.91        | 67.22       | 40.00           | 62.13          | _56.02_       | _39.29_           | 49.73     | 37.21     | 31.25        | 15.12     | 14.17     | _60.00_      |
| CogVLM2          | 26.71           | 23.80          | 7.692              | 31.67        | 30.09       | 16.67           | 35.61          | 29.79         | 11.76             | 7.87      | 7.86      | 4.615        | 14.61     | /         | 7.692        |
| Gemini-1.5-flash | 27.45           | 25.72          | 8.421              | 83.64        | 77.34       | 0.0             | 32.80          | 25.27         | 12.90             | 5.01      | 4.88      | 12.07        | 15.18     | /         | 9.091        |
| GPT-4o           | 58.27           | 56.21          | _50.00_            | 82.86        | 77.00       | _50.00_         | 59.67          | 56.34         | 27.27             | 44.52     | 34.17     | _40.00_      | 19.17     | 18.48     | 33.33        |
| **MJ-VIDEO**     | **78.41**       | **71.22**      | **79.05**          | **87.50**    | **81.84**   | **83.33**       | _68.60_        | **58.53**     | **58.82**         | **95.36** | **53.57** | **58.46**    | **86.92** | 55.97     | **69.23**    |

#### Results of Overall Video Preference Evaluation

The **best** test results are highlighted in bold, and the *second-best* results are in italic. 
_Strict_ treats undecided cases as incorrect, while _tie-aware_ assigns 0.5 for ties in calculating accuracy.  

| Model         | \datasetname (Strict) | \datasetname (Tie-aware) | Safesora-test (Strict) | Safesora-test (Tie-aware) | GenAI-Bench (Strict) | GenAI-Bench (Tie-aware) |
| ------------- | --------------------- | ------------------------ | ---------------------- | ------------------------- | -------------------- | ----------------------- |
| InternVL2-2B  | 5.93                  | 47.88                    | 4.60                   | 50.30                     | 13.71                | 55.43                   |
| InternVL2-4B  | 13.55                 | 49.15                    | 11.74                  | 50.91                     | 39.00                | 61.79                   |
| InternVL2-8B  | 16.95                 | 47.88                    | 14.29                  | 53.09                     | 36.85                | 62.43                   |
| InternVL2-26B | 22.88                 | 53.81                    | 10.41                  | 52.00                     | 31.86                | 55.64                   |
| Qwen-VL-2B    | 13.33                 | 48.09                    | 13.18                  | 51.27                     | 27.29                | 56.71                   |
| Qwen-VL-7B    | 17.14                 | 47.62                    | 14.58                  | 52.41                     | 20.57                | 51.36                   |
| MiniCPM       | 30.51                 | 53.39                    | 25.30                  | 52.54                     | 47.43                | 60.21                   |
| CogVLM2       | 8.47                  | 47.46                    | 9.56                   | 52.48                     | 21.29                | 56.29                   |
| VideoScore    | *58.47*               | *58.47*                  | *55.33*                | *55.51*                   | *69.14*              | *69.14*                 |
| Gemini        | 2.66                  | 48.67                    | 2.66                   | 48.67                     | 21.45                | 50.71                   |
| GPT-4o        | 35.35                 | 54.6                     | 35.35                  | 54.6                      | 48.85                | 59.14                   |
| **MJ-VIDEO**  | **68.75**             | **68.75**                | **64.16**              | **64.16**                 | **70.28**            | **70.28**               |

#### Evaluation of Video Models Across Human Evaluation and Automated Evaluation on VBench

Human evaluation assesses **Video Quality** and **Text-to-Video Alignment**. 
Automated evaluation on VBench evaluates **Imaging Quality (IQ)**, **Human Action (HA)**, **Scene (S)**, and **Overall Consistency (OC)**.  

| Model         | Quality (Human) | Align (Human) | IQ (Auto) | HA (Auto) | S (Auto)  | OC (Auto) |
| ------------- | --------------- | ------------- | --------- | --------- | --------- | --------- |
| VideoCrafter2 | 56.30           | 68.80         | *67.04*   | 90.00     | 54.00     | **28.39** |
| VideoScore    | *64.50*         | *74.80*       | 65.03     | *92.00*   | *54.79*   | *28.38*   |
| **MJ-VIDEO**  | **69.90**       | **79.20**     | **67.89** | **94.00** | **55.09** | 28.19     |

# Cite

Please cite us using the following bibtex
```bibtex
@misc{tong2025mjvideofinegrainedbenchmarkingrewarding,
      title={MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences in Video Generation}, 
      author={Haibo Tong and Zhaoyang Wang and Zhaorun Chen and Haonian Ji and Shi Qiu and Siwei Han and Kexin Geng and Zhongkai Xue and Yiyang Zhou and Peng Xia and Mingyu Ding and Rafael Rafailov and Chelsea Finn and Huaxiu Yao},
      year={2025},
      eprint={2502.01719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.01719}, 
}
```
