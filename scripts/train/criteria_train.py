import argparse
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from ..data_processor.dataset import VideoDataset, VideoDataCollator
from ..model import InternVLChatRewardModeling, InternVLChatRewardModelingConfig
import os
from accelerate import Accelerator
import pandas as pd

accelerator = Accelerator()
sigmoid = torch.nn.Sigmoid()

def parse_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--train_json_path', type=str, default='../../datas/train.json', help="Path to the training data JSON file")
    parser.add_argument('--test_json_path', type=str, default='../../datas/test.json', help="Path to the test data JSON file")
    parser.add_argument('--model_name', type=str, default="OpenGVLab/InternVL2-2B", help="Model name or path")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to the checkpoint file (optional)")
    parser.add_argument('--num_segments', type=int, default=8, help="Frames of Videos")
    parser.add_argument('--output_dir', type=str, default='../../checkpoints/criteria_output_mse_three_epoch', help="Directory to save the model")
    parser.add_argument('--logging_dir', type=str, default='../../logs/criteria_logs', help="Directory for logging")

    # Training Arguments
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size for evaluation")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--logging_steps', type=int, default=3, help="Steps between logging")
    parser.add_argument('--save_strategy', type=str, default='epoch', choices=['epoch', 'steps'], help="Save strategy")
    parser.add_argument('--eval_steps', type=int, default=1, help="Number of steps between evaluations")
    parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['steps', 'epoch'], help="Evaluation strategy")
    parser.add_argument('--save_total_limit', type=int, default=3, help="Maximum number of saved models")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Steps to accumulate gradients before updating")
    parser.add_argument('--report_to', type=str, default="tensorboard", help="Where to report metrics")
    parser.add_argument('--logging_first_step', type=bool, default=True, help="Whether to log the first step")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "constant"], help="Type of learning rate scheduler")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=25, help="Number of warmup steps")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay")
    parser.add_argument('--ddp_find_unused_parameters', type=bool, default=True, help="Whether to find unused parameters in DDP")
    parser.add_argument('--remove_unused_columns', type=bool, default=False, help="Whether to remove unused columns in dataset")
    parser.add_argument('--bf16', type=bool, default=True, help="Use bfloat16 precision for training")

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

# Custom Trainer to override the loss calculation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, eps = 1e-5, focal_alpha=[
        0.35, 0.35, 0.35, 0.35, 0.35, 0.5, 0.6, 0.6, 0.65, 0.65, 0.35, 0.65, 0.65, 0.55, 0.55, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.4, 0.45, 0.45, 0.3
    ], mse=False):
        batch_size, frames, channel, height, weight = inputs["video_0_pixel_values"].shape
        video_0_pixel_values = inputs["video_0_pixel_values"].view(-1, channel, height, weight)
        video_1_pixel_values = inputs["video_1_pixel_values"].view(-1, channel, height, weight)
        output_video_0 = model(
            video_0_pixel_values, 
            inputs["video_0_input_ids"],
            inputs["video_0_attention_mask"]
        )
        output_video_1 = model(
            video_1_pixel_values, 
            inputs["video_1_input_ids"],
            inputs["video_1_attention_mask"]
        )
        if mse:
            rewards_video_0 = output_video_0.rewards.flatten()
            rewards_video_1 = output_video_1.rewards.flatten()
        else:
            rewards_video_0 = sigmoid(output_video_0.rewards.flatten())
            rewards_video_1 = sigmoid(output_video_1.rewards.flatten())
        rewards_ground_truth_video_0 = inputs["video_0_criteria_score"].flatten()
        rewards_ground_truth_video_1 = inputs["video_1_criteria_score"].flatten()
        criteria_related_video_0 = inputs["video_0_criteria_related"].flatten()
        criteria_related_video_1 = inputs["video_1_criteria_related"].flatten()
        focal_alpha = torch.tensor(focal_alpha).to(rewards_video_0.device)
        length = rewards_ground_truth_video_0.shape[0]
        if not mse:
            # 用logistic
            loss_video_0 = -(rewards_ground_truth_video_0 * torch.log(rewards_video_0 + eps) * focal_alpha + (1 - rewards_ground_truth_video_0) * torch.log(1 - rewards_video_0 + eps) * (1 - focal_alpha)) * criteria_related_video_0
            loss_video_1 = -(rewards_ground_truth_video_1 * torch.log(rewards_video_1 + eps) * focal_alpha + (1 - rewards_ground_truth_video_1) * torch.log(1 - rewards_video_1 + eps) * (1 - focal_alpha)) * criteria_related_video_1
        else:
            # 用MSE loss
            loss_video_0 = (rewards_video_0 - rewards_ground_truth_video_0) ** 2 / length
            loss_video_1 = (rewards_video_1 - rewards_ground_truth_video_1) ** 2 / length
        loss = loss_video_0 + loss_video_1
        loss = loss.sum()
        return loss

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        mse = False
        """
        Override the evaluate method to ensure the input data is processed correctly during evaluation.
        """
        model = self.model
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model.eval()
        nb_eval_steps = 0
        correct_tensor = None
        total_tensor = None
        TP = None
        FP = None
        TN = None
        FN = None
        # Min_tensor = None
        # Max_tensor = None
        # Sum_tensor = None
        # Sum_2_tensor = None
        total_num = 0

        # Data storage for the Excel file
        evaluation_data = []
        device_id = None

        for batch in eval_dataloader:
            batch_size, frames, channel, height, weight = batch["video_0_pixel_values"].shape
            video_0_pixel_values = batch["video_0_pixel_values"].view(-1, channel, height, weight)
            video_1_pixel_values = batch["video_1_pixel_values"].view(-1, channel, height, weight)

            output_video_0 = model(
                video_0_pixel_values, 
                batch["video_0_input_ids"],
                batch["video_0_attention_mask"]
            )

            output_video_1 = model(
                video_1_pixel_values, 
                batch["video_1_input_ids"],
                batch["video_1_attention_mask"]
            )
            device_id = batch["video_0_pixel_values"].device.index if batch["video_0_pixel_values"].device.index is not None else 0
            rewards_video_0 = output_video_0.rewards.flatten()
            rewards_video_1 = output_video_1.rewards.flatten()
            # del(output_video_0)
            # del(output_video_1)
            rewards_ground_truth_video_0 = batch["video_0_criteria_score"].flatten()
            rewards_ground_truth_video_1 = batch["video_1_criteria_score"].flatten()
            criteria_related_video_0 = batch["video_0_criteria_related"].flatten()
            criteria_related_video_1 = batch["video_1_criteria_related"].flatten()
            
            # if Sum_tensor is None:
            if correct_tensor is None:
                correct_tensor = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                total_tensor = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                TP = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                FP = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                TN = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                FN = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                Sum_tensor = torch.zeros(rewards_video_0.shape, device='cpu')
                Sum_2_tensor = torch.zeros(rewards_video_0.shape, device='cpu')
                Min_tensor = torch.full_like(rewards_video_0, 1000, device='cpu')
                Max_tensor = torch.full_like(rewards_video_0, -1000, device='cpu')

            Min_tensor = torch.minimum(Min_tensor, rewards_video_0.to('cpu'))
            Min_tensor = torch.minimum(Min_tensor, rewards_video_1.to('cpu'))
            Max_tensor = torch.maximum(Max_tensor, rewards_video_0.to('cpu'))
            Max_tensor = torch.maximum(Max_tensor, rewards_video_1.to('cpu'))
            Sum_tensor += (rewards_video_0.to('cpu') + rewards_video_1.to('cpu'))
            Sum_2_tensor += (rewards_video_0.to('cpu') ** 2 + rewards_video_1.to('cpu') ** 2)
            total_num += 2
            

            mask_video_0 = criteria_related_video_0 != 0
            mask_video_1 = criteria_related_video_1 != 0
            rewards_video_0 = rewards_video_0 > 0
            rewards_video_1 = rewards_video_1 > 0
            if mse:
                rewards_ground_truth_video_0 = rewards_ground_truth_video_0 > 0
                rewards_ground_truth_video_1 = rewards_ground_truth_video_1 > 0
                
            correct_video_0 = (rewards_video_0 == rewards_ground_truth_video_0) & mask_video_0
            correct_video_1 = (rewards_video_1 == rewards_ground_truth_video_1) & mask_video_1

            predicted_positives_0 = (rewards_video_0 == 1)
            predicted_positives_1 = (rewards_video_1 == 1)
            predicted_negatives_0 = (rewards_video_0 == 0)
            predicted_negatives_1 = (rewards_video_1 == 0)
            actual_positives_0 = (rewards_ground_truth_video_0 == 1)
            actual_positives_1 = (rewards_ground_truth_video_1 == 1)
            actual_negatives_0 = (rewards_ground_truth_video_0 == 0)
            actual_negatives_1 = (rewards_ground_truth_video_1 == 0)

            TP += ((predicted_positives_0 & actual_positives_0) & mask_video_0) + ((predicted_positives_1 & actual_positives_1) & mask_video_1)
            FP += ((predicted_positives_0 & actual_negatives_0) & mask_video_0) + ((predicted_positives_1 & actual_negatives_1) & mask_video_1)
            TN += ((predicted_negatives_0 & actual_negatives_0) & mask_video_0) + ((predicted_negatives_1 & actual_negatives_1) & mask_video_1)
            FN += ((predicted_negatives_0 & actual_positives_0) & mask_video_0) + ((predicted_negatives_1 & actual_positives_1) & mask_video_1)

            correct_tensor += correct_video_0
            correct_tensor += correct_video_1
            total_tensor += mask_video_0
            total_tensor += mask_video_1
            
            nb_eval_steps += 1
            accelerator.free_memory()

        accuracy = correct_tensor.sum().item() / total_tensor.sum().item() if total_tensor.sum().item() > 0 else 0
        accuracy_dim = correct_tensor / total_tensor
        recall = TP.sum().item() / (TP + FN).sum().item() if (TP + FN).sum().item() > 0 else 0
        recall_dim = TP / (TP + FN)
        precision = TP.sum().item() / (TP + FP).sum().item() if (TP + FP).sum().item() > 0 else 0
        precision_dim = TP / (TP + FP)
        F1_dim = 2 * (recall_dim * precision_dim) / (recall_dim + precision_dim)
        F1_score = 2 * (recall * precision) / (recall + precision)

        output = {
            f"{metric_key_prefix}_accuracy": accuracy,
        }
        
        # Save the results to the evaluation_data list for later writing to the Excel file
        evaluation_data.append({
            "Metric": "Accuracy", "Value": accuracy
        })
        evaluation_data.append({
            "Metric": "Precision", "Value": precision
        })
        evaluation_data.append({
            "Metric": "Recall", "Value": recall
        })
        evaluation_data.append({
            "Metric": "F1 Score", "Value": F1_score
        })

        average_tensor = Sum_tensor / total_num
        std_tensor = torch.sqrt((Sum_2_tensor + average_tensor ** 2 * total_num - 2 * Sum_tensor * average_tensor) / (total_num - 1))

        # Add per-dimension results
        for idx in range(accuracy_dim.shape[-1]):
            evaluation_data.append({
                "Metric": f"Accuracy (dim {idx})", "Value": accuracy_dim[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Precision (dim {idx})", "Value": precision_dim[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Recall (dim {idx})", "Value": recall_dim[idx].item()
            })
            evaluation_data.append({
                "Metric": f"F1 Score (dim {idx})", "Value": F1_dim[idx].item()
            })

        # Add TP, FP, TN, FN values for each dimension
        for idx in range(TN.shape[-1]):
            evaluation_data.append({
                "Metric": f"TP (dim {idx})", "Value": TP[idx].item()
            })
            evaluation_data.append({
                "Metric": f"FP (dim {idx})", "Value": FP[idx].item()
            })
            evaluation_data.append({
                "Metric": f"TN (dim {idx})", "Value": TN[idx].item()
            })
            evaluation_data.append({
                "Metric": f"FN (dim {idx})", "Value": FN[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Mean (dim {idx})", "Value": average_tensor[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Min (dim {idx})", "Value": Min_tensor[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Max (dim {idx})", "Value": Max_tensor[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Std (dim {idx})", "Value": std_tensor[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Sum (dim {idx})", "Value": Sum_tensor[idx].item()
            })
            evaluation_data.append({
                "Metric": f"Sum^2 (dim {idx})", "Value": Sum_2_tensor[idx].item()
            })

        # Add sum values for TP, FP, TN, FN
        evaluation_data.append({
            "Metric": "TP Sum", "Value": TP.sum().item()
        })
        evaluation_data.append({
            "Metric": "FP Sum", "Value": FP.sum().item()
        })
        evaluation_data.append({
            "Metric": "TN Sum", "Value": TN.sum().item()
        })
        evaluation_data.append({
            "Metric": "FN Sum", "Value": FN.sum().item()
        })
        evaluation_data.append({
            "Metric": "Count", "Value": total_num
        })

        # Convert to DataFrame and save to Excel
        df = pd.DataFrame(evaluation_data)
        df.to_excel(f"criteria_train_evaluation_results_{device_id}.xlsx", index=False)

        return output
        # return 0


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    # 递归冻结子模块的参数
    for child in model.children():
        freeze_model(child)

def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True
    # 递归冻结子模块的参数
    for child in model.children():
        activate_model(child)

def prepare_model_for_training(model):
    freeze_model(model)
    activate_model(model.regression_layer)
    activate_model(model.model.language_model)
    return model

def find_safetensors_file(checkpoint_folder):
    files = os.listdir(checkpoint_folder)
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_folder}")
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

    model = InternVLChatRewardModeling(name=args.model_name, config=config)
    if args.checkpoint_path is not None:
        try:
            from safetensors.torch import load_file
            safetensors_file = find_safetensors_file(args.checkpoint_path)
            checkpoint = load_file(safetensors_file)
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
    model = prepare_model_for_training(model)

    # Load datasets
    train_dataset = VideoDataset(
        json_path=args.train_json_path,
        tokenizer=tokenizer,
        config=config,
        generation_config=generation_config,
        num_segments=args.num_segments,
        check=True
    )

    test_dataset = VideoDataset(
        json_path=args.test_json_path,
        tokenizer=tokenizer,
        config=config,
        generation_config=generation_config,
        num_segments=args.num_segments,
        check=True
    )

    data_collactor = VideoDataCollator(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=args.report_to,
        logging_first_step=args.logging_first_step,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        remove_unused_columns=args.remove_unused_columns,
        bf16=args.bf16
    )

    # Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collactor,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
