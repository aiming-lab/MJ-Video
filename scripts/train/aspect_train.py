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
    parser.add_argument('--checkpoint_path', type=str, default="../../checkpoints/criteria_output_mse_three_epoch/checkpoint-201", help="Path to the checkpoint file (optional)")
    parser.add_argument('--num_segments', type=int, default=8, help="Frames of Videos")
    parser.add_argument('--output_dir', type=str, default='../../checkpoints/aspect_output_mse_three_epoch', help="Directory to save the model")
    parser.add_argument('--logging_dir', type=str, default='../../logs/aspect_logs', help="Directory for logging")

    # Training Arguments
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size for evaluation")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--logging_steps', type=int, default=4, help="Steps between logging")
    parser.add_argument('--save_strategy', type=str, default='epoch', choices=['epoch', 'steps'], help="Save strategy")
    parser.add_argument('--eval_steps', type=int, default=1, help="Number of steps between evaluations")
    parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['steps', 'epoch'], help="Evaluation strategy")
    parser.add_argument('--save_total_limit', type=int, default=2, help="Maximum number of saved models")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Steps to accumulate gradients before updating")
    parser.add_argument('--report_to', type=str, default="tensorboard", help="Where to report metrics")
    parser.add_argument('--logging_first_step', type=bool, default=True, help="Whether to log the first step")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "constant"], help="Type of learning rate scheduler")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=50, help="Number of warmup steps")
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
    def compute_loss(self, model, inputs, num_items_in_batch, eps = 1e-5, beta=1, alpha=[0.3, 1, 1, 0.5], focal_alpha_stage_1=[
        0.35, 0.35, 0.35, 0.35, 0.35, 0.5, 0.6, 0.6, 0.65, 0.65, 0.35, 0.65, 0.65, 0.55, 0.55, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.4, 0.45, 0.45, 0.3
    ], focal_alpha_stage_2=[0.4, 0.4, 0.43, 0.2, 0.3], mse=True):
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

        # stage 1 loss
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
        focal_alpha_stage_1 = torch.tensor(focal_alpha_stage_1).to(rewards_video_0.device)
        length = rewards_ground_truth_video_0.shape[0]
        
        if mse:
            # 用mse
            loss_video_0 = (rewards_video_0 - rewards_ground_truth_video_0) ** 2 / length
            loss_video_1 = (rewards_video_1 - rewards_ground_truth_video_1) ** 2 / length
        else:
            # 用logistic
            loss_video_0 = -(rewards_ground_truth_video_0 * torch.log(rewards_video_0 + eps) * focal_alpha_stage_1 + (1 - rewards_ground_truth_video_0) * torch.log(1 - rewards_video_0 + eps) * (1 - focal_alpha_stage_1)) * criteria_related_video_0
            loss_video_1 = -(rewards_ground_truth_video_1 * torch.log(rewards_video_1 + eps) * focal_alpha_stage_1 + (1 - rewards_ground_truth_video_1) * torch.log(1 - rewards_video_1 + eps) * (1 - focal_alpha_stage_1)) * criteria_related_video_1

        stage_one_loss = (loss_video_0 + loss_video_1).sum()


        # 第一部分的loss：由于我们有chosen和reject的具体loss，所以log(p)(1-p)的损失可以加上
        if mse:
            aspect_score_video_0 = output_video_0.aspect_scores.flatten()
            aspect_score_video_1 = output_video_1.aspect_scores.flatten()
        else:
            aspect_score_video_0 = sigmoid(output_video_0.aspect_scores.flatten())
            aspect_score_video_1 = sigmoid(output_video_1.aspect_scores.flatten())
        aspect_score_ground_truth_video_0 = inputs["video_0_aspect_score"].flatten()
        aspect_score_ground_truth_video_1 = inputs["video_1_aspect_score"].flatten()
        video_0_aspect_related = inputs["video_0_aspect_related"].flatten()
        video_1_aspect_related = inputs["video_1_aspect_related"].flatten()
        focal_alpha_stage_2 = torch.tensor(focal_alpha_stage_2).to(rewards_video_0.device)
        length = aspect_score_ground_truth_video_1.shape[0]
        if mse:
            stage_two_loss_video_0 = (aspect_score_video_0 - aspect_score_ground_truth_video_0) ** 2 / length
            stage_two_loss_video_1 = (aspect_score_video_1 - aspect_score_ground_truth_video_1) ** 2 / length
            stage_two_loss = (stage_two_loss_video_0 + stage_two_loss_video_1).sum()
        else:
            # 用logistic
            loss_logistic_video_0 = -(aspect_score_ground_truth_video_0 * torch.log(aspect_score_video_0 + eps) * focal_alpha_stage_2 + (1 - aspect_score_ground_truth_video_0) * torch.log(1 - aspect_score_video_0 + eps) * (1 - focal_alpha_stage_2)) * video_0_aspect_related
            loss_logistic_video_1 = -(aspect_score_ground_truth_video_1 * torch.log(aspect_score_video_1 + eps) * focal_alpha_stage_2 + (1 - aspect_score_ground_truth_video_1) * torch.log(1 - aspect_score_video_1 + eps) * (1 - focal_alpha_stage_2)) * video_1_aspect_related
            loss_logistic = loss_logistic_video_0 + loss_logistic_video_1
            aspect_num = video_0_aspect_related.sum().item() + video_1_aspect_related.sum().item()
            loss_logistic = loss_logistic.sum() / aspect_num if aspect_num > 0 else 0

        # 第二部分loss：bt loss (yysy，这东西其实和logistic的数学形式很像)
        aspect_reward_video_0 = output_video_0.aspect_scores.flatten()
        aspect_reward_video_1 = output_video_1.aspect_scores.flatten()
        # chosen为video_0时，preference为0，chosen为video_1时，preference为1
        # 假设为video_0(直接计算相除之后的结果，分子归一，防止溢出) [batch_size, aspect_num]
        video_0_prefer = 1 / (1 + torch.exp(beta * (aspect_reward_video_1 - aspect_reward_video_0)))
        # 假设为video_1(直接计算相除之后的结果，分子归一，防止溢出) [batch_size, aspect_num]
        video_1_prefer = 1 / (1 + torch.exp(beta * (aspect_reward_video_0 - aspect_reward_video_1)))

        aspect_preference_ground_truth = inputs["aspect_preference"].flatten() # [batch_size, aspect_num]
        aspect_mask = inputs["aspect_mask"].flatten()

        bt_loss = - torch.log((1 - aspect_preference_ground_truth) * video_0_prefer + aspect_preference_ground_truth * video_1_prefer) * aspect_mask
        bt_loss = bt_loss.sum() / aspect_mask.sum().item() if aspect_mask.sum().item() > 0 else 0

        # 第三部分loss：gating对于无关量需要分配权重为0
        # gating loss
        if not mse:
            aspect_gating_video_0 = output_video_0.aspect_weights.flatten()
            aspect_gating_video_1 = output_video_1.aspect_weights.flatten()

            mask_ground_truth_video_0 = inputs["video_0_criteria_related"].flatten()
            mask_ground_truth_video_1 = inputs["video_1_criteria_related"].flatten()

            gating_loss_video_0 = - (1 - mask_ground_truth_video_0) * torch.log(1 - aspect_gating_video_0 + eps)
            gating_loss_video_1 = - (1 - mask_ground_truth_video_1) * torch.log(1 - aspect_gating_video_1 + eps)
            gating_loss = gating_loss_video_0 + gating_loss_video_1
            gating_loss = gating_loss.sum() / batch_size

            loss = alpha[0] * stage_one_loss + alpha[1] * loss_logistic + alpha[2] * bt_loss + alpha[3] * gating_loss
        else:
            loss = alpha[0] * stage_one_loss + alpha[1] * stage_two_loss + alpha[2] * bt_loss

        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        mse = True
        """
        Override the evaluate method to ensure the input data is processed correctly during evaluation.
        """
        model = self.model
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model.eval()
        nb_eval_steps = 0
        correct_tensor_aspect = None
        total_tensor_aspect = None
        TP_aspect = None
        FP_aspect = None
        TN_aspect = None
        FN_aspect = None
        correct_tensor_criteria = None
        total_tensor_criteria = None
        TP_criteria = None
        FP_criteria = None
        TN_criteria = None
        FN_criteria = None
        
        evaluation_data_aspect = []
        evaluation_data_criteria = []
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
            aspect_score_ground_truth_video_0 = batch["video_0_aspect_score"].flatten()
            aspect_score_ground_truth_video_1 = batch["video_1_aspect_score"].flatten()
            
            # Aspect evaluation
            aspect_score_video_0 = (output_video_0.aspect_scores.flatten() > 0)
            aspect_score_video_1 = (output_video_1.aspect_scores.flatten() > 0)
            if mse:
                aspect_score_ground_truth_video_0 = (batch["video_0_aspect_score"].flatten() > 0)
                aspect_score_ground_truth_video_1 = (batch["video_1_aspect_score"].flatten() > 0)

            aspect_related_video_0 = batch["video_0_aspect_related"].flatten()
            aspect_related_video_1 = batch["video_1_aspect_related"].flatten()
            
            # Criteria evaluation
            rewards_video_0 = (output_video_0.rewards.flatten() > 0)
            rewards_video_1 = (output_video_1.rewards.flatten() > 0)
            rewards_ground_truth_video_0 = batch["video_0_criteria_score"].flatten()
            rewards_ground_truth_video_1 = batch["video_1_criteria_score"].flatten()
            if mse:
                rewards_ground_truth_video_0 = (batch["video_0_criteria_score"].flatten() > 0)
                rewards_ground_truth_video_1 = (batch["video_1_criteria_score"].flatten() > 0)
            criteria_related_video_0 = batch["video_0_criteria_related"].flatten()
            criteria_related_video_1 = batch["video_1_criteria_related"].flatten()
            
            # Initialize tensors if first batch
            if correct_tensor_aspect is None:
                correct_tensor_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
                total_tensor_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
                TP_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
                FP_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
                TN_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
                FN_aspect = torch.zeros(aspect_score_video_0.shape).to(aspect_score_video_0.device)
            if correct_tensor_criteria is None:
                correct_tensor_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                total_tensor_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                TP_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                FP_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                TN_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)
                FN_criteria = torch.zeros(criteria_related_video_0.shape).to(criteria_related_video_0.device)

            # Calculate aspect metrics
            TP_aspect, FP_aspect, TN_aspect, FN_aspect, correct_tensor_aspect, total_tensor_aspect = self.calculate_metrics(aspect_score_video_0, aspect_score_ground_truth_video_0, aspect_related_video_0, correct_tensor_aspect, total_tensor_aspect, TP_aspect, FP_aspect, TN_aspect, FN_aspect)
            TP_aspect, FP_aspect, TN_aspect, FN_aspect, correct_tensor_aspect, total_tensor_aspect = self.calculate_metrics(aspect_score_video_1, aspect_score_ground_truth_video_1, aspect_related_video_1, correct_tensor_aspect, total_tensor_aspect, TP_aspect, FP_aspect, TN_aspect, FN_aspect)
            
            # Calculate criteria metrics
            TP_criteria, FP_criteria, TN_criteria, FN_criteria, correct_tensor_criteria, total_tensor_criteria = self.calculate_metrics(rewards_video_0, rewards_ground_truth_video_0, criteria_related_video_0, correct_tensor_criteria, total_tensor_criteria, TP_criteria, FP_criteria, TN_criteria, FN_criteria)
            TP_criteria, FP_criteria, TN_criteria, FN_criteria, correct_tensor_criteria, total_tensor_criteria = self.calculate_metrics(rewards_video_1, rewards_ground_truth_video_1, criteria_related_video_1, correct_tensor_criteria, total_tensor_criteria, TP_criteria, FP_criteria, TN_criteria, FN_criteria)
            
            nb_eval_steps += 1

        # Calculate final metrics and save results
        accuracy_aspect = self.save_metrics(correct_tensor_aspect, total_tensor_aspect, TP_aspect, FP_aspect, TN_aspect, FN_aspect, evaluation_data_aspect, "aspect_evaluation_results", device_id)
        accuracy_criteria = self.save_metrics(correct_tensor_criteria, total_tensor_criteria, TP_criteria, FP_criteria, TN_criteria, FN_criteria, evaluation_data_criteria, "criteria_evaluation_results", device_id)

        output = {
            f"{metric_key_prefix}_accuracy": accuracy_aspect,
        }
        return output

    def calculate_metrics(self, predictions, ground_truth, mask, correct_tensor, total_tensor, TP, FP, TN, FN):
        mask_active = mask != 0
        predicted_positives = (predictions == 1)
        predicted_negatives = (predictions == 0)
        actual_positives = (ground_truth == 1)
        actual_negatives = (ground_truth == 0)
        
        correct = (predictions == ground_truth) & mask_active
        TP += (predicted_positives & actual_positives) & mask_active
        FP += (predicted_positives & actual_negatives) & mask_active
        TN += (predicted_negatives & actual_negatives) & mask_active
        FN += (predicted_negatives & actual_positives) & mask_active
        
        correct_tensor += correct
        total_tensor += mask_active

        return TP, FP, TN, FN, correct_tensor, total_tensor

    def save_metrics(self, correct_tensor, total_tensor, TP, FP, TN, FN, evaluation_data, file_name, device_id):
        accuracy = correct_tensor.sum().item() / total_tensor.sum().item() if total_tensor.sum().item() > 0 else 0
        recall = TP.sum().item() / (TP + FN).sum().item() if (TP + FN).sum().item() > 0 else 0
        precision = TP.sum().item() / (TP + FP).sum().item() if (TP + FP).sum().item() > 0 else 0
        F1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

        accuracy_dim = correct_tensor / total_tensor
        recall_dim = TP / (TP + FN)
        precision_dim = TP / (TP + FP)
        F1_dim = 2 * (recall_dim * precision_dim) / (recall_dim + precision_dim)

        evaluation_data.append({"Metric": "Accuracy", "Value": accuracy})
        evaluation_data.append({"Metric": "Precision", "Value": precision})
        evaluation_data.append({"Metric": "Recall", "Value": recall})
        evaluation_data.append({"Metric": "F1 Score", "Value": F1_score})

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
        for idx in range(TP.shape[-1]):
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
        # Save to Excel
        df = pd.DataFrame(evaluation_data)
        df.to_excel(f"{file_name}_{device_id}.xlsx", index=False)
        return accuracy


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for child in model.children():
        freeze_model(child)

def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True
    for child in model.children():
        activate_model(child)

def prepare_model_for_training(model):
    freeze_model(model)
    activate_model(model.criteria_gating)
    activate_model(model.regression_layer)
    activate_model(model.model.language_model)
    return model

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

    model = InternVLChatRewardModeling(name=args.model_name, config=config)

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

    trainer.train()


if __name__ == "__main__":
    main()
