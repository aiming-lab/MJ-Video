from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from internvl2 import InternVLChatModel, InternLM2PreTrainedModel, get_conv_template, InternVLChatConfig

from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.utils import ModelOutput
from transformers.utils import add_start_docstrings_to_model_forward


class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 10,
                logit_scale: float = 1., hidden_dim: int = 1024, n_hidden: int = 3):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        # Apply the conditional ReLU using the expanded mask
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]

    def forward_wo_softmax(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        # Apply the conditional ReLU using the expanded mask
        return x


# for internvl2
# text = '<|im_end|><|im_start|>assistant\n'
# tokenizer(text, return_tensors='pt')
token_pattern = [92542, 92543,   525, 11353,   364]

def find_token_for_gating(lst, ):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j:j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


@dataclass
class CustomOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        hidden_state (`Tuple[torch.FloatTensor]` of length `config.num_hidden_layers`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        prompt_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The embeddings of the prompt tokens.
        gating_output (`torch.FloatTensor` of shape `(batch_size, config.num_objectives)`):
            The logits for the gating network.
        score (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            The final reward score.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Same as score
    """

    rewards: torch.FloatTensor = None
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    prompt_embedding: Optional[torch.FloatTensor] = None
    criteria_gating_output: Optional[torch.FloatTensor] = None
    aspect_gating_output: Optional[torch.FloatTensor] = None
    aspect_weights: Optional[torch.FloatTensor]=None
    score: Optional[torch.FloatTensor] = None
    weighted_scores: Optional[torch.FloatTensor]=None
    aspect_scores: Optional[torch.FloatTensor]=None


class InternVLChatRewardModelingConfig(InternVLChatConfig):
    def __init__(self, internVLChatConfigName_or_path=None, **kwargs):
        """
        自定义配置类，继承自 InternVLChatConfig，并支持额外的参数。
        :param internVLChatConfigName_or_path: 可选的预训练配置路径或名称。
        :param kwargs: 其他参数，用于初始化自定义属性。
        """
        super().__init__(**kwargs)

        # 初始化自定义属性
        self.num_objectives = kwargs.get('num_objectives', 0)
        self.num_aspects = kwargs.get('num_aspects', 0)
        self.aspect2criteria = kwargs.get('aspect2criteria', {})
        self.gating_temperature = kwargs.get('gating_temperature', 1.0)
        self.gating_hidden_dim = kwargs.get('gating_hidden_dim', 1024)
        self.gating_n_hidden = kwargs.get('gating_n_hidden', 3)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        重载 from_pretrained 方法以支持加载自定义属性。
        """
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.num_objectives = kwargs.get('num_objectives', config.num_objectives)
        config.num_aspects = kwargs.get('num_aspects', config.num_aspects)
        config.aspect2criteria = kwargs.get('aspect2criteria', config.aspect2criteria)
        config.gating_temperature = kwargs.get('gating_temperature', config.gating_temperature)
        config.gating_hidden_dim = kwargs.get('gating_hidden_dim', config.gating_hidden_dim)
        config.gating_n_hidden = kwargs.get('gating_n_hidden', config.gating_n_hidden)
        return config

    def to_dict(self):
        """
        重载 to_dict 方法以确保自定义属性被保存。
        """
        output = super().to_dict()
        output.update({
            "num_objectives": self.num_objectives,
            "num_aspects": self.num_aspects,
            "aspect2criteria": self.aspect2criteria
        })
        return output



class InternVLChatRewardModeling(nn.Module):
    def __init__(self, name: str, config):
        super().__init__()

        self.num_labels = config.num_labels
        self.model = InternVLChatModel.from_pretrained(name)
        # 禁用 vision_model 的梯度
        config_dict = config.to_dict()
        # aspects * criteria = total criteria
        self.num_objectives = config_dict['num_objectives']
        # aspects
        self.num_aspects = config_dict['num_aspects']
        # aspect 2 criteria
        # for example, aspect2criteria = {0: [0, 1], 1: [2, 3]}
        self.aspect2criteria: dict[int, list[int]] = config_dict['aspect2criteria']
        # sanity check
        assert len(self.aspect2criteria) == self.num_aspects
        assert sum(len(v) for v in self.aspect2criteria.values()) == self.num_objectives
        temp = []
        for k in self.aspect2criteria.values(): temp += k
        assert sum(len(set(v)) for v in self.aspect2criteria.values()) == len(set(temp))

        hidden_size = config.llm_config.hidden_size

        self.regression_layer = nn.Linear(hidden_size, self.num_objectives, bias=False)
        # Not using torch.eye because it is not supported in BF16
        I = torch.zeros(self.num_objectives, self.num_objectives)
        I[range(self.num_objectives), range(self.num_objectives)] = 1.
        self.reward_transform_matrix = nn.Parameter(I)
        self.reward_transform_matrix.requires_grad = False

        # aspect moe
        self.aspect_gating = GatingNetwork(hidden_size, self.num_aspects,
                                    temperature=config_dict["gating_temperature"], 
                                    hidden_dim=config_dict["gating_hidden_dim"], 
                                    n_hidden=config_dict["gating_n_hidden"])

        # criteria moe
        self.criteria_gating = GatingNetwork(hidden_size, config.num_objectives,
                                    temperature=config_dict["gating_temperature"],
                                    hidden_dim=config_dict["gating_hidden_dim"],
                                    n_hidden=config_dict["gating_n_hidden"])
        
        # config
        self.config = config

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.model.forward(
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
            past_key_values,
            labels,
            use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        tokens_hidden_states = transformer_outputs.hidden_states[-1]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(tokens_hidden_states.device)

                # debug
                if False:
                    print(sequence_lengths)
            else:
                sequence_lengths = -1

        dummy_iterator = torch.arange(batch_size, device=tokens_hidden_states.device)
        hidden_states = tokens_hidden_states[dummy_iterator, sequence_lengths]
        # assert hidden_states.shape == (batch_size, self.config.hidden_size)
        rewards = self.regression_layer(hidden_states)
        rewards = rewards @ self.reward_transform_matrix

        gating_token_positions = [find_token_for_gating(ids.tolist()) for ids in input_ids]
        prompt_embedding = tokens_hidden_states[dummy_iterator, gating_token_positions, :]

        # aspect gating [bsz, num_aspects]
        aspect_gating_output = self.aspect_gating(prompt_embedding)
        # criteria gating
        # gating wo softmax
        criteria_gating_output = self.criteria_gating.forward_wo_softmax(prompt_embedding)
        aspect_weights = {}
    
        # deal for aspect
        for aspect, criteria_indices in self.aspect2criteria.items():
            # get criteria dim
            aspect_output = criteria_gating_output[:, criteria_indices]
            
            # use softmax
            aspect_weights[aspect] = F.softmax(aspect_output / self.criteria_gating.temperature, dim=-1) * self.criteria_gating.logit_scale[0]

        aspect_num = len(self.aspect2criteria)  # 计算有多少个aspect
        # 初始化一个张量来保存每个aspect的分数
        aspect_scores = torch.zeros(batch_size, aspect_num).to(aspect_output.device)
        
        # 对每个aspect进行加权操作
        for i, (aspect, criteria_indices) in enumerate(self.aspect2criteria.items()):
            # 提取该aspect的权重
            aspect_weight = aspect_weights[aspect]
            
            # 提取rewards中对应criteria的分数
            aspect_rewards = rewards[:, criteria_indices]
            
            # 对每个criteria分数应用权重并求和，得到该aspect的分数
            weighted_scores = (aspect_rewards * aspect_weight).sum(dim=-1)
            
            # 将结果存入对应的列
            aspect_scores[:, i] = weighted_scores
        score = (aspect_scores * aspect_gating_output).sum(dim=-1)
        # Initialize a list to collect the aspect weights
        aspect_weights_list = []

        # Iterate through the aspect_weights dictionary
        for aspect, weight in aspect_weights.items():
            aspect_weights_list.append(weight)

        final_aspect_weights = torch.cat(aspect_weights_list, dim=-1)

        return CustomOutput(
            rewards=rewards,
            hidden_state=hidden_states,
            prompt_embedding=prompt_embedding,
            criteria_gating_output=criteria_gating_output,
            aspect_gating_output=aspect_gating_output,
            aspect_weights=final_aspect_weights,
            weighted_scores=weighted_scores,
            aspect_scores=aspect_scores,
            score=score
        )