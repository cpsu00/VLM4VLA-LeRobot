# Copyright 2026 The HuggingFace Inc. team and the Google DeepMind team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
import logging
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.robovlm.configuration_robovlm import RoboVLMConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)

class MLPTanhHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSigmoidHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            # Sigmoid is applied in loss calculation / inference
        )

    def forward(self, x):
        return self.mlp(x)


class FCDecoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):
        super().__init__()
        self.down_sample = down_sample
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        self.actions = MLPTanhHead(self.hidden_size * latent, fwd_pred_next_n * (self.action_dim - 1))
        self.gripper = MLPSigmoidHead(self.hidden_size * latent, fwd_pred_next_n)
        self.mlp = nn.Sequential(
            nn.Linear(in_features * latent, in_features * latent // 2),
            nn.ReLU(),
            nn.Linear(in_features * latent // 2, hidden_size * latent),
        )
        
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
            
        self._initialize_param(self)

    def _initialize_param(self, model):
        with torch.no_grad():
            for m in model.children():
                if hasattr(m, "weight") and m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        m.bias.fill_(0)
                else:
                    self._initialize_param(m)

    def forward(self, tok_seq, **kwargs):
        # tok_seq: (bs, latent_num, hidden_size)
        bs, n_tok, tok_dim = tok_seq.shape
        
        if self.down_sample == "pooling":
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = tok_seq.reshape(bs, -1)
        elif self.down_sample == "none":
            tok_seq = tok_seq.reshape(bs, -1)
        else:
            raise NotImplementedError(f"Downsample {self.down_sample} not supported.")

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        
        # Reshape to (bs, chunk_size, dim)
        actions = actions.reshape(bs, self.fwd_pred_next_n, self.action_dim - 1)
        gripper = gripper.reshape(bs, self.fwd_pred_next_n)
        
        return actions, gripper


class RoboVLMPolicy(PreTrainedPolicy):
    config_class = RoboVLMConfig
    name = "robovlm"

    def __init__(self, config: RoboVLMConfig, **kwargs):
        super().__init__(config)
        self.config = config
        
        # Initialize processor and VLM backbone
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        self.hidden_size = self.vlm.config.hidden_size
        
        # Learnable action query token
        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.action_token, std=0.02)
        
        # Action decoder head
        self.act_head = FCDecoder(
            in_features=self.hidden_size,
            hidden_size=1024,
            action_dim=config.action_dim,
            down_sample=config.down_sample,
            latent=config.latent_num,
            fwd_pred_next_n=config.chunk_size,
        )
        
        if config.freeze_backbone:
            self.vlm.requires_grad_(False)
            
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = batch["action"].shape[0] if "action" in batch else next(iter(batch.values())).shape[0]
        
        # LeRobot images are in range [0, 1] and shape (bs, n_obs_steps, 3, H, W)
        static_imgs = batch["observation.images.static"]
        
        # Convert to PIL images for the HF Qwen processor
        images_list = []
        for i in range(bs):
            img_static = TF.to_pil_image(static_imgs[i, -1].cpu())
            if self.config.use_hand_rgb:
                img_gripper = TF.to_pil_image(batch["observation.images.gripper"][i, -1].cpu())
                images_list.append([img_static, img_gripper])
            else:
                images_list.append([img_static])
            
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]
            
        if self.config.use_hand_rgb:
            image_placeholder = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            
        texts = [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{image_placeholder}"
            f"What action should the robotic arm take to {task}<|im_end|>\n"
            "<|im_start|>assistant\n"
            for task in tasks
        ]
        
        inputs = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"].type(self.vlm.visual.dtype)
        image_grid_thw = inputs["image_grid_thw"]
        
        # 1. Embed text tokens
        input_embeds = self.vlm.model.embed_tokens(input_ids)
        
        # 2. Get vision embeddings
        image_embeds = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        
        # 3. Replace image token placeholders with visual features
        mask = input_ids == self.vlm.config.image_token_id
        image_mask = mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
        input_embeds = input_embeds.masked_scatter(image_mask, image_embeds.to(input_embeds.dtype))
        
        # 4. Append action query tokens to the end of the sequence
        action_tokens = self.action_token.view(1, 1, -1).expand(bs, self.config.latent_num, -1)
        multimodal_embeds = torch.cat([input_embeds, action_tokens], dim=1)
        
        action_attn_mask = torch.ones((bs, self.config.latent_num), dtype=attention_mask.dtype, device=attention_mask.device)
        multimodal_attention_mask = torch.cat([attention_mask, action_attn_mask], dim=1)
        
        action_token_mask = torch.zeros(multimodal_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        action_token_mask[:, -self.config.latent_num:] = True
        
        return multimodal_embeds, multimodal_attention_mask, action_token_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bs = batch["action"].shape[0]
        
        # Format VLM inputs
        multimodal_embeds, multimodal_attention_mask, action_token_mask = self._prepare_inputs(batch)
        
        # Run backbone forward
        output = self.vlm(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            output_hidden_states=True,
        )
        
        output_hs = output.hidden_states[-1]
        action_hs = output_hs[action_token_mask].reshape(bs, self.config.latent_num, -1)
        
        # Output actions and gripper logits
        pred_actions, pred_gripper = self.act_head(action_hs)
        
        # Calculate loss
        gt_actions = batch["action"] # (bs, chunk_size, 7)
        gt_pose = gt_actions[..., :6]
        gt_gripper = gt_actions[..., -1]
        
        pose_loss = F.huber_loss(pred_actions, gt_pose)
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper, gt_gripper)
        
        loss = pose_loss + 0.01 * gripper_loss
        
        return {
            "loss": loss,
            "loss_arm": pose_loss,
            "loss_gripper": gripper_loss,
        }

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        
        if len(self._action_queue) == 0:
            # Predict a new chunk of actions
            multimodal_embeds, multimodal_attention_mask, action_token_mask = self._prepare_inputs(batch)
            output = self.vlm(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                inputs_embeds=multimodal_embeds,
                use_cache=False,
                output_hidden_states=True,
            )
            output_hs = output.hidden_states[-1]
            action_hs = output_hs[action_token_mask].reshape(1, self.config.latent_num, -1)
            pred_actions, pred_gripper = self.act_head(action_hs)
            
            # Combine arm actions and gripper activation probability
            pred_gripper = torch.sigmoid(pred_gripper).unsqueeze(-1)
            pred_chunk = torch.cat([pred_actions, pred_gripper], dim=-1).squeeze(0) # (chunk_size, 7)
            
            # Queue the chunk
            self._action_queue.extend(pred_chunk.cpu())
            
        return self._action_queue.popleft().to(self.device)
