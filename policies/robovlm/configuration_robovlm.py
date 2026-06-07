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

from dataclasses import dataclass, field
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

@PreTrainedConfig.register_subclass("robovlm")
@dataclass
class RoboVLMConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 10 # fwd_pred_next_n (or window/action chunk)
    n_action_steps: int = 10
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    
    # Backbone and model specs
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_vlm_weights: bool = True
    freeze_backbone: bool = False
    
    # Tokenizer / Text length
    tokenizer_max_length: int = 256
    
    # Action Head Configuration
    act_head_type: str = "FCDecoder"
    latent_num: int = 1
    action_dim: int = 7
    down_sample: str = "none"
    
    # Training presets
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    
    scheduler_warmup_steps: int = 0
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 0.0
    
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )
        
    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
        
    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list:
        return [0]
        
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))
        
    @property
    def reward_delta_indices(self) -> None:
        return None
