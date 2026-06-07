import os
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

logger = logging.getLogger(__name__)

class CalvinEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dataset_path: str,
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 200,
        observation_height: int = 200,
        camera_name_mapping: dict[str, str] | None = None,
        episode_length: int = 360,
        eval_sequences_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.episode_length = episode_length
        
        if camera_name_mapping is None:
            camera_name_mapping = {
                "static": "static",
                "gripper": "gripper",
            }
        self.camera_name_mapping = camera_name_mapping
        
        try:
            from calvin_env.envs.play_table_env import get_env
            from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
            self.get_env = get_env
            self.get_env_state_for_initial_condition = get_env_state_for_initial_condition
        except ImportError as e:
            raise ImportError(
                "Failed to import calvin dependencies. Please ensure the calvin repository is "
                "cloned and installed, or added to your PYTHONPATH."
            ) from e
            
        if eval_sequences_path is None:
            eval_sequences_path = os.environ.get(
                "CALVIN_EVAL_SEQUENCES",
                "/home/cpsu/workspace/VLM4VLA/configs/data/calvin/eval_sequences.json"
            )
            
        if os.path.exists(eval_sequences_path):
            with open(eval_sequences_path, "r") as f:
                self.eval_sequences = json.load(f)
        else:
            self.eval_sequences = []
            logger.warning(f"Could not load evaluation sequences from {eval_sequences_path}")
            
        self.current_seq_idx = 0
        self._env = None
        self._max_episode_steps = episode_length * 5
        
        images = {}
        for cam in ["static", "gripper"]:
            mapped_name = self.camera_name_mapping[cam]
            h = self.observation_height if cam == "static" else 84
            w = self.observation_width if cam == "static" else 84
            images[mapped_name] = spaces.Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8,
            )
            
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict(images),
            "robot_state": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
        })
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
    def _ensure_env(self):
        if self._env is None:
            self._env = self.get_env(self.dataset_path, show_gui=False)
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_env()
        
        if len(self.eval_sequences) == 0:
            raise ValueError("No evaluation sequences loaded.")
            
        initial_state, self.task_sequence = self.eval_sequences[self.current_seq_idx]
        self.current_seq_idx = (self.current_seq_idx + 1) % len(self.eval_sequences)
        
        try:
            import hydra
            from omegaconf import OmegaConf
            calvin_models_path = os.environ.get("CALVIN_MODELS_PATH", "/home/cpsu/workspace/VLM4VLA/calvin/calvin_models")
            conf_path = Path(calvin_models_path) / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
            task_cfg = OmegaConf.load(conf_path)
            self.task_oracle = hydra.utils.instantiate(task_cfg)
        except Exception as e:
            logger.warning(f"Could not load Calvin task oracle callback: {e}")
            self.task_oracle = None
            
        robot_obs, scene_obs = self.get_env_state_for_initial_condition(initial_state)
        self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        
        self.active_subtask_idx = 0
        self.active_subtask_steps = 0
        self.tasks_completed = 0
        self.start_info = self._env.get_info()
        
        obs = self._get_obs()
        info = {
            "is_success": False,
            "task": self.task_sequence[self.active_subtask_idx],
        }
        return obs, info
        
    @property
    def task(self) -> str:
        if hasattr(self, "task_sequence") and self.active_subtask_idx < len(self.task_sequence):
            return self.task_sequence[self.active_subtask_idx]
        return ""
        
    @property
    def task_description(self) -> str:
        return self.task
        
    def step(self, action):
        self._ensure_env()
        calvin_action = action.copy()
        calvin_action[-1] = 1.0 if action[-1] > 0.5 else -1.0
        
        raw_obs, _, _, current_info = self._env.step(calvin_action)
        self.active_subtask_steps += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        
        active_subtask = self.task_sequence[self.active_subtask_idx]
        
        subtask_success = False
        if self.task_oracle is not None:
            current_task_info = self.task_oracle.get_task_info_for_set(self.start_info, current_info, {active_subtask})
            if len(current_task_info) > 0:
                subtask_success = True
                
        if subtask_success:
            self.tasks_completed += 1
            reward = 1.0
            
            self.active_subtask_idx += 1
            self.active_subtask_steps = 0
            self.start_info = self._env.get_info()
            
            if self.active_subtask_idx >= len(self.task_sequence):
                terminated = True
        elif self.active_subtask_steps >= 360:
            terminated = True
            
        obs = self._get_obs()
        info = {
            "task": self.task,
            "tasks_completed": self.tasks_completed,
            "is_success": terminated and self.tasks_completed == 5,
        }
        return obs, reward, terminated, truncated, info
        
    def _get_obs(self):
        raw_obs = self._env.get_obs()
        static_name = self.camera_name_mapping["static"]
        gripper_name = self.camera_name_mapping["gripper"]
        
        return {
            "pixels": {
                static_name: raw_obs["rgb_obs"]["rgb_static"],
                gripper_name: raw_obs["rgb_obs"]["rgb_gripper"],
            },
            "robot_state": raw_obs["robot_obs"].astype(np.float32),
        }
        
    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
