import os
import argparse
from pathlib import Path

def patch_file(file_path: Path, target: str, replacement: str, check_str: str):
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist. Skipping.")
        return False
        
    content = file_path.read_text()
    if check_str in content:
        print(f"Already patched: {file_path}")
        return True
        
    if target not in content:
        print(f"Error: Target pattern not found in {file_path}. Cannot patch.")
        return False
        
    new_content = content.replace(target, replacement, 1)
    file_path.write_text(new_content)
    print(f"Successfully patched: {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Patch LeRobot to register RoboVLM policy and CalvinEnv.")
    parser.add_argument("--lerobot_dir", type=str, required=True, help="Path to your lerobot repository root")
    args = parser.parse_args()
    
    lerobot_dir = Path(args.lerobot_dir)
    src_dir = lerobot_dir / "src" / "lerobot"
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist. Please check your --lerobot_dir path.")
        return
        
    # Copy files
    print("Copying policy and environment files...")
    os.system(f"mkdir -p {src_dir}/policies/robovlm")
    os.system(f"cp -r policies/robovlm/* {src_dir}/policies/robovlm/")
    os.system(f"cp envs/calvin.py {src_dir}/envs/calvin.py")
    
    # 1. Patch factory.py imports
    factory_path = src_dir / "policies" / "factory.py"
    target_import = "from .smolvla.configuration_smolvla import SmolVLAConfig\nfrom .tdmpc.configuration_tdmpc import TDMPCConfig"
    repl_import = "from .smolvla.configuration_smolvla import SmolVLAConfig\nfrom .robovlm.configuration_robovlm import RoboVLMConfig\nfrom .tdmpc.configuration_tdmpc import TDMPCConfig"
    patch_file(factory_path, target_import, repl_import, "RoboVLMConfig")
    
    # 2. Patch factory.py policy class mapping
    target_cls = '    elif name == "vla_jepa":\n        from .vla_jepa.modeling_vla_jepa import VLAJEPAPolicy\n\n        return VLAJEPAPolicy'
    repl_cls = '    elif name == "vla_jepa":\n        from .vla_jepa.modeling_vla_jepa import VLAJEPAPolicy\n\n        return VLAJEPAPolicy\n    elif name == "robovlm":\n        from .robovlm.modeling_robovlm import RoboVLMPolicy\n\n        return RoboVLMPolicy'
    patch_file(factory_path, target_cls, repl_cls, '"robovlm"')
    
    # 3. Patch factory.py config mapping
    target_cfg = '    elif policy_type == "vla_jepa":\n        return VLAJEPAConfig(**kwargs)'
    repl_cfg = '    elif policy_type == "vla_jepa":\n        return VLAJEPAConfig(**kwargs)\n    elif policy_type == "robovlm":\n        return RoboVLMConfig(**kwargs)'
    patch_file(factory_path, target_cfg, repl_cfg, '"robovlm"')
    
    # 4. Patch factory.py pre/post processors
    target_proc = '    elif isinstance(policy_cfg, VLAJEPAConfig):\n        from .vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors\n\n        processors = make_vla_jepa_pre_post_processors(\n            config=policy_cfg,\n            dataset_stats=kwargs.get("dataset_stats"),\n        )'
    repl_proc = '    elif isinstance(policy_cfg, VLAJEPAConfig):\n        from .vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors\n\n        processors = make_vla_jepa_pre_post_processors(\n            config=policy_cfg,\n            dataset_stats=kwargs.get("dataset_stats"),\n        )\n    elif isinstance(policy_cfg, RoboVLMConfig):\n        from .robovlm.processor_robovlm import make_robovlm_pre_post_processors\n\n        processors = make_robovlm_pre_post_processors(\n            config=policy_cfg,\n            dataset_stats=kwargs.get("dataset_stats"),\n        )'
    patch_file(factory_path, target_proc, repl_proc, "RoboVLMConfig")
    
    # 5. Patch env configs.py
    env_configs_path = src_dir / "envs" / "configs.py"
    target_env = '    def create_envs(self, n_envs: int, use_async_envs: bool = True):\n        from lerobot.envs.robomme import create_robomme_envs\n\n        env_cls = _make_vec_env_cls(use_async_envs, n_envs)\n        return create_robomme_envs(\n            task=self.task,\n            n_envs=n_envs,\n            action_space_type=self.action_space,\n            dataset=self.dataset_split,\n            episode_length=self.episode_length,\n            task_ids=self.task_ids,\n            env_cls=env_cls,\n        )'
    
    repl_env = target_env + """\n\n\n@EnvConfig.register_subclass("calvin")\n@dataclass\nclass CalvinEnv(EnvConfig):\n    dataset_path: str = "/home/cpsu/workspace/VLM4VLA/calvin/dataset/validation"\n    obs_type: str = "pixels"\n    render_mode: str = "rgb_array"\n    observation_width: int = 200\n    observation_height: int = 200\n    camera_name_mapping: dict[str, str] = field(\n        default_factory=lambda: {\n            "static": "static",\n            "gripper": "gripper",\n        }\n    )\n    episode_length: int = 360\n    eval_sequences_path: str | None = None\n    features: dict[str, PolicyFeature] = field(default_factory=dict)\n    features_map: dict[str, str] = field(\n        default_factory=lambda: {\n            ACTION: ACTION,\n            "pixels/static": f"{OBS_IMAGES}.static",\n            "pixels/gripper": f"{OBS_IMAGES}.gripper",\n            "robot_state": OBS_STATE,\n        }\n    )\n\n    def __post_init__(self):\n        self.features = {\n            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),\n            "pixels/static": PolicyFeature(type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)),\n            "pixels/gripper": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),\n            "robot_state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),\n        }\n\n    @property\n    def gym_kwargs(self) -> dict:\n        return {\n            "dataset_path": self.dataset_path,\n            "obs_type": self.obs_type,\n            "render_mode": self.render_mode,\n            "observation_width": self.observation_width,\n            "observation_height": self.observation_height,\n            "camera_name_mapping": self.camera_name_mapping,\n            "episode_length": self.episode_length,\n            "eval_sequences_path": self.eval_sequences_path,\n        }\n\n    def create_envs(self, n_envs: int, use_async_envs: bool = False):\n        from lerobot.envs.calvin import CalvinEnv as GymCalvinEnv\n\n        env_cls = _make_vec_env_cls(use_async_envs, n_envs)\n        \n        def env_fn():\n            return GymCalvinEnv(**self.gym_kwargs)\n            \n        return {\n            "calvin": {\n                0: env_cls([env_fn for _ in range(n_envs)])\n            }\n        }\n"""
    patch_file(env_configs_path, target_env, repl_env, '"calvin"')
    
    print("Patching complete!")

if __name__ == "__main__":
    main()
