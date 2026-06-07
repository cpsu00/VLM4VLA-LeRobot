# VLM4VLA (RoboVLM) & Calvin Integration for Hugging Face LeRobot

This repository provides the complete implementation to port the **VLM4VLA (RoboVLM)** policy architecture (VLM backbone + `FCDecoder` continuous action head) and the **Calvin** sequential evaluation benchmark directly into the **Hugging Face LeRobot** framework. 

---

## Repository Structure

* `policies/robovlm/`: Implementation of `RoboVLMPolicy`, `RoboVLMConfig`, and `make_robovlm_pre_post_processors`.
* `envs/calvin.py`: Gymnasium-compatible wrapper (`CalvinEnv`) that runs the underlying Calvin Pybullet simulator and manages 5-step sequential task transitions internally.
* `convert_calvin_to_lerobot.py`: Converter utility that transforms the raw Calvin `.npz` dataset frames and instructions into the Hugging Face `LeRobotDataset` format.
* `patch_lerobot.py`: Script to automatically copy files and patch a LeRobot installation to register the new policy and environment.

---

## Getting Started: Installation & Setup

### Step 1: Clone and install LeRobot
Ensure you have cloned the LeRobot repository and have your environment activated:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### Step 2: Apply the Patch
Run the patch script from this repository to copy the policy/environment files and register them in LeRobot's factories:
```bash
python patch_lerobot.py --lerobot_dir /path/to/lerobot
```
This script will copy all the required files and patch:
* `lerobot/policies/factory.py` (registers `robovlm` configuration, model, and processors)
* `lerobot/envs/configs.py` (registers `calvin` environment configuration)

### Step 3: Calvin Dataset Conversion
To load the Calvin dataset in LeRobot, convert the raw Calvin training/validation directories to the standard `LeRobotDataset` format:
```bash
python convert_calvin_to_lerobot.py \
  --calvin_dir /path/to/calvin_dataset \
  --output_dir /path/to/save_lerobot_ds \
  --split training

python convert_calvin_to_lerobot.py \
  --calvin_dir /path/to/calvin_dataset \
  --output_dir /path/to/save_lerobot_ds \
  --split validation
```

---

## Running Training & Evaluation in LeRobot

### 1. Training RoboVLM on Calvin
Once the dataset is converted, training can be started using LeRobot's standard training CLI:

```bash
lerobot-train \
  --policy.type=robovlm \
  --policy.vlm_model_name="Qwen/Qwen2.5-VL-3B-Instruct" \
  --dataset.repo_id=local/calvin_training \
  --env.type=calvin \
  --env.dataset_path=/path/to/calvin_dataset/validation \
  --steps=200000 \
  --batch_size=8 \
  --eval_freq=5000 \
  --eval.n_episodes=50 \
  --output_dir=./outputs/train/robovlm_calvin
```

### 2. Evaluation
To run evaluation rollouts on a saved checkpoint:

```bash
lerobot-eval \
  --policy.path=./outputs/train/robovlm_calvin/checkpoints/last/policy \
  --env.type=calvin \
  --env.dataset_path=/path/to/calvin_dataset/validation \
  --eval.n_episodes=50
```
During evaluation, the environment will run 5 consecutive subtask rollouts for each episode without resets. The logged `eval_tracker.pc_success` represents the percentage of sequences where all 5 tasks succeeded in succession.
