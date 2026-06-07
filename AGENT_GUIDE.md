# Agent Guide: VLM4VLA & Calvin Integration for Hugging Face LeRobot

This guide is written for future autonomous agents and developers building upon, training, or evaluating the **VLM4VLA (RoboVLM)** policy and **Calvin** Gym environment in the **Hugging Face LeRobot** framework.

---

## 1. Architecture Design Philosophy

Rather than writing custom trainers or modifying LeRobot's core training loop, we maintain the design principle of **integration by registration**:
1. **Model Backbone + Head (`robovlm`)**: The VLM backbone (specifically `Qwen2.5-VL-3B-Instruct`) and the continuous `FCDecoder` action head are encapsulated within `RoboVLMPolicy` and registered as a standard LeRobot policy subclass.
2. **Sequential Benchmark (`calvin`)**: Gym environments are expected to be episodic and standard in LeRobot. We encapsulate Calvin's custom 5-task sequential evaluation protocol (which evaluates consecutive tasks without environment resets) entirely inside a custom Gymnasium environment wrapper (`CalvinEnv`).

---

## 2. Codebase & Directory Structure

* **`policies/robovlm/`**
  * `modeling_robovlm.py`: Contains the `RoboVLMPolicy` subclass of `PreTrainedPolicy`. It embeds text/vision tokens via `Qwen2.5-VL-3B-Instruct`, appends learnable action queries at the end of the token sequence, feeds the corresponding hidden states into `FCDecoder`, and outputs continuous arm actions (tanh-activated) and gripper states.
  * `configuration_robovlm.py`: Contains `RoboVLMConfig`. Note that it implements a no-op `validate_features()` method to satisfy the abstract requirements of LeRobot's `PreTrainedConfig`.
  * `processor_robovlm.py`: Standard minimal pre- and post-processors to feed raw observations and retrieve policy outputs.
* **`envs/calvin.py`**
  * Implements `CalvinEnv` wrapping the Calvin play table simulator.
  * Dynamically queries the Calvin task oracle (using OmegaConf/Hydra configs) to identify task transitions and success.
  * Runs 5 consecutive tasks in sequence per episode.
* **`convert_calvin_to_lerobot.py`**
  * Parses Calvin dataset directories (e.g. `task_ABC_D`), extracts RGB frames (static & wrist cameras), states, actions, and language instructions, and compiles them into a standard `LeRobotDataset` bundle.
* **`patch_lerobot.py`**
  * Automates registering the policy, config classes, and environment configurations into any standard clone of `huggingface/lerobot`.

---

## 3. Critical Implementation Quirks & Gotchas

### Python Version Incompatibility
> [!IMPORTANT]
> The official Hugging Face `lerobot` repository uses Python 3.12+ type parameter syntax (e.g., `[T: JsonLike]`) in its core modules.
> Running LeRobot commands under python versions older than 3.12 will result in `SyntaxError` crashes. Ensure your virtualenv is running **Python 3.12+**.
>
> *(A verified workspace environment is available on this system at `/home/cpsu/workspace/robo4d_vla/.venv_eval`)*

### Action & Gripper Normalization
* **LeRobot convention**: Gripper control ranges from `0.0` (fully closed) to `1.0` (fully open).
* **Calvin convention**: Relative gripper actions range from `-1.0` (closed) to `1.0` (open).
* **Converter Handling**: `convert_calvin_to_lerobot.py` rescales raw Calvin gripper labels:
  $$\text{action}_{\text{lerobot}} = \frac{\text{action}_{\text{calvin}} + 1.0}{2.0}$$
* **Environment Handling**: `CalvinEnv.step()` correctly remaps actions back:
  $$\text{action}_{\text{calvin}} = \text{action}_{\text{lerobot}} \times 2.0 - 1.0$$

### Sequential Evaluation Metric
During evaluation (`lerobot-eval`), the logged metric `eval_tracker.pc_success` represents the percentage of completed sequences where **all 5 tasks** were successfully executed in a row without a single failure or reset.

---

## 4. Setup, Training, and Evaluation Runbook

### Step 1: Install & Patch LeRobot
First, clone the LeRobot repository, install it, and run the patcher script:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

# Patch the installed LeRobot
python /path/to/VLM4VLA-LeRobot/patch_lerobot.py --lerobot_dir .
```

### Step 2: Convert Calvin Datasets
Convert raw Calvin `.npz` files to the `LeRobotDataset` format:
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

### Step 3: Run Training
Initiate the policy training using the standard LeRobot CLI:
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

### Step 4: Run Evaluation
To test the policy on the Calvin sequential benchmark:
```bash
lerobot-eval \
  --policy.path=./outputs/train/robovlm_calvin/checkpoints/last/policy \
  --env.type=calvin \
  --env.dataset_path=/path/to/calvin_dataset/validation \
  --eval.n_episodes=50
```
