# Bimanual SmolVLA on SO-101: Two-Arm Coordination with Vision-Language-Action Models

<p align="center">
  <img src="docs/setup_photo.jpg" alt="Bimanual SO-101 Setup" width="600"/>
</p>

A complete pipeline for teaching **two SO-101 robot arms** to coordinate on a bimanual box-passing task using **SmolVLA** (Vision-Language-Action model) with natural language instructions.

**Task:** The right arm passes a red bowl (containing a box) to the left arm. The left arm picks up the box and places it in the green bowl.

This builds on the single-arm [SmolVLA_MRL2Bootcamp](https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp) project, extending it to bimanual coordination -- a significantly harder problem that requires both arms to work together in a shared workspace.

---

## Table of Contents

1. [Demo Videos](#demo-videos)
2. [Why Bimanual?](#why-bimanual)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Setup](#software-setup)
5. [Hardware Setup & Wiring](#hardware-setup--wiring)
6. [Identifying USB Ports](#identifying-usb-ports)
7. [Calibration](#calibration)
8. [Camera Configuration](#camera-configuration)
9. [Step 1: Record Bimanual Demonstrations](#step-1-record-bimanual-demonstrations)
10. [Step 2: Train SmolVLA (Bimanual)](#step-2-train-smolvla-bimanual)
11. [Step 3: Run Bimanual Inference](#step-3-run-bimanual-inference)
12. [Architecture Deep Dive](#architecture-deep-dive)
13. [Critical Learnings & Bugs](#critical-learnings--bugs)
14. [Training Details](#training-details)
15. [HuggingFace Resources](#huggingface-resources)
16. [Repository Structure](#repository-structure)
17. [Acknowledgments](#acknowledgments)

---

## Demo Videos

### Bimanual Inference Runs

These videos show the trained SmolVLA model autonomously controlling both arms:

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3628.mp4" controls width="640"></video>

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3631.mp4" controls width="640"></video>

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3623.mp4" controls width="640"></video>

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3612.mp4" controls width="640"></video>

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3634.mp4" controls width="640"></video>

<video src="https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual/raw/main/docs/video/IMG_3636.mp4" controls width="640"></video>

---

## Why Bimanual?

Single-arm manipulation covers most tabletop tasks, but real-world robotics often requires **two-arm coordination**:

- **Handover tasks** -- one arm holds, another picks
- **Assembly tasks** -- one stabilizes, another inserts
- **Large object manipulation** -- both arms grip and move together

The key challenge is that the model must learn **coordinated 12-DOF control** (6 joints per arm) from a single policy, using shared visual input to decide what each arm should do. SmolVLA's VLM backbone provides the semantic understanding ("pass the red bowl"), while the action expert learns the bimanual coordination.

---

## Hardware Requirements

### Robot Arms (4 total)

| Arm | Role | Description |
|-----|------|-------------|
| **Left Follower** | Autonomous | SO-101 (6-DOF + gripper, Feetech STS3215 servos) |
| **Right Follower** | Autonomous | SO-101 (identical) |
| **Left Leader** | Teleoperation | SO-101 (for human demonstrations + manual reset) |
| **Right Leader** | Teleoperation | SO-101 (identical) |

### Cameras (3 total)

| Camera | Mount Location | Index | Resolution |
|--------|---------------|-------|------------|
| Left wrist cam | Left follower arm wrist | 0 | 640x480 |
| Right wrist cam | Right follower arm wrist | 1 | 640x480 |
| Overhead webcam | Tripod above workspace | 2 | 640x480 |

### Other Hardware

- **3 colored bowls** (red, green, blue) and a **box** to pick up
- **Mac with Apple Silicon** (M1/M2/M3/M4) for local inference
- **USB hub** -- at least 4 USB ports for the arm serial connections + 3 for cameras
- **Cloud GPU** (RunPod A100 recommended) for training

---

## Software Setup

### 1. Clone this repository

```bash
git clone https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp_Bimanual.git
cd SmolVLA_MRL2Bootcamp_Bimanual
```

### 2. Set up LeRobot

LeRobot must be installed from source with the bimanual robot support (`bi_so_follower`). Clone and install from the main project:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
cd ..
```

> **Important:** The `bi_so_follower` and `bi_so_leader` robot types are required. If your LeRobot version doesn't include them, use the vendored copy from the [main project](https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp).

### 3. Set your HuggingFace token

```bash
export HF_TOKEN=your_token_here
```

---

## Hardware Setup & Wiring

### Workspace Layout

```
                    [Overhead Webcam - Camera 2]
                           |
                           v
    +---------------------------------------------+
    |                 WORKSPACE                    |
    |                                              |
    |   [Green Bowl]     [Box in       [Blue Bowl] |
    |                   Red Bowl]                   |
    |                                              |
    |  LEFT ARM                        RIGHT ARM   |
    |  (Follower)                      (Follower)  |
    |  [Wrist Cam 0]                  [Wrist Cam 1]|
    +------+------------------------------+--------+
           |                              |
    LEFT LEADER                     RIGHT LEADER
    (for teleop)                    (for teleop)
```

### USB Connections (4 serial ports + 3 cameras = 7 USB connections)

Each SO-101 arm connects via a USB-to-serial adapter (CH340/CH341). You need to identify which port maps to which arm.

**My port assignments (yours will differ):**

| Device | USB Serial Port |
|--------|----------------|
| Left Follower | `/dev/tty.wchusbserial5AE60830811` |
| Right Follower | `/dev/tty.wchusbserial5AE60840931` |
| Left Leader | `/dev/tty.wchusbserial5A7C1167331` |
| Right Leader | `/dev/tty.wchusbserial5AE60829961` |

---

## Identifying USB Ports

With 4 arms connected, figuring out which serial port belongs to which arm is non-trivial. Use the included port identification script:

```bash
source activate.sh
python scripts/identify_ports.py
```

This connects to two ports simultaneously and continuously reads motor positions. **Physically move one arm by hand** -- the port showing changing values is the one connected to that arm. Repeat for all four arms.

### Alternative method

```bash
# Find all connected serial ports
ls /dev/tty.wchusbserial*
```

Then disconnect/reconnect one arm at a time to identify each port.

> **Learning:** This was one of the most time-consuming parts of the bimanual setup. Label your USB cables and ports once identified! I used tape labels on each cable.

---

## Calibration

Calibrate all four arms before first use. Calibration is saved to files and only needs to be done once per arm.

```bash
source activate.sh

# Calibrate left follower
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.wchusbserial5AE60830811 \
  --robot.id=bimanual_left

# Calibrate right follower
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.wchusbserial5AE60840931 \
  --robot.id=bimanual_right

# Calibrate left leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.wchusbserial5A7C1167331 \
  --teleop.id=bimanual_left

# Calibrate right leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.wchusbserial5AE60829961 \
  --teleop.id=bimanual_right
```

Follow the on-screen instructions to move each joint to its limits. The calibration IDs (`bimanual_left`, `bimanual_right`) must match between follower and leader for each side.

---

## Camera Configuration

### Test your cameras

```bash
python show_cameras.py
```

### The triple-camera naming convention

The bimanual system uses 3 cameras. The naming gets tricky because the `BiSOFollower` class adds `left_`/`right_` prefixes to camera names, creating double-prefixed names:

| Physical Camera | Name in SO-101 Arm | Bimanual Prefixed Name | SmolVLA Expects |
|----------------|-------------------|----------------------|-----------------|
| Left wrist cam | `left_wrist` | `left_left_wrist` | `camera1` |
| Right wrist cam | `right_wrist` | `right_right_wrist` | `camera2` |
| Overhead webcam | `webcam` | `right_webcam` | `camera3` |

The overhead webcam is physically attached to the right arm's camera config (index 2) since LeRobot associates cameras with specific arms.

**The rename map that handles this:**

```json
{
  "observation.images.left_left_wrist": "observation.images.camera1",
  "observation.images.right_right_wrist": "observation.images.camera2",
  "observation.images.right_webcam": "observation.images.camera3"
}
```

> **Critical:** This rename map must be identical in both training and inference. A mismatch causes silent failures -- the model receives garbage input and outputs erratic actions.

---

## Step 1: Record Bimanual Demonstrations

Record teleoperation episodes where you simultaneously control both leader arms. This is the hardest part -- bimanual teleoperation requires practice!

### Recording protocol

| Parameter | Value |
|-----------|-------|
| Episodes | 5+ (more is better) |
| Episode duration | 60 seconds max |
| Reset time | 15 seconds between episodes |
| Recording FPS | 30 Hz |
| Cameras | 3 (left wrist + right wrist + overhead) at 640x480 |
| Action dimensions | 12 (6 per arm) |

### Run the recording script

```bash
./record_bimanual_box.sh
```

Or manually:

```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port=/dev/tty.wchusbserial5AE60830811 \
  --robot.left_arm_config.cameras='{"left_wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.port=/dev/tty.wchusbserial5AE60840931 \
  --robot.right_arm_config.cameras='{"right_wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "webcam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=bi_so_leader \
  --teleop.id=bimanual \
  --teleop.left_arm_config.port=/dev/tty.wchusbserial5A7C1167331 \
  --teleop.right_arm_config.port=/dev/tty.wchusbserial5AE60829961 \
  --dataset.repo_id="RajatDandekar/so101_bimanual_box_pass" \
  --dataset.single_task="Right arm passes the red bowl with the box to the left arm, left arm picks up the box and places it in the green bowl" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=5 \
  --display_data=true \
  --resume=true
```

### What to do during recording

1. Both follower arms mirror the leader arms in real time.
2. **Right hand (leader):** Pick up the red bowl containing the box, pass/present it toward the left arm's workspace.
3. **Left hand (leader):** Receive the bowl, pick up the box from it, and place the box into the green bowl.
4. Press the record button (right arrow) to end the episode.
5. **Reset the workspace** during the 15-second window -- put the box back in the red bowl, bowls back to starting positions.

### Tips for good bimanual demonstrations

- **Practice the motion without recording** several times first. Bimanual teleoperation is awkward at first.
- **Keep motions slow and deliberate.** Fast, jerky bimanual motions confuse the model.
- **Consistency is key** -- approach angles, grasp positions, and handover timing should be similar across episodes.
- The task phrase matters: use a descriptive sentence that captures the full bimanual coordination.
- **Use `--resume=true`** to add more episodes to an existing dataset without overwriting.

### What gets recorded per timestep

```
observation.state:                     [12 floats]  -> 6 joints per arm (left + right)
observation.images.left_left_wrist:    [480x640x3]  -> left wrist camera
observation.images.right_right_wrist:  [480x640x3]  -> right wrist camera
observation.images.right_webcam:       [480x640x3]  -> overhead camera
action:                                [12 floats]  -> leader arm positions (6 per arm)
task:                                  string       -> the task description
```

The dataset is pushed to HuggingFace as [`RajatDandekar/so101_bimanual_box_pass`](https://huggingface.co/datasets/RajatDandekar/so101_bimanual_box_pass).

---

## Step 2: Train SmolVLA (Bimanual)

Training fine-tunes the pretrained `lerobot/smolvla_base` model on your bimanual demonstrations. The VLM backbone retains its language understanding; the action expert learns 12-DOF bimanual joint-space control.

### Train on RunPod (recommended)

Rent a GPU on [RunPod](https://runpod.io):

| GPU | VRAM | Batch Size | Training Time |
|-----|------|------------|---------------|
| A100 80GB | 80 GB | 64 | ~1 hour (10K steps) |
| A40 48GB | 48 GB | 32 | ~2 hours |
| 4090 24GB | 24 GB | 16 | ~3.5 hours |

**Setup on RunPod (run once after pod starts):**

```bash
apt-get update && apt-get install -y ffmpeg
pip install "lerobot[smolvla]"
huggingface-cli login --token $HF_TOKEN
```

**Run training:**

```bash
bash train_smolvla_bimanual_runpod.sh
```

Or manually:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="RajatDandekar/smolvla_bimanual_box_pass" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --dataset.repo_id="RajatDandekar/so101_bimanual_box_pass" \
  --batch_size=64 \
  --steps=10000 \
  --num_workers=8 \
  --save_freq=2000 \
  --log_freq=50 \
  --output_dir=outputs/train/smolvla_bimanual_box_pass \
  --rename_map='{"observation.images.left_left_wrist": "observation.images.camera1", "observation.images.right_right_wrist": "observation.images.camera2", "observation.images.right_webcam": "observation.images.camera3"}'
```

### Key differences from single-arm training

| Parameter | Single Arm | Bimanual |
|-----------|-----------|----------|
| Action dimensions | 6 | **12** (6 per arm) |
| Cameras | 2 | **3** (left wrist, right wrist, overhead) |
| Rename map entries | 2 | **3** |
| Training steps | 30,000 | **10,000** (fewer demos, less diversity) |
| Observation state | 6 floats | **12 floats** (6 per arm) |

### The rename map is essential

The camera names in the dataset (`left_left_wrist`, `right_right_wrist`, `right_webcam`) don't match what SmolVLA expects (`camera1`, `camera2`, `camera3`). The `--rename_map` flag handles this. **Forgetting it or getting it wrong causes silent failures.**

The trained model is automatically pushed to HuggingFace as [`RajatDandekar/smolvla_bimanual_box_pass`](https://huggingface.co/RajatDandekar/smolvla_bimanual_box_pass).

---

## Step 3: Run Bimanual Inference

Deploy the trained model for autonomous bimanual operation:

```bash
./run_bimanual_inference.sh
```

Or manually:

```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port="/dev/tty.wchusbserial5AE60830811" \
  --robot.left_arm_config.cameras='{"left_wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.port="/dev/tty.wchusbserial5AE60840931" \
  --robot.right_arm_config.cameras='{"right_wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "webcam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=bi_so_leader \
  --teleop.id=bimanual \
  --teleop.left_arm_config.port="/dev/tty.wchusbserial5A7C1167331" \
  --teleop.right_arm_config.port="/dev/tty.wchusbserial5AE60829961" \
  --policy.path="RajatDandekar/smolvla_bimanual_box_pass" \
  --policy.device=mps \
  --dataset.repo_id="RajatDandekar/eval_so101_bimanual_box_pass" \
  --dataset.single_task="Right arm passes the red bowl with the box to the left arm, left arm picks up the box and places it in the green bowl" \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=1 \
  --dataset.rename_map='{"observation.images.left_left_wrist": "observation.images.camera1", "observation.images.right_right_wrist": "observation.images.camera2", "observation.images.right_webcam": "observation.images.camera3"}' \
  --dataset.push_to_hub=false \
  --display_data=true
```

### What happens during bimanual inference

1. All 3 cameras capture frames at 30 Hz (640x480).
2. SmolVLA receives the 3 images + language instruction + 12-DOF joint state.
3. The model outputs an **action chunk** -- 50 future joint positions for **both** arms (12 values each).
4. Actions are queued and executed at the control rate.
5. When the queue empties, the model runs again.
6. The control loop runs at ~2.7 Hz on Apple Silicon MPS (expected -- SmolVLA is a 500M parameter VLM).

### Important inference notes

- **`--policy.device=mps`** for Apple Silicon Mac. Use `cuda` on Linux/RunPod.
- The **leader arms are still connected** during inference for manual reset between episodes. You can also grab them for safety stops.
- **Episode time is 600 seconds** (10 minutes) to give the bimanual task enough time.
- The eval dataset recording (`--dataset.repo_id`) lets you replay and review inference runs later.

---

## Architecture Deep Dive

### How BiSOFollower Works

The bimanual system uses a custom `BiSOFollower` robot class in LeRobot that wraps two independent `SOFollower` arms:

```
BiSOFollower
|-- left_arm: SOFollower
|   |-- Motors: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
|   +-- Cameras: left_wrist (index 0)
+-- right_arm: SOFollower
    |-- Motors: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    +-- Cameras: right_wrist (index 1), webcam (index 2)
```

**Observation flow:**
1. `BiSOFollower.get_observation()` calls both arms' `get_observation()`
2. Prefixes all left arm keys with `left_` and right arm keys with `right_`
3. Returns a merged dict: `{"left_shoulder_pan.pos": ..., "right_shoulder_pan.pos": ..., "left_left_wrist": <image>, ...}`

**Action flow:**
1. Policy outputs a 12-dim action tensor
2. `make_robot_action()` maps it to named keys
3. `BiSOFollower.send_action()` splits by prefix, strips the prefix, and sends to each arm independently

### SmolVLA Model Architecture for Bimanual

| Component | Details |
|-----------|---------|
| VLM backbone | SmolVLM2-500M-Video-Instruct (frozen) |
| Action expert | Lightweight transformer head (trained) |
| State projection | Linear layer mapping 12-DOF state (trained) |
| Input | 3 images (512x512 resized) + task string + 12-DOF state |
| Output | 50 action chunks x 12 DOF = 600 values per forward pass |
| Total params | ~906 MB on disk |
| Inference device | MPS (Apple Silicon) or CUDA |

---

## Critical Learnings & Bugs

These are hard-won lessons from building the bimanual pipeline. If you're building a similar system, read these carefully.

### 1. The Double-Prefix Camera Naming Problem

**Problem:** The `BiSOFollower` class adds `left_`/`right_` prefixes to ALL keys from each arm, including camera names. But the camera in the arm config is already named `left_wrist`. So the final observation key becomes `left_left_wrist` -- a double prefix.

**Impact:** If you naively set the rename map to `left_wrist -> camera1`, it won't match anything. The model receives no image for that camera and outputs garbage.

**Fix:** The rename map must use the double-prefixed names:
```json
{"observation.images.left_left_wrist": "observation.images.camera1"}
```

### 2. Port Identification with 4 Arms

**Problem:** With 4 USB serial devices connected, macOS assigns port names based on the USB-serial chip ID, not the physical port. Unplugging and replugging can (rarely) change assignments.

**Fix:** Use `scripts/identify_ports.py` to physically identify each port by wiggling motors. Label cables immediately after identification.

### 3. Bimanual Teleoperation is Hard

**Problem:** Operating two leader arms simultaneously requires significant motor coordination. Early demonstrations had jerky, uncoordinated motion that confused the model.

**Fix:**
- Practice the task 5-10 times without recording
- Start with simple motions -- the right arm moves first, then the left
- Use sequential rather than simultaneous motions when possible
- Discard bad episodes freely -- quality >> quantity

### 4. Camera Index Stability

**Problem:** USB camera indices (0, 1, 2) can shift if cameras are plugged in a different order or the system reassigns them after sleep/wake.

**Fix:**
- Always plug cameras in the same order
- Test with `python show_cameras.py` before recording/inference
- Consider using `index_or_path` with the camera's device path instead of integer index for production setups

### 5. 3-Camera SmolVLA -- camera3 Support

**Problem:** The base SmolVLA model is designed for up to 3 cameras (`camera1`, `camera2`, `camera3`). The single-arm setup uses 2. For bimanual, we need all 3.

**Fix:** The rename map simply maps the third camera to `camera3` and SmolVLA handles it natively. No model architecture changes needed.

### 6. Action Space Doubles

**Problem:** The bimanual policy outputs 12-dimensional actions (6 per arm) instead of 6. If the training data has the wrong action ordering or dimensions, the model silently learns garbage.

**Fix:** LeRobot's `BiSOFollower` automatically handles the 12-DOF split. The dataset features declare 12 action dimensions, and the training pipeline picks this up. Just make sure your recording script uses `--robot.type=bi_so_follower`.

### 7. Startup Delay -- Arms Struggle to Get Up

**Problem:** When inference starts, the arms are in a resting/collapsed position. The first policy action demands a large joint movement. The Feetech STS3215 servos have P_Coefficient=16 (reduced to prevent shakiness), which makes them slow to overcome gravity with large position jumps.

**Workaround:** Manually position the arms near the starting pose before running inference. The servos handle small incremental movements well; it's only the initial large jump from rest that struggles.

### 8. Control Loop Runs at ~2.7 Hz (Expected)

**Problem:** The control loop warns about running slower than 30 Hz. This is expected on Apple Silicon MPS -- SmolVLA inference takes ~370ms per forward pass.

**Why it still works:** Action chunking produces 50 actions per forward pass. Between model calls, the queued actions execute at the target FPS. The robot motion appears smooth despite the low model call frequency. Each model call produces ~1.7 seconds of smooth actions.

### 9. All Single-Arm Bugs Still Apply

The [10 critical bugs from the single-arm project](https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp#critical-bugs--fixes) all still apply:
- State vector must be in motor order (not alphabetical)
- Use `predict_action()` from `control_utils` -- never reimplement the pipeline
- Don't call `.float()` on SmolVLA
- Camera resolution must be 640x480 native
- The rename map is required in both training and inference

### 10. Fewer Training Steps Needed

**Observation:** With 5 bimanual episodes (vs 45 single-arm episodes), we only train for 10,000 steps (vs 30,000). The model converges faster because:
- Less diversity in demonstrations (one task, one motion pattern)
- The VLM backbone is frozen -- only the action expert trains
- The state projection layer for 12-DOF state trains quickly

Over-training on few demos can cause overfitting. Monitor the training loss and stop early if it plateaus.

---

## Training Details

Full training configuration from the [`smolvla_bimanual_box_pass`](https://huggingface.co/RajatDandekar/smolvla_bimanual_box_pass) model:

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `lerobot/smolvla_base` |
| Total training steps | 10,000 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Betas | [0.9, 0.95] |
| Weight decay | 1e-10 |
| Gradient clip norm | 10.0 |
| Scheduler | Cosine decay with warmup |
| Warmup steps | 1,000 |
| Decay steps | 30,000 |
| Minimum LR | 2.5e-6 |
| AMP (mixed precision) | Enabled |
| Save frequency | Every 2,000 steps |

### What Gets Trained vs Frozen

| Component | Trained? |
|-----------|----------|
| Vision encoder (SigLIP) | Frozen |
| VLM backbone (SmolVLM2-500M) | Frozen (first 16 layers used) |
| Action expert head | **Trained** |
| State projection (12-DOF input) | **Trained** |

### Action Chunking Config

| Parameter | Value |
|-----------|-------|
| Chunk size | 50 |
| n_action_steps | 50 |
| n_obs_steps | 1 |
| Action shape | [12] (6 per arm) |
| State shape | [12] (6 per arm) |

### Image Preprocessing

- Input resolution: 640x480 (native capture)
- Model resize: 512x512 with padding
- Normalization: ImageNet mean/std
- No data augmentation

---

## HuggingFace Resources

| Resource | Repo ID |
|----------|---------|
| Bimanual training dataset | [`RajatDandekar/so101_bimanual_box_pass`](https://huggingface.co/datasets/RajatDandekar/so101_bimanual_box_pass) |
| Bimanual trained model | [`RajatDandekar/smolvla_bimanual_box_pass`](https://huggingface.co/RajatDandekar/smolvla_bimanual_box_pass) |
| Base SmolVLA model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Single-arm dataset (reference) | [`RajatDandekar/so101_box_to_bowl`](https://huggingface.co/datasets/RajatDandekar/so101_box_to_bowl) |
| Single-arm model (reference) | [`RajatDandekar/smolvla_box_to_bowl`](https://huggingface.co/RajatDandekar/smolvla_box_to_bowl) |

---

## Repository Structure

```
.
|-- README.md                           # This file
|-- activate.sh                         # Environment activation
|
|-- record_bimanual_box.sh              # Record bimanual demonstrations
|-- train_smolvla_bimanual_runpod.sh    # Train on RunPod (A100)
|-- run_bimanual_inference.sh           # Run autonomous bimanual inference
|
|-- show_cameras.py                     # Camera display utility (test your 3 cameras)
|-- scripts/
|   +-- identify_ports.py              # Identify which USB port = which arm
|
+-- docs/
    |-- setup_photo.jpg                 # Hardware setup photo
    +-- video/
        |-- IMG_3612.mp4               # Inference demo video
        |-- IMG_3623.mp4               # Inference demo video
        |-- IMG_3628.mp4               # Inference demo video
        |-- IMG_3631.mp4               # Inference demo video
        |-- IMG_3634.mp4               # Inference demo video
        +-- IMG_3636.mp4              # Inference demo video
```

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace -- the robotics framework that makes all of this possible
- [SmolVLA](https://huggingface.co/papers/2506.01844) -- the compact VLA model architecture
- [SO-101](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio -- the open-source robot arm design
- [SmolVLA_MRL2Bootcamp](https://github.com/RajatDandekar/SmolVLA_MRL2Bootcamp) -- the single-arm predecessor project

---

*Built by Rajat Dandekar, March 2026*
