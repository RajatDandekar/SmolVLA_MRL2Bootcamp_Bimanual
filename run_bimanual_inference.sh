#!/bin/zsh
# =============================================================================
# Run bimanual inference with trained SmolVLA model on SO-101 robots
# Both arms autonomously execute the learned bimanual task.
# Usage: ./run_bimanual_inference.sh
# =============================================================================

export HF_TOKEN=$HF_TOKEN
export HF_HOME="/Users/rajatdandekar/Desktop/Robotics/.hf_cache"
export HF_LEROBOT_CALIBRATION="/Users/rajatdandekar/.cache/huggingface/lerobot/calibration"

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
# Trained bimanual model on HuggingFace
POLICY_REPO_ID="RajatDandekar/smolvla_bimanual_box_pass"

# Robot ports
LEFT_FOLLOWER_PORT="/dev/tty.wchusbserial5AE60830811"
RIGHT_FOLLOWER_PORT="/dev/tty.wchusbserial5AE60840931"

# Leader arms for manual reset between episodes
LEFT_LEADER_PORT="/dev/tty.wchusbserial5A7C1167331"
RIGHT_LEADER_PORT="/dev/tty.wchusbserial5AE60829961"

# Dataset to save inference recordings (optional — for review/debugging)
DATASET_REPO_ID="RajatDandekar/eval_so101_bimanual_box_pass"

TASK="Right arm passes the red bowl with the box to the left arm, left arm picks up the box and places it in the green bowl"
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/activate.sh"

# Remove cached eval dataset to avoid conflicts
rm -rf ~/.cache/huggingface/lerobot/${DATASET_REPO_ID}
rm -rf "${HF_HOME}/lerobot/${DATASET_REPO_ID}"

echo "=== Running BIMANUAL inference with trained SmolVLA policy ==="
echo "Policy: ${POLICY_REPO_ID}"
echo "Task:   ${TASK}"
echo ""

lerobot-record \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port="${LEFT_FOLLOWER_PORT}" \
  --robot.left_arm_config.cameras='{"left_wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --robot.right_arm_config.port="${RIGHT_FOLLOWER_PORT}" \
  --robot.right_arm_config.cameras='{"right_wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "webcam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=bi_so_leader \
  --teleop.id=bimanual \
  --teleop.left_arm_config.port="${LEFT_LEADER_PORT}" \
  --teleop.right_arm_config.port="${RIGHT_LEADER_PORT}" \
  --policy.path="${POLICY_REPO_ID}" \
  --policy.device=mps \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=1 \
  --dataset.rename_map='{"observation.images.left_left_wrist": "observation.images.camera1", "observation.images.right_right_wrist": "observation.images.camera2", "observation.images.right_webcam": "observation.images.camera3"}' \
  --dataset.push_to_hub=false \
  --display_data=true
