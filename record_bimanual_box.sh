#!/bin/zsh
# =============================================================================
# Record bimanual SO-101 episodes: Pass box from red bowl to green bowl
#
# Task: Right arm passes the bowl, left arm picks up box and places in green bowl
#
# Usage:
#   ./record_bimanual_box.sh
# =============================================================================

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_bimanual_box_pass"
NUM_EPISODES=5
TASK="Right arm passes the red bowl with the box to the left arm, left arm picks up the box and places it in the green bowl"
# ─────────────────────────────────────────────────────────────────────────────


echo "============================================"
echo "  BIMANUAL Recording"
echo "  Task:        $TASK"
echo "  Episodes:    $NUM_EPISODES"
echo "  Dataset:     $DATASET_REPO_ID"
echo "============================================"
echo ""

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
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --display_data=true \
  --resume=true

echo ""
echo "=== Done recording bimanual episodes! ==="
