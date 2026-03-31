#!/bin/bash
# =============================================================================
# SmolVLA Training on RunPod — Bimanual Box Pass Task
# Task: Right arm passes red bowl, left arm picks box and places in green bowl
#
# SETUP ON RUNPOD (run once after pod starts):
#   apt-get update && apt-get install -y ffmpeg
#   pip install "lerobot[smolvla]"
#   huggingface-cli login --token $HF_TOKEN
#
# Then run:
#   bash train_smolvla_bimanual_runpod.sh
# =============================================================================

export HF_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_bimanual_box_pass"
OUTPUT_REPO_ID="RajatDandekar/smolvla_bimanual_box_pass"

BATCH_SIZE=64
STEPS=10000
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Training SmolVLA on bimanual box pass dataset ==="
echo "  Dataset:  $DATASET_REPO_ID"
echo "  Output:   $OUTPUT_REPO_ID"
echo "  Batch:    $BATCH_SIZE"
echo "  Steps:    $STEPS"
echo ""

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${OUTPUT_REPO_ID}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size=${BATCH_SIZE} \
  --steps=${STEPS} \
  --num_workers=8 \
  --save_freq=2000 \
  --log_freq=50 \
  --output_dir=outputs/train/smolvla_bimanual_box_pass \
  --rename_map='{"observation.images.left_left_wrist": "observation.images.camera1", "observation.images.right_right_wrist": "observation.images.camera2", "observation.images.right_webcam": "observation.images.camera3"}'

echo ""
echo "=== Done! Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
