#!/usr/bin/env bash
# Fine-tune one X-VLA policy for one atomic reset skill (paper: 2 policies x 30-50 demos per skill).
# Demos: record with LeRobot (lerobot-record) as one dataset per skill, task string = the skill instruction.
#   pip install "lerobot[xvla]"
# Usage: bash scripts/train_xvla_skill.sh <hf_dataset_repo_id> <scene> <skill_name> <policy_idx 0|1> [steps]
set -euo pipefail
DATASET=$1; SCENE=$2; SKILL=$3; IDX=$4; STEPS=${5:-20000}
OUT=ckpt/${SCENE}/${SKILL}/p${IDX}
lerobot-train \
  --dataset.repo_id="$DATASET" \
  --output_dir="outputs/xvla_${SCENE}_${SKILL}_p${IDX}" \
  --job_name="xvla_${SCENE}_${SKILL}_p${IDX}" \
  --policy.path="lerobot/xvla-base" \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --steps="$STEPS" \
  --seed=$((IDX + 1))
mkdir -p "$(dirname "$OUT")"
ln -sfn "$(pwd)/outputs/xvla_${SCENE}_${SKILL}_p${IDX}/checkpoints/last/pretrained_model" "$OUT"
echo "linked $OUT"
# The paper does not say how the two policies differ; here they differ by seed (and you may use
# different demo subsets). halter/skills/xvla_skill.py uses p1 on a retry after p0 fails.
