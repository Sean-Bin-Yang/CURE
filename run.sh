#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_ID="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT="CURE_train.py"

# ============================================================
# Common hyperparameters
# ============================================================
WD="5e-4"
EMB="144"
EPOCHS="2000"
DROPOUT="0.1"

TASKS=("checkIn" "crime" "serviceCall")
CITIES=("NY" "Chi" "SF")

# lambda_causal search values
LAMBDA_CAUSALS=("0")

for CITY in "${CITIES[@]}"; do

  if [ "${CITY}" = "NY" ]; then
    LR="6e-4"
  elif [ "${CITY}" = "Chi" ]; then
    LR="5e-4"
  elif [ "${CITY}" = "SF" ]; then
    LR="5e-4"
  else
    echo "Unknown city: ${CITY}"
    exit 1
  fi

  for LAMBDA_CAUSAL in "${LAMBDA_CAUSALS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      echo "=========================================="
      echo "Running CURE lambda_causal search"
      echo "GPU_ID=${GPU_ID}"
      echo "CITY=${CITY}"
      echo "TASK=${TASK}"
      echo "EMB=${EMB}"
      echo "LR=${LR}"
      echo "WD=${WD}"
      echo "LAMBDA_CAUSAL=${LAMBDA_CAUSAL}"
      echo "EPOCHS=${EPOCHS}"
      echo "DROPOUT=${DROPOUT}"
      echo "=========================================="

      python "${SCRIPT}" \
        --city "${CITY}" \
        --task "${TASK}" \
        --learning_rate "${LR}" \
        --weight_decay "${WD}" \
        --lambda_causal "${LAMBDA_CAUSAL}" \
        --embedding_size "${EMB}" \
        --epochs "${EPOCHS}" \
        --dropout "${DROPOUT}"
    done
  done
done
