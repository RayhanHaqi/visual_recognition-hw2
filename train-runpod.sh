#!/bin/bash

Q=${1:-1000}       # Argumen 1: Queries (Default: 1000)
BS=${2:-24}        # Argumen 2: Batch Size (Default: 24)
LR=${3:-5e-5}      # Argumen 3: Learning Rate (Default: 5e-5)
WD=${4:-5e-4}      # Argumen 4: Weight Decay (Default: 5e-4)
GPU=${5:-0}       # Argumen 5: GPU ID (Default: 0)
WORKER=${6:-6}    # Argumen 6: Number of Workers (Default: 6)
EOF=${7:-0.0001}  # Argumen 7: EOS Coefficient (Default: 0.0001)

RUN_NAME="q${Q}_bs${BS}_lr${LR}_wd${WD}_eof${EOF}_runpod"

echo "=================================================="
echo "🚀 STARTING PIPELINE: $RUN_NAME"
echo "=================================================="

echo "⏳ [1/3] Starting Training..."
python train.py \
  --queries $Q \
  --batch_size $BS \
  --lr $LR \
  --wd $WD \
  --epochs 50 \
  --gpu $GPU \
  --workers $WORKER \
  --eof $EOF \
  --run_name $RUN_NAME && \

echo "⏳ [2/3] Starting Inference..."
python submission.py \
  --queries $Q \
  ./checkpoints/${RUN_NAME}_best.pth && \

echo "🧹 [EXTRA] Deleting checkpoint files (.pth)..."
rm -rf ./checkpoints/*.pth && \

echo "⏳ [3/3] Saving to GitHub safely..."

git add -A
git commit -m "Auto-save: Done training $RUN_NAME"
git pull origin master --rebase
git push origin master

echo "🎉 PIPELINE DONE!"


sleep 60 

if [ -z "$RUNPOD_POD_ID" ]; then
    echo "ℹ️  RUNPOD_POD_ID is not detected. Skipping Pod deletion."
else
    echo "💥 Time's up! Deleting Pod $RUNPOD_POD_ID..."
    runpodctl remove pod $RUNPOD_POD_ID
fi