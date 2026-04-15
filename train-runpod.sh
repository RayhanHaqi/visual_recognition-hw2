#!/bin/bash

# --- BACA ARGUMEN DARI TERMINAL ---
# Format: $1 (Argumen ke-1), $2 (Argumen ke-2), dst.
# Tanda :- artinya "Nilai Default" jika kamu lupa memasukkan argumen.

Q=${1:-1000}       # Argumen 1: Queries (Default: 1000)
BS=${2:-24}        # Argumen 2: Batch Size (Default: 24)
LR=${3:-5e-5}      # Argumen 3: Learning Rate (Default: 5e-5)
WD=${4:-5e-4}      # Argumen 4: Weight Decay (Default: 5e-4)
GPU=${5:-0}       # Argumen 5: GPU ID (Default: 0)
WORKER=${6:-6}    # Argumen 6: Number of Workers (Default: 6)

RUN_NAME="q${Q}_bs${BS}_lr${LR}_wd${WD}"

echo "=================================================="
echo "🚀 MEMULAI PIPELINE: $RUN_NAME"
echo "=================================================="

# 1. Training
echo "⏳ [1/3] Memulai Training..."
python train.py \
  --queries $Q \
  --batch_size $BS \
  --lr $LR \
  --wd $WD \
  --epochs 30 \
  --gpu $GPU \
  --workers $WORKER \
  --run_name $RUN_NAME && \

# 2. Inference / Submission
echo "⏳ [2/3] Memulai Inference..."
python submission.py \
  --queries $Q \
  ./checkpoints/${RUN_NAME}_best.pth && \

# --- FASE BARU: PEMBERSIHAN ---
echo "🧹 [EXTRA] Menghapus file checkpoint raksasa (.pth)..."
rm -rf ./checkpoints/*.pth && \

# 3. Git Push yang Aman (Parallel-Safe)
echo "⏳ [3/3] Menyimpan ke GitHub secara aman..."

git add -A
git commit -m "Auto-save: Done training $RUN_NAME"
git pull origin master --rebase
git push origin master

echo "🎉 PIPELINE SELESAI!"

# --- 4. AUTO-DESTRUCT SEQUENCE (Menghemat Saldo) ---
echo "⚠️ PERINGATAN: Pod ini akan dihancurkan secara permanen dalam 60 detik!"
echo "🛑 Tekan [Ctrl+C] SEKARANG jika kamu ingin membatalkan penghancuran."

# Menunggu 1 menit
sleep 60 

# Mengecek apakah variabel RUNPOD_POD_ID ada (mencegah error jika dijalankan di lokal)
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "ℹ️  RUNPOD_POD_ID tidak terdeteksi. Melewati proses penghapusan Pod (Mungkin kamu sedang di PC Lokal)."
else
    echo "💥 Waktu habis! Menghapus Pod $RUNPOD_POD_ID..."
    runpodctl remove pod $RUNPOD_POD_ID
fi