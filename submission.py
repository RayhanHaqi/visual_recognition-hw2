import os
import json
import zipfile
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           
CHECKPOINT_PATH = "./checkpoints/paper_v1_best.pth"

# Direktori Output Baru
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

IMG_SIZE = 640
NUM_CLASSES = 10
CONF_THRESHOLD = 0.05                      
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Ekstrak nama model dari CHECKPOINT_PATH (misal: "rtdetr_r50_best")
MODEL_NAME = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]

# ==========================================
# 2. DEFINISI MODEL
# ==========================================
class RTDETRv2_R50_System(nn.Module):
    def __init__(self, num_classes=10):
        super(RTDETRv2_R50_System, self).__init__()
        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, images):
        return self.model(pixel_values=images)

# ==========================================
# 3. FUNGSI UTAMA INFERENCE
# ==========================================
def main():
    print(f"🚀 Memulai Inference menggunakan {DEVICE}...")

    # 1. Inisialisasi Model & Load Best Weights
    model = RTDETRv2_R50_System(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"📥 Loading weights dari: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 2. Transformasi Input
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📁 Ditemukan {len(image_files)} gambar untuk diprediksi.")

    # 3. Loop Inference
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Processing Test Images"):
            img_path = os.path.join(TEST_IMG_DIR, file_name)
            
            # Ambil Image ID
            try:
                image_id = int(os.path.splitext(file_name)[0])
            except ValueError:
                image_id = file_name 

            # Ambil Resolusi Asli
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size 
            
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(input_tensor)
            
            logits = outputs.logits[0]  
            boxes = outputs.pred_boxes[0] 

            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)

            keep = scores > CONF_THRESHOLD
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]

            # 4. Konversi BBox & Simpan ke List
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                cx, cy, norm_w, norm_h = box.tolist()
                
                w = norm_w * orig_w
                h = norm_h * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                pred_dict = {
                    "image_id": image_id,
                    "category_id": int(label.item()) + 1,  # KEMBALIKAN KE 1-10
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                }
                predictions.append(pred_dict)

    # ==========================================
    # 4. PENYIMPANAN & ZIPPING (FITUR BARU)
    # ==========================================
    json_path = os.path.join(SUBMISSION_DIR, "pred.json")
    zip_path = os.path.join(SUBMISSION_DIR, f"{MODEL_NAME}.zip")

    # Simpan ke pred.json sementara di dalam folder submission
    print(f"\n💾 Menyimpan JSON sementara...")
    with open(json_path, 'w') as f:
        json.dump(predictions, f)

    # Bungkus pred.json ke dalam file .zip
    print(f"🗜️ Melakukan zipping file menjadi: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # arcname="pred.json" memastikan struktur folder tidak ikut masuk ke dalam zip
        zipf.write(json_path, arcname="pred.json")

    # (Opsional) Hapus file pred.json mentah jika kamu hanya butuh file zip-nya
    os.remove(json_path) 

    print("✅ Selesai! File zip kamu sudah siap di folder ./submission untuk diunggah ke CodaBench.")

if __name__ == "__main__":
    main()