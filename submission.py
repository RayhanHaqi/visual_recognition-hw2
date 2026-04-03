import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           # Path ke folder gambar test
TEST_JSON_PATH = "./datasets/test.json"    # Path ke test.json (untuk ambil image_id)
CHECKPOINT_PATH = "./checkpoints/rtdetr_r50_best.pth"
OUTPUT_JSON = "pred.json"                  # Nama file wajib dari TA

IMG_SIZE = 640
NUM_CLASSES = 10
CONF_THRESHOLD = 0.05                      # Threshold rendah agar mAP CodaBench maksimal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DEFINISI MODEL (Sama dengan saat training)
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
    print(f"🚀 Memulai Generasi Submission menggunakan {DEVICE}...")

    # 1. Load Data Test JSON (Untuk mendapatkan Info Resolusi Asli & Image ID)
    with open(TEST_JSON_PATH, 'r') as f:
        test_coco = json.load(f)
    
    # Buat dictionary agar mudah mencari info gambar berdasarkan file_name
    image_info_dict = {img['file_name']: img for img in test_coco['images']}

    # 2. Inisialisasi Model & Load Best Weights
    model = RTDETRv2_R50_System(num_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"📥 Loading weights dari: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 3. Transformasi Input
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 4. Loop Inference
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Processing Test Images"):
            img_path = os.path.join(TEST_IMG_DIR, file_name)
            
            # Ambil metadata gambar asli (PENTING untuk scaling bbox)
            if file_name not in image_info_dict:
                continue
            img_info = image_info_dict[file_name]
            image_id = img_info['id']
            orig_w = img_info['width']
            orig_h = img_info['height']

            # Buka dan transform gambar
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Tambah batch dimensi [1, C, H, W]

            # Prediksi dengan model
            with torch.amp.autocast('cuda'):
                outputs = model(input_tensor)
            
            logits = outputs.logits[0]  # [300, num_classes]
            boxes = outputs.pred_boxes[0] # [300, 4] format: [cx, cy, w, h] normalized 0-1

            # Hitung Probabilitas (Sigmoid untuk focal loss/DETR)
            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)

            # Filter prediksi yang skornya melebihi threshold
            keep = scores > CONF_THRESHOLD
            
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]

            # 5. Konversi BBox & Simpan ke List
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                cx, cy, norm_w, norm_h = box.tolist()
                
                # Un-normalize & Ubah ke format COCO [x_min, y_min, w, h]
                w = norm_w * orig_w
                h = norm_h * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                # Format akhir untuk 1 objek
                pred_dict = {
                    "image_id": image_id,
                    "category_id": int(label.item()) + 1,  # KEMBALIKAN KE 1-10
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                }
                predictions.append(pred_dict)

    # 6. Simpan ke pred.json
    print(f"\n💾 Menyimpan {len(predictions)} prediksi ke {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f)

    print("✅ Selesai! Kamu bisa men-zip file pred.json ini dan mengunggahnya ke CodaBench.")

if __name__ == "__main__":
    main()