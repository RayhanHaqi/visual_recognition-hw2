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
CHECKPOINT_PATH = "./checkpoints/rtdetr_r50_best.pth"
OUTPUT_JSON = "pred.json"                  

IMG_SIZE = 640
NUM_CLASSES = 10
CONF_THRESHOLD = 0.05                      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"🚀 Memulai Inference langsung dari folder gambar menggunakan {DEVICE}...")

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
    
    # Ambil semua file gambar dari folder test
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📁 Ditemukan {len(image_files)} gambar untuk diprediksi.")

    # 3. Loop Inference
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Processing Test Images"):
            img_path = os.path.join(TEST_IMG_DIR, file_name)
            
            # --- MENDAPATKAN IMAGE ID DARI NAMA FILE ---
            # Sistem evaluasi COCO biasanya mengharuskan image_id berupa integer (angka).
            # Kita pecah "123.jpg" menjadi angka 123.
            try:
                image_id = int(os.path.splitext(file_name)[0])
            except ValueError:
                # Jika nama filenya bukan angka (misal "test_A.jpg"), kita pakai stringnya saja.
                image_id = file_name 

            # --- MENDAPATKAN RESOLUSI ASLI ---
            # Buka gambar dan langsung catat ukuran aslinya untuk un-normalize bounding box nanti
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size 
            
            # Siapkan tensor untuk model
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # Prediksi dengan model
            with torch.amp.autocast('cuda'):
                outputs = model(input_tensor)
            
            logits = outputs.logits[0]  
            boxes = outputs.pred_boxes[0] 

            # Hitung Probabilitas
            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)

            # Filter prediksi yang skornya melebihi threshold (0.05)
            keep = scores > CONF_THRESHOLD
            
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]

            # 4. Konversi BBox & Simpan ke List
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                cx, cy, norm_w, norm_h = box.tolist()
                
                # Un-normalize & Ubah ke format COCO [x_min, y_min, w, h]
                # Kita gunakan orig_w dan orig_h yang didapat dari PIL tadi
                w = norm_w * orig_w
                h = norm_h * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                # Format prediksi untuk 1 objek
                pred_dict = {
                    "image_id": image_id,
                    "category_id": int(label.item()) + 1,  # KEMBALIKAN KE 1-10
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                }
                predictions.append(pred_dict)

    # 5. Simpan ke pred.json
    print(f"\n💾 Menyimpan total {len(predictions)} objek terdeteksi ke {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f)

    print("✅ Selesai! File pred.json sudah siap disubmit.")

if __name__ == "__main__":
    main()