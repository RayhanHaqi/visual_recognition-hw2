import os
import json
import zipfile
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection
from tqdm import tqdm
from torchvision.ops import nms

# ==========================================
# 1. KONFIGURASI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           
# PASTIKAN INI ADALAH BOBOT BASELINE PERTAMA KAMU YANG DAPAT SKOR 0.38
CHECKPOINT_PATH = "./checkpoints/rtdetr_r50_best.pth" 

SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_CLASSES = 10
CONF_THRESHOLD = 0.05 
# MULTI-SCALE TTA (Test Time Augmentation)
SCALES = [480, 640, 800] 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0] + "_TTA"

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

print(f"🔄 Memuat bobot dari: {CHECKPOINT_PATH}")
model = RTDETRv2_R50_System(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==========================================
# 3. INFERENCE DENGAN TTA & NMS
# ==========================================
predictions = []
image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"🚀 Memulai prediksi TTA (Multi-scale: {SCALES}) pada {len(image_files)} gambar...")
progress_bar = tqdm(image_files)

with torch.no_grad():
    for filename in progress_bar:
        img_path = os.path.join(TEST_IMG_DIR, filename)
        image_id = int(os.path.splitext(filename)[0])
        
        orig_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_image.size
        
        all_boxes = []
        all_scores = []
        all_labels = []

        # 1. TTA LOOP (Menebak di berbagai ukuran)
        for size in SCALES:
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                outputs = model(img_tensor)
            
            logits = outputs.logits
            boxes = outputs.pred_boxes
            probs = logits.sigmoid()
            
            scores, labels = probs[0].max(dim=-1)
            keep = scores > CONF_THRESHOLD
            
            f_boxes = boxes[0][keep]
            f_scores = scores[keep]
            f_labels = labels[keep]
            
            # Kembalikan koordinat TTA ke resolusi asli gambar
            for box, score, label in zip(f_boxes, f_scores, f_labels):
                cx, cy, norm_w, norm_h = box.tolist()
                
                x_min = (cx * orig_w) - ((norm_w * orig_w) / 2)
                y_min = (cy * orig_h) - ((norm_h * orig_h) / 2)
                x_max = x_min + (norm_w * orig_w)
                y_max = y_min + (norm_h * orig_h)
                
                all_boxes.append([x_min, y_min, x_max, y_max])
                all_scores.append(score.item())
                all_labels.append(label.item())

        # 2. GABUNGKAN PREDIKSI DENGAN NMS
        if len(all_boxes) > 0:
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32).to(DEVICE)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32).to(DEVICE)
            labels_tensor = torch.tensor(all_labels, dtype=torch.int64).to(DEVICE)

            # Lakukan NMS per kelas digit untuk membersihkan kotak yang bertumpuk
            unique_labels = torch.unique(labels_tensor)
            final_boxes, final_scores, final_labels = [], [], []
            
            for ul in unique_labels:
                idx = (labels_tensor == ul).nonzero(as_tuple=True)[0]
                cls_boxes = boxes_tensor[idx]
                cls_scores = scores_tensor[idx]
                
                # IoU threshold 0.5 untuk menggabungkan kotak TTA yang saling tindih
                keep_idx = nms(cls_boxes, cls_scores, iou_threshold=0.5) 
                
                for k in keep_idx:
                    bx = cls_boxes[k].cpu().tolist()
                    final_boxes.append(bx)
                    final_scores.append(cls_scores[k].cpu().item())
                    final_labels.append(ul.cpu().item())

            # 3. FORMAT KE JSON CODABENCH
            for bx, sc, lb in zip(final_boxes, final_scores, final_labels):
                x_min, y_min, x_max, y_max = bx
                w = x_max - x_min
                h = y_max - y_min
                
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(lb) + 1, 
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(sc, 4)
                })

# ==========================================
# 4. PENYIMPANAN & ZIPPING
# ==========================================
json_path = os.path.join(SUBMISSION_DIR, "pred.json")
zip_path = os.path.join(SUBMISSION_DIR, f"{MODEL_NAME}.zip")

print(f"\n💾 Menyimpan JSON...")
with open(json_path, 'w') as f:
    json.dump(predictions, f)

print(f"🗜️ Melakukan zipping file menjadi: {zip_path}...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(json_path, arcname="pred.json")

print(f"✅ Selesai! Silakan submit file {zip_path} ke CodaBench.")