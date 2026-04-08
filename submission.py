import os
import json
import zipfile
import torch
import argparse
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrV2ForObjectDetection, RTDetrV2Config
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. FUNGSI SMART LOADER
# ==========================================
def load_smart_model(ckpt_path):
    name = os.path.basename(ckpt_path).lower()
    
    # Deteksi Queries
    q = 1500 if "tcocr" in name else 1000 if "dptext" in name else 300
    
    # Deteksi Arsitektur Base
    cfg_base = "PekingU/rtdetr_r50vd" if "v1" in name else "PekingU/rtdetr_v2_r50vd"
    
    if "v1" in name:
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrForObjectDetection.from_pretrained(cfg_base, config=cfg, ignore_mismatched_sizes=True)
    else:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrV2ForObjectDetection.from_pretrained(cfg_base, config=cfg, ignore_mismatched_sizes=True)
    
    print(f"✅ Memuat: {name} | Arsitektur: {'V1' if 'v1' in name else 'V2'} | Queries: {q}")
    
    # ==========================================
    # ⚠️ PERBAIKAN: MENCUKUR PREFIX "model."
    # ==========================================
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    cleaned_state_dict = {}
    
    for key, value in state_dict.items():
        # Jika kunci berawalan "model.", potong 6 karakter pertamanya
        if key.startswith('model.'):
            cleaned_key = key[6:] 
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
            
    # Load state dict yang sudah bersih
    model.load_state_dict(cleaned_state_dict)
    # ==========================================
    
    return model.to(DEVICE).eval()

# ==========================================
# 3. FUNGSI UTAMA (INFERENCE)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Default Standard Submission")
    parser.add_argument("ckpt", type=str, help="Path ke file checkpoint (.pth)")
    args = parser.parse_args()

    # Muat model secara otomatis
    model = load_smart_model(args.ckpt)
    MODEL_NAME = os.path.splitext(os.path.basename(args.ckpt))[0]
    
    predictions = []
    
    trans = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for img_name in tqdm(os.listdir(TEST_IMG_DIR)):
        img_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        with torch.no_grad():
            tensor = trans(pil_img).unsqueeze(0).to(DEVICE)
            out = model(tensor)
            
            probs = out.logits.sigmoid()[0]
            scores, labels = probs.max(dim=-1)
            
            # Threshold Default
            mask = scores > 0.05

            for box, score, label in zip(out.pred_boxes[0][mask], scores[mask], labels[mask]):
                cx, cy, nw, nh = box.tolist()
                w = nw * orig_w
                h = nh * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()) + 1,  # Dikembalikan ke format 1-10
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                })

    # ==========================================
    # 4. PENYIMPANAN & ZIPPING
    # ==========================================
    json_path = os.path.join(SUBMISSION_DIR, "pred.json")
    zip_path = os.path.join(SUBMISSION_DIR, f"default_{MODEL_NAME}.zip")

    print(f"\n💾 Menyimpan JSON sementara...")
    with open(json_path, 'w') as f:
        json.dump(predictions, f)

    print(f"🗜️ Melakukan zipping file menjadi: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, "pred.json")
        
    print(f"🎉 Selesai! File submission default siap digunakan.")

if __name__ == "__main__": 
    main()