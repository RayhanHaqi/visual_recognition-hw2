import os
import json
import zipfile
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrV2ForObjectDetection, RTDetrV2Config
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI DIREKTORI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. FUNGSI SMART LOADER
# ==========================================
def load_smart_model(ckpt_path, device):
    """
    Otomatis mendeteksi V1/V2 dan jumlah Queries dari nama file checkpoint.
    """
    name = os.path.basename(ckpt_path).lower()
    
    # Deteksi Queries: Jika ada kata "tcocr" = 1500, "dptext" = 1000, sisanya 300
    if "tcocr" in name:
        q = 1500
        approach = "TC-OCR"
    elif "dptext" in name:
        q = 1000
        approach = "DPText"
    else:
        q = 300
        approach = "Baseline"
        
    is_v2 = "v2" in name
    cfg_base = "PekingU/rtdetr_v2_r50vd" if is_v2 else "PekingU/rtdetr_r50vd"
    
    # Bangun Model sesuai konfigurasi yang dideteksi
    if is_v2:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrV2ForObjectDetection(config=cfg)
    else:
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrForObjectDetection(config=cfg)
        
    print(f"✅ Deteksi Otomatis Checkpoint: {name}")
    print(f"   ├─ Arsitektur : {'V2' if is_v2 else 'V1'}")
    print(f"   ├─ Pendekatan : {approach}")
    print(f"   └─ Queries    : {q}")

    # ==========================================
    # ROBUST WEIGHT LOADER (Anti Mismatch Error)
    # ==========================================
    state_dict = torch.load(ckpt_path, map_location=device)
    cleaned_dict = {}
    
    # Membersihkan prefix "model." jika sebelumnya model dibungkus (wrapped)
    for k, v in state_dict.items():
        if k.startswith('model.'):
            cleaned_dict[k[6:]] = v
        else:
            cleaned_dict[k] = v
            
    model.load_state_dict(cleaned_dict, strict=True)
    return model.to(device).eval(), name.split('.')[0]

# ==========================================
# 3. MAIN LOOP INFERENCE
# ==========================================
def main():
    # Setup Positional Argument
    parser = argparse.ArgumentParser(description="Generate CodaBench Submission")
    parser.add_argument("ckpt", type=str, help="Path ke file checkpoint (.pth)")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model, base_name = load_smart_model(args.ckpt, DEVICE)

    # Augmentasi Test (Hanya Resize + Normalisasi)
    trans = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    predictions = []
    
    print("\n🔍 Memulai proses deteksi pada gambar test...")
    for img_name in tqdm(os.listdir(TEST_IMG_DIR)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_id = int(os.path.splitext(img_name)[0])
        img_pil = Image.open(os.path.join(TEST_IMG_DIR, img_name)).convert("RGB")
        orig_w, orig_h = img_pil.size
        
        tensor = trans(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model(tensor)
            
            probs = out.logits.sigmoid()[0]
            scores, labels = probs.max(dim=-1)
            
            # Filter Confidence > 0.05
            mask = scores > 0.05

            for box, score, label in zip(out.pred_boxes[0][mask], scores[mask], labels[mask]):
                cx, cy, nw, nh = box.tolist()
                
                # Kembalikan ke koordinat absolut gambar asli
                w = nw * orig_w
                h = nh * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()) + 1,  # Format kelas 1-10 (Sesuai COCO)
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                })

    # ==========================================
    # 4. PENYIMPANAN & ZIPPING
    # ==========================================
    json_path = os.path.join(SUBMISSION_DIR, "pred.json")
    zip_path = os.path.join(SUBMISSION_DIR, f"submission_{base_name}.zip")

    print(f"\n💾 Menyimpan format JSON...")
    with open(json_path, 'w') as f:
        json.dump(predictions, f)

    print(f"🗜️ Melakukan kompresi ke file ZIP...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "pred.json")
        
    os.remove(json_path) # Bersihkan file json sementara
    
    print(f"🎉 SUKSES! File siap diupload ke CodaBench: {zip_path}")

if __name__ == "__main__":
    main()