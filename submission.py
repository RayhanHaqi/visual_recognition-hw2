import os
import json
import zipfile
import argparse
import torch
import gc
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
# 2. FUNGSI ROBUST MODEL LOADER
# ==========================================
def load_smart_model(ckpt_path, device):
    """
    Otomatis mendeteksi arsitektur, queries, dan membersihkan mismatch state_dict.
    """
    name = os.path.basename(ckpt_path).lower()
    
    # Deteksi Queries berdasarkan nama file (Heuristic)
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
    
    # Inisialisasi Model Kosong
    if is_v2:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrV2ForObjectDetection(config=cfg)
    else:
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrForObjectDetection(config=cfg)
        
    print(f"✅ Inisialisasi Arsitektur: {'V2' if is_v2 else 'V1'} ({approach})")

    # --- PROSES LOADING WEIGHTS ---
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Jika file adalah Full Checkpoint (hasil script Ultimate), ambil dict modelnya saja
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Pembersihan Prefix Sesuai Struktur Hugging Face
    new_state_dict = {}
    for k, v in state_dict.items():
        # Kasus 1: Menghapus prefix 'model.' jika dobel (model.model.xxx)
        if k.startswith('model.model.'):
            name_key = k[6:] 
        # Kasus 2: Menambah prefix 'model.' jika file pth tidak punya tapi model butuh
        elif not k.startswith('model.'):
            name_key = 'model.' + k
        else:
            name_key = k
        new_state_dict[name_key] = v

    # Load dengan strict=False untuk menghindari crash jika ada layer non-esensial yang beda
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(unexpected) > 0:
        print(f"⚠️  Catatan: Ada {len(unexpected)} layer tak terduga (diabaikan).")
    if len(missing) > 0:
        print(f"⚠️  Peringatan: Ada {len(missing)} layer yang tidak terisi!")

    return model.to(device).eval(), name.split('.')[0]

# ==========================================
# 3. MAIN LOOP INFERENCE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate CodaBench Submission")
    parser.add_argument("ckpt", type=str, help="Path ke file checkpoint (.pth)")
    parser.add_argument("--gpu", type=int, default=1, help="ID GPU yang digunakan")
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Model dengan cerdas
    model, base_name = load_smart_model(args.ckpt, DEVICE)

    # Preprocessing (Sesuai dengan standar RT-DETR)
    trans = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    predictions = []
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n🔍 Melakukan deteksi pada {len(test_files)} gambar...")
    
    with torch.no_grad():
        for img_name in tqdm(test_files):
            img_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            
            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size
            
            # Persiapan Tensor
            input_tensor = trans(img_pil).unsqueeze(0).to(DEVICE)
            
            # Forward Pass
            with torch.amp.autocast('cuda'):
                outputs = model(input_tensor)
            
            # Post-processing
            logits = outputs.logits[0]
            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # Threshold Confidence Rendah (Sesuai standar evaluasi COCO)
            keep = scores > 0.05
            
            boxes = outputs.pred_boxes[0][keep]
            confidences = scores[keep]
            class_ids = labels[keep]

            for box, score, label in zip(boxes, confidences, class_ids):
                cx, cy, nw, nh = box.tolist()
                
                # Konversi dari Center-Normalized [cx, cy, w, h] 
                # ke COCO Format [x_min, y_min, w, h] absolut
                w = nw * orig_w
                h = nh * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()) + 1, # Label 1-10
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                })

    # ==========================================
    # 4. PENYIMPANAN & ZIPPING
    # ==========================================
    json_name = "pred.json"
    json_path = os.path.join(SUBMISSION_DIR, json_name)
    zip_path = os.path.join(SUBMISSION_DIR, f"submission_{base_name}.zip")

    print(f"\n💾 Menyimpan {len(predictions)} deteksi ke JSON...")
    with open(json_path, 'w') as f:
        json.dump(predictions, f)

    print(f"🗜️ Membuat file ZIP untuk CodaBench...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, json_name)
        
    os.remove(json_path) # Bersihkan file json sementara
    
    print(f"🎉 SELESAI! File submission Anda: {zip_path}")
    
    # Cleanup memori
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()