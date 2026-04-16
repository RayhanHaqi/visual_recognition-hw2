import os
import json
import zipfile
import argparse
import torch
import gc
from PIL import Image
from torchvision import transforms
from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI DIREKTORI
# ==========================================
TEST_IMG_DIR = "./datasets/test"           
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. FUNGSI ROBUST MODEL LOADER (V2 DEFAULT)
# ==========================================
def load_smart_model(ckpt_path, device, queries):
    name = os.path.basename(ckpt_path).lower()
    
    # Langsung gunakan konfigurasi RT-DETR v2
    cfg_base = "PekingU/rtdetr_v2_r50vd"
    cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
    cfg.num_queries = queries
    model = RTDetrV2ForObjectDetection(config=cfg)
        
    print(f"✅ Arsitektur: RT-DETR v2 | Queries: {queries}")

    # --- LOGIKA REPAIR STATE DICT ---
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Kasus: File punya prefix 'model.model.' (akibat nested saving)
        if k.startswith('model.model.'):
            name_key = k[6:] 
        # Kasus: File tidak punya prefix 'model.' padahal Transformers butuh
        elif not k.startswith('model.'):
            name_key = 'model.' + k
        else:
            name_key = k
        new_state_dict[name_key] = v

    # Load dengan strict=False agar tidak crash karena layer statistik (running_mean dsb)
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(device).eval(), name.split('.')[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="Path ke file .pth hasil training")
    parser.add_argument("--queries", type=int, required=True, help="Wajib diisi! Jumlah kueri yang digunakan saat training (misal: 300, 1000, 1500)")
    parser.add_argument("--gpu", type=str, default="0", help="ID GPU yang digunakan (misal: 0 atau 1)")
    args = parser.parse_args()

    # Logika pemilihan device yang lebih robust
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.gpu}")
    else:
        DEVICE = torch.device("cpu")
        
    print(f"🚀 Inference menggunakan device: {DEVICE}")
    # Pass argument queries ke loader
    model, base_name = load_smart_model(args.ckpt, DEVICE, args.queries)

    trans = transforms.Compose([
        transforms.Resize((320, 640)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    predictions = []
    test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with torch.no_grad():
        for img_name in tqdm(test_files, desc="Inference"):
            img_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size
            
            input_tensor = trans(img_pil).unsqueeze(0).to(DEVICE)
            
            # with torch.amp.autocast('cuda'):
            outputs = model(input_tensor)
            
            probs = outputs.logits[0].sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # Gunakan threshold rendah untuk memaksimalkan Recall di CodaBench
            keep = scores > 0.15 
            
            boxes = outputs.pred_boxes[0][keep]
            confidences = scores[keep]
            class_ids = labels[keep]

            for box, score, label in zip(boxes, confidences, class_ids):
                cx, cy, nw, nh = box.tolist()
                w, h = nw * orig_w, nh * orig_h
                x_min = (cx * orig_w) - (w / 2)
                y_min = (cy * orig_h) - (h / 2)

                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()) + 1,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                })

    json_name = "pred.json"
    json_path = os.path.join(SUBMISSION_DIR, json_name)
    with open(json_path, 'w') as f: json.dump(predictions, f)

    # Zip file sesuai format yang diminta CodaBench
    zip_path = os.path.join(SUBMISSION_DIR, f"submission_{base_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, json_name)
    
    os.remove(json_path)
    print(f"🎉 Submission siap: {zip_path}")

if __name__ == "__main__": main()