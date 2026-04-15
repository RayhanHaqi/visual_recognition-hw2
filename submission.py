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
# 2. FUNGSI ROBUST MODEL LOADER
# ==========================================
def load_smart_model(ckpt_path, device, queries):
    name = os.path.basename(ckpt_path).lower()
    
    cfg_base = "PekingU/rtdetr_v2_r50vd"
    cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
    cfg.num_queries = queries
    model = RTDetrV2ForObjectDetection(config=cfg)
        
    print(f"✅ Arsitektur: RT-DETR v2 | Queries: {queries}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'): new_state_dict[k[6:]] = v
        else: new_state_dict[k] = v
            
    model.model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, os.path.splitext(os.path.basename(ckpt_path))[0]

# ==========================================
# 3. MAIN INFERENCE LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="Path ke file .pth hasil training")
    parser.add_argument("--queries", type=int, required=True, help="Jumlah kueri (misal: 300, 1000, 1500)")
    parser.add_argument("--gpu", type=str, default="0", help="ID GPU (misal: 0 atau 1)")
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference menggunakan device: {DEVICE}")

    model, base_name = load_smart_model(args.ckpt, DEVICE, args.queries)

    val_transforms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    predictions = []
    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]

    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Inference"):
            img_path = os.path.join(TEST_IMG_DIR, img_name)
            img_id = int(os.path.splitext(img_name)[0])
            
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            
            input_tensor = val_transforms(image).unsqueeze(0).to(DEVICE)
            
            outputs = model(input_tensor)
            
            probs = outputs.logits[0].sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # --- PROTEKSI 1: THRESHOLD KETAT ---
            # Buang tebakan ngawur, hanya ambil yang yakin > 15%
            keep = scores > 0.15 
            
            boxes = outputs.pred_boxes[0][keep]
            confidences = scores[keep]
            class_ids = labels[keep]

            # --- PROTEKSI 2: TOP-K FILTERING ---
            # COCO mAP akan hancur jika mengirim lebih dari 300 kotak per gambar
            if len(confidences) > 300:
                topk_indices = confidences.topk(300)[1]
                boxes = boxes[topk_indices]
                confidences = confidences[topk_indices]
                class_ids = class_ids[topk_indices]

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

    zip_path = os.path.join(SUBMISSION_DIR, f"submission_{base_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, json_name)
        
    os.remove(json_path)
    print(f"🎉 Submission siap: {zip_path}")

if __name__ == "__main__": main()