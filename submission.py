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
# 2. FUNGSI ROBUST MODEL LOADER (FIXED)
# ==========================================
def load_smart_model(ckpt_path, device):
    name = os.path.basename(ckpt_path).lower()
    
    # Deteksi Konfigurasi
    if "tcocr" in name: q = 1500
    elif "dptext" in name: q = 500 # Sesuaikan dengan script train-dptext terakhir kita
    else: q = 300
        
    is_v2 = "v2" in name
    cfg_base = "PekingU/rtdetr_v2_r50vd" if is_v2 else "PekingU/rtdetr_r50vd"
    
    if is_v2:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrV2ForObjectDetection(config=cfg)
    else:
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=10)
        cfg.num_queries = q
        model = RTDetrForObjectDetection(config=cfg)
        
    print(f"✅ Arsitektur: {'V2' if is_v2 else 'V1'} | Queries: {q}")

    # --- LOGIKA REPAIR STATE DICT ---
    checkpoint = torch.load(ckpt_path, map_location='cpu')
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
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}")
    model, base_name = load_smart_model(args.ckpt, DEVICE)

    trans = transforms.Compose([
        transforms.Resize((640, 640)), 
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
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_tensor)
            
            probs = outputs.logits[0].sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # Gunakan threshold rendah untuk memaksimalkan mAP di CodaBench
            keep = scores > 0.01 
            
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

    zip_path = os.path.join(SUBMISSION_DIR, f"submission_{base_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, json_name)
    
    os.remove(json_path)
    print(f"🎉 Submission siap: {zip_path}")

if __name__ == "__main__": main()