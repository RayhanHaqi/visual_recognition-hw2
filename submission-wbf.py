import os, json, zipfile, torch, argparse
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrV2ForObjectDetection, RTDetrV2Config
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

TEST_IMG_DIR = "./datasets/test"
DEVICE = torch.device("cuda:1")

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

def main():
    parser = argparse.ArgumentParser(description="WBF Only Submission")
    # ⚠️ Menggunakan Positional Argument (tanpa --)
    parser.add_argument("ckpt", type=str, help="Path ke file checkpoint (.pth)")
    args = parser.parse_args()

    model = load_smart_model(args.ckpt)
    model_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    predictions = []
    trans = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for img_name in tqdm(os.listdir(TEST_IMG_DIR)):
        img_pil = Image.open(os.path.join(TEST_IMG_DIR, img_name)).convert("RGB")
        w_orig, h_orig = img_pil.size
        
        with torch.no_grad():
            out = model(trans(img_pil).unsqueeze(0).to(DEVICE))
            probs = out.logits.sigmoid()[0]; s, l = probs.max(dim=-1); mask = s > 0.01 
            b_norm = [[max(0,b[0]-b[2]/2), max(0,b[1]-b[3]/2), min(1,b[0]+b[2]/2), min(1,b[1]+b[3]/2)] for b in out.pred_boxes[0][mask].tolist()]
            
            f_b, f_s, f_l = weighted_boxes_fusion([b_norm], [s[mask].tolist()], [l[mask].tolist()], iou_thr=0.45, skip_box_thr=0.05)

            for b, score, label in zip(f_b, f_s, f_l):
                predictions.append({
                    "image_id": int(os.path.splitext(img_name)[0]), "category_id": int(label) + 1,
                    "bbox": [round(b[0]*w_orig, 2), round(b[1]*h_orig, 2), round((b[2]-b[0])*w_orig, 2), round((b[3]-b[1])*h_orig, 2)],
                    "score": round(float(score), 4)
                })

    with open("pred.json", 'w') as f: json.dump(predictions, f)
    zip_name = f"wbf_{model_name}.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z: z.write("pred.json")
    print(f"🎉 Selesai! File tersimpan sebagai: {zip_name}")

if __name__ == "__main__": main()