import os, json, zipfile, torch
from PIL import Image
from torchvision import transforms
from transformers import RTDetrForObjectDetection, RTDetrConfig
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

# === KONFIGURASI ===
CKPT_TCOCR = "./checkpoints/tcocr_v1_best.pth"   # Sesuaikan nama file Anda
CKPT_DPTEXT = "./checkpoints/dptext_v1_best.pth" # Sesuaikan nama file Anda
DEVICE = torch.device("cuda:1")

def load_clean_model(ckpt_path, queries):
    cfg = RTDetrConfig.from_pretrained("PekingU/rtdetr_r50vd", num_labels=10)
    cfg.num_queries = queries
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", config=cfg, ignore_mismatched_sizes=True)
    
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    cleaned = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model.to(DEVICE).eval()

def main():
    print("📦 Memuat TC-OCR (1500 Queries)...")
    m_tcocr = load_clean_model(CKPT_TCOCR, 1500)
    print("📦 Memuat DPText (1000 Queries)...")
    m_dptext = load_clean_model(CKPT_DPTEXT, 1000)

    predictions = []
    trans = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for img_name in tqdm(os.listdir("./datasets/test")):
        img_id = int(os.path.splitext(img_name)[0])
        img_pil = Image.open(os.path.join("./datasets/test", img_name)).convert("RGB")
        w_orig, h_orig = img_pil.size
        boxes_list, scores_list, labels_list = [], [], []

        with torch.no_grad():
            tensor = trans(img_pil).unsqueeze(0).to(DEVICE)
            
            # Prediksi TC-OCR
            out1 = m_tcocr(tensor)
            s1, l1 = out1.logits.sigmoid()[0].max(dim=-1); mask1 = s1 > 0.05
            b1 = [[max(0,b[0]-b[2]/2), max(0,b[1]-b[3]/2), min(1,b[0]+b[2]/2), min(1,b[1]+b[3]/2)] for b in out1.pred_boxes[0][mask1].tolist()]
            boxes_list.append(b1); scores_list.append(s1[mask1].tolist()); labels_list.append(l1[mask1].tolist())

            # Prediksi DPText
            out2 = m_dptext(tensor)
            s2, l2 = out2.logits.sigmoid()[0].max(dim=-1); mask2 = s2 > 0.05
            b2 = [[max(0,b[0]-b[2]/2), max(0,b[1]-b[3]/2), min(1,b[0]+b[2]/2), min(1,b[1]+b[3]/2)] for b in out2.pred_boxes[0][mask2].tolist()]
            boxes_list.append(b2); scores_list.append(s2[mask2].tolist()); labels_list.append(l2[mask2].tolist())

        # FUSION DUA MODEL (Bobot Sama)
        f_b, f_s, f_l = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=[1, 1], iou_thr=0.45, skip_box_thr=0.05)

        for b, score, label in zip(f_b, f_s, f_l):
            predictions.append({
                "image_id": img_id, "category_id": int(label) + 1,
                "bbox": [round(b[0]*w_orig, 2), round(b[1]*h_orig, 2), round((b[2]-b[0])*w_orig, 2), round((b[3]-b[1])*h_orig, 2)],
                "score": round(float(score), 4)
            })

    with open("pred.json", 'w') as f: json.dump(predictions, f)
    with zipfile.ZipFile("submission/ensemble_tcocr_dptext.zip", 'w', zipfile.ZIP_DEFLATED) as z: z.write("pred.json")
    print(f"🎉 Selesai! File tersimpan sebagai: submission/ensemble_tcocr_dptext.zip")

if __name__ == "__main__": main()