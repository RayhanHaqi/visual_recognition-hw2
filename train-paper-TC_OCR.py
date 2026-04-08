import os, json, csv, torch, gc
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrV2ForObjectDetection, RTDetrV2Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import OneCycleLR

# ================= KONFIGURASI =================
DATA_ROOT_TRAIN, ANN_FILE_TRAIN = "./datasets/train", "./datasets/train.json"
DATA_ROOT_VAL, ANN_FILE_VAL = "./datasets/valid", "./datasets/valid.json"
CHECKPOINT_DIR, LOG_DIR = "./checkpoints", "./logs"
NUM_CLASSES, INITIAL_BATCH_SIZE, MAX_LR, EPOCHS, IMG_SIZE = 10, 32, 5e-5, 20, 640
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

# ================= DATA & AUGMENTASI =================
tcocr_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        # ⚠️ KEMBALI KE CARA LAMA YANG AMAN
        orig_w, orig_h = image.size 
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        # ⚠️ KEMBALI MENGEMBALIKAN 5 ELEMEN
        return image, target, id, orig_w, orig_h

def collate_fn(batch):
    # ⚠️ KEMBALI KE CARA LAMA YANG AMAN
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    image_ids = [item[2] for item in batch]
    orig_sizes = [(item[3], item[4]) for item in batch]
    return images, targets, image_ids, orig_sizes

def prepare_targets(targets, sizes, dev):
    res = []
    for i, t in enumerate(targets):
        b, l = [], []
        for o in t:
            x, y, w, h = o['bbox']
            if w > 0 and h > 0:
                b.append([(x+w/2)/sizes[i][0], (y+h/2)/sizes[i][1], w/sizes[i][0], h/sizes[i][1]])
                l.append(o['category_id'] - 1)
        if b: res.append({"boxes": torch.tensor(b, dtype=torch.float32).to(dev), "class_labels": torch.tensor(l, dtype=torch.long).to(dev)})
    return res

# ================= MODEL INJECTION (TC-OCR) =================
class TCOCR_DETR(nn.Module):
    def __init__(self, version="v1"):
        super().__init__()
        if version == "v1": config = RTDetrConfig.from_pretrained("PekingU/rtdetr_r50vd", num_labels=NUM_CLASSES)
        else: config = RTDetrV2Config.from_pretrained("PekingU/rtdetr_v2_r50vd", num_labels=NUM_CLASSES)
        
        config.num_queries = 1500
        config.matcher_bbox_cost = 6.0
        config.matcher_giou_cost = 3.0
        
        if version == "v1": self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", config=config, ignore_mismatched_sizes=True)
        else: self.model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd", config=config, ignore_mismatched_sizes=True)
    def forward(self, img, labels=None): return self.model(pixel_values=img, labels=labels)

# ================= TRAINING LOOP =================
def train_one_epoch(model, ema_model, loader, optimizer, scaler, scheduler, device):
    model.train(); total_loss = 0; pb = tqdm(loader, desc="Training")
    for img, tgt, _, sz in pb:
        img = img.to(device); tgt = prepare_targets(tgt, sz, device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'): loss = model(img, labels=tgt).loss
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer); scaler.update(); scheduler.step(); ema_model.update_parameters(model)
        total_loss += loss.item(); pb.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
    return total_loss / len(loader)

# ⚠️ FUNGSI EVALUATE YANG SUDAH DIPERBAIKI (BEBAS BUG PYCOCOTOOLS)
def evaluate(model, loader, device, val_json, v):
    model.eval(); preds = []; pb = tqdm(loader, desc="Val EMA")
    with torch.no_grad():
        for img, tgt, ids, sz in pb:
            img = img.to(device); out = model(img, labels=prepare_targets(tgt, sz, device))
            probs = out.logits.sigmoid()
            for i in range(len(img)):
                scores, labels = probs[i].max(dim=-1); keep = scores > 0.05
                for box, s, l in zip(out.pred_boxes[i][keep], scores[keep], labels[keep]):
                    cx, cy, nw, nh = box.tolist(); w, h = nw * sz[i][0], nh * sz[i][1]
                    preds.append({"image_id": ids[i], "category_id": int(l.item())+1, "bbox": [round(cx*sz[i][0]-w/2, 2), round(cy*sz[i][1]-h/2, 2), round(w, 2), round(h, 2)], "score": round(s.item(), 4)})
    if not preds: return 0, {"mAP_50_95":0, "mAP_50":0, "mAP_75":0, "mAP_S":0, "mAP_M":0}
    
    with open(f"tmp_dptext_{v}.json", 'w') as f: json.dump(preds, f)
    import io; from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(COCO(val_json), COCO(val_json).loadRes(f"tmp_dptext_{v}.json"), 'bbox')
        
        # ⚠️ PERBAIKAN 1: Wajib memasukkan angka 100 agar pycocotools tidak error
        max_queries = 1000  # Karena skrip ini DPText
        coco_eval.params.maxDets = [1, 10, 100, max_queries] 
        
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    os.remove(f"tmp_dptext_{v}.json")
    
    # ⚠️ PERBAIKAN 2: Ekstraksi manual spesifik untuk batas max_queries kita
    m_5095 = coco_eval._summarize(1, maxDets=max_queries)
    m_50   = coco_eval._summarize(1, iouThr=.5, maxDets=max_queries)
    m_75   = coco_eval._summarize(1, iouThr=.75, maxDets=max_queries)
    m_S    = coco_eval._summarize(1, areaRng='small', maxDets=max_queries)
    m_M    = coco_eval._summarize(1, areaRng='medium', maxDets=max_queries)

    # ⚠️ PERBAIKAN 3: Filter nilai -1.0 menjadi 0 agar logic _best.pth berjalan
    metrics = {
        "mAP_50_95": m_5095 if m_5095 > -1 else 0,
        "mAP_50": m_50 if m_50 > -1 else 0,
        "mAP_75": m_75 if m_75 > -1 else 0,
        "mAP_S": m_S if m_S > -1 else 0,
        "mAP_M": m_M if m_M > -1 else 0,
    }
    return 0, metrics

def main():
    for version in ["v1", "v2"]:
        print(f"\n{'='*50}\n🚀 TC-OCR LOOP | MODEL: {version.upper()}\n{'='*50}")
        bs = INITIAL_BATCH_SIZE; log_file = f"{LOG_DIR}/log_tcocr_{version}.csv"
        
        # ⚠️ HEADER CSV DIPERBARUI
        if not os.path.exists(log_file): 
            with open(log_file, 'w') as f:
                f.write("Epoch,TrainLoss,ValLoss,mAP_50_95,mAP_50,mAP_75,mAP_Small,mAP_Medium\n")
        
        model = TCOCR_DETR(version).to(DEVICE)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
        opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4); scaler = torch.amp.GradScaler('cuda')
        
        tr_ds = CustomCocoDetection(DATA_ROOT_TRAIN, ANN_FILE_TRAIN, transform=tcocr_transform)
        vl_ds = CustomCocoDetection(DATA_ROOT_VAL, ANN_FILE_VAL, transform=val_transform)
        tr_ld = DataLoader(tr_ds, bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
        vl_ld = DataLoader(vl_ds, bs, shuffle=False, collate_fn=collate_fn, num_workers=8)
        sch = OneCycleLR(opt, max_lr=MAX_LR, steps_per_epoch=len(tr_ld), epochs=EPOCHS)
        
        best_map = 0; ep = 0
        while ep < EPOCHS:
            try:
                print(f"\nEpoch {ep+1}/{EPOCHS} (Batch: {bs})")
                tl = train_one_epoch(model, ema, tr_ld, opt, scaler, sch, DEVICE)
                
                # ⚠️ MENERIMA DICTIONARY METRIKS
                _, metrics = evaluate(ema, vl_ld, DEVICE, ANN_FILE_VAL, version)
                
                mAP_utama = metrics['mAP_50_95']
                print(f"📈 T_Loss: {tl:.4f} | mAP(50-95): {mAP_utama:.4f} | mAP_50: {metrics['mAP_50']:.4f} | mAP_S: {metrics['mAP_S']:.4f}")
                
                # ⚠️ MENULIS SEMUA METRIKS KE CSV
                with open(log_file, 'a') as f: 
                    f.write(f"{ep+1},{tl:.4f},0,{mAP_utama:.4f},{metrics['mAP_50']:.4f},{metrics['mAP_75']:.4f},{metrics['mAP_S']:.4f},{metrics['mAP_M']:.4f}\n")
                
                if mAP_utama > best_map:
                    best_map = mAP_utama; torch.save(ema.module.state_dict(), f"{CHECKPOINT_DIR}/tcocr_{version}_best.pth")
                torch.save(ema.module.state_dict(), f"{CHECKPOINT_DIR}/tcocr_{version}_last.pth")
                ep += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("⚠️ OOM! Kurangi Batch..."); bs -= 4; model.zero_grad(); opt.zero_grad(); torch.cuda.empty_cache(); gc.collect()
                    if bs < 1: break
                    tr_ld = DataLoader(tr_ds, bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
                    vl_ld = DataLoader(vl_ds, bs, shuffle=False, collate_fn=collate_fn, num_workers=8)
                    sch = OneCycleLR(opt, max_lr=MAX_LR, steps_per_epoch=len(tr_ld), epochs=EPOCHS, last_epoch=(ep * len(tr_ld)) - 1)
                else: raise e
        
        print(f"🧹 Membersihkan {version.upper()} dari VRAM...")
        del model, ema, opt, scaler, sch, tr_ld, vl_ld; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__": main()