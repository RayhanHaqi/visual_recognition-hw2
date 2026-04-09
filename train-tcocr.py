import os, json, argparse, gc, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrV2Config, RTDetrV2ForObjectDetection
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ================= KONFIGURASI =================
DATA_ROOT_TRAIN, ANN_FILE_TRAIN = "./datasets/train", "./datasets/train.json"
DATA_ROOT_VAL, ANN_FILE_VAL = "./datasets/valid", "./datasets/valid.json"
CHECKPOINT_DIR, LOG_DIR = "./checkpoints", "./log"
IMG_SIZE, NUM_CLASSES, LR, INITIAL_BATCH_SIZE = 640, 10, 1e-4, 64

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ================= AUGMENTASI TC-OCR =================
# Ditambahkan RandomAffine(translate) untuk mensimulasikan pergeseran/crop ringan
tcocr_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]; img = self._load_image(id); tgt = self._load_target(id)
        if self.transforms is not None: img, tgt = self.transforms(img, tgt)
        return img, tgt, id, img.size[0], img.size[1]

def collate_fn(batch):
    return torch.stack([i[0] for i in batch], 0), [i[1] for i in batch], [i[2] for i in batch], [(i[3], i[4]) for i in batch]

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

# ================= BUILDER STRICT RULE (1500 QUERIES) =================
def build_tcocr_model(version="v1", device="cuda:1"):
    print(f"🛠️ Membangun Model Strict TC-OCR: {version.upper()} (1500 Queries)")
    cfg_base = "PekingU/rtdetr_r50vd" if version == "v1" else "PekingU/rtdetr_v2_r50vd"
    
    if version == "v1":
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=NUM_CLASSES); cfg.num_queries = 1500
        model = RTDetrForObjectDetection(config=cfg)
    else:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=NUM_CLASSES); cfg.num_queries = 1500
        model = RTDetrV2ForObjectDetection(config=cfg)

    pretrained = RTDetrForObjectDetection.from_pretrained(cfg_base) if version == "v1" else RTDetrV2ForObjectDetection.from_pretrained(cfg_base)
    model.load_state_dict({k: v for k, v in pretrained.state_dict().items() if 'backbone' in k}, strict=False)
    
    for name, param in model.named_parameters():
        param.requires_grad = False if 'backbone' in name else True
            
    del pretrained; gc.collect()
    return model.to(device)

def validate_and_evaluate(model, loader, device, val_json_path, version):
    model.eval(); total_loss = 0; predictions = []
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device); formatted_targets = prepare_targets(targets, orig_sizes, device)
            with torch.amp.autocast('cuda'):
                outputs = model(images, labels=formatted_targets)
                total_loss += outputs.loss.item()
            
            probs = outputs.logits.sigmoid()
            for i in range(len(images)):
                img_id, (orig_w, orig_h) = image_ids[i], orig_sizes[i]
                scores, labels = probs[i].max(dim=-1); keep = scores > 0.05 
                for box, score, label in zip(outputs.pred_boxes[i][keep], scores[keep], labels[keep]):
                    cx, cy, norm_w, norm_h = box.tolist()
                    w, h = norm_w * orig_w, norm_h * orig_h
                    predictions.append({
                        "image_id": img_id, "category_id": int(label.item()) + 1, 
                        "bbox": [round((cx * orig_w) - (w / 2), 2), round((cy * orig_h) - (h / 2), 2), round(w, 2), round(h, 2)],
                        "score": round(score.item(), 4)
                    })

    if len(predictions) > 0:
        tmp = f"tmp_{version}.json"
        with open(tmp, 'w') as f: json.dump(predictions, f)
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(COCO(val_json_path), COCO(val_json_path).loadRes(tmp), 'bbox')
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        os.remove(tmp) 
        return total_loss/len(loader), coco_eval.stats[0]
    return total_loss/len(loader), 0.0

# ================= MAIN LOOP =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v1", "v2"]); parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    DEVICE = torch.device("cuda:1")
    run_name = f"tcocr_{args.version}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    train_ds = CustomCocoDetection(DATA_ROOT_TRAIN, ANN_FILE_TRAIN, transform=tcocr_transform)
    val_ds = CustomCocoDetection(DATA_ROOT_VAL, ANN_FILE_VAL, transform=val_transform)
    
    bs = INITIAL_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=8)

    model = build_tcocr_model(args.version, DEVICE)
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(active_params, lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # SETUP EMA (Exponential Moving Average)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 0.01, 5), CosineAnnealingLR(optimizer, args.epochs-5, 1e-6)], [5])

    best_map = 0.0; epoch = 1
    while epoch <= args.epochs:
        try:
            model.train(); total_loss = 0
            pb = tqdm(train_loader, desc=f"TC-OCR Ep {epoch}/{args.epochs} (BS: {bs})")
            for images, targets, _, orig_sizes in pb:
                images = images.to(DEVICE); formatted_targets = prepare_targets(targets, orig_sizes, DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'): loss = model(images, labels=formatted_targets).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(active_params, 0.1)
                scaler.step(optimizer); scaler.update()
                
                # UPDATE EMA SETIAP BATCH
                ema_model.update_parameters(model)
                total_loss += loss.item(); pb.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # EVALUASI MENGGUNAKAN EMA MODEL (Bukan model biasa)
            v_loss, mAP = validate_and_evaluate(ema_model.module, val_loader, DEVICE, ANN_FILE_VAL, args.version)
            
            print(f"📈 Ep {epoch} | mAP_50-95 (EMA): {mAP:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            writer.add_scalar('Metrics/mAP_50_95', mAP, epoch)
            
            if mAP > best_map:
                best_map = mAP; torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_best.pth")
                print(f"🏆 New Best mAP_50-95: {best_map:.4f} - Model disimpan!")
            # torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_last.pth")
            
            scheduler.step(); epoch += 1
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"⚠️ OOM! Turunkan Batch Size ke {bs - 4}..."); bs -= 4
                if bs < 1: break
                model.zero_grad(); optimizer.zero_grad(); torch.cuda.empty_cache(); gc.collect()
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
                val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=8)
            else: raise e
    writer.close()

if __name__ == "__main__": main()