import os, json, argparse, gc, torch, sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrV2Config, RTDetrV2ForObjectDetection

# ==========================================
# 1. EARLY STOPPING CLASS
# ==========================================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, warm_up=20):
        self.patience, self.min_delta, self.warm_up = patience, min_delta, warm_up
        self.counter, self.best_score = 0, None
        self.early_stop = False

    def __call__(self, current_score, epoch):
        if epoch < self.warm_up: return False
        if self.best_score is None: self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        return self.early_stop

# ==========================================
# 2. CONFIG & TRANSFORM
# ==========================================
DATA_ROOT_TRAIN, ANN_FILE_TRAIN = "./datasets/train", "./datasets/train.json"
DATA_ROOT_VAL, ANN_FILE_VAL = "./datasets/valid", "./datasets/valid.json"
CHECKPOINT_DIR, LOG_DIR = "./checkpoints", "./log"
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

IMG_SIZE, NUM_CLASSES, LR = 640, 10, 1e-4
PHYSICAL_BATCH_SIZE = 4  # Aman untuk 16GB VRAM
GRAD_ACCUM_STEPS = 8     # Effective Batch Size = 32

baseline_train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]; image = self._load_image(id); target = self._load_target(id)
        orig_w, orig_h = image.size 
        if self.transforms is not None: image, target = self.transforms(image, target)
        return image, target, id, orig_w, orig_h

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

# ==========================================
# 3. BUILD MODEL (UNFROZEN + MEMORY OPTIM)
# ==========================================
def build_model(version, device):
    print(f"🛠️ Membangun Model Baseline UNFROZEN: {version.upper()}")
    cfg_base = "PekingU/rtdetr_r50vd" if version == "v1" else "PekingU/rtdetr_v2_r50vd"
    cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=NUM_CLASSES) if version == "v1" else RTDetrV2Config.from_pretrained(cfg_base, num_labels=NUM_CLASSES)
    cfg.num_queries = 300 
    
    model = RTDetrForObjectDetection(config=cfg) if version == "v1" else RTDetrV2ForObjectDetection(config=cfg)
    pretrained = RTDetrForObjectDetection.from_pretrained(cfg_base) if version == "v1" else RTDetrV2ForObjectDetection.from_pretrained(cfg_base)
    model.load_state_dict({k: v for k, v in pretrained.state_dict().items() if 'backbone' in k}, strict=False)
    
    model.model.gradient_checkpointing_enable() # Hemat VRAM
    for p in model.parameters(): p.requires_grad = True # Unfreeze semua
    
    del pretrained; gc.collect()
    return model.to(device)

def validate_and_evaluate(model, loader, device, val_json_path, version):
    model.eval(); predictions = []
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device); formatted_targets = prepare_targets(targets, orig_sizes, device)
            with torch.amp.autocast('cuda'): outputs = model(images)
            probs = outputs.logits.sigmoid()
            for i in range(len(images)):
                img_id, (orig_w, orig_h) = image_ids[i], orig_sizes[i]
                scores, labels = probs[i].max(dim=-1); keep = scores > 0.05 
                for box, score, label in zip(outputs.pred_boxes[i][keep], scores[keep], labels[keep]):
                    cx, cy, norm_w, norm_h = box.tolist()
                    w, h = norm_w * orig_w, norm_h * orig_h
                    predictions.append({"image_id": img_id, "category_id": int(label.item()) + 1, "bbox": [round((cx*orig_w)-(w/2),2), round((cy*orig_h)-(h/2),2), round(w,2), round(h,2)], "score": round(score.item(), 4)})
    if not predictions: return 0.0
    tmp = f"tmp_base_{version}.json"
    with open(tmp, 'w') as f: json.dump(predictions, f)
    coco_gt = COCO(val_json_path); coco_eval = COCOeval(coco_gt, coco_gt.loadRes(tmp), 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    os.remove(tmp); return coco_eval.stats[0]

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v1", "v2"])
    parser.add_argument("--gpu", type=int, default=0, help="ID GPU (contoh: 0 atau 1)")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args(); DEVICE = torch.device(f"cuda:{args.gpu}")
    
    run_name = f"unfrozen_baseline_{args.version}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    print(f"🚀 Menjalankan {run_name} di {DEVICE}")
    
    train_loader = DataLoader(CustomCocoDetection(DATA_ROOT_TRAIN, ANN_FILE_TRAIN, transform=baseline_train_transform), batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(CustomCocoDetection(DATA_ROOT_VAL, ANN_FILE_VAL, transform=val_transform), batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)

    model = build_model(args.version, DEVICE)
    
    # Differential LR
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": LR},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": LR * 0.1},
    ], weight_decay=1e-4)
    
    scaler = torch.amp.GradScaler('cuda')
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 0.01, 5), CosineAnnealingLR(optimizer, args.epochs-5, 1e-6)], [5])
    early_stopper = EarlyStopping(patience=30)

    best_map, epoch = 0.0, 1
    while epoch <= args.epochs:
        model.train(); total_loss = 0; optimizer.zero_grad()
        pb = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {epoch}/{args.epochs}")
        for i, (images, targets, _, orig_sizes) in pb:
            images = images.to(DEVICE); formatted_targets = prepare_targets(targets, orig_sizes, DEVICE)
            with torch.amp.autocast('cuda'):
                loss = model(images, labels=formatted_targets).loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pb.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
        
        mAP = validate_and_evaluate(model, val_loader, DEVICE, ANN_FILE_VAL, args.version)
        print(f"📈 Ep {epoch} | mAP: {mAP:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        writer.add_scalar('Metrics/mAP_50_95', mAP, epoch); scheduler.step()
        
        if mAP > best_map:
            best_map = mAP; torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_best.pth")
            print(f"🏆 Best mAP baru: {best_map:.4f} - Tersimpan!")
            
        if early_stopper(mAP, epoch):
            print(f"🛑 [AUTO-EXIT] Early Stopping aktif. Model stagnan di mAP {early_stopper.best_score:.4f}"); break
        epoch += 1
    writer.close()

if __name__ == "__main__": main()