import os
import json
import argparse
import gc
import torch
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
# 1. KONFIGURASI HYPERPARAMETER & PATH
# ==========================================
DATA_ROOT_TRAIN, ANN_FILE_TRAIN = "./datasets/train", "./datasets/train.json"
DATA_ROOT_VAL, ANN_FILE_VAL = "./datasets/valid", "./datasets/valid.json"
CHECKPOINT_DIR, LOG_DIR = "./checkpoints", "./log"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

IMG_SIZE = 640
NUM_CLASSES = 10
LR = 1e-4  # Hardcoded sebagai Hyperparameter
INITIAL_BATCH_SIZE = 64 # Start dari 32

# ==========================================
# 2. DATASET & AUGMENTASI
# ==========================================
baseline_train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        orig_w, orig_h = image.size 
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, id, orig_w, orig_h

def collate_fn(batch):
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
        if b: 
            res.append({"boxes": torch.tensor(b, dtype=torch.float32).to(dev), "class_labels": torch.tensor(l, dtype=torch.long).to(dev)})
    return res

# ==========================================
# 3. BUILDER MODEL "STRICT RULE"
# ==========================================
def build_strict_model(version="v1", device="cuda:1"):
    print(f"🛠️ Membangun Model Strict Baseline: {version.upper()}")
    cfg_base = "PekingU/rtdetr_r50vd" if version == "v1" else "PekingU/rtdetr_v2_r50vd"
    
    if version == "v1":
        cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=NUM_CLASSES)
        cfg.num_queries = 300
        model = RTDetrForObjectDetection(config=cfg)
    else:
        cfg = RTDetrV2Config.from_pretrained(cfg_base, num_labels=NUM_CLASSES)
        cfg.num_queries = 300
        model = RTDetrV2ForObjectDetection(config=cfg)

    print("📥 Membedah Pretrained Weights HuggingFace...")
    pretrained_model = RTDetrForObjectDetection.from_pretrained(cfg_base) if version == "v1" else RTDetrV2ForObjectDetection.from_pretrained(cfg_base)
    
    pretrained_dict = pretrained_model.state_dict()
    backbone_dict = {k: v for k, v in pretrained_dict.items() if 'backbone' in k}
    print(f"💉 Menyuntikkan {len(backbone_dict)} layer Backbone ke model bayi...")
    
    model.load_state_dict(backbone_dict, strict=False)
    
    frozen_layers = 0
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            frozen_layers += 1
        else:
            param.requires_grad = True 
            
    print(f"🧊 Berhasil membekukan {frozen_layers} layer parameter Backbone.")
    del pretrained_model; gc.collect()
    
    return model.to(device)

# ==========================================
# 4. FUNGSI EVALUASI
# ==========================================
def validate_and_evaluate(model, loader, device, val_json_path, version):
    model.eval()
    total_loss = 0
    predictions = []
    
    pb = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in pb:
            images = images.to(device)
            formatted_targets = prepare_targets(targets, orig_sizes, device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images, labels=formatted_targets)
                loss = outputs.loss
                total_loss += loss.item()
            
            probs = outputs.logits.sigmoid()
            for i in range(len(images)):
                img_id, (orig_w, orig_h) = image_ids[i], orig_sizes[i]
                scores, labels = probs[i].max(dim=-1)
                keep = scores > 0.05 
                
                for box, score, label in zip(outputs.pred_boxes[i][keep], scores[keep], labels[keep]):
                    cx, cy, norm_w, norm_h = box.tolist()
                    w, h = norm_w * orig_w, norm_h * orig_h
                    x_min, y_min = (cx * orig_w) - (w / 2), (cy * orig_h) - (h / 2)
                    predictions.append({
                        "image_id": img_id, "category_id": int(label.item()) + 1, 
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                        "score": round(score.item(), 4)
                    })

    val_loss = total_loss / len(loader)
    map_50_95, map_50, map_75 = 0.0, 0.0, 0.0

    if len(predictions) > 0:
        tmp_json = f"tmp_val_{version}.json"
        with open(tmp_json, 'w') as f: json.dump(predictions, f)
        import io; from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            coco_gt = COCO(val_json_path)
            coco_dt = coco_gt.loadRes(tmp_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        map_50_95, map_50, map_75 = coco_eval.stats[0], coco_eval.stats[1], coco_eval.stats[2]
        os.remove(tmp_json) 
        
    return val_loss, map_50_95, map_50, map_75

# ==========================================
# 5. MAIN TRAINING LOOP DENGAN OOM HANDLER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Strict Baseline RT-DETR")
    parser.add_argument("version", type=str, choices=["v1", "v2"], help="Arsitektur yang ingin dilatih (v1 atau v2)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_name = f"strict_baseline_{args.version}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    # 1. Load Dataset
    train_dataset = CustomCocoDetection(DATA_ROOT_TRAIN, ANN_FILE_TRAIN, transform=baseline_train_transform)
    val_dataset = CustomCocoDetection(DATA_ROOT_VAL, ANN_FILE_VAL, transform=val_transform)
    
    current_batch_size = INITIAL_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # 2. Load Model & Optimizer
    model = build_strict_model(args.version, DEVICE)
    active_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(active_parameters, lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Setup Scheduler (Dihitung PER EPOCH, bukan per batch)
    warmup_epochs = 5
    sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    best_map = 0.0
    print(f"\n🚀 MEMULAI TRAINING: {args.version.upper()} | Epochs: {args.epochs} | Base Batch: {INITIAL_BATCH_SIZE}")
    
    epoch = 1
    while epoch <= args.epochs:
        try:
            model.train()
            total_train_loss = 0
            pb = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (BS: {current_batch_size})")
            
            for images, targets, _, orig_sizes in pb:
                images = images.to(DEVICE)
                formatted_targets = prepare_targets(targets, orig_sizes, DEVICE)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images, labels=formatted_targets)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(active_parameters, max_norm=0.1)
                
                scaler.step(optimizer)
                scaler.update()
                    
                total_train_loss += loss.item()
                pb.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- VALIDASI ---
            val_loss, map_50_95, map_50, map_75 = validate_and_evaluate(model, val_loader, DEVICE, ANN_FILE_VAL, args.version)
            
            # --- STEP SCHEDULER DI AKHIR EPOCH ---
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            print(f"📈 Ep {epoch} | T_Loss: {avg_train_loss:.4f} | V_Loss: {val_loss:.4f} | mAP_50-95: {map_50_95:.4f} | LR: {current_lr:.6f}")
            
            # --- LOGGING & SAVE ---
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Metrics/mAP_50_95', map_50_95, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            if map_50_95 > best_map:
                best_map = map_50_95
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_best.pth")
                print(f"🏆 New Best mAP_50-95: {best_map:.4f} - Model disimpan!")
                
            # torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_last.pth")
            
            epoch += 1 # Lanjut ke epoch berikutnya jika aman
            
        # ==========================================
        # PENANGANAN OOM (OUT OF MEMORY)
        # ==========================================
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n⚠️ OOM Terdeteksi! Menurunkan Batch Size dari {current_batch_size} menjadi {current_batch_size - 4}...")
                current_batch_size -= 4
                
                # Bersihkan VRAM
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                
                if current_batch_size < 1:
                    print("❌ Batch size terlalu kecil (kurang dari 1). Menghentikan training.")
                    break
                
                # Buat ulang DataLoader dengan batch size baru
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
                val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
                
                print(f"🔄 Mengulang Epoch {epoch} dengan Batch Size {current_batch_size}...\n")
                continue # Langsung ulangi iterasi `while` tanpa menambah epoch
            else:
                raise e # Jika error lain, hentikan program

    writer.close()
    print(f"\n🎉 Selesai! Buka tensorboard dengan: tensorboard --logdir={LOG_DIR}")

if __name__ == "__main__":
    main()