import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import gc

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import OneCycleLR

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
DATA_ROOT_TRAIN = "./datasets/train"      
ANN_FILE_TRAIN = "./datasets/train.json"  
DATA_ROOT_VAL = "./datasets/valid"          
ANN_FILE_VAL = "./datasets/valid.json"      

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

NUM_CLASSES = 10                     
INITIAL_BATCH_SIZE = 32               
MAX_LR = 1e-4 
EPOCHS = 10
STAGE_2_START_EPOCH = 8 # Babak 2 (Tanpa Augmentasi) dimulai di Epoch 16
IMG_SIZE = 640                       
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 2. DATA PREPARATION
# ==========================================
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

def prepare_targets_for_detr(targets, orig_sizes, device):
    formatted_targets = []
    for i, target in enumerate(targets):
        boxes, labels = [], []
        orig_w, orig_h = orig_sizes[i]
        for obj in target:
            x, y, w, h = obj['bbox']
            if w <= 0 or h <= 0: continue
            cx, cy = (x + w / 2) / orig_w, (y + h / 2) / orig_h
            nw, nh = w / orig_w, h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(obj['category_id'] - 1) 
        if len(boxes) > 0:
            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "class_labels": torch.tensor(labels, dtype=torch.long).to(device)
            })
    return formatted_targets

# 🛠️ STRATEGI 1: TWO-STAGE AUGMENTATION
transform_stage_1 = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_clean = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. MODEL SELECTION
# ==========================================
class PaperDETR_System(nn.Module):
    def __init__(self, version="v1", num_classes=10):
        super(PaperDETR_System, self).__init__()
        if version == "v1":
            self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            self.model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd", num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, images, labels=None):
        return self.model(pixel_values=images, labels=labels)

# ==========================================
# 4. TRAINING & EVALUATION LOGIC
# ==========================================
def train_one_epoch(model, ema_model, loader, optimizer, scaler, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    for images, targets, _, orig_sizes in progress_bar:
        images = images.to(device)
        formatted_targets = prepare_targets_for_detr(targets, orig_sizes, device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images, labels=formatted_targets)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        ema_model.update_parameters(model)
        total_loss += loss.item()
        
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})
    return total_loss / len(loader)

def validate_and_evaluate(model, loader, device, val_json_path, version):
    model.eval()
    total_loss = 0
    predictions = []
    
    progress_bar = tqdm(loader, desc="Validation (EMA)")
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in progress_bar:
            images = images.to(device)
            formatted_targets = prepare_targets_for_detr(targets, orig_sizes, device)
            with torch.amp.autocast('cuda'):
                outputs = model(images, labels=formatted_targets)
                loss = outputs.loss
                total_loss += loss.item()
            
            logits, boxes = outputs.logits, outputs.pred_boxes
            probs = logits.sigmoid()
            
            for i in range(len(images)):
                img_id, (orig_w, orig_h) = image_ids[i], orig_sizes[i]
                scores, labels = probs[i].max(dim=-1)
                keep = scores > 0.05 
                for box, score, label in zip(boxes[i][keep], scores[keep], labels[keep]):
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
        tmp_json = f"tmp_val_preds_{version}_ema.json"
        with open(tmp_json, 'w') as f: json.dump(predictions, f)
        coco_gt = COCO(val_json_path)
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(tmp_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        map_50_95, map_50, map_75 = coco_eval.stats[0], coco_eval.stats[1], coco_eval.stats[2]
        os.remove(tmp_json) 
        
    return val_loss, map_50_95, map_50, map_75

# ==========================================
# 5. MAIN LOOP (EKSEKUSI V1 LALU V2)
# ==========================================
def main():
    versions_to_train = ["v1", "v2"]
    
    for version in versions_to_train:
        print(f"\n{'='*50}")
        print(f"🚀 MEMULAI TRAINING PAPER VERSION UNTUK: {version.upper()}")
        print(f"{'='*50}\n")
        
        # Reset Batch Size untuk setiap model
        current_batch_size = INITIAL_BATCH_SIZE
        
        # Inisialisasi File Log Baru untuk Masing-masing Versi
        log_file = f"{LOG_DIR}/training_log_paper_{version}.csv"
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Stage", "Train_Loss", "Val_Loss", "mAP_50_95", "mAP_50", "mAP_75"])

        # Inisialisasi Model & EMA Baru
        model = PaperDETR_System(version=version, num_classes=NUM_CLASSES).to(DEVICE)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        
        # Reset Dataset Transforms ke Babak 1 (Augmented)
        train_dataset = CustomCocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform_stage_1)
        val_dataset = CustomCocoDetection(root=DATA_ROOT_VAL, annFile=ANN_FILE_VAL, transform=transform_clean)
        
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
        
        # Init OneCycleLR Baru
        scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
        
        best_map = 0.0 
        epoch = 0
        
        while epoch < EPOCHS:
            try:
                current_stage = "Stage 1 (Augmented)"
                # Pindah ke Babak 2
                if epoch == STAGE_2_START_EPOCH:
                    print(f"\n🔔 [MODEL {version.upper()}] MEMASUKI BABAK 2: Augmentasi Dimatikan (Clean Fine-Tuning)!")
                    # RE-INISIALISASI DATASET & LOADER AGAR PYTORCH MEMBUNGKUSNYA DENGAN BENAR
                    train_dataset = CustomCocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform_clean)
                    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
                
                if epoch >= STAGE_2_START_EPOCH:
                    current_stage = "Stage 2 (Clean)"

                print(f"\n--- Model: {version.upper()} | Epoch {epoch+1}/{EPOCHS} [{current_stage}] (Batch: {current_batch_size}) ---")
                
                train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, scaler, scheduler, DEVICE)
                val_loss, map_50_95, map_50, map_75 = validate_and_evaluate(ema_model, val_loader, DEVICE, ANN_FILE_VAL, version)

                print(f"📈 T_Loss: {train_loss:.4f} | EMA_V_Loss: {val_loss:.4f} | EMA_mAP_50-95: {map_50_95:.4f}")

                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, current_stage, train_loss, val_loss, map_50_95, map_50, map_75])

                if map_50_95 > best_map:
                    best_map = map_50_95
                    torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/paper_{version}_best.pth")
                    print(f"⭐ Best EMA Model {version.upper()} Saved! (New Highest mAP: {best_map:.4f})")
                    
                torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/paper_{version}_last.pth")

                epoch += 1 

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n⚠️ DETECTED CUDA OOM on {version.upper()}! Reducing batch size...")
                    model.zero_grad(); optimizer.zero_grad()
                    torch.cuda.empty_cache(); gc.collect()
                    
                    current_batch_size -= 4
                    if current_batch_size < 1: 
                        print(f"❌ Batch size terlalu kecil, membatalkan {version.upper()}")
                        break
                    
                    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
                    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
                    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, last_epoch=(epoch * len(train_loader)) - 1)
                    
                    print(f"🔄 Retrying Epoch {epoch+1} with Batch Size: {current_batch_size}")
                    continue 
                else:
                    raise e 
        
        # ==========================================
        # ⚠️ PENGHAPUSAN MEMORI GPU SEBELUM PINDAH VERSI
        # ==========================================
        print(f"\n🧹 Menyelesaikan {version.upper()}. Membersihkan VRAM...")
        del model
        del ema_model
        del optimizer
        del scaler
        del scheduler
        del train_loader
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()
        print(f"✅ VRAM Bersih! Bersiap untuk versi selanjutnya...\n")

if __name__ == "__main__":
    main()