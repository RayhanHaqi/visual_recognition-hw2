import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import gc
from pathlib import Path

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
DATA_ROOT_TRAIN = "./datasets/train"      
ANN_FILE_TRAIN = "./datasets/train.json"  
DATA_ROOT_VAL = "./datasets/valid"          
ANN_FILE_VAL = "./datasets/valid.json"      

CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "./logs/training_log_freeze.csv"
NUM_CLASSES = 10                     
INITIAL_BATCH_SIZE = 32               
LR = 1e-4    # Kita bisa pakai LR lumayan besar karena Backbone aman
EPOCHS = 20  # Cukup 20 karena hanya otak depan yang belajar
IMG_SIZE = 640                       
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP_50_95", "mAP_50"])

# ==========================================
# 2. DATA PREPARATION (STANDAR & BERSIH)
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
        boxes = []
        labels = []
        orig_w, orig_h = orig_sizes[i]
        
        for obj in target:
            x, y, w, h = obj['bbox']
            if w <= 0 or h <= 0: continue
            
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            
            boxes.append([cx, cy, nw, nh])
            labels.append(obj['category_id'] - 1) 
            
        if len(boxes) > 0:
            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "class_labels": torch.tensor(labels, dtype=torch.long).to(device)
            })
    return formatted_targets

# ==========================================
# 3. MODEL (DENGAN FREEZE BACKBONE)
# ==========================================
class RTDETRv2_R50_System(nn.Module):
    def __init__(self, num_classes=10):
        super(RTDETRv2_R50_System, self).__init__()
        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # ❄️ SENJATA RAHASIA: FREEZE BACKBONE ❄️
        print("\n❄️ Membekukan (Freezing) Backbone ResNet-50...")
        frozen_params = 0
        for name, param in self.model.named_parameters():
            # Kunci semua parameter yang bernama 'backbone'
            if "backbone" in name:
                param.requires_grad = False
                frozen_params += 1
        print(f"✅ {frozen_params} layer backbone berhasil dibekukan! Hanya layer Transformer yang akan belajar.\n")

    def forward(self, images, labels=None):
        return self.model(pixel_values=images, labels=labels)

# ==========================================
# 4. TRAINING LOGIC
# ==========================================
def train_one_epoch(model, loader, optimizer, scaler, device):
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
        
        # Mencegah Gradient Explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

def validate_and_evaluate(model, loader, device, val_json_path):
    model.eval()
    total_loss = 0
    predictions = []
    
    progress_bar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in progress_bar:
            images = images.to(device)
            formatted_targets = prepare_targets_for_detr(targets, orig_sizes, device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images, labels=formatted_targets)
                loss = outputs.loss
                total_loss += loss.item()
            
            logits = outputs.logits
            boxes = outputs.pred_boxes
            probs = logits.sigmoid()
            
            for i in range(len(images)):
                img_id = image_ids[i]
                orig_w, orig_h = orig_sizes[i]
                
                scores, labels = probs[i].max(dim=-1)
                keep = scores > 0.01 
                
                f_boxes = boxes[i][keep]
                f_scores = scores[keep]
                f_labels = labels[keep]
                
                for box, score, label in zip(f_boxes, f_scores, f_labels):
                    cx, cy, norm_w, norm_h = box.tolist()
                    w = norm_w * orig_w
                    h = norm_h * orig_h
                    x_min = (cx * orig_w) - (w / 2)
                    y_min = (cy * orig_h) - (h / 2)
                    
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(label.item()) + 1, 
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                        "score": round(score.item(), 4)
                    })

    val_loss = total_loss / len(loader)
    map_50_95, map_50 = 0.0, 0.0

    if len(predictions) > 0:
        tmp_json = "tmp_val_preds_fz.json"
        with open(tmp_json, 'w') as f:
            json.dump(predictions, f)
            
        coco_gt = COCO(val_json_path)
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(tmp_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        map_50_95 = coco_eval.stats[0] 
        map_50 = coco_eval.stats[1]    
        os.remove(tmp_json) 
        
    return val_loss, map_50_95, map_50

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    global INITIAL_BATCH_SIZE
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = RTDETRv2_R50_System(num_classes=NUM_CLASSES).to(DEVICE)
    
    # ⚠️ PENTING: Optimizer hanya boleh melatih parameter yang TIDAK dibekukan (requires_grad=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_map = 0.0 

    epoch = 0
    while epoch < EPOCHS:
        try:
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (Batch Size: {INITIAL_BATCH_SIZE}) ---")
            
            train_dataset = CustomCocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)

            val_dataset = CustomCocoDetection(root=DATA_ROOT_VAL, annFile=ANN_FILE_VAL, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)

            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
            val_loss, map_50_95, map_50 = validate_and_evaluate(model, val_loader, DEVICE, ANN_FILE_VAL)

            print(f"📈 T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | mAP_50-95: {map_50_95:.4f}")

            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, val_loss, map_50_95, map_50])

            if map_50_95 > best_map:
                best_map = map_50_95
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{Path(__file__).stem}_best.pth")
                print(f"⭐ Best Model Saved! (New Highest mAP: {best_map:.4f})")
                
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{Path(__file__).stem}_last.pth")
            epoch += 1 

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("\n⚠️ DETECTED CUDA OOM! Reducing batch size...")
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                
                INITIAL_BATCH_SIZE -= 4
                if INITIAL_BATCH_SIZE < 1:
                    print("❌ Error: Batch size cannot be reduced further.")
                    break
                print(f"🔄 Retrying Epoch {epoch+1} with Batch Size: {INITIAL_BATCH_SIZE}")
                continue 
            else:
                raise e 

if __name__ == "__main__":
    main()