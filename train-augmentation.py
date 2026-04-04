import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import gc
import numpy as np
from PIL import Image
import cv2

# Import Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
DATA_ROOT_TRAIN = "./datasets/train"      
ANN_FILE_TRAIN = "./datasets/train.json"  
DATA_ROOT_VAL = "./datasets/valid"          
ANN_FILE_VAL = "./datasets/valid.json"      

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
LOG_FILE = "./logs/training_log_aug.csv"    # Nama log baru agar tidak menimpa yang lama
NUM_CLASSES = 10                     
INITIAL_BATCH_SIZE = 32               
LR = 1e-4
EPOCHS = 20
IMG_SIZE = 640                       
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP_50_95", "mAP_50", "mAP_75"])

# ==========================================
# 2. HEAVY DATA AUGMENTATION (ALBUMENTATIONS)
# ==========================================
# Augmentasi untuk Training
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    # Manipulasi Warna & Pencahayaan
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
    # Manipulasi Lensa Kamera (Buram / Noise)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MedianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(10.0, 50.0)),
    ], p=0.4),
    # Manipulasi Geometri (Geser, Zoom, Putar Sedikit)
    # Rotasi dibatasi 15 derajat agar 6 tidak jadi 9
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, 
                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128), p=0.5),
    # Normalisasi standar PyTorch
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.2))

# Augmentasi untuk Validasi (Hanya Resize & Normalize)
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# ==========================================
# 3. DATA PREPARATION CUSTOM
# ==========================================
# ==========================================
# 3. DATA PREPARATION CUSTOM (DENGAN CLIPPING)
# ==========================================
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transforms = transform

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        orig_w, orig_h = image.size 
        image_np = np.array(image.convert("RGB"))

        # Ekstrak Box dan Label dengan CLIPPING
        boxes, category_ids = [], []
        for obj in target:
            x, y, w, h = obj['bbox']
            
            # 1. Pastikan kotak tidak tumpah ke kiri/atas (minimal 0)
            x_min = max(0.0, float(x))
            y_min = max(0.0, float(y))
            
            # 2. Pastikan kotak tidak tumpah ke kanan/bawah (maksimal lebar/tinggi gambar)
            x_max = min(float(orig_w), float(x + w))
            y_max = min(float(orig_h), float(y + h))
            
            # 3. Hitung lebar dan tinggi baru setelah digunting
            new_w = x_max - x_min
            new_h = y_max - y_min
            
            # 4. Hanya masukkan kotak jika masih memiliki luas (tidak hilang)
            if new_w > 0.0 and new_h > 0.0:
                boxes.append([x_min, y_min, new_w, new_h])
                category_ids.append(obj['category_id'] - 1) 
        
        if self.transforms is not None:
            # Jika dataset kosong dari kotak valid, skip augmentasi yang bikin error
            if len(boxes) == 0:
                transformed = self.transforms(image=image_np, bboxes=[], category_ids=[])
            else:
                transformed = self.transforms(image=image_np, bboxes=boxes, category_ids=category_ids)
            
            image_tensor = transformed['image']
            boxes = transformed['bboxes']
            category_ids = transformed['category_ids']
        else:
            image_tensor = image_np 

        return image_tensor, boxes, category_ids, id, orig_w, orig_h

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [{"boxes": item[1], "labels": item[2]} for item in batch]
    image_ids = [item[3] for item in batch]
    orig_sizes = [(item[4], item[5]) for item in batch]
    return images, targets, image_ids, orig_sizes

def prepare_targets_for_detr(targets, device, img_size=640):
    formatted_targets = []
    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]
        
        norm_boxes = []
        for (x, y, w, h) in boxes:
            # Albumentations sudah me-resize ke 640, jadi kita cukup bagi 640
            cx = (x + w / 2) / img_size
            cy = (y + h / 2) / img_size
            nw = w / img_size
            nh = h / img_size
            norm_boxes.append([cx, cy, nw, nh])
            
        if len(norm_boxes) > 0:
            formatted_targets.append({
                "boxes": torch.tensor(norm_boxes, dtype=torch.float32).to(device),
                "class_labels": torch.tensor(labels, dtype=torch.long).to(device)
            })
        else:
            # Jaga-jaga jika semua kotak hilang terpotong augmentasi
            formatted_targets.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32).to(device),
                "class_labels": torch.empty((0,), dtype=torch.long).to(device)
            })
    return formatted_targets

# ==========================================
# 4. MODEL & TRAINING LOGIC
# ==========================================
class RTDETRv2_R50_System(nn.Module):
    def __init__(self, num_classes=10):
        super(RTDETRv2_R50_System, self).__init__()
        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, images, labels=None):
        return self.model(pixel_values=images, labels=labels)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    
    for images, targets, _, _ in progress_bar:
        images = images.to(device)
        formatted_targets = prepare_targets_for_detr(targets, device, IMG_SIZE)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images, labels=formatted_targets)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

def validate_and_evaluate(model, loader, device, val_json_path):
    model.eval()
    total_loss = 0
    predictions = []
    
    progress_bar = tqdm(loader, desc="Validation & mAP Calc")
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in progress_bar:
            images = images.to(device)
            formatted_targets = prepare_targets_for_detr(targets, device, IMG_SIZE)
            
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
    map_50_95, map_50, map_75 = 0.0, 0.0, 0.0

    if len(predictions) > 0:
        tmp_json = "tmp_val_preds_aug.json"
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
        map_75 = coco_eval.stats[2]    
        
        os.remove(tmp_json) 
        
    return val_loss, map_50_95, map_50, map_75

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    global INITIAL_BATCH_SIZE
    
    model = RTDETRv2_R50_System(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_map = 0.0 

    epoch = 0
    while epoch < EPOCHS:
        try:
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (Batch Size: {INITIAL_BATCH_SIZE}) ---")
            
            # Gunakan Albumentations transform di Dataset
            train_dataset = CustomCocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)

            val_dataset = CustomCocoDetection(root=DATA_ROOT_VAL, annFile=ANN_FILE_VAL, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)

            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
            
            val_loss, map_50_95, map_50, map_75 = validate_and_evaluate(model, val_loader, DEVICE, ANN_FILE_VAL)

            print(f"📈 T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | mAP_50-95: {map_50_95:.4f} | mAP_50: {map_50:.4f}")

            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, val_loss, map_50_95, map_50, map_75])

            if map_50_95 > best_map:
                best_map = map_50_95
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/rtdetr_r50_best.pth")
                print(f"⭐ Best Model Saved! (New Highest mAP: {best_map:.4f})")
                
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/rtdetr_r50_last.pth")

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