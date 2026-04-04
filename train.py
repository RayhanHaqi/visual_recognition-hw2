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

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
DATA_ROOT_TRAIN = "./datasets/train"      
ANN_FILE_TRAIN = "./datasets/train.json"  
DATA_ROOT_VAL = "./datasets/valid"          
ANN_FILE_VAL = "./datasets/valid.json"      

CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "training_log.csv"
NUM_CLASSES = 10                     
INITIAL_BATCH_SIZE = 16               
LR = 1e-4
EPOCHS = 20
IMG_SIZE = 640                       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Inisialisasi File Log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP_50_95", "mAP_50", "mAP_75"])

# ==========================================
# 2. DATA PREPARATION CUSTOM
# ==========================================
class CustomCocoDetection(CocoDetection):
    """
    Subclass khusus agar DataLoader juga mengembalikan Image ID dan 
    Resolusi Asli gambar, yang sangat dibutuhkan untuk menghitung mAP.
    """
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        orig_w, orig_h = image.size # Catat ukuran asli
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target, id, orig_w, orig_h

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    image_ids = [item[2] for item in batch]
    orig_sizes = [(item[3], item[4]) for item in batch]
    return images, targets, image_ids, orig_sizes

def prepare_targets_for_detr(targets, device, img_size=640):
    formatted_targets = []
    for target in targets:
        boxes = []
        labels = []
        for obj in target:
            x, y, w, h = obj['bbox']
            if w <= 0 or h <= 0: continue
            
            cx = (x + w / 2) / img_size
            cy = (y + h / 2) / img_size
            nw = w / img_size
            nh = h / img_size
            
            boxes.append([cx, cy, nw, nh])
            labels.append(obj['category_id'] - 1) 
            
        if len(boxes) > 0:
            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "class_labels": torch.tensor(labels, dtype=torch.long).to(device)
            })
    return formatted_targets

# ==========================================
# 3. MODEL INITIALIZATION
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

# ==========================================
# 4. TRAINING & VALIDATION LOGIC
# ==========================================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    
    # Perhatikan ada _, _ untuk menampung id dan size yang tidak dipakai saat training
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
    """
    Fungsi ini menghitung Val Loss DAN menghitung mAP layaknya CodaBench.
    """
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
            
            # --- EKSTRAKSI UNTUK mAP ---
            logits = outputs.logits
            boxes = outputs.pred_boxes
            probs = logits.sigmoid()
            
            for i in range(len(images)):
                img_id = image_ids[i]
                orig_w, orig_h = orig_sizes[i]
                
                scores, labels = probs[i].max(dim=-1)
                keep = scores > 0.01 # Gunakan threshold rendah agar mAP akurat
                
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
                        "category_id": int(label.item()) + 1, # Kembali ke 1-10
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                        "score": round(score.item(), 4)
                    })

    val_loss = total_loss / len(loader)
    map_50_95, map_50, map_75 = 0.0, 0.0, 0.0

    # --- HITUNG mAP MENGGUNAKAN PYCOCOTOOLS ---
    if len(predictions) > 0:
        tmp_json = "tmp_val_preds.json"
        with open(tmp_json, 'w') as f:
            json.dump(predictions, f)
            
        coco_gt = COCO(val_json_path)
        coco_dt = coco_gt.loadRes(tmp_json)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Ambil metrik dari stats array COCOeval
        map_50_95 = coco_eval.stats[0] # Average Precision  (AP) @[ IoU=0.50:0.95 ]
        map_50 = coco_eval.stats[1]    # Average Precision  (AP) @[ IoU=0.50      ]
        map_75 = coco_eval.stats[2]    # Average Precision  (AP) @[ IoU=0.75      ]
        
        os.remove(tmp_json) # Bersihkan file sementara
        
    return val_loss, map_50_95, map_50, map_75

# ==========================================
# 5. MAIN EXECUTION WITH AUTO-OOM RECOVERY
# ==========================================
def main():
    global INITIAL_BATCH_SIZE
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = RTDETRv2_R50_System(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    best_map = 0.0 # Sekarang patokan terbaik adalah mAP, bukan Loss

    epoch = 0
    while epoch < EPOCHS:
        try:
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (Batch Size: {INITIAL_BATCH_SIZE}) ---")
            
            # Gunakan CustomCocoDetection
            train_dataset = CustomCocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)

            val_dataset = CustomCocoDetection(root=DATA_ROOT_VAL, annFile=ANN_FILE_VAL, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)

            # 1. Training
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
            
            # 2. Validation & mAP
            val_loss, map_50_95, map_50, map_75 = validate_and_evaluate(model, val_loader, DEVICE, ANN_FILE_VAL)

            print(f"📈 T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | mAP_50-95: {map_50_95:.4f} | mAP_50: {map_50:.4f}")

            # 3. Simpan Log ke CSV
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, val_loss, map_50_95, map_50, map_75])

            # 4. Simpan Model Terbaik berdasarkan mAP
            if map_50_95 > best_map:
                best_map = map_50_95
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/rtdetr_r50_best.pth")
                print(f"⭐ Best Model Saved! (New Highest mAP: {best_map:.4f})")
                
            # Simpan juga model last epoch buat jaga-jaga
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