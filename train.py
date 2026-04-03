import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from transformers import RTDetrForObjectDetection
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
NUM_CLASSES = 10                     
INITIAL_BATCH_SIZE = 16               # Kita mulai dari 8, jika OOM akan turun ke 4
LR = 1e-4
EPOCHS = 20
IMG_SIZE = 640                       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 2. DATA PREPARATION (Target Formatting)
# ==========================================
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
            labels.append(obj['category_id'] - 1) # Koreksi label 1-10 -> 0-9
            
        if len(boxes) > 0:
            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "class_labels": torch.tensor(labels, dtype=torch.long).to(device)
            })
    return formatted_targets

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

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
    for images, targets in progress_bar:
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
    best_val_loss = float('inf')

    epoch = 0
    while epoch < EPOCHS:
        try:
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (Batch Size: {INITIAL_BATCH_SIZE}) ---")
            
            # Re-inisialisasi DataLoader dengan Batch Size saat ini
            train_dataset = CocoDetection(root=DATA_ROOT_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

            val_dataset = CocoDetection(root=DATA_ROOT_VAL, annFile=ANN_FILE_VAL, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

            # Jalankan Training
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
            
            # Jalankan Validasi (Opsional: Masukkan ke try-except juga jika perlu)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Validation"):
                    images = images.to(DEVICE)
                    formatted_targets = prepare_targets_for_detr(targets, DEVICE, IMG_SIZE)
                    outputs = model(images, labels=formatted_targets)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            print(f"📈 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/rtdetr_r50_best.pth")
                print("⭐ Best Model Saved!")

            epoch += 1 # Lanjut ke epoch berikutnya jika sukses

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("\n⚠️ DETECTED CUDA OOM! Reducing batch size...")
                
                # Bersihkan memori secara paksa
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                
                # Kurangi batch size
                INITIAL_BATCH_SIZE -= 4
                
                if INITIAL_BATCH_SIZE < 1:
                    print("❌ Error: Batch size cannot be reduced further (already < 1).")
                    break
                
                print(f"🔄 Retrying Epoch {epoch+1} with Batch Size: {INITIAL_BATCH_SIZE}")
                continue # Ulangi iterasi 'while' pada epoch yang sama
            else:
                raise e # Jika error lain, tetap tampilkan error-nya

if __name__ == "__main__":
    main()