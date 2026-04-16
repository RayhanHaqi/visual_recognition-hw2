import os, json, argparse, gc, torch, sys, csv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import OneCycleLR

# --- EARLY STOPPING LOGIC ---
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, warm_up=10, collapse_threshold=0.40):
        self.patience = patience
        self.min_delta = min_delta
        self.warm_up = warm_up 
        self.collapse_threshold = collapse_threshold 
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, epoch):
        # 1. SELALU CATAT REKOR TERTINGGI (Meskipun masih dalam masa warm-up)
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0  # Reset kesabaran jika rekor pecah
        
        # 2. JANGAN EKSEKUSI STOPPING JIKA MASIH WARM-UP
        if epoch < self.warm_up: 
            return False
        
        # 3. SATPAM BERAKSI (Cek Kolaps)
        # Sekarang dia membandingkan dengan rekor asli (0.46+), bukan skor ampas saat dia baru bangun
        if current_score < (self.collapse_threshold * self.best_score):
            print(f"\n🚨 [EMERGENCY STOP] Epoch {epoch}: Model Collapse Detected (Skor hancur > 60%)!")
            self.early_stop = True
            return True
            
        # 4. SATPAM BERAKSI (Cek Stagnasi)
        if current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n⏹️ [EARLY STOP] Tidak ada perbaikan mAP selama {self.patience} epoch.")
                self.early_stop = True
                return True
                
        return False
# --- DATASET & COLLATOR ---
class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]; img = self._load_image(id); tgt = self._load_target(id); orig_w, orig_h = img.size 
        if self.transforms: img, tgt = self.transforms(img, tgt)
        return img, tgt, id, orig_w, orig_h

def collate_fn(batch): 
    return torch.stack([i[0] for i in batch], 0), [i[1] for i in batch], [i[2] for i in batch], [(i[3], i[4]) for i in batch]

def prepare_targets(targets, sizes, dev):
    res = []
    for i, t in enumerate(targets):
        b, l = [], []
        for o in t:
            x,y,w,h = o['bbox']
            if w>0 and h>0:
                b.append([(x+w/2)/sizes[i][0], (y+h/2)/sizes[i][1], w/sizes[i][0], h/sizes[i][1]])
                l.append(o['category_id']-1)
        if b: res.append({"boxes": torch.tensor(b, dtype=torch.float32).to(dev), "class_labels": torch.tensor(l, dtype=torch.long).to(dev)})
    return res

# --- EVALUATION ---
def validate_and_evaluate(model, loader, device, val_json_path, run_name):
    model.eval(); predictions = []; total_v_loss = 0.0
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device); formatted_targets = prepare_targets(targets, orig_sizes, device)
            outputs = model(images, labels=formatted_targets)
            total_v_loss += outputs.loss.item()
            probs = outputs.logits.sigmoid()
            for i in range(len(images)):
                img_id, (orig_w, orig_h) = image_ids[i], orig_sizes[i]
                scores, labels = probs[i].max(-1); keep = scores > 0.05
                for box, score, label in zip(outputs.pred_boxes[i][keep], scores[keep], labels[keep]):
                    cx,cy,nw,nh = box.tolist(); w,h = nw*orig_w, nh*orig_h
                    predictions.append({"image_id":img_id, "category_id":int(label)+1, "bbox":[round((cx*orig_w)-(w/2),2), round((cy*orig_h)-(h/2),2), round(w,2), round(h,2)], "score":round(float(score),4)})
    avg_v_loss = total_v_loss / len(loader)
    if not predictions: return avg_v_loss, 0.0, 0.0
    tmp = f"tmp_eval_{run_name}.json" 
    with open(tmp,'w') as f: json.dump(predictions,f)
    coco_gt = COCO(val_json_path); coco_eval = COCOeval(coco_gt, coco_gt.loadRes(tmp), 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    os.remove(tmp); return avg_v_loss, coco_eval.stats[0], coco_eval.stats[1]

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4) 
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, )
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    DEVICE = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    csv_path = f"./log/{args.run_name}_log.csv"

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'mAP_50_95', 'mAP_50', 'LR'])

    # --- UNIFIED AUGMENTATION ---
    transform = transforms.Compose([
        transforms.Resize((320, 640)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    config = RTDetrV2Config.from_pretrained("PekingU/rtdetr_v2_r50vd", num_labels=10, num_queries=args.queries)
    # config.eos_coefficient = 0.0005
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd", config=config, ignore_mismatched_sizes=True).to(DEVICE)
    
    # --- RESET WEIGHTS (Train Transformer from Scratch) ---
    print(f"\n🔄 Resetting Transformer weights with {args.queries} queries...")
    model.model.encoder.apply(model._init_weights)
    model.model.decoder.apply(model._init_weights)
    print("✅ ResNet (Backbone) preserved, Transformer is ready to learn from scratch!\n")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr * 0.1},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.wd)
    scaler = torch.amp.GradScaler('cuda')
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    early_stopper = EarlyStopping()

    train_loader = DataLoader(CustomCocoDetection(f"{args.data_path}/train", f"{args.data_path}/train.json", transform), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.workers) 
    val_loader = DataLoader(CustomCocoDetection(f"{args.data_path}/valid", f"{args.data_path}/valid.json", transform), batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.workers)

    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.2)
    best_map, epoch = 0.0, 1

    while epoch <= args.epochs:
        try:
            model.train(); total_loss = 0; optimizer.zero_grad()
            pb = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
            for i, (images, targets, _, sizes) in pb:
                images = images.to(DEVICE); targets = prepare_targets(targets, sizes, DEVICE)
                with torch.amp.autocast('cuda'): loss = model(images, labels=targets).loss
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step(); ema_model.update_parameters(model); total_loss += loss.item()
            
            v_loss, mAP, mAP50 = validate_and_evaluate(ema_model.module, val_loader, DEVICE, f"{args.data_path}/valid.json", args.run_name)
            
            print(f"📊 Epoch {epoch} Summary | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {v_loss:.4f} | Val mAP: {mAP:.4f} | Val mAP@50: {mAP50:.4f}")
            
            with open(csv_path, 'a', newline='') as f: 
                csv.writer(f).writerow([epoch, round(total_loss/len(train_loader),4), round(v_loss,4), round(mAP,4), round(mAP50,4), optimizer.param_groups[0]['lr']])
            
            if mAP > best_map:
                print(f"🏆 NEW BEST MODEL! mAP improved from {best_map:.4f} to {mAP:.4f}. Saving checkpoint...")
                best_map = mAP; torch.save(ema_model.module.state_dict(), f"{args.save_path}/{args.run_name}_best.pth")
            else:
                print(f"⏳ No improvement in mAP. Current Best: {best_map:.4f}.")
            
            if early_stopper(mAP, epoch): break
            epoch += 1
            print("-" * 100)
            print("-" * 100)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ OUT OF MEMORY (OOM) Detected at Epoch {epoch}! Clearing GPU cache to prevent crash...")
                torch.cuda.empty_cache(); gc.collect(); break
            else: raise e

if __name__ == "__main__": main()