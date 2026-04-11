import os, json, argparse, gc, torch, sys, csv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrV2Config, RTDetrV2ForObjectDetection
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, warm_up=20):
        self.patience, self.min_delta, self.warm_up = patience, min_delta, warm_up
        self.counter, self.best_score = 0, None; self.early_stop = False
    def __call__(self, current_score, epoch):
        if epoch < self.warm_up: return False
        if self.best_score is None: self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = current_score; self.counter = 0
        return self.early_stop

DATA_ROOT_TRAIN, ANN_FILE_TRAIN = "./datasets/train", "./datasets/train.json"
DATA_ROOT_VAL, ANN_FILE_VAL = "./datasets/valid", "./datasets/valid.json"
CHECKPOINT_DIR, LOG_DIR = "./checkpoints", "./log"
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

IMG_SIZE, NUM_CLASSES, LR = 640, 10, 1e-4

PHYSICAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4

tcocr_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]; img = self._load_image(id); tgt = self._load_target(id); orig_w, orig_h = img.size 
        if self.transforms: img, tgt = self.transforms(img, tgt)
        return img, tgt, id, orig_w, orig_h

def collate_fn(batch): return torch.stack([i[0] for i in batch], 0), [i[1] for i in batch], [i[2] for i in batch], [(i[3], i[4]) for i in batch]

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

def build_model(version, device):
    print(f"🛠️ Membangun Model TC-OCR UNFROZEN: {version.upper()}")
    cfg_base = "PekingU/rtdetr_r50vd" if version == "v1" else "PekingU/rtdetr_v2_r50vd"
    cfg = RTDetrConfig.from_pretrained(cfg_base, num_labels=NUM_CLASSES) if version=="v1" else RTDetrV2Config.from_pretrained(cfg_base, num_labels=NUM_CLASSES)
    cfg.num_queries = 1500 
    model = RTDetrForObjectDetection(config=cfg) if version=="v1" else RTDetrV2ForObjectDetection(config=cfg)
    pretrained = RTDetrForObjectDetection.from_pretrained(cfg_base) if version=="v1" else RTDetrV2ForObjectDetection.from_pretrained(cfg_base)
    model.load_state_dict({k: v for k, v in pretrained.state_dict().items() if 'backbone' in k}, strict=False)
    for p in model.parameters(): p.requires_grad = True
    del pretrained; gc.collect()
    return model.to(device)

def validate_and_evaluate(model, loader, device, val_json_path, version):
    model.eval(); predictions = []; total_v_loss = 0.0
    with torch.no_grad():
        for images, targets, image_ids, orig_sizes in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device); formatted_targets = prepare_targets(targets, orig_sizes, device)
            with torch.amp.autocast('cuda'): 
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
    if not predictions: return avg_v_loss, 0.0, 0.0, 0.0
    
    tmp = f"tmp_tcocr_{version}.json"
    with open(tmp,'w') as f: json.dump(predictions,f)
    coco_gt = COCO(val_json_path); coco_eval = COCOeval(coco_gt, coco_gt.loadRes(tmp), 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    os.remove(tmp); return avg_v_loss, coco_eval.stats[0], coco_eval.stats[1], coco_eval.stats[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v1", "v2"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args(); DEVICE = torch.device(f"cuda:{args.gpu}")
    
    global PHYSICAL_BATCH_SIZE, GRAD_ACCUM_STEPS
    EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
    
    run_name = f"unfrozen_tcocr_{args.version}"
    csv_path = os.path.join(LOG_DIR, f"{run_name}.csv")
    print(f"🚀 Menjalankan {run_name} di {DEVICE}")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'mAP_50_95_EMA', 'mAP_50_EMA', 'mAP_75_EMA', 'LR'])

    model = build_model(args.version, DEVICE)
    optimizer = torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if "backbone" not in n], "lr":LR}, 
        {"params":[p for n,p in model.named_parameters() if "backbone" in n], "lr":LR*0.1}
    ], weight_decay=1e-4)
    
    scaler = torch.amp.GradScaler('cuda')
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=5), 
            CosineAnnealingLR(optimizer, T_max=args.epochs-5, eta_min=1e-6)
        ], 
        milestones=[5]
    )
    early_stopper = EarlyStopping(patience=30)

    best_map, epoch = 0.0, 1

    while epoch <= args.epochs:
        try:
            train_loader = DataLoader(CustomCocoDetection(DATA_ROOT_TRAIN, ANN_FILE_TRAIN, tcocr_transform), batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)
            val_loader = DataLoader(CustomCocoDetection(DATA_ROOT_VAL, ANN_FILE_VAL, val_transform), batch_size=PHYSICAL_BATCH_SIZE, collate_fn=collate_fn, num_workers=8)

            model.train(); total_loss = 0; optimizer.zero_grad()
            pb = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {epoch}/{args.epochs} (BS: {PHYSICAL_BATCH_SIZE})")
            for i, (images, targets, _, sizes) in pb:
                images = images.to(DEVICE); targets = prepare_targets(targets, sizes, DEVICE)
                with torch.amp.autocast('cuda'): 
                    loss = model(images, labels=targets).loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                
                if (i+1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    ema_model.update_parameters(model)
                    
                total_loss += loss.item() * GRAD_ACCUM_STEPS
                pb.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
                
            avg_loss = total_loss / len(train_loader)
            v_loss, mAP, mAP50, mAP75 = validate_and_evaluate(ema_model.module, val_loader, DEVICE, ANN_FILE_VAL, args.version)
            cur_lr = optimizer.param_groups[0]['lr']
            
            print(f"📈 Ep {epoch} | T_Loss: {avg_loss:.4f} | V_Loss: {v_loss:.4f} | mAP: {mAP:.4f} | mAP50: {mAP50:.4f}")
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, round(avg_loss, 4), round(v_loss, 4), round(mAP, 4), round(mAP50, 4), round(mAP75, 4), cur_lr])
                
            scheduler.step()
            
            if mAP > best_map: 
                best_map = mAP; torch.save(ema_model.module.state_dict(), f"{CHECKPOINT_DIR}/{run_name}_best.pth")
                print(f"🏆 Best mAP baru: {best_map:.4f} - Tersimpan!")
                
            if early_stopper(mAP, epoch): break
            epoch += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad(); torch.cuda.empty_cache(); gc.collect()
                print(f"\n⚠️ OOM Terdeteksi di Epoch {epoch}! Menyelamatkan VRAM...")
                if PHYSICAL_BATCH_SIZE <= 1:
                    print("❌ GAGAL: Batch Size sudah 1 tapi tetap OOM. Hentikan training."); break
                PHYSICAL_BATCH_SIZE = max(1, PHYSICAL_BATCH_SIZE - 2)
                GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE)
                print(f"🔄 Mengurangi Batch Size menjadi: {PHYSICAL_BATCH_SIZE} | Accumulation: {GRAD_ACCUM_STEPS}\n")
            else: raise e

if __name__ == "__main__": main()