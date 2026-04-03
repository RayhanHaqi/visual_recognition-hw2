import json
import os
from pycocotools.coco import COCO

def validate_coco_dataset(ann_file, img_root, num_classes=10):
    print(f"--- Memulai Validasi Dataset: {ann_file} ---")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # 1. Cek Struktur Dasar
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            print(f"❌ Error: Kunci '{key}' tidak ditemukan di JSON!")
            return

    # 2. Cek Categories
    print(f"Found {len(data['categories'])} categories.")
    cat_ids = [cat['id'] for cat in data['categories']]
    print(f"Category IDs: {cat_ids}")

    # 3. Validasi Gambar & Anotasi
    error_count = 0
    invalid_boxes = 0
    out_of_range_labels = 0
    missing_images = 0
    
    image_dict = {img['id']: img for img in data['images']}
    
    # Cek Keberadaan File Fisik
    print("Checking image files...")
    for img in data['images']:
        img_path = os.path.join(img_root, img['file_name'])
        if not os.path.exists(img_path):
            # print(f"⚠️ Gambar Hilang: {img['file_name']}")
            missing_images += 1
            error_count += 1

    # Cek Anotasi
    print("Checking annotations...")
    for ann in data['annotations']:
        # A. Cek Label
        if ann['category_id'] >= num_classes or ann['category_id'] < 0:
            # print(f"❌ Label Out of Range: ID {ann['category_id']} di Image ID {ann['image_id']}")
            out_of_range_labels += 1
            error_count += 1
            
        # B. Cek Bounding Box [x, y, w, h]
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            # print(f"❌ BBox Ilegal (w/h <= 0): {ann['bbox']} di Image ID {ann['image_id']}")
            invalid_boxes += 1
            error_count += 1
            
        # C. Cek apakah image_id ada di list images
        if ann['image_id'] not in image_dict:
            print(f"❌ Orphan Annotation: Image ID {ann['image_id']} tidak punya data gambar!")
            error_count += 1

    print("\n--- HASIL VALIDASI ---")
    if error_count == 0:
        print("✅ Dataset Bersih! Aman untuk ditraining.")
    else:
        print(f"❌ Ditemukan {error_count} masalah total:")
        print(f"   - Label di luar range (0-{num_classes-1}): {out_of_range_labels}")
        print(f"   - Bounding Box rusak (w/h <= 0): {invalid_boxes}")
        print(f"   - File gambar tidak ditemukan di folder: {missing_images}")
        print("\nSaran: Perbaiki data di atas atau tambahkan logika 'filter' di DataLoader.")

if __name__ == "__main__":
    # Ganti path sesuai foldermu
    validate_coco_dataset(
        ann_file="./datasets/train.json", 
        img_root="./datasets/train", 
        num_classes=10
    )