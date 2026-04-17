import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def generate_plots(csv_filename):
    if not os.path.exists(csv_filename):
        print(f"❌ Error: File '{csv_filename}' tidak ditemukan.")
        print("Pastikan nama file sudah benar dan kamu berada di folder yang tepat.")
        return

    print(f"📊 Membaca data dari {csv_filename}...")
    df = pd.read_csv(csv_filename)

    required_columns = ['Epoch', 'Train_Loss', 'Val_Loss', 'mAP_50_95', 'mAP_50']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Error: Kolom '{col}' tidak ditemukan di dalam file CSV.")
            return

    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')
    
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.close()
    print("✅ Berhasil menyimpan: loss_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df['Epoch'], df['mAP_50_95'], label='mAP@50-95', marker='o', linewidth=2, color='#2ca02c')
    plt.plot(df['Epoch'], df['mAP_50'], label='mAP@50', marker='s', linewidth=2, color='#d62728')
    
    plt.title('Validation Mean Average Precision (mAP)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('map_curve.png', dpi=300)
    plt.close()
    print("✅ Berhasil menyimpan: map_curve.png")

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Generate Loss and mAP plots from RT-DETR training log CSV.")
    
    # Menambahkan argumen wajib: nama file csv
    parser.add_argument("csv_file", type=str, help="Nama file CSV log training (contoh: log.csv)")
    
    # Parsing argumen dari terminal
    args = parser.parse_args()
    
    # Menjalankan fungsi dengan argumen yang diberikan
    generate_plots(args.csv_file)