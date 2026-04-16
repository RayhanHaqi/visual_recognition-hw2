import os
import subprocess
from pathlib import Path

def check_dataset_exists(extract_dir):
    train_json = os.path.join(extract_dir, "train.json")
    valid_json = os.path.join(extract_dir, "valid.json")
    return os.path.exists(train_json) and os.path.exists(valid_json)

def install_pigz():
    print("\n⚡ Mengecek ketersediaan pigz (Parallel GZIP)...")
    try:
        subprocess.run(["pigz", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("   ✅ pigz sudah terpasang.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⏳ pigz belum terpasang. Menginstal pigz untuk ekstraksi super cepat...")
        try:
            # Mencoba install menggunakan apt (untuk Debian/Ubuntu Linux seperti RunPod)
            subprocess.run(["apt-get", "update", "-y"], check=False, stdout=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "pigz"], check=True)
            print("   ✅ pigz berhasil diinstal!")
        except Exception as e:
            print(f"   ⚠️ Gagal menginstal pigz secara otomatis. Akan mencoba fallback. Error: {e}")

def setup_git_credentials():
    print("\n🔧 5. Mengatur Konfigurasi Kredensial Git & LFS...")
    
    try:
        # Identitas & Credential Helper
        subprocess.run(["git", "config", "--global", "user.email", "RayhanHaqi@github.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "RayhanHaqi"], check=True)
        subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
        print("   ✅ Identitas Git & Credential Helper berhasil diset.")
        
        # LFS Setup & Pull
        subprocess.run(["git", "lfs", "install"], check=True)
        # Menarik file LFS asli jika repositori baru saja di-clone
        subprocess.run(["git", "lfs", "pull"], check=False) # check=False karena kalau blm ada commit lfs bisa error
        print("   ✅ Git LFS siap digunakan.")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Gagal mengatur kredensial Git: {e}")

if __name__ == "__main__":
    print("==================================================")
    print("🚀 MEMULAI SETUP ENVIRONMENT DETEKSI DIGIT")
    print("==================================================")

    # Direktori target ekstraksi
    extract_dir = "datasets"
    dataset_tar = "datasets.tar.gz"

    if check_dataset_exists(extract_dir):
        print(f"\n📦 Dataset sudah ada dan lengkap di folder '{extract_dir}'. Melewati ekstraksi.")
    else:
        if not os.path.exists(dataset_tar):
            print(f"\n❌ File {dataset_tar} tidak ditemukan! Pastikan file LFS sudah di-pull.")
        else:
            # Pastikan pigz terinstal
            install_pigz()
            
            print(f"\n📦 Mengekstrak {dataset_tar} menggunakan pigz (Multi-core Extraction)...")
            os.makedirs(extract_dir, exist_ok=True)
            try:
                # Menggunakan tar dengan command -I pigz untuk dekompresi paralel
                subprocess.run(["tar", "-I", "pigz", "-xf", dataset_tar, "-C", extract_dir], check=True)
                
                print("\n   ✅ Ekstraksi selesai dengan sukses dan cepat!")
                os.remove(dataset_tar)
                print("   🧹 File .tar.gz telah dihapus untuk menghemat ruang.")

            except Exception as e:
                print(f"\n   ❌ Terjadi kesalahan saat ekstraksi dengan pigz: {e}")

    # Memanggil konfigurasi Git LFS & OS
    setup_git_credentials()

    # --- FUNGSI BARU: CHMOD +X UNTUK SCRIPT BASH ---
    print("\n📜 6. Memeriksa script eksekusi (train.sh & train-runpod.sh)...")
    
    scripts = ["train.sh", "train-runpod.sh"]
    for script_name in scripts:
        if os.path.exists(script_name):
            try:
                subprocess.run(["chmod", "+x", script_name], check=True)
                print(f"   ✅ Izin eksekusi otomatis ditambahkan ke {script_name}.")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Gagal menambahkan izin eksekusi ke {script_name}: {e}")
        else:
            print(f"   ℹ️ Script {script_name} belum ada, melewati tahap chmod.")

    print("\n🎉 SETUP SELESAI! Kamu siap melakukan training.")