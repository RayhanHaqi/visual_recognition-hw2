import os
import subprocess
import tarfile
from pathlib import Path

# Coba import tqdm, jika belum ada install otomatis
try:
    from tqdm import tqdm
except ImportError:
    subprocess.run(["pip", "install", "-q", "tqdm"])
    from tqdm import tqdm

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
            subprocess.run(["apt-get", "update", "-y"], check=False, stdout=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "pigz"], check=True, stdout=subprocess.DEVNULL)
            print("   ✅ pigz berhasil diinstal!")
        except Exception as e:
            print(f"   ⚠️ Gagal menginstal pigz. Pastikan kamu memiliki akses root/sudo. Error: {e}")

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
        print("   ✅ Git LFS berhasil diinisialisasi dan siap digunakan.")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Terjadi kesalahan saat konfigurasi Git/LFS: {e}")

def setup_environment():
    extract_dir = "datasets"
    dataset_tar = "nycu-hw2-data.tar.gz"
    url = "https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view"

    print("=======================================")
    print("🚀 AUTOMATIC ENVIRONMENT SETUP")
    print("=======================================\n")

    print("📦 1. Memeriksa dependencies...")
    try:
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        pass

    print("\n🔍 2. Memeriksa keberadaan Dataset...")
    if check_dataset_exists(extract_dir):
        print("   ✅ Dataset lengkap! Melewati Download & Ekstraksi.")
    else:
        print("   ⚠️ Dataset belum lengkap. Memulai Download...")
        
        # --- DOWNLOAD MENGGUNAKAN GDOWN TETAP ADA ---
        if not os.path.exists(dataset_tar):
            print("\n⬇️ 3. Mengunduh dataset dari Google Drive...")
            subprocess.run(["gdown", url], check=True)
        else:
            print(f"   ✅ File {dataset_tar} sudah ada, lanjut ke ekstraksi.")

        # --- EKSTRAKSI DENGAN PIGZ (SUPER CEPAT) ---
        print(f"\n🗜️ 4. Mempersiapkan Ekstraksi...")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Install pigz jika belum ada
        install_pigz()

        print(f"   📦 Mengekstrak {dataset_tar} menggunakan pigz (Multi-core)...")
        try:
            # Menggunakan tar dengan opsi -I pigz dan -C untuk menentukan direktori tujuan
            # Catatan: Opsi --strip-components=1 digunakan jika struktur tar-nya memiliki folder akar tambahan (seperti logika if len(parts) > 1 di script lama)
            # Jika tar-nya langsung berisi 'train.json', hapus '--strip-components=1'
            
            # Kita gunakan cara paling aman, mengekstrak langsung ke folder datasets
            subprocess.run(["tar", "-I", "pigz", "-xf", dataset_tar, "-C", extract_dir], check=True)
            
            print("\n   ✅ Ekstraksi selesai dengan sukses!")
            os.remove(dataset_tar)
            print("   🧹 File .tar.gz telah dihapus.")

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
            print(f"   ℹ️ Script {script_name} belum dibuat, melewati tahap chmod.")

    print("\n🎉 Setup Selesai! Kamu siap jalankan training.")

if __name__ == "__main__":
    setup_environment()