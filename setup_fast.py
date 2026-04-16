import os
import subprocess
import tarfile
import shutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

def setup_git_credentials():
    print("\n🔧 5. Mengatur Konfigurasi Kredensial Git...")
    
    try:
        # Identitas & Credential Helper
        subprocess.run(["git", "config", "--global", "user.email", "RayhanHaqi@github.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "RayhanHaqi"], check=True)
        subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
        print("   ✅ Identitas Git & Credential Helper berhasil diset.")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Terjadi kesalahan saat konfigurasi Git: {e}")

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
        
        # --- DOWNLOAD ---
        if not os.path.exists(dataset_tar):
            print("\n⬇️ 3. Mengunduh dataset...")
            subprocess.run(["gdown", url], check=True)
        else:
            print(f"   ✅ File {dataset_tar} sudah ada, lanjut ke ekstraksi.")

        # --- EKSTRAKSI DENGAN PIGZ + MULTI-THREADING ---
        print(f"\n🗜️ 4. Mempersiapkan Ekstraksi Cepat...")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            # Check and install pigz if not available
            import shutil
            if shutil.which("pigz") is None:
                print("   📥 Menginstall pigz untuk dekompresi paralel...")
                try:
                    subprocess.run(["apt-get", "update", "-qq"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    subprocess.run(["apt-get", "install", "-y", "-qq", "pigz"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL,
                                   check=True)
                    print("   ✅ pigz berhasil diinstall!")
                except subprocess.CalledProcessError:
                    print("   ⚠️ Gagal install pigz, menggunakan ekstraksi standar...")

            # Decompress with pigz (parallel decompression)
            decompressed_tar = dataset_tar.replace(".tar.gz", ".tar")
            
            if shutil.which("pigz") and not os.path.exists(decompressed_tar):
                print("   🚀 Dekompresi paralel dengan pigz...")
                subprocess.run(["pigz", "-d", "-k", dataset_tar], check=True)
                print("   ✅ Dekompresi selesai!")
                use_decompressed = True
            else:
                print("   ⚠️ Menggunakan file terkompresi langsung...")
                decompressed_tar = dataset_tar
                use_decompressed = False

            # Read tar file and prepare members
            print("   📋 Membaca index file tar...")
            tar_mode = "r:" if use_decompressed else "r:gz"
            
            with tarfile.open(decompressed_tar, tar_mode) as tar:
                all_members = tar.getmembers()
                members_to_extract = []
                
                for member in all_members:
                    parts = Path(member.name).parts
                    if len(parts) > 1:
                        member.name = str(Path(*parts[1:]))
                        
                        dest_path = os.path.abspath(os.path.join(extract_dir, member.name))
                        if not dest_path.startswith(os.path.abspath(extract_dir)):
                            continue

                        member.uid = 0; member.gid = 0
                        member.uname = "root"; member.gname = "root"
                        members_to_extract.append(member)

                print(f"   📦 Menemukan {len(members_to_extract)} file.")
                
            # Multi-threaded extraction
            print("   ⚡ Memulai ekstraksi multi-threaded...")
            from concurrent.futures import ThreadPoolExecutor
            import multiprocessing
            
            def extract_single_member(member):
                """Extract a single member from tar file"""
                with tarfile.open(decompressed_tar, tar_mode) as tar:
                    tar.extract(member, path=extract_dir)
            
            # Use optimal number of workers (don't over-parallelize)
            max_workers = min(multiprocessing.cpu_count(), 8)
            print(f"   🔧 Menggunakan {max_workers} thread paralel...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(
                    executor.map(extract_single_member, members_to_extract),
                    total=len(members_to_extract),
                    desc="Mengekstrak",
                    unit="file",
                    ncols=80
                ))
                
            print("\n   ✅ Ekstraksi selesai dengan sukses!")
            
            # Cleanup
            os.remove(dataset_tar)
            print("   🧹 File .tar.gz telah dihapus.")
            
            if use_decompressed and os.path.exists(decompressed_tar):
                os.remove(decompressed_tar)
                print("   🧹 File .tar dekompresi telah dihapus.")

        except Exception as e:
            print(f"\n   ❌ Terjadi kesalahan saat ekstraksi: {e}")

    # Memanggil konfigurasi Git LFS & OS
    setup_git_credentials()

    # --- FUNGSI BARU: CHMOD +X UNTUK SCRIPT BASH ---
    print("\n📜 6. Memeriksa script eksekusi (train.sh)...")
    script_name = "train.sh"
    if os.path.exists(script_name):
        try:
            subprocess.run(["chmod", "+x", script_name], check=True)
            print(f"   ✅ Izin eksekusi otomatis ditambahkan ke {script_name}.")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Gagal menambahkan izin eksekusi: {e}")
    else:
        print(f"   ℹ️ Script {script_name} belum dibuat, melewati tahap chmod.")

    script_name = "train-runpod.sh"
    if os.path.exists(script_name):
        try:
            subprocess.run(["chmod", "+x", script_name], check=True)
            print(f"   ✅ Izin eksekusi otomatis ditambahkan ke {script_name}.")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Gagal menambahkan izin eksekusi: {e}")
    else:
        print(f"   ℹ️ Script {script_name} belum dibuat, melewati tahap chmod.")

    print("\n🎉 Setup Selesai! Kamu siap jalankan training.")

if __name__ == "__main__":
    setup_environment()