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

        # --- EKSTRAKSI DENGAN PROGRESS BAR ---
        print(f"\n🗜️ 4. Mempersiapkan Ekstraksi (Membaca index file tar)...")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with tarfile.open(dataset_tar, "r:gz") as tar:
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

                print(f"   📦 Menemukan {len(members_to_extract)} file. Memulai ekstraksi...")
                
                # INI DIA MAGIC-NYA: PROGRESS BAR
                for member in tqdm(members_to_extract, desc="Mengekstrak", unit="file"):
                    tar.extract(member, path=extract_dir)
                
            print("\n   ✅ Ekstraksi selesai dengan sukses!")
            os.remove(dataset_tar)
            print("   🧹 File .tar.gz telah dihapus.")

        except Exception as e:
            print(f"\n   ❌ Terjadi kesalahan saat ekstraksi: {e}")

    print("\n🎉 Setup Selesai!")

if __name__ == "__main__":
    setup_environment()