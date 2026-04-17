import os
import subprocess
import tarfile
from pathlib import Path

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
    print("\n🔧 5. Configuring Git Credentials...")
    
    try:
        subprocess.run(["git", "config", "--global", "user.email", "anon@users.noreply.github.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "anon"], check=True)
        subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
        print("   ✅ Git identity and credential helper configured.")
        
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "pull"], check=False)
        print("   ✅ Git LFS successfully configured.")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error configuring Git credentials: {e}")

def setup_environment():
    extract_dir = "datasets"
    dataset_tar = "nycu-hw2-data.tar.gz"
    url = "https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view"

    print("=======================================")
    print("🚀 AUTOMATIC ENVIRONMENT SETUP")
    print("=======================================\n")

    print("📦 1. Checking dependencies...")
    try:
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        pass

    print("\n🔍 2. Checking dataset existence...")
    if check_dataset_exists(extract_dir):
        print("   ✅ Dataset is complete. Skipping download and extraction.")
    else:
        print("   ⚠️ Dataset is not complete. Starting download...")
        
        # --- DOWNLOAD ---
        if not os.path.exists(dataset_tar):
            print("\n⬇️ 3. Downloading dataset...")
            subprocess.run(["gdown", url], check=True)
        else:
            print(f"   ✅ File {dataset_tar} is already downloaded. Skipping download.")

        # --- EKSTRAKSI DENGAN PROGRESS BAR ---
        print(f"\n🗜️ 4. Preparing Extraction (Reading tar index file)...")
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

                print(f"   📦 Found {len(members_to_extract)} file. Starting extraction...")
                
                # INI DIA MAGIC-NYA: PROGRESS BAR
                for member in tqdm(members_to_extract, desc="Extracting", unit="file"):
                    tar.extract(member, path=extract_dir)
                
            print("\n   ✅ Extraction completed successfully!")
            os.remove(dataset_tar)
            print("   🧹 File .tar.gz has been deleted.")

        except Exception as e:
            print(f"\n   ❌ Error during extraction: {e}")

    setup_git_credentials()

    print("\n📜 6. Checking script execution permissions (train.sh)...")
    script_name = "train.sh"
    if os.path.exists(script_name):
        try:
            subprocess.run(["chmod", "+x", script_name], check=True)
            print(f"   ✅ Automatic execution permissions added to {script_name}.")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error adding execution permissions: {e}")
    else:
        print(f"   ℹ️ Script {script_name} not created, skipping chmod step.")
    script_name = "train-runpod.sh"
    if os.path.exists(script_name):
        try:
            subprocess.run(["chmod", "+x", script_name], check=True)
            print(f"   ✅ Automatic execution permissions added to {script_name}.")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error adding execution permissions: {e}")
    else:
        print(f"   ℹ️ Script {script_name} not created, skipping chmod step.")
    print("\n🎉 Setup is done. Ready to run training.")

if __name__ == "__main__":
    setup_environment()
