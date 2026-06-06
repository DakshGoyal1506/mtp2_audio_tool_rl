import os
import subprocess
from huggingface_hub import snapshot_download

def main():
    repo_id = "cvssp/WavCaps"
    folder_in_repo = "Zip_files/AudioSet_SL"
    local_dir = "audioset_zips"
    extract_dir = "audioset"

    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading from {repo_id}/{folder_in_repo}...")
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{folder_in_repo}/*",
        local_dir=local_dir
    )

    zip_path = os.path.join(downloaded_path, folder_in_repo, "AudioSet_SL.zip")

    print(f"Downloaded to {downloaded_path}.")
    print(f"Extracting {zip_path} into {extract_dir}/...")

    # Use 7z to extract as it natively handles multi-part zip archives perfectly.
    # If 7z is not available, we fallback to unzip (though unzip can sometimes struggle with multi-part zips).
    try:
        subprocess.run(["7z", "x", zip_path, f"-o{extract_dir}"], check=True)
        print("Extraction complete.")
    except FileNotFoundError:
        print("7z not found. Trying with unzip...")
        # Unzip might ask for the subsequent volumes or combine them automatically depending on version.
        subprocess.run(["unzip", "-q", zip_path, "-d", extract_dir], check=True)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    main()
