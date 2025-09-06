
"""
Simple prep script for the Kaggle "hardhat-vest-dataset-v3".
- If local folder exists, just filter label files to keep only class 0 (helmet).
- If it doesn't exist, download via Kaggle CLI and unzip, then filter.
- Edits labels IN-PLACE (keeps empty files for negatives).

Default local path: dataset
"""

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "muhammetzahitaydn/hardhat-vest-dataset-v3"
LOCAL_DIR = Path("dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def log(msg: str):
    print(f"[prep] {msg}")

def run(cmd):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)

def ensure_dataset():
    if LOCAL_DIR.exists():
        log(f"Found local dataset at: {LOCAL_DIR}")
        return
 
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"Local dataset not found. Downloading from Kaggle: {DATASET_SLUG}")
    run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(LOCAL_DIR.parent), "-o"])

    for z in LOCAL_DIR.parent.glob("*.zip"):
        log(f"Unzipping: {z.name}")
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(LOCAL_DIR.parent)
        z.unlink(missing_ok=True)

def filter_label_file(path: Path) -> int:
    """Keep only lines where class == 0. Returns number of kept lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        # Defensive: class id can appear as '0' or '0.0'
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if cls == 0:
            parts[0] = "0"
            kept.append(" ".join(parts))

    path.write_text("\n".join(kept), encoding="utf-8")
    return len(kept)

def filter_all_labels():
    log("Filtering labels to keep only class 0 (helmet)...")
    label_files = list(LOCAL_DIR.rglob("*.txt"))
    if not label_files:
        log("No label files (.txt) found. Nothing to filter.")
        return

    total_files = 0
    total_kept = 0
    for i, lbl in enumerate(label_files, 1):
        kept = filter_label_file(lbl)
        total_files += 1
        total_kept += kept
        if i % 200 == 0:
            log(f"Processed {i} label files...")

    log(f"Done. Files processed: {total_files}, total helmet boxes kept: {total_kept}")

def main():
    try:
        ensure_dataset()
        filter_all_labels()
        log("All set!")
        log(f"Folder: {LOCAL_DIR.resolve()}")
    except subprocess.CalledProcessError as e:
        log("Command failed. Make sure Kaggle CLI is installed and authenticated.")
        log(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
