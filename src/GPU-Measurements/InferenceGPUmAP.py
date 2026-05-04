import os
import torch
from ultralytics import YOLO

# 1. Absolute Paths - Corrected based on your screenshot
MODEL_PATH = "/Users/israanhayle/Thesis/exports/M1-best.pt"
DATA_DIR = "/Users/israanhayle/Thesis/Unseen-Test-Data"

def first_run():
    # 2. Teleport: Change working directory to where the data and YAML live
    # This ensures "test/images" in your YAML points to the correct folder
    if os.path.exists(DATA_DIR):
        os.chdir(DATA_DIR)
        print(f"✅ Working directory changed to: {os.getcwd()}")
    else:
        print(f"❌ Error: Could not find directory {DATA_DIR}")
        return

    # 3. Load the model (using your M2-best.pt variable)
    model = YOLO(MODEL_PATH)

    # 4. Force M4 GPU (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 5. Run Validation
    # We use split='val' because in your YAML, 'val' points to 'test/images'
    # 'test/images' is the folder shown in your screenshot
    results = model.val(
        data="data.yaml",   # Look for data.yaml inside Unseen-Test-Data
        device=device, 
        split='val',        # Targets the 'val: test/images' line in your YAML
        imgsz=1280,         # Kept your high-resolution setting
        verbose=True
    )

    # 6. Print Results for your Thesis
    print("\n" + "="*30)
    print("      FIRST RUN RESULTS      ")
    print("="*30)
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    first_run()