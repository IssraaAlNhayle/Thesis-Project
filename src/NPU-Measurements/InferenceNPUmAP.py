import os
from ultralytics import YOLO


MODEL_PATH = "/Users/israanhayle/Thesis/exports/M4-lowimgrz.mlpackage"
DATA_DIR = "/Users/israanhayle/Thesis/Unseen-Test-Data"

def evaluate_npu():

    if os.path.exists(DATA_DIR):
        os.chdir(DATA_DIR)
        print(f"✅ Working directory changed to: {os.getcwd()}")
    else:
        print(f"❌ Error: Could not find directory {DATA_DIR}")
        return

    print(f"🚀 Loading CoreML model. Inference will route through Apple's ML framework...")
    model = YOLO(MODEL_PATH)

    results = model.val(
        data="data.yaml",   
        split='val',        
        imgsz=640,         
        verbose=True
    )

    # 5. Print Results for your Thesis
    print("\n" + "="*30)
    print("      NPU (CoreML) RESULTS      ")
    print("="*30)
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    evaluate_npu()