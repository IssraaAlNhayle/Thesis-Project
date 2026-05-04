import os
import time
import torch
from ultralytics import YOLO

# 1. Absolute Paths
MODEL_PATH = "/Users/israanhayle/Thesis/exports/M1-best.pt"
DATA_DIR = "/Users/israanhayle/Thesis/Unseen-Test-Data"

# 2. Timing Variables (in seconds)
WARMUP_TIME = 5 * 60       # 5 minutes
MEASUREMENT_TIME = 5 * 60  # 5 minutes

def run_power_load():
    os.chdir(DATA_DIR)
    
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Preparing to generate load on: {device} (M4 GPU)")

    # ---------------------------------------------------------
    # PHASE 1: Thermal Warm-up (5 Minutes)
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("🔥 PHASE 1: WARM-UP (5 MINUTES)")
    print("Do not record power yet. Letting the SoC heat up...")
    print("="*40)
    
    warmup_start = time.time()
    while time.time() - warmup_start < WARMUP_TIME:
        model.predict(source="test/images", device=device, verbose=False, imgsz=1280)
        
        # Print a gentle heartbeat so you know it's working
        elapsed = int(time.time() - warmup_start)
        if elapsed % 60 == 0 and elapsed > 0:
            print(f"Warm-up: {elapsed // 60} minute(s) elapsed...")

    # ---------------------------------------------------------
    # PHASE 2: Measurement Window (5 Minutes)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚨 ACTION REQUIRED: START POWERMETRICS NOW! 🚨")
    print("🚨 Go to Terminal 2 and run the sudo command! 🚨")
    print("="*50 + "\n")
    
    # Give you 5 seconds to switch windows and hit enter
    for i in range(5, 0, -1):
        print(f"Starting measurement phase in {i}...")
        time.sleep(1)

    print("\n📊 PHASE 2: MEASURING (5 MINUTES)...")
    measure_start = time.time()
    
    while time.time() - measure_start < MEASUREMENT_TIME:
        model.predict(source="test/images", device=device, verbose=False, imgsz=1280)
        
        elapsed = int(time.time() - measure_start)
        if elapsed % 60 == 0 and elapsed > 0:
            print(f"Measurement: {elapsed // 60} minute(s) recorded...")

    print("\n✅ DONE! You can now stop the powermetrics logger.")

if __name__ == "__main__":
    run_power_load()