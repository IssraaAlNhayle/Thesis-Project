import os
import time
import torch
import numpy as np
from ultralytics import YOLO

# 1. Absolute Paths
MODEL_PATH = "/Users/israanhayle/Thesis/exports/best-M0.pt"
DATA_DIR = "/Users/israanhayle/Thesis/Unseen-Test-Data"

def measure_latency():
    # 2. Teleport to the Data Directory
    os.chdir(DATA_DIR)
    
    # 3. Load Model and Force MPS
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Benchmarking on: {device} (M4 GPU)")

    # 4. Phase 1: Warm-up (50 Cycles)
    # These are discarded to avoid recording "cold" start times
    print("Stretching the M4 (Warm-up: 50 cycles)...")
    for _ in range(50):
        # We use stream=True for memory efficiency in large loops
        model.predict(source="test/images", device=device, verbose=False, imgsz=1280)

    # 5. Phase 2: Measurement (200 Cycles)
    print("Recording Latency (Measurement: 200 cycles)...")
    latencies = []
    
    for cycle in range(200):
        # Measure the time for a full pass over the 84 images
        start_time = time.perf_counter()
        
        # Inference
        model.predict(source="test/images", device=device, verbose=False, imgsz=1280)
        
        # Calculate per-image latency for this pass
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        per_image_latency = total_time_ms / 84
        latencies.append(per_image_latency)
        
        if (cycle + 1) % 50 == 0:
            print(f"Completed {cycle + 1}/200 cycles...")

    # 6. Statistical Reporting for Thesis
    print("\n" + "="*40)
    print("      LATENCY RESULTS (PER IMAGE)      ")
    print("="*40)
    print(f"Mean Latency:    {np.mean(latencies):.2f} ms")
    print(f"Median (P50):    {np.median(latencies):.2f} ms")
    print(f"95th Perc (P95): {np.percentile(latencies, 95):.2f} ms")
    print(f"99th Perc (P99): {np.percentile(latencies, 99):.2f} ms")
    print(f"Std Deviation:   {np.std(latencies):.2f} ms")
    print("="*40 + "\n")

if __name__ == "__main__":
    measure_latency()