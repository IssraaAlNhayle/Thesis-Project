import os
import time
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "/Users/israanhayle/Thesis/exports/M4-lowimgrz.mlpackage" 
DATA_DIR = "/Users/israanhayle/Thesis/Unseen-Test-Data"

def measure_npu_latency():
    if os.path.exists(DATA_DIR):
        os.chdir(DATA_DIR)
    else:
        print(f"❌ Error: Could not find directory {DATA_DIR}")
        return

    print("🚀 Loading CoreML model. Inference will route through the Apple Neural Engine (ANE)...")
    model = YOLO(MODEL_PATH)

    print("Stretching the M4 NPU (Warm-up: 50 cycles)...")
    for _ in range(50):
        model.predict(source="test/images", verbose=False, imgsz=640)

    print("Recording Latency (Measurement: 200 cycles)...")
    latencies = []
    
    for cycle in range(200):
        start_time = time.perf_counter()

        model.predict(source="test/images", verbose=False, imgsz=640)
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        per_image_latency = total_time_ms / 84
        latencies.append(per_image_latency)
        
        if (cycle + 1) % 50 == 0:
            print(f"Completed {cycle + 1}/200 cycles...")

    # 6. Statistical Reporting for Thesis
    print("\n" + "="*40)
    print("    NPU LATENCY RESULTS (PER IMAGE)    ")
    print("="*40)
    print(f"Mean Latency:    {np.mean(latencies):.2f} ms")
    print(f"Median (P50):    {np.median(latencies):.2f} ms")
    print(f"95th Perc (P95): {np.percentile(latencies, 95):.2f} ms")
    print(f"99th Perc (P99): {np.percentile(latencies, 99):.2f} ms")
    print(f"Std Deviation:   {np.std(latencies):.2f} ms")
    print("="*40 + "\n")

if __name__ == "__main__":
    measure_npu_latency()