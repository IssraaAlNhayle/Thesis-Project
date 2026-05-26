import time, os, itertools
from ultralytics import YOLO


MODEL_PATH = "src/Training/runs/detect/11N-KD_Training_results/11N-KD/weights/best.pt"
IMGSZ      = 640


crashed = "src/Unseen-Test-Data-Roboflow/Crashed-unseen-data/test/images"
intact  = "src/Unseen-Test-Data-Roboflow/Intact-unseen-data/test/images"
images  = [os.path.join(crashed, f) for f in os.listdir(crashed)] + \
          [os.path.join(intact,  f) for f in os.listdir(intact)]
image_cycle = itertools.cycle(images)

IS_PT  = MODEL_PATH.endswith('.pt')
DEVICE = "mps" if IS_PT else "cpu"

model = YOLO(MODEL_PATH)
if IS_PT:
    model.to('mps')

print(f"Model : {MODEL_PATH}")
print(f"imgsz : {IMGSZ}")
print(f"Device: {'mps' if IS_PT else 'CoreML/ANE'}")
print(f"\nStart powermetrics in the other terminal NOW, then press Enter to begin.")
input()

results = []

for run in range(1, 6):
    print(f"\n--- Run {run}/5 ---")
    count = 0
    start = time.perf_counter()
    while True:
        model.predict(next(image_cycle), imgsz=IMGSZ, device=DEVICE, verbose=False)
        count += 1
        elapsed = time.perf_counter() - start
        if elapsed >= 10.0 and count >= 10:
            break
    ips = count / elapsed
    results.append((count, elapsed))
    print(f"  Inferences: {count}  |  Duration: {elapsed:.1f}s  |  IPS: {ips:.2f}")
    print(f"  → Note the avg power from powermetrics for this run window")

print(f"\n--- DONE — stop powermetrics now ---")
print(f"\nFor each run, calculate:")
print(f"  Energy/inference (µJ) = (avg_total_power_mW × duration_s × 1000) / inferences")
print(f"  Then take the MEDIAN of the 5 values.")
print(f"\nRun durations: {[f'{r[1]:.1f}s ({r[0]} iters)' for r in results]}")
