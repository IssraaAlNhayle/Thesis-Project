import time, statistics, os, itertools
from ultralytics import YOLO

MODEL_PATH = "src/Training/runs/detect/11N-Base_Training_results/11N-Base/weights/best.mlpackage"
IMGSZ      = 640


crashed = "src/Unseen-Test-Data-Roboflow/Crashed-unseen-data/test/images"
intact  = "src/Unseen-Test-Data-Roboflow/Intact-unseen-data/test/images"
images  = [os.path.join(crashed, f) for f in os.listdir(crashed)] + \
          [os.path.join(intact,  f) for f in os.listdir(intact)]
image_cycle = itertools.cycle(images)

IS_PT = MODEL_PATH.endswith('.pt')
DEVICE = "mps" if IS_PT else "cpu"

model = YOLO(MODEL_PATH)
if IS_PT:
    model.to('mps')
    print(f"Device: {next(model.model.parameters()).device}")
else:
    print("Device: CoreML (cpu — ANE eligible)")

ips_per_round = []

for run in range(1, 6):
    count = 0
    start = time.perf_counter()
    while True:
        model.predict(next(image_cycle), imgsz=IMGSZ, device=DEVICE, verbose=False)
        count += 1
        elapsed = time.perf_counter() - start
        if elapsed >= 10.0 and count >= 10:
            break
    ips = count / elapsed
    ips_per_round.append(ips)
    print(f"  Run {run}: {ips:.2f} IPS  ({count} iters in {elapsed:.1f}s)")

median_ips = statistics.median(ips_per_round)
print(f"\nModel : {MODEL_PATH}")
print(f"imgsz : {IMGSZ}")
print(f"Median IPS    : {median_ips:.2f}")
print(f"Median latency: {1000/median_ips:.1f} ms/image")