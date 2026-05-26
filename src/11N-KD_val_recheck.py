from ultralytics import YOLO

MODEL_PATH = "src/Training/runs/detect/11N-KD_Training_results/11N-KD/weights/best.pt"
DATA_PATH  = "Training-V2-dataset-Roboflow/data.yaml"

model = YOLO(MODEL_PATH)
metrics = model.val(
    data=DATA_PATH,
    imgsz=1280,
    device="mps",
    split="val",
    plots=True,
    save_json=False,
    project="runs/detect",
    name="11N-KD_val_recheck",
)

print(f"\nmAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")
print(f"Per-class mAP50: {metrics.box.maps}")
