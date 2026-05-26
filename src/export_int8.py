from ultralytics import YOLO

MODEL_PATH = "src/Training/runs/detect/11N-KD_Training_results/11N-KD/weights/best.pt"
DATA_YAML  = "Training-V2-dataset-Roboflow/data.yaml"
IMGSZ      = 1280

model = YOLO(MODEL_PATH)
model.export(format="coreml", int8=True, imgsz=IMGSZ, data=DATA_YAML)
