from ultralytics import YOLO

def main():
    model = YOLO("yolo11l.pt")
    model.train(
        data="/Users/israanhayle/Thesis/Training-V2-dataset-Roboflow/data.yaml",
        epochs=200,
        imgsz=1280,
        device="mps",
        batch=2,
        half=False,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.001,
        mixup=0.15,
        hsv_h=0.3,
        hsv_s=0.9,
        hsv_v=0.5,
        fliplr=0.5,
        flipud=0.3,
        scale=0.7,
        translate=0.3,
        degrees=45,
        cls=1.5,
        dropout=0.1,
        patience=50,
        project="11L-Base_Training_results",
        name="11L-Base",
        plots=True,
    )
main()
