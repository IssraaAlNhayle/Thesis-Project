from ultralytics import YOLO
def main():
    model = YOLO("yolo11n.pt") 
    model.train(
    data="/Users/israanhayle/thesis/Data/data.yaml",
    epochs=100 ,
    imgsz=1280 ,
    device="mps",
    batch=2 ,
    half=False,
    project="Training_results",
    name="M0 Model" ,
    plots=True ,
     )
main()