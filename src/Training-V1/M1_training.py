#code before pausing the training at epoch 18
# from ultralytics import YOLO
# def main():
#     model = YOLO("yolo11l.pt") 
#     model = YOLO("runs/detect/train/weights/last.pt")
#     model.train(
#         data="/Users/israanhayle/Movies/Plastic_Detection_Project.v1-m1-model.yolov11/data.yaml",
#         epochs=100 ,
#         imgsz=1280 ,
#         device="mps",
#         batch=2 ,
#         half=False ,
#         project="Training_results",
#         name="M1 Model" ,
#         plots=True ,
#     )
# main()

#code after resuming the training
from ultralytics import YOLO
def main():
    model = YOLO("weights/last.pt")
    model.train(resume=True)

main()

