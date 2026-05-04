import torch 
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

KD_CONFIG = {
    "teacher_path": "../exports/M1-best.pt",
    "alpha": 0.7,         
    "temperature": 2.0    
}

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
    
        if overrides is None:
            overrides = {}
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        self.teacher = YOLO(KD_CONFIG["teacher_path"]).model.to(self.device).eval()
        self.alpha = float(KD_CONFIG["alpha"])
        self.temp = float(KD_CONFIG["temperature"])
        LOGGER.info(f"Distillation Engine Ready: Alpha={self.alpha}, Temp={self.temp}")

    def criterion(self, preds, batch):
        """Calculates combined YOLO (Hard) + Distillation (Soft) loss."""
        yolo_loss, loss_items = super().criterion(preds, batch)
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])
        kd_loss = 0
        for i in range(len(preds)):
            p_s = F.log_softmax(preds[i] / self.temp, dim=-1)
            p_t = F.softmax(teacher_preds[i] / self.temp, dim=-1)
            kd_loss += F.kl_div(p_s, p_t, reduction='batchmean') * (self.temp ** 2)

        total_loss = (self.alpha * yolo_loss) + ((1.0 - self.alpha) * kd_loss)
        return total_loss, loss_items

if __name__ == '__main__':
    train_args = dict(
        model='yolo11n.pt',
        data='../Data/data.yaml',
        epochs=400,
        imgsz=1280,       
        batch=2,              
        cache=False,           
        device='mps',          
        patience=50,
        project='../runs/detect/Training_results',
        name='M2_Model_Distilled',
        exist_ok=True
    )
 
    trainer = KDDetectionTrainer(overrides=train_args)
    trainer.train()