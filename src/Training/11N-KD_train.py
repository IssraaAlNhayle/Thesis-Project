import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

TEACHER_PATH = "/Users/israanhayle/Thesis/src/Training/runs/detect/11S-Teacher_Training_results/11S-Teacher/weights/best.pt"
DATA_PATH    = "/Users/israanhayle/Thesis/Training-V2-dataset-Roboflow/data.yaml"


KD_ALPHA       = 0.5   
KD_TEMPERATURE = 2.0   



class _StudentHook:
    def __init__(self, cache, idx):
        self.cache = cache
        self.idx = idx
    def __call__(self, _, __, output):
        self.cache[self.idx] = output


class _TeacherHook:
    def __init__(self, cache, idx):
        self.cache = cache
        self.idx = idx
    def __call__(self, _, __, output):
        self.cache[self.idx] = output.detach()


class _KDLossWrapper:
    def __init__(self, original_loss, teacher_model,
                 student_cache, teacher_cache, alpha, temperature):
        self.original_loss  = original_loss
        self.teacher_model  = teacher_model
        self._student_cache = student_cache
        self._teacher_cache = teacher_cache
        self.kd_alpha       = alpha
        self.kd_temperature = temperature

    def __call__(self, batch, preds=None):
        task_loss, loss_items = self.original_loss(batch, preds)
        if preds is not None or self.teacher_model is None:
            return task_loss, loss_items

        with torch.no_grad():
            self.teacher_model(batch["img"])

        kd = self._compute_kd_loss()
        combined = (1.0 - self.kd_alpha) * task_loss + self.kd_alpha * kd
        return combined, loss_items

    def _compute_kd_loss(self):
        T     = self.kd_temperature
        total = 0.0
        n     = len(self._student_cache)

        if n == 0:
            return torch.tensor(0.0)

        for i in range(n):
            s  = self._student_cache[i].float()   
            t  = self._teacher_cache[i].float()   
            nc = s.shape[1]

            s_flat  = s.permute(0, 2, 3, 1).reshape(-1, nc)  
            t_flat  = t.permute(0, 2, 3, 1).reshape(-1, nc)

            s_log  = F.log_softmax(s_flat / T, dim=1)
            t_soft = F.softmax(t_flat / T, dim=1)

            total = total + F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)

        return total / n

class KDDetectionTrainer(DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.kd_alpha       = KD_ALPHA
        self.kd_temperature = KD_TEMPERATURE
        self.teacher_model  = None
        self._student_cache = {}
        self._teacher_cache = {}

    def _setup_train(self):
        """
        Override _setup_train so KD is attached AFTER super() creates ModelEMA.
        EMA deepcopies the clean model — our KD wrapper is added afterwards.
        """
        super()._setup_train()          
        self._load_teacher()
        self._register_hooks()
        self._wrap_loss()

    def _load_teacher(self):
        teacher = YOLO(TEACHER_PATH).model
        teacher = teacher.float().to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher_model = teacher
        print(f"[KD] Teacher loaded  ▸  alpha={self.kd_alpha}, T={self.kd_temperature}")

    def _register_hooks(self):
        s_detect = self.model.model[-1]
        t_detect = self.teacher_model.model[-1]

        for i in range(len(s_detect.cv3)):
            s_detect.cv3[i].register_forward_hook(_StudentHook(self._student_cache, i))
            t_detect.cv3[i].register_forward_hook(_TeacherHook(self._teacher_cache, i))

        print(f"[KD] Hooks registered on {len(s_detect.cv3)} detection scales.")

    def _wrap_loss(self):
        self.model.loss = _KDLossWrapper(
            self.model.loss,
            self.teacher_model,
            self._student_cache,
            self._teacher_cache,
            self.kd_alpha,
            self.kd_temperature,
        )



def main():
    overrides = {
        "model":         "yolo11n.pt",
        "data":          DATA_PATH,
        "epochs":        200,
        "imgsz":         1280,
        "device":        "mps",
        "batch":         4,
        "half":          False,
        "cos_lr":        True,
        "warmup_epochs": 5,
        "weight_decay":  0.001,
        "mixup":         0.15,
        "hsv_h":         0.3,
        "hsv_s":         0.9,
        "hsv_v":         0.5,
        "fliplr":        0.5,
        "flipud":        0.3,
        "scale":         0.7,
        "translate":     0.3,
        "degrees":       45,
        "cls":           1.5,
        "patience":      50,
        "project":       "11N-KD_Training_results",
        "name":          "11N-KD",
        "plots":         True,
    }

    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
