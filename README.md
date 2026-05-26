# Energy-Efficient Real-Time Plastic Identification

Companion code for the bachelor thesis *"Energy-Efficient Real-Time Plastic Identification: A Comparative Trade-off Analysis of GPU-based and NPU-optimized Inference"* (IU International University of Applied Sciences, 2026).

The thesis evaluates the sequential application of **Knowledge Distillation (KD)**, **Post-Training Quantization (PTQ)** and **input image resolution downscaling** on a YOLO11n object detection model deployed for plastic bottle classification (PET_Pfand, PET_no_Pfand, HDPE) on Apple Silicon.

## Repository structure

```
src/
├── Training/
│   ├── 11L-Base_train.py        # YOLO11l baseline (OOM at epoch 118)
│   ├── 11S-Teacher_train.py     # YOLO11s teacher
│   ├── 11N-Base_train.py        # YOLO11n baseline
│   └── 11N-KD_train.py          # YOLO11n student with response-based KD (custom DetectionTrainer)
├── 11N-Base_val_recheck.py      # Re-validate nano baseline at imgsz=1280
├── 11N-KD_val_recheck.py        # Re-validate KD student at imgsz=1280
├── export_int8.py               # PTQ export to CoreML INT8 via Ultralytics + coremltools
├── latency_test.py              # Five-run median IPS / latency benchmark (MLPerf Tiny style)
├── power_test.py                # Inference loop synchronized with macOS powermetrics
├── parse_power.py               # Parse CPU / GPU / ANE power from powermetrics log
└── pareto_front.py              # Multi-objective Pareto front visualization
```

## Model nomenclature (Tab. 2 in the thesis)

| Name              | Description                              |
|-------------------|------------------------------------------|
| `11L-Base`        | YOLO11 Large baseline (1280 px)          |
| `11S-Teacher`     | YOLO11 Small / KD teacher (1280 px)      |
| `11N-Base`        | YOLO11 Nano baseline (1280 px)           |
| `11N-KD`          | YOLO11 Nano student distilled from `11S-Teacher` |

The eight evaluated nano variants combine `{Base, KD} x {FP32, INT8} x {1280px, 640px}`.

## Environment

- macOS Sequoia 15.7.4
- Python 3.12.7, PyTorch 2.11.0, Ultralytics, coremltools
- Apple M4 (MacBook Air, 16 GB unified memory), MPS backend

## Dataset

Custom dataset (~1,100 annotated bottle images, 3 classes) curated and hosted on Roboflow. Dataset access and configuration are managed via the Roboflow project linked in the thesis (Appendix B).

## Notes on reproducibility

- All training runs used a fixed random seed (`seed=1`) — see thesis Limitations (section 3.5).
- The KD hyperparameters in `11N-KD_train.py` are `KD_ALPHA = 0.5` and `KD_TEMPERATURE = 2.0`, following Hinton et al. (2015).
- Absolute paths inside the scripts point to the author's local directory layout. Update `TEACHER_PATH`, `MODEL_PATH` and `DATA_PATH` to match your local environment before running.
- Pretrained YOLO11 weights (`yolo11n.pt`, `yolo11s.pt`, `yolo11l.pt`) are downloaded automatically by Ultralytics on first use and are not committed to this repository.

## Reference

Al Nhayle, I. (2026). *Energy-Efficient Real-Time Plastic Identification: A Comparative Trade-off Analysis of GPU-based and NPU-optimized Inference* [Bachelor thesis]. IU International University of Applied Sciences.
