# Raspberry Pi Zero 2 W: MobileNetV3 (ONNX Runtime / C++) Results

## What we’re doing
- Run **on-device inference** using a **MobileNetV3 ONNX** model with **ONNX Runtime (C++)**.
- Use a simple pipeline: **load image → preprocess (MobileNet-style) → ORT inference → prediction**.
- Evaluate on a folder of labeled images to validate **deployment accuracy + throughput** on target hardware.

## Device: Raspberry Pi Zero 2 W (capabilities)
- **CPU**: Quad-core ARM Cortex-A53 (1GHz-class)
- **Memory**: 512MB RAM
- **Acceleration**: No dedicated NPU; inference is **CPU-bound** (useful baseline for “worst-case” edge performance)
- **Power/IO**: Suited for always-on, low-power deployments; Wi‑Fi for remote monitoring if needed

## Command + dataset
Command:
`./bin/phc_evaluate_mobilenet ./model/mobilenet_v3.onnx ./data/test`

Evaluated: **8106 images** from `./data/test`

## Quantization status (current)
- **Quantization**: **Not enabled in this run** — results above are for the **FP32** model (`mobilenet_v3.onnx`).
- **Optional**: The repo includes `quantize_mobilenet_onnx.py` to produce a **dynamic INT8 weight-quantized** ONNX model (e.g. `mobilenet_v3_int8.onnx`) for potentially smaller/faster CPU inference on the Pi.

## Results (ONNX / C++)
Overall metrics:
- **Accuracy**: 0.9983 (99.8273%)
- **Balanced accuracy**: 0.9978 (99.7830%)
- **Precision (diseased)**: 0.9988 (99.8810%)
- **Recall (diseased)**: 0.9988 (99.8810%)
- **F1-score (diseased)**: 0.9988 (99.8810%)
- **Specificity (healthy)**: 0.9968 (99.6850%)

Confusion matrix (rows=actual, cols=predicted):

|  | Pred: healthy | Pred: diseased |
| --- | ---:| ---:|
| **Actual: healthy** | 2215 | 7 |
| **Actual: diseased** | 7 | 5877 |

TN: 2215, FP: 7, FN: 7, TP: 5877

## Performance (end-to-end, 8106 images)
Includes **load + preprocess + ORT**:
- **Total**: 1009.689 s
- **Avg / image**: 124.561 ms
- **Throughput**: 8.028 img/s

