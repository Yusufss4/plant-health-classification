# Edge deployment report: Plant health classification (MobileNet V3) on Raspberry Pi Zero 2 W

## 1. Introduction and motivation

Deploying machine learning models **on the edge**—close to sensors and users rather than in a data center—matters when latency, privacy, connectivity, or power budgets rule out always-on cloud inference. A common pattern is to export a trained model to a portable format and run it with a lightweight runtime on a small single-board computer or microcontroller-class device.

This work focuses on **edge deployment of a plant health classifier**: given a leaf image, the system predicts whether the plant appears **healthy** or **diseased**. The **motivation** is twofold:

- **Engineering**: validate that a compact CNN (**MobileNet V3**) can run end-to-end on constrained hardware (**Raspberry Pi Zero 2 W**) with acceptable accuracy and measurable throughput.
- **Learning**: understand the **role of quantization and INT8-oriented optimization** in that setting. Quantization reduces numeric precision (often from FP32 to INT8) to shrink model size and speed up CPU inference; the trade-off is potential accuracy loss. The repository supports **dynamic INT8 weight quantization** of the ONNX model as an optional next step; the results documented here establish an **FP32 ONNX baseline** on the Pi so that a future INT8 run can be compared directly for accuracy, latency, and throughput.

---

## 2. Description of the selected system, software, and application

### 2.1 Target hardware

**Raspberry Pi Zero 2 W** was chosen as a **low-cost, resource-constrained** edge node:

| Aspect | Characteristic |
| --- | --- |
| CPU | Quad-core ARM Cortex-A53 (~1 GHz class) |
| RAM | 512 MB |
| Acceleration | No dedicated NPU; inference is **CPU-bound** (a strong stress test for “worst-case” edge CPUs) |
| Role | Baseline for always-on or field-style deployments where power and cost dominate |

### 2.2 Software stack

- **Model**: **MobileNet V3** exported to **ONNX** (`mobilenet_v3.onnx`).
- **Runtime**: **ONNX Runtime (ORT)** invoked from a **C++** evaluation binary (`phc_evaluate_mobilenet`), so performance reflects a realistic native deployment path rather than a Python-only loop.
- **Pipeline**: **Load image → MobileNet-style preprocessing → ORT session run → argmax / metrics** over a labeled test folder.

### 2.3 Application

A **binary image classifier** for plant health: input is a file path (batch evaluation over a directory), output is per-image predictions and aggregate metrics (accuracy, balanced accuracy, precision/recall/F1 for the diseased class, specificity, confusion matrix). The same logical task is also evaluated on a **development machine** with **PyTorch + CUDA** for reference; numbers differ mainly because of hardware throughput, not because the task definition changes.

---

## 3. Requirements of the system, software, and application

### 3.1 Functional requirements

- Load a **MobileNet V3 ONNX** model and run **single-image inference** with preprocessing consistent with training/export.
- Evaluate on a **fixed labeled test set** (here: **8106** images under `./data/test`) and report **classification metrics** and a **confusion matrix**.
- **Optional**: support an **INT8-quantized ONNX** variant (e.g. produced by `quantize_mobilenet_onnx.py`) to study quantization effects; this report’s Pi numbers are the **FP32** baseline.

### 3.2 Non-functional requirements

- **Deployability**: run on **512 MB RAM** and a **quad-core A53** CPU without GPU.
- **Reproducibility**: documented commands and paths so results can be re-run on the same dataset and model artifacts.
- **Observability**: separate **accuracy** from **latency/throughput** (end-to-end time includes load, preprocess, and ORT).

### 3.3 Environment and artifacts

- **On device**: ONNX model under `./model/`, test data under `./data/test`, C++ binary e.g. `./bin/phc_evaluate_mobilenet`.
- **For comparison**: PyTorch checkpoint `checkpoints/mobilenet_v3_best.pth`, `evaluate.py` with **CUDA** on a capable workstation (environment example: Ubuntu 24.04 LTS on **WSL2**, NVIDIA GPU — see §5.2).

---

## 4. Designed system, software, and application

### 4.1 Design rationale

- **MobileNet V3** balances accuracy and parameter count (~1.5M parameters in the recorded checkpoint) for edge use.
- **ONNX + ORT** gives a **deployment-oriented** graph and a runtime tuned for many backends; **C++** keeps the measurement closer to production embedding than a Python script alone.
- **FP32 baseline first**, then **INT8** (dynamic weight quantization or other schemes) as a controlled experiment on **quantization effects**—smaller files, faster math on CPU, with accuracy tracked against this baseline.

### 4.2 Processing flow

1. **Input**: image files from the test directory with known labels.
2. **Preprocess**: resize/normalize as required by MobileNet (same convention as training export).
3. **Inference**: ORT executes the ONNX graph on the CPU.
4. **Output**: predicted class per image; aggregation yields global metrics and confusion counts.

### 4.3 Quantization (experimental track)

The repo can generate a **dynamic INT8 weight-quantized** ONNX model (e.g. `mobilenet_v3_int8.onnx`). That path is the intended hook for comparing **INT8 optimization** against the FP32 ONNX run on identical data and binary, isolating **precision format** as the main variable.

---

## 5. Experimental results

Raw measurement logs and commands for this deployment are kept separately in [`rpi_zero_2w_mobilenet_onnx_cpp.md`](rpi_zero_2w_mobilenet_onnx_cpp.md).

### 5.1 Raspberry Pi Zero 2 W — ONNX Runtime (C++)

**Command**

`./bin/phc_evaluate_mobilenet ./model/mobilenet_v3.onnx ./data/test`

**Quantization status (this run)**

- **FP32** model (`mobilenet_v3.onnx`). INT8 was **not** used in these Pi numbers.

**Classification metrics (8106 images)**

| Metric | Value |
| --- | --- |
| Accuracy | 0.9983 (99.83%) |
| Balanced accuracy | 0.9978 (99.78%) |
| Precision (diseased) | 0.9988 (99.88%) |
| Recall (diseased) | 0.9988 (99.88%) |
| F1 (diseased) | 0.9988 (99.88%) |
| Specificity (healthy) | 0.9968 (99.69%) |

**Confusion matrix** (rows = actual, cols = predicted)

|  | Pred: healthy | Pred: diseased |
| --- | ---:| ---:|
| **Actual: healthy** | 2215 | 7 |
| **Actual: diseased** | 7 | 5877 |

TN: 2215, FP: 7, FN: 7, TP: 5877

**End-to-end performance** (load + preprocess + ORT, 8106 images)

| Measure | Value |
| --- | --- |
| Total time | 1009.689 s |
| Avg / image | 124.561 ms |
| Throughput | 8.028 img/s |

### 5.2 Local development machine — PyTorch (`evaluate.py`) reference

**Environment** (example from this repo’s WSL2 session): Ubuntu 24.04 LTS, Linux 6.6.x, WSL2; **CPU**: AMD Ryzen 7 5800H; **GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM, driver 595.97).

**Command**

`python evaluate.py --model mobilenet_v3` — checkpoint `checkpoints/mobilenet_v3_best.pth`, **CUDA**, 8106 test samples.

**Metrics**

| Metric | Value |
| --- | --- |
| Accuracy | 0.9980 (99.80%) |
| Balanced accuracy | 0.9972 (99.72%) |
| Precision (diseased) | 0.9983 (99.83%) |
| Recall (diseased) | 0.9990 (99.90%) |
| F1 (diseased) | 0.9986 (99.86%) |
| Specificity (healthy) | 0.9955 (99.55%) |
| MCC | 0.9950 |
| ROC-AUC | 1.0000 |
| Parameters | 1,519,906 |

**Confusion matrix**

|  | Pred: healthy | Pred: diseased |
| --- | ---:| ---:|
| **Actual: healthy** | 2212 | 10 |
| **Actual: diseased** | 6 | 5878 |

TN: 2212, FP: 10, FN: 6, TP: 5878

**Timing** (full eval loop)

| Measure | Value |
| --- | --- |
| Total | 10.7903 s |
| Avg / image | 1.331 ms |
| Throughput | 751.23 img/s |

---

## 6. Brief conclusion

The **Pi Zero 2 W + FP32 ONNX + ORT (C++)** setup meets strong **accuracy** on the held-out test set while exposing a **CPU-bound latency regime** (~125 ms/image end-to-end) suitable for comparing future **INT8-quantized** models on the same pipeline. The **motivation** for quantization experiments is clear: preserve accuracy while improving **size and speed** on edge CPUs; this document’s FP32 edge run is the reference point for that comparison.
