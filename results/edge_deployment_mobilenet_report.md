# Edge deployment report: Plant health classification (MobileNet V3) on Raspberry Pi Zero 2 W

## 1. Introduction and motivation

Deploying machine learning models **on the edge**—close to sensors and users rather than in a data center—matters when latency, privacy, connectivity, or power budgets rule out always-on cloud inference. A common pattern is to export a trained model to a portable format and run it with a lightweight runtime on a small single-board computer or microcontroller-class device.

This work focuses on **edge deployment of a plant health classifier**: given a leaf image, the system predicts whether the plant appears **healthy** or **diseased**. The **motivation** includes:

- **Engineering**: validate that a compact CNN (**MobileNet V3**) can run end-to-end on constrained hardware (**Raspberry Pi Zero 2 W**) with acceptable accuracy and measurable throughput.
- **Learning**: understand the **role of quantization and INT8-oriented optimization** in that setting. Quantization reduces numeric precision (often from FP32 to INT8) to shrink model size and speed up CPU inference; the trade-off is potential accuracy loss. Documented results establish an **FP32 ONNX baseline** on the Pi so a later **INT8** deployment can be compared on accuracy, latency, and throughput.
- **Real-world validation**: models trained on curated leaf image collections still need to be tested **in the field**—**real leaves**, **real lighting**, and capture from a **physical camera** rather than only files from a dataset. That step reveals how well the classifier tolerates blur, exposure, angle, background clutter, and sensor noise, and it motivates iterating on capture setup and preprocessing before trusting the system in production.

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

- **Model**: **MobileNet V3** exported to **ONNX** for deployment.
- **Runtime**: **ONNX Runtime** driven from a **native (C++)** inference path so measurements reflect realistic embedded-style execution rather than an interpreted-only loop.
- **Pipeline**: **Image in → MobileNet-style preprocessing → ONNX Runtime inference → class scores / label**.
- **Camera-based UI**: a **web portal** on the device (or reachable on the local network) shows **camera view**, **live predictions**, **summary metrics**, and **throughput (e.g. FPS or images per second)**.

### 2.3 Application

In practice the app does one thing: from a picture of a leaf it guesses whether the plant looks **healthy** or **unhealthy**.

For a **demo**, you open the **web page** on your phone or laptop (on the same Wi‑Fi as the device). You show **healthy** and **unhealthy** plants to the camera—or step through example images—and everyone can see the **label** the model picked and **how many frames per second** it is running.

---

## 3. Requirements of the system, software, and application

### 3.1 Functional requirements (including architecture)


- Open a **web portal** and see **inference results** (predicted class and, where applicable, confidence).
- See **performance at a glance**: **FPS** (or equivalent throughput) and, if useful, **per-frame or rolling latency**, updated as the system runs.
- Run **single-image** or **stream/batch** inference through the same stack so lab tests and field-style runs share one pipeline.

**Architecture (logical components)**

| Layer | Responsibility |
| --- | --- |
| **Presentation** | Web UI (portal) for status, predictions, FPS, and optional history or session summary. |
| **Application / API** | Serves the UI and exposes inference status and metrics to it (e.g. HTTP or WebSocket). |
| **Inference** | Loads the **MobileNet V3 ONNX** graph, applies consistent preprocessing, runs **ONNX Runtime** on the device CPU. |
| **Acquisition (optional path)** | Feeds frames from storage or from a **camera** into the preprocessor; same graph for file replay and live capture when implemented. |
| **Quantization track** | Same topology with **INT8** weights for comparison against the **FP32** baseline—accuracy and speed measured under identical UI and metrics. |

**Offline / validation path (for benchmarking)**

- Ability to score a **fixed labeled test split** and report **classification metrics** (accuracy, balanced accuracy, precision/recall/F1 for the diseased class, specificity) and a **confusion matrix**, independent of the portal, for rigorous comparison between FP32 and INT8 builds.

### 3.2 Non-functional requirements

- **Deployability**: run on **512 MB RAM** and a **quad-core A53** CPU without a GPU on the edge device.
- **Reproducibility**: evaluation procedures and hardware context are recorded so accuracy and throughput numbers remain comparable across model variants.
- **Observability**: **accuracy** (from labeled runs) is separable from **latency / throughput** (portal and benchmarks); end-to-end timing includes load, preprocess, and runtime where applicable.

---

## 4. Designed system, software, and application

### 4.1 Design rationale

- **MobileNet V3** balances accuracy and a small parameter count (on the order of **1.5M** parameters) for edge use.
- **ONNX + ONNX Runtime** provides a **deployment-oriented** graph and a runtime tuned for multiple backends; a **native** caller keeps measurements close to how the stack would ship in production.
- **FP32 baseline first**, then **INT8** (or similar) as a controlled experiment: smaller artifacts and faster integer math on CPU, with accuracy tracked against the same test protocol.
- **Web portal** as the default way to **see results and FPS** aligns the system with operator expectations and simplifies demos and field checks.

### 4.2 Processing flow

1. **Input**: image from file, batch folder, or (when available) camera frame.
2. **Preprocess**: resize and normalize to match training/export conventions.
3. **Inference**: ONNX Runtime executes the graph on the CPU.
4. **Output**: class prediction (and optional scores); metrics feed the **portal** and any batch evaluator.
5. **Presentation**: portal displays **prediction**, **FPS / throughput**, and optional aggregates.

### 4.3 Quantization (experimental track)

A reduced-precision build (e.g. **INT8** weight quantization) targets **smaller footprint and faster inference** on the same CPU, with **accuracy and FPS** compared to the **FP32** baseline under the same preprocessing and portal-facing metrics.

---

## 5. Experimental results

The tables below summarize **benchmark-style** runs: **edge** (Raspberry Pi Zero 2 W, ONNX Runtime via native code on **FP32** ONNX) and **development workstation** (PyTorch on **CUDA**). Sample counts refer to one held-out test split of **8106** images.

### 5.1 Raspberry Pi Zero 2 W — ONNX Runtime (native), FP32

**Quantization status for this run**: **FP32** ONNX only; **INT8** was not used for these figures.

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

**End-to-end performance** (load + preprocess + runtime, 8106 images)

| Measure | Value |
| --- | --- |
| Total time | 1009.689 s |
| Avg / image | 124.561 ms |
| Throughput | 8.028 img/s |

### 5.2 Development workstation — PyTorch on GPU (reference)

**Environment** (example): Ubuntu 24.04 LTS on **WSL2**; **CPU**: AMD Ryzen 7 5800H; **GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM).

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

**Timing** (full evaluation loop)

| Measure | Value |
| --- | --- |
| Total | 10.7903 s |
| Avg / image | 1.331 ms |
| Throughput | 751.23 img/s |

---

## 6. Brief conclusion

The **Pi Zero 2 W** configuration with **FP32 ONNX** and **ONNX Runtime** achieves **strong accuracy** on the held-out test split while operating in a **CPU-bound** regime near **8 images/s** end-to-end for that benchmark. That establishes a clear baseline for **INT8** and other optimizations. **Field trials with a real camera** remain important to validate robustness beyond curated dataset images.

---

## 7. TODO / future work

- [ ] **Camera pipeline**: integrate a **live camera** path (capture, orientation, ROI) feeding the same preprocess → ONNX Runtime stack as file-based evaluation.
- [ ] **Real-world leaf trials**: systematic tests **outdoors / in greenhouses** with **natural lighting** and varied distances, not only dataset-style images.
- [ ] **Optimization**: pursue **INT8** (or similar) quantization, runtime flags, and threading tuned for Cortex-A53; re-measure **accuracy vs FPS** on device.
- [ ] **Portal hardening**: ensure **FPS**, latency, and error states are visible and actionable when inference or the camera degrades.
- [ ] **Calibration / thresholds**: optional score calibration or adjustable operating point for diseased vs healthy when moving from lab to field.
