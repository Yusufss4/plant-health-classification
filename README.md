# Plant Health Classification — Edge Deployment on Raspberry Pi Zero 2 W

Train a lightweight **MobileNet-v3-Small** classifier for plant leaf health, export it to **ONNX**, and run inference on a **Raspberry Pi Zero 2 W** with **ONNX Runtime (C++)**. The model predicts three classes: **healthy**, **diseased**, and **background** (non-leaf / empty frame).

The workflow is **train → export → deploy**. PyTorch handles training and validation on a development machine; the Pi runs a native C++ stack for batch evaluation, single-image inference, and optional live camera preview over HTTP.

---

## Overview

| Stage | Where it runs | What happens |
|-------|----------------|--------------|
| Data prep | Dev PC | Download PlantVillage + background images; 3-class folder layout |
| Training | Dev PC (GPU optional) | Fine-tune MobileNet-v3 on 224×224 RGB |
| Evaluation | Dev PC | Test-set metrics and confusion matrix (PyTorch) |
| Export | Dev PC | ONNX with embedded class metadata |
| Inference | Pi Zero 2 W (or PC for dev) | Preprocess → ORT → label + timing |
| Live demo | Pi (optional) | Camera + MJPEG/SSE web UI (`live_infer_web`) |

**Target hardware:** Raspberry Pi Zero 2 W (quad-core Cortex-A53, 512 MB RAM), typically with **Camera Module 3** and 64-bit Raspberry Pi OS.

**Benchmarks and deployment notes:** [`results/rpi_zero_2w_mobilenet_onnx_cpp.md`](results/rpi_zero_2w_mobilenet_onnx_cpp.md) (on-device accuracy and throughput) and [`results/edge_deployment_mobilenet_report.md`](results/edge_deployment_mobilenet_report.md).

---

## Model and classes

- **Architecture:** MobileNet-v3-Small (`torchvision`), ~2.5M parameters, ImageNet-pretrained backbone.
- **Input:** NCHW float32, shape `[1, 3, 224, 224]`, ImageNet normalization (same in Python and C++).
- **Output:** logits for 3 classes.

| Index | Label | Meaning |
|-------|--------|---------|
| 0 | `healthy` | Healthy leaf |
| 1 | `diseased` | Diseased leaf |
| 2 | `background` | Non-leaf / scene without a leaf |

Class order is fixed across Python training, checkpoint metadata, ONNX `metadata_props`, and C++ inference.

---

## Repository layout

```
plant-health-classification/
├── data/                    # train/val/test (created by prepare scripts)
├── checkpoints/             # .pth and .onnx (gitignored)
├── models/
│   ├── mobilenet_v3.py      # Backbone + factory
│   └── registry.py          # Registered models + training hyperparameters
├── utils/                   # Data loaders, metrics, plots
├── train.py                 # Train MobileNet-v3
├── evaluate.py              # PyTorch test-set evaluation
├── export_mobilenet_onnx.py # Export ONNX for C++
├── prepare_data.py          # PlantVillage download + split
├── prepare_background_data.py
├── cpp/                     # ONNX Runtime inference (see cpp/README.md)
├── scripts/                 # ORT download, deploy, parity validation
├── web/live/                # HTML embedded into live_infer_web
└── results/                 # Pi benchmarks and deployment notes
```

---

## Requirements

### Development machine (training + export)

- **Python 3.10+** recommended
- **pip** packages: see [`requirements.txt`](requirements.txt)
  - PyTorch, torchvision (MobileNet)
  - TensorFlow + TFDS (dataset download only)
  - ONNX, onnxruntime (export and parity checks)
  - scikit-learn, matplotlib, etc. (metrics and plots)
- **Disk:** several GB for PlantVillage + COCO background subset
- **GPU:** optional but speeds up training

### Build machine (C++ / cross-compile)

- **CMake 3.16+**, C++17 compiler
- **ONNX Runtime** CPU build for your host (`linux-x64`) and/or Pi (`linux-aarch64`) — [`scripts/download_onnxruntime.sh`](scripts/download_onnxruntime.sh)
- **Cross-compile to Pi:** `aarch64-linux-gnu` toolchain + sysroot — [`cpp/README.md`](cpp/README.md) and [`cpp/toolchains/rpi-aarch64/`](cpp/toolchains/rpi-aarch64/)
- **Live camera web UI:** libcamera development headers (`-DENABLE_LIBCAMERA=ON`)

### Raspberry Pi Zero 2 W

- **64-bit** Raspberry Pi OS (aarch64)
- ONNX model file and `phc_*` binaries (via [`scripts/deploy_rpi_zero2w.sh`](scripts/deploy_rpi_zero2w.sh) or a native Pi build)
- `libonnxruntime.so` on `LD_LIBRARY_PATH` or next to binaries

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone <repo-url> plant-health-classification
cd plant-health-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

**Leaf images (PlantVillage)** — maps 38 fine-grained labels to healthy vs diseased, then splits 70% / 15% / 15%:

```bash
python prepare_data.py
# Reproducible split:
python prepare_data.py --seed 42
```

**Background class** — adds `background/` under train, val, and test:

```bash
python prepare_background_data.py
```

**Verify layout:**

```bash
python prepare_data.py --verify-only
python prepare_data.py --verify-only --require-background
```

Expected structure:

```
data/
├── train/{healthy,diseased,background}/
├── val/{healthy,diseased,background}/
└── test/{healthy,diseased,background}/
```

---

## Workflow

### Train (PyTorch)

```bash
python train.py
# equivalent:
python train.py --model mobilenet_v3
```

Default hyperparameters (in [`models/registry.py`](models/registry.py)):

| Setting | Value |
|---------|--------|
| Epochs | 15 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Dropout | 0.2 |
| Checkpoint | `checkpoints/mobilenet_v3_3cls_best.pth` |

Checkpoints store `model_type`, `num_classes`, `class_names`, and weights for export and evaluation.

### Evaluate (PyTorch, dev machine)

```bash
python evaluate.py
```

Loads `checkpoints/mobilenet_v3_3cls_best.pth`, runs the test loader, prints metrics and a confusion matrix.

### Export ONNX

```bash
python export_mobilenet_onnx.py
# → checkpoints/mobilenet_v3_3cls.onnx
```

Class names and `num_classes` are written into ONNX `metadata_props` for the C++ tools.

### Build and test C++ locally (x86_64)

```bash
bash scripts/download_onnxruntime.sh linux-x64
export ONNXRUNTIME_ROOT="$(pwd)/third_party/onnxruntime/onnxruntime-linux-x64-1.24.4"

cd cpp
cmake --preset local-release
cmake --build --preset local-release
ctest --test-dir build/local-release --output-on-failure

export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}"
./build/local-release/phc_infer_mobilenet ../checkpoints/mobilenet_v3_3cls.onnx /path/to/leaf.jpg
./build/local-release/phc_evaluate_mobilenet ../checkpoints/mobilenet_v3_3cls.onnx ../data/test
```

**Python vs C++ parity** (same preprocessed tensor):

```bash
bash scripts/validate_cpp_inference.sh /path/to/leaf.jpg
```

C++ architecture, cross-compilation, unit tests, and the live web UI: **[`cpp/README.md`](cpp/README.md)**.

### Deploy to Raspberry Pi Zero 2 W

1. Cross-build with the **aarch64** ONNX Runtime package (see [`cpp/README.md`](cpp/README.md)), or build natively on the Pi.
2. Copy artifacts:

```bash
scripts/deploy_rpi_zero2w.sh --host <pi-hostname-or-ip>
```

3. On the Pi:

```bash
cd ~/phc_deploy
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH}"
./bin/phc_evaluate_mobilenet model/mobilenet_v3_3cls.onnx data/test
```

**Live preview** (requires `live_infer_web` built with libcamera):

```bash
./bin/live_infer_web ./model/mobilenet_v3_3cls.onnx --port 8080
# Open http://<pi-ip>:8080/
```

---

## End-to-end checklist

1. `python prepare_data.py [--seed 42]`
2. `python prepare_background_data.py`
3. `python train.py`
4. `python export_mobilenet_onnx.py`
5. `python evaluate.py` (optional sanity check on PC)
6. Build C++ (`cpp/`), run unit tests (`ctest`), validate parity (`scripts/validate_cpp_inference.sh`)
7. Deploy to Pi (`scripts/deploy_rpi_zero2w.sh`) and run `phc_evaluate_mobilenet` or `live_infer_web`

---

## C++ tools

| Binary | Purpose |
|--------|---------|
| `phc_infer_mobilenet` | Single image → class label |
| `phc_evaluate_mobilenet` | Folder of `healthy/` / `diseased/` / `background/` → metrics + timing |
| `live_infer_web` | Camera + in-process HTTP (MJPEG + SSE); built with `-DENABLE_LIBCAMERA=ON` |

Preprocessing (224×224, ImageNet norm, NCHW) lives in `cpp/src/preprocess/`. Inference uses a shared ORT wrapper under `cpp/src/inference_ort/`.

---

## Extending training with another backbone

The training stack is model-agnostic via a small registry:

1. Add `models/your_model.py` with `create_*_model(num_classes, ...)`.
2. Register it in [`models/registry.py`](models/registry.py) (factory + epochs, batch size, lr, dropout).
3. Train with `python train.py --model <key>` and add a matching ONNX export script and C++ preprocess path for Pi deployment.

---

## Scripts reference

| Script | Role |
|--------|------|
| [`scripts/download_onnxruntime.sh`](scripts/download_onnxruntime.sh) | Fetch ORT release (`linux-x64` or `linux-aarch64`) |
| [`scripts/validate_cpp_inference.sh`](scripts/validate_cpp_inference.sh) | Compare Python ORT vs C++ on the same tensor |
| [`scripts/deploy_rpi_zero2w.sh`](scripts/deploy_rpi_zero2w.sh) | rsync binaries, ONNX, and test data to the Pi |
| [`scripts/sync_rpi_sysroot.sh`](scripts/sync_rpi_sysroot.sh) | Sync sysroot for cross-compilation |
| [`scripts/dump_ort_reference.py`](scripts/dump_ort_reference.py) | Save NCHW tensor for C++ `--tensor-bin` tests |

---

## License and attribution

Dataset sources: **PlantVillage** (via TensorFlow Datasets) for leaf images; **COCO**-derived backgrounds for the third class. See preparation scripts for download details.
