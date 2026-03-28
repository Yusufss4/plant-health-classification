# C++ inference (ONNX Runtime)

This folder builds:

- **`infer_mobilenet`** — single-image inference (same as a forward pass in Python).
- **`evaluate_mobilenet`** — walks `data/test/healthy` and `data/test/diseased` (same layout as `evaluate.py`), computes accuracy, precision, recall, F1, balanced accuracy, specificity, confusion counts, and timing (load + preprocess + ORT), similar to `utils/evaluation.py` + `print_evaluation_results`.

Shared preprocessing and ONNX Runtime calls live in `mobilenet_common`.

## Contract (must match training)

- Input name: `input`, shape `[1, 3, 224, 224]`, float32, NCHW.
- Output name: `logits`, shape `[1, 2]`.
- RGB image preprocessing: resize 224×224 (bilinear), scale to `[0,1]`, then ImageNet normalize per channel:
  - mean `[0.485, 0.456, 0.406]`
  - std `[0.229, 0.224, 0.225]`
- Class indices: `0` = healthy, `1` = diseased.

## 1. Export ONNX (from repo root)

```bash
python export_mobilenet_onnx.py
```

Produces `checkpoints/mobilenet_v3.onnx`.

## 2. Install ONNX Runtime (dev machine: Linux x86_64)

Download the CPU build and point CMake at it:

```bash
bash scripts/download_onnxruntime.sh linux-x64
export ONNXRUNTIME_ROOT="$(pwd)/third_party/onnxruntime/onnxruntime-linux-x64-1.17.3"
```

Adjust `ONNXRUNTIME_ROOT` if you use a different version or install path.

## 3. Build (local)

```bash
cd cpp
cmake -B build -S .
cmake --build build
```

Run (library path may be required):

```bash
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}"
./build/infer_mobilenet ../checkpoints/mobilenet_v3.onnx /path/to/leaf.jpg
```

### Test-set evaluation (like `python evaluate.py --model mobilenet_v3`)

From repo root, with `data/test/healthy` and `data/test/diseased` populated:

```bash
./build/evaluate_mobilenet ../checkpoints/mobilenet_v3.onnx ../data/test
```

Default second argument is `data/test` if omitted (run from `cpp/`). Metrics are for the **diseased** positive class (precision/recall/F1), plus specificity for **healthy**, matching the binary sklearn defaults in Python.

### Exact parity with Python (same float tensor)

To verify the C++ runtime matches ONNX Runtime on **identical** input data (bypasses resize differences between languages):

```bash
bash scripts/validate_cpp_inference.sh /path/to/image.jpg
```

This writes a float32 tensor with `scripts/dump_ort_reference.py` (torchvision preprocessing) and runs `infer_mobilenet ... --tensor-bin ...`.

## 4. Raspberry Pi Zero 2 W (aarch64)

1. Use **64-bit** Raspberry Pi OS so you can use the `linux-aarch64` ONNX Runtime package.
2. On the Pi (or download elsewhere and copy):

   ```bash
   bash scripts/download_onnxruntime.sh linux-aarch64
   export ONNXRUNTIME_ROOT="$(pwd)/third_party/onnxruntime/onnxruntime-linux-aarch64-1.17.3"
   ```

3. Install a C++17 compiler and CMake: `sudo apt install cmake g++`.
4. Copy the repo (or only `cpp/`, `checkpoints/mobilenet_v3.onnx`, `scripts/`).
5. Build as in section 3 on the Pi.
6. Run with `LD_LIBRARY_PATH` set to `${ONNXRUNTIME_ROOT}/lib`.

**Memory:** Pi Zero 2 W has 512 MB RAM. Enable swap if linking OOMs. If inference is too heavy, export a quantized model:

```bash
python quantize_mobilenet_onnx.py --output checkpoints/mobilenet_v3_int8.onnx
```

Use `mobilenet_v3_int8.onnx` as the first argument to `infer_mobilenet`. Re-check accuracy on a few labeled images.

## 5. ABI note

Link and run against the **same** ONNX Runtime major version on the machine that builds the binary and at runtime (copy the `libonnxruntime.so*` next to the binary or set `LD_LIBRARY_PATH`).
