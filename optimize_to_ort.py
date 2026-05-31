"""
Convert an ONNX model into a pre-optimized ORT-format (.ort) model.

ORT format applies ONNX Runtime's graph optimizations *offline* and serializes
the result, so the target process (e.g. the Pi Zero 2 W) skips ONNX parsing and
the graph-optimization passes at every startup. This shaves real wall-clock time
off the first inference and lowers peak memory on the 512 MB board.

Run it on whichever .onnx you intend to deploy:

    python optimize_to_ort.py --input checkpoints/mobilenet_v3_3cls.onnx

The .ort file is written next to the input (same stem, .ort extension). The C++
engine detects the .ort extension and loads it with graph optimizations disabled
(they are already baked in by this script using ORT_ENABLE_ALL).

Requires: pip install onnxruntime
"""

import argparse
import os
import shutil
from pathlib import Path


def convert(input_path: str) -> str:
    from onnxruntime.tools import convert_onnx_models_to_ort

    model_path = Path(input_path).resolve()
    out_dir = model_path.parent

    # Fixed style bakes in the full (ORT_ENABLE_ALL) graph optimizations and
    # writes "<stem>.ort" next to the source model.
    convert_onnx_models_to_ort.convert_onnx_models_to_ort(
        model_path_or_dir=model_path,
        optimization_styles=[convert_onnx_models_to_ort.OptimizationStyle.Fixed],
    )

    stem = model_path.stem
    desired = out_dir / f"{stem}.ort"
    if desired.is_file():
        return str(desired)

    # Fall back to locating the generated .ort if the naming differs.
    candidates = [
        out_dir / f
        for f in os.listdir(out_dir)
        if f.startswith(stem) and f.endswith(".ort")
    ]
    if not candidates:
        raise RuntimeError(
            f"convert_onnx_models_to_ort produced no .ort file for {input_path}"
        )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    if newest != desired:
        shutil.move(str(newest), str(desired))
    return str(desired)


def main():
    p = argparse.ArgumentParser(
        description="Convert ONNX to pre-optimized ORT (.ort) format"
    )
    p.add_argument("--input", required=True, help="Path to the .onnx model")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Missing {args.input}")

    out = convert(args.input)
    print(f"Wrote ORT-format model: {out}")
    print("Deploy this .ort with the C++ engine (it loads .ort with opts disabled).")


if __name__ == "__main__":
    main()
