#!/usr/bin/env python3
"""
Landseer adapter: PyTorch (standard form) <-> TensorFlow/Keras.

Input direction (runs before a TensorFlow tool):
    /input/config_model.py [+ /input/model.pt]  ->  /output/model_tf/model.keras
                                                     /output/model_tf/weight_map.json

    The network defined by config_model.py is converted to a trainable Keras
    model with nobuco (layer by layer, so the tool can train it). Which Keras
    weight holds which PyTorch tensor, and with which axis permutation, is
    found by converting once with unique marker values in every tensor and
    matching them; that map is written to weight_map.json for the output
    direction. The real weights (model.pt, or the fresh initialisation of
    config_model.py) are then copied in and both models are compared on a
    batch of the task's data.

Output direction (runs after the tool):
    /input/model_tf/model.keras (trained by the tool) + /input/model_tf/weight_map.json
        ->  /output/model.pt   (state_dict for config_model.config(), as PyTorch tools write it)
            /output/conversion_report.json

Both directions fail (non-zero exit) when the converted model does not match
the source model within tolerance.
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MAP_FORMAT = "landseer-torch-keras-map/1"


def log(msg):
    print(f"[adapter] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Landseer PyTorch <-> TensorFlow adapter")
    parser.add_argument("--direction", choices=["input", "output"],
                        default=os.environ.get("LANDSEER_ADAPTER", "input"),
                        help="input: PyTorch -> Keras before the tool; output: Keras -> PyTorch after it")
    parser.add_argument("--input", default=os.environ.get("INPUT_DIR", "/input"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--model-dir", default="model_tf",
                        help="Directory (relative to input/output) holding the Keras model and weight map")
    parser.add_argument("--weights", choices=["auto", "none", "required"], default="auto",
                        help="input direction: load /input/model.pt (auto: if present; none: keep the "
                             "fresh initialisation of config_model.py)")
    parser.add_argument("--input-shape", default="",
                        help="Per-sample input shape C,H,W (default: from data.npy, else 3,32,32)")
    parser.add_argument("--trace-batch", type=int, default=11,
                        help="Batch size used while tracing; reshapes with this leading dim become dynamic")
    parser.add_argument("--check-samples", type=int, default=256, help="Samples used for the parity check")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_config_model(input_dir: Path):
    """Import config_model.config() from the task input (or /app, where workers also mount it)."""
    import importlib.util
    for candidate in (input_dir / "config_model.py", Path("/app/config_model.py")):
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("config_model", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            log(f"model architecture from {candidate}")
            return module.config
    raise SystemExit(f"config_model.py not found in {input_dir} or /app")


def to_nchw_float(x: np.ndarray) -> np.ndarray:
    """Coerce an image batch to NCHW float32 in [0, 1] (the standard dataset layout)."""
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def sample_batch(input_dir: Path, n: int, input_shape, seed: int):
    """(x, y) from test_data.npy / data.npy (first n samples), or random inputs when absent."""
    for data_name, label_name in (("test_data.npy", "test_labels.npy"), ("data.npy", "labels.npy")):
        data_path = input_dir / data_name
        if data_path.exists():
            data = np.load(data_path, mmap_mode="r")
            x = to_nchw_float(np.array(data[:n]))
            y = None
            if (input_dir / label_name).exists():
                y = np.array(np.load(input_dir / label_name, mmap_mode="r")[:n]).astype(np.int64)
            log(f"parity check on {len(x)} samples from {data_name}")
            return x, y
    rng = np.random.default_rng(seed)
    log(f"no dataset in input; parity check on {n} random inputs")
    return rng.random((n, *input_shape), dtype=np.float32), None


def infer_input_shape(input_dir: Path, arg: str):
    if arg:
        return tuple(int(v) for v in arg.split(","))
    for name in ("data.npy", "test_data.npy"):
        path = input_dir / name
        if path.exists():
            return tuple(to_nchw_float(np.array(np.load(path, mmap_mode="r")[:1])).shape[1:])
    return (3, 32, 32)


def unwrap_state_dict(loaded):
    """Accept the checkpoint shapes Landseer tools write (see containers/evals/common/model_loader.py)."""
    import torch
    if isinstance(loaded, torch.nn.Module):
        return loaded.state_dict()
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), torch.nn.Module):
        return loaded["model"].state_dict()
    if not isinstance(loaded, dict):
        raise SystemExit(f"Unsupported checkpoint type {type(loaded).__name__}")
    if all(k.startswith("module.") for k in loaded):
        loaded = {k[len("module."):]: v for k, v in loaded.items()}
    return loaded


def torch_logits(model, x: np.ndarray, batch: int = 128) -> np.ndarray:
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(model(torch.from_numpy(x[i:i + batch])).cpu().numpy())
    return np.concatenate(outs)


def keras_logits(model, x: np.ndarray, batch: int = 128) -> np.ndarray:
    outs = []
    for i in range(0, len(x), batch):
        outs.append(np.asarray(model(x[i:i + batch], training=False)))
    return np.concatenate(outs)


def compare(reference: np.ndarray, converted: np.ndarray, labels, atol: float, rtol: float) -> dict:
    diff = float(np.max(np.abs(reference - converted)))
    limit = atol + rtol * float(np.max(np.abs(reference)))
    report = {
        "samples": int(len(reference)),
        "max_abs_logit_diff": diff,
        "tolerance": limit,
        "prediction_agreement": float(np.mean(reference.argmax(1) == converted.argmax(1))),
        "ok": bool(diff <= limit),
    }
    if labels is not None:
        report["reference_accuracy"] = float(np.mean(reference.argmax(1) == labels))
        report["converted_accuracy"] = float(np.mean(converted.argmax(1) == labels))
    return report


# ---------------------------------------------------------------------------
# Input direction: PyTorch -> Keras
# ---------------------------------------------------------------------------

def _register_batch_aware_reshape(trace_batch: int):
    """
    Make view/reshape keep the batch dimension dynamic.

    nobuco records ``x.view(x.size(0), -1)`` with the traced batch size as a
    constant, so the Keras model would only accept that batch size. Tracing
    with an unusual batch size and replacing a leading dimension equal to it
    with -1 keeps the model batch-agnostic.
    """
    import nobuco
    import tensorflow as tf
    import torch
    from nobuco.commons import ChannelOrderingStrategy

    def batch_shape(x, shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        shape = [int(s) for s in shape]
        if shape and shape[0] == trace_batch:
            rest = shape[1:]
            if -1 in rest:
                known = int(np.prod([s for s in rest if s != -1])) or 1
                rest = [int(np.prod(x.shape[1:])) // known if s == -1 else s for s in rest]
            shape = [-1] + rest
        return shape

    @nobuco.converter(torch.Tensor.view, torch.Tensor.reshape, torch.reshape,
                      channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
    def converter_reshape(self, *shape):
        def func(self, *shape):
            return tf.reshape(self, batch_shape(self, shape))
        return func


def match_weights(keras_model, markers: dict) -> list:
    """Find, for every Keras weight, the marked PyTorch tensor and axis permutation it holds."""
    by_size = {}
    for key, value in markers.items():
        by_size.setdefault(value.size, []).append(key)
    entries, used = [], set()
    for weight in keras_model.weights:
        value = weight.numpy()
        found = None
        for key in by_size.get(value.size, []):
            if key in used:
                continue
            source = markers[key]
            for perm in itertools.permutations(range(source.ndim)):
                candidate = np.transpose(source, perm)
                if candidate.shape == value.shape and np.array_equal(candidate, value):
                    found = (key, list(perm))
                    break
            if found:
                break
        if found is None:
            raise SystemExit(f"Keras weight {weight.name} {tuple(value.shape)} does not correspond to any "
                             "PyTorch tensor (the converter changed it beyond a transpose)")
        used.add(found[0])
        entries.append({"keras": weight.name, "torch": found[0], "perm": found[1],
                        "trainable": bool(weight.trainable)})
    return entries


def run_input(args) -> int:
    import nobuco
    import tensorflow as tf
    import torch
    from nobuco import ChannelOrder

    input_dir, output_dir = Path(args.input), Path(args.output)
    model_dir = output_dir / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = load_config_model(input_dir)().eval()

    weights_path = input_dir / "model.pt"
    weights_source = "config_model.py initialisation"
    if args.weights == "required" and not weights_path.exists():
        raise SystemExit(f"--weights required but {weights_path} does not exist")
    if args.weights != "none" and weights_path.exists():
        state = unwrap_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        model.load_state_dict(state)
        weights_source = str(weights_path)
    log(f"weights: {weights_source}")

    real_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    generator = torch.Generator().manual_seed(12345)
    # Positive, distinct values (BN running_var must stay > 0 while tracing).
    markers = {k: (torch.rand(v.shape, generator=generator, dtype=torch.float32) + 0.5).to(v.dtype)
               for k, v in real_state.items() if v.is_floating_point()}
    model.load_state_dict({**real_state, **markers})

    input_shape = infer_input_shape(input_dir, args.input_shape)
    _register_batch_aware_reshape(args.trace_batch)
    dummy = torch.rand(args.trace_batch, *input_shape)
    start = time.time()
    keras_model = nobuco.pytorch_to_keras(
        model,
        args=[dummy],
        input_shapes={dummy: (None, *input_shape)},
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH,
        save_trace_html=False,
    )
    log(f"converted with nobuco in {time.time() - start:.1f}s: {len(keras_model.weights)} Keras weights, "
        f"{len(real_state)} PyTorch tensors")

    entries = match_weights(keras_model, {k: v.numpy() for k, v in markers.items()})
    mapped = {e["torch"] for e in entries}
    torch_only = sorted(k for k in real_state if k not in mapped)
    unmapped_float = [k for k in torch_only if real_state[k].is_floating_point()]
    if unmapped_float:
        log(f"WARNING: PyTorch tensors with no Keras counterpart (kept from the source model): {unmapped_float}")

    by_name = {w.name: w for w in keras_model.weights}
    for entry in entries:
        by_name[entry["keras"]].assign(np.transpose(real_state[entry["torch"]].numpy(), entry["perm"]))
    model.load_state_dict(real_state)

    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            cfg = layer.get_config()
            log(f"BatchNormalization momentum={cfg['momentum']} epsilon={cfg['epsilon']} (first BN layer)")
            break

    x, y = sample_batch(input_dir, args.check_samples, input_shape, args.seed)
    report = compare(torch_logits(model, x), keras_logits(keras_model, x), y, args.atol, args.rtol)
    log(f"parity PyTorch vs Keras: {json.dumps(report)}")
    if not report["ok"]:
        raise SystemExit("Converted Keras model does not match the PyTorch model")

    keras_model.save(model_dir / "model.keras")
    weight_map = {
        "format": MAP_FORMAT,
        "input_shape": list(input_shape),
        "channel_order": "NCHW",
        "weights_source": weights_source,
        "entries": entries,
        "torch_only": torch_only,
        "parity": report,
    }
    (model_dir / "weight_map.json").write_text(json.dumps(weight_map, indent=2))
    log(f"wrote {model_dir / 'model.keras'} and weight_map.json ({len(entries)} mapped weights)")
    return 0


# ---------------------------------------------------------------------------
# Output direction: Keras -> PyTorch
# ---------------------------------------------------------------------------

def run_output(args) -> int:
    import tensorflow as tf
    import torch

    input_dir, output_dir = Path(args.input), Path(args.output)
    model_dir = input_dir / args.model_dir
    keras_path = model_dir / "model.keras"
    map_path = model_dir / "weight_map.json"
    for path in (keras_path, map_path):
        if not path.exists():
            raise SystemExit(f"{path} not found: the tool must write {args.model_dir}/model.keras and the "
                             "input adapter must have written weight_map.json")

    weight_map = json.loads(map_path.read_text())
    if weight_map.get("format") != MAP_FORMAT:
        raise SystemExit(f"Unsupported weight map format {weight_map.get('format')!r}")

    keras_model = tf.keras.models.load_model(keras_path, compile=False)
    by_name = {w.name: w for w in keras_model.weights}
    missing = [e["keras"] for e in weight_map["entries"] if e["keras"] not in by_name]
    if missing:
        raise SystemExit(f"Trained Keras model lacks weights listed in the weight map: {missing[:5]}")

    model = load_config_model(input_dir)()
    state = model.state_dict()
    for entry in weight_map["entries"]:
        value = by_name[entry["keras"]].numpy()
        inverse = list(np.argsort(entry["perm"]))
        target = state[entry["torch"]]
        converted = torch.from_numpy(np.ascontiguousarray(np.transpose(value, inverse))).to(target.dtype)
        if converted.shape != target.shape:
            raise SystemExit(f"{entry['torch']}: shape {tuple(converted.shape)} != {tuple(target.shape)}")
        state[entry["torch"]] = converted
    model.load_state_dict(state, strict=True)
    model.eval()

    input_shape = tuple(weight_map.get("input_shape") or (3, 32, 32))
    x, y = sample_batch(input_dir, args.check_samples, input_shape, args.seed)
    report = compare(keras_logits(keras_model, x), torch_logits(model, x), y, args.atol, args.rtol)
    log(f"parity Keras vs PyTorch: {json.dumps(report)}")
    if not report["ok"]:
        raise SystemExit("Converted PyTorch model does not match the trained Keras model")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "conversion_report.json").write_text(json.dumps({
        "source": str(keras_path),
        "target": str(output_dir / "model.pt"),
        "source_framework": "tensorflow",
        "target_framework": "pytorch",
        "mapped_weights": len(weight_map["entries"]),
        "parity": report,
    }, indent=2))
    log(f"wrote {output_dir / 'model.pt'}")
    return 0


def main() -> int:
    args = parse_args()
    log(f"direction={args.direction} input={args.input} output={args.output}")
    return run_input(args) if args.direction == "input" else run_output(args)


if __name__ == "__main__":
    sys.exit(main())
