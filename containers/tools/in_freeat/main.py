#!/usr/bin/env python3
"""
Free Adversarial Training ("Adversarial Training for Free!", Shafahi et al. 2019) for Landseer.

TensorFlow implementation of the training loop of
https://github.com/ais2397/free_adv_train_fork (free_train.py), run on the
pipeline's own architecture instead of the repo's fixed WRN-28-10:

- The input adapter (pytorch_to_tensorflow) converts config_model.py into a
  trainable Keras model at /input/model_tf/model.keras.
- Each minibatch is replayed ``replay_m`` times. Every replay does one forward/
  backward pass on x + delta; the same gradients update the weights (SGD with
  momentum, weight decay on conv/dense kernels) and the perturbation
  (delta <- clip(delta + epsilon * sign(grad_delta), -epsilon, epsilon)).
  delta carries over between minibatches, as in the original.
- The trained model is written to /output/model_tf/model.keras; the output
  adapter converts it to /output/model.pt.

Data follows the Landseer contract: /input/{data,labels,test_data,test_labels}.npy,
images NCHW float32 in [0, 1]. ``--epsilon`` and ``--step_size``-style values
use the original repo's 0-255 pixel scale (default epsilon 8.0 = 8/255).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def pair(arg):
    return [float(x) for x in arg.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="Free adversarial training (TensorFlow) for Landseer")
    parser.add_argument("--input_dir", default=os.environ.get("INPUT_DIR", "/input"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--model_dir", default="model_tf",
                        help="Directory (relative to input/output) holding model.keras")
    parser.add_argument("--dataset", "-d", default="cifar10", help="Dataset name (logged only)")
    # Arguments of the original free_train.py / config.py.
    parser.add_argument("--tf_seed", type=int, default=451760341)
    parser.add_argument("--np_seed", type=int, default=216105420)
    parser.add_argument("--train_steps", type=int, default=0,
                        help="Total optimizer steps including replays (original default 80000). "
                             "0 = derive from --epochs")
    parser.add_argument("--epochs", type=float, default=26,
                        help="Passes over the training set, each minibatch replayed m times "
                             "(26 at m=8, batch 128 is ~the original 80000 steps)")
    parser.add_argument("--out_steps", "-o", type=int, default=100)
    parser.add_argument("--summary_steps", type=int, default=500)
    parser.add_argument("--train_batch_size", "-b", type=int, default=128)
    parser.add_argument("--step_size_schedule", nargs="+", type=pair, default=None,
                        help="Learning rate schedule as step,lr pairs, e.g. 0,0.1 40000,0.01 60000,0.001. "
                             "Default: 0.1, /10 at 50%% and 75%% of training (the original's "
                             "40000/60000 of 80000)")
    parser.add_argument("--weight_decay", "-w", type=float, default=0.0002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--replay_m", "-m", type=int, default=8)
    parser.add_argument("--epsilon", "-e", type=float, default=8.0, help="L-inf radius on the 0-255 scale")
    parser.add_argument("--eval_examples", type=int, default=10000)
    parser.add_argument("--eval_size", type=int, default=128)
    # Landseer additions.
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable random crop (pad 4) + horizontal flip of the original CIFAR input pipeline")
    parser.add_argument("--max_train_examples", type=int, default=0, help="Use only the first N training samples")
    return parser.parse_args()


def load_split(input_dir: Path, data_name: str, label_name: str, limit: int = 0):
    x = np.load(input_dir / data_name, mmap_mode="r")
    y = np.load(input_dir / label_name, mmap_mode="r")
    if limit:
        x, y = x[:limit], y[:limit]
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    return np.ascontiguousarray(x, dtype=np.float32), np.asarray(y).astype(np.int64).reshape(-1)


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Zero-pad 4, random 32x32 crop and random horizontal flip (cifar10_input.AugmentedCIFAR10Data)."""
    n, c, h, w = batch.shape
    padded = np.pad(batch, ((0, 0), (0, 0), (4, 4), (4, 4)))
    out = np.empty_like(batch)
    tops = rng.integers(0, 9, size=n)
    lefts = rng.integers(0, 9, size=n)
    flips = rng.random(n) < 0.5
    for i in range(n):
        img = padded[i, :, tops[i]:tops[i] + h, lefts[i]:lefts[i] + w]
        out[i] = img[:, :, ::-1] if flips[i] else img
    return out


def accuracy(model, x: np.ndarray, y: np.ndarray, batch: int) -> float:
    if len(x) == 0:
        return float("nan")
    correct = 0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch], training=False)
        correct += int(np.sum(np.argmax(np.asarray(logits), 1) == y[i:i + batch]))
    return correct / len(x)


def main() -> int:
    args = parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output)
    tf.random.set_seed(args.tf_seed)
    np.random.seed(args.np_seed)
    rng = np.random.default_rng(args.np_seed)

    model_path = input_dir / args.model_dir / "model.keras"
    if not model_path.exists():
        print(f"{model_path} not found. in_freeat trains the pipeline's config_model.py architecture and needs "
              "the pytorch_to_tensorflow input adapter (configs/tools.yaml) to provide it.", file=sys.stderr)
        return 2
    model = tf.keras.models.load_model(model_path, compile=False)

    x_train, y_train = load_split(input_dir, "data.npy", "labels.npy", args.max_train_examples)
    x_test, y_test = load_split(input_dir, "test_data.npy", "test_labels.npy", args.eval_examples)
    batch_size = min(args.train_batch_size, len(x_train))
    batches_per_epoch = len(x_train) // batch_size
    train_steps = args.train_steps or max(1, int(round(args.epochs * batches_per_epoch * args.replay_m)))

    if args.step_size_schedule:
        schedule = args.step_size_schedule
    else:
        schedule = [[0, 0.1], [int(train_steps * 0.5), 0.01], [int(train_steps * 0.75), 0.001]]
    boundaries = [int(s[0]) for s in schedule][1:]
    values = [float(s[1]) for s in schedule]
    learning_rate = (tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
                     if boundaries else values[0])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=args.momentum)

    epsilon = args.epsilon / 255.0
    pert = tf.Variable(tf.zeros((batch_size, *x_train.shape[1:]), tf.float32), trainable=False,
                       name="instance_perturbation")
    # The original decays variables named 'DW': conv and fully connected kernels, not BN parameters.
    decay_vars = [v for v in model.trainable_variables if "kernel" in v.name]

    @tf.function
    def free_step(x, y):
        with tf.GradientTape() as tape:
            tape.watch(pert)
            logits = model(tf.clip_by_value(x + pert, 0.0, 1.0), training=True)
            mean_xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            decay = tf.add_n([tf.nn.l2_loss(v) for v in decay_vars]) if decay_vars else 0.0
            total_loss = mean_xent + args.weight_decay * decay
        grads = tape.gradient(total_loss, [pert] + model.trainable_variables)
        pert.assign(tf.clip_by_value(pert + epsilon * tf.sign(grads[0]), -epsilon, epsilon))
        optimizer.apply_gradients(zip(grads[1:], model.trainable_variables))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32))
        return total_loss, acc

    print(f"\n********** free training for epsilon={args.epsilon:.1f} using m_replay={args.replay_m} **********\n")
    print(f"dataset: {args.dataset}  train samples: {len(x_train)}  test samples: {len(x_test)}  "
          f"batch: {batch_size}  train steps: {train_steps}  lr schedule: {schedule}", flush=True)

    start = time.time()
    order = rng.permutation(len(x_train))
    cursor = 0
    for step in range(train_steps):
        if step % args.replay_m == 0:
            if cursor + batch_size > len(order):
                order, cursor = rng.permutation(len(x_train)), 0
            idx = order[cursor:cursor + batch_size]
            cursor += batch_size
            x_batch = x_train[idx]
            if not args.no_augment:
                x_batch = augment(x_batch, rng)
            x_batch, y_batch = tf.constant(x_batch), tf.constant(y_train[idx])
        loss, acc = free_step(x_batch, y_batch)
        if step % args.summary_steps == 0 or step % args.out_steps == 0 or step == train_steps - 1:
            print(f"Step {step}:    ({datetime.now()})  loss {float(loss):.4f}  "
                  f"training adv-batch accuracy {float(acc) * 100:.4}%", flush=True)

    train_time = time.time() - start
    test_acc = accuracy(model, x_test, y_test, args.eval_size)
    train_acc = accuracy(model, x_train[:len(x_test)], y_train[:len(x_test)], args.eval_size)
    print(f"natural accuracy: train {train_acc * 100:.2f}%  test {test_acc * 100:.2f}%  "
          f"({train_time:.1f}s training)", flush=True)

    model_out = output_dir / args.model_dir
    model_out.mkdir(parents=True, exist_ok=True)
    model.save(model_out / "model.keras")
    (output_dir / "freeat_metrics.json").write_text(json.dumps({
        "tool": "in_freeat",
        "framework": f"tensorflow=={tf.__version__}",
        "epsilon_255": args.epsilon,
        "replay_m": args.replay_m,
        "train_steps": train_steps,
        "train_batch_size": batch_size,
        "train_samples": int(len(x_train)),
        "natural_train_accuracy": train_acc,
        "natural_test_accuracy": test_acc,
        "train_time_s": train_time,
    }, indent=2))
    print(f"saved {model_out / 'model.keras'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
