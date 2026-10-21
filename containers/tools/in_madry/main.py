#!/usr/bin/env python3
"""
PGD adversarial training ("Towards Deep Learning Models Resistant to Adversarial Attacks",
Madry et al. 2018) for Landseer.

TensorFlow 2 port of the training loop of the official code
https://github.com/MadryLab/cifar10_challenge (train.py, pgd_attack.LinfPGDAttack,
cifar10_input.DataSubset/AugmentedCIFAR10Data at f15682d), run on the pipeline's own
architecture instead of the repo's fixed wide ResNet:

- The input adapter (pytorch_to_tensorflow, selected by the image's model_input=architecture
  label) converts config_model.py into a trainable Keras model at /input/model_tf/model.keras.
- Every step: draw a batch (reshuffle when fewer than a batch remain), augment it (zero-pad
  2 pixels per side, random crop, random horizontal flip), replace it by its PGD adversarial
  example (random start in the epsilon ball, ``num_steps`` signed-gradient steps of
  ``step_size`` on the summed cross-entropy, projected onto the ball and the valid pixel
  range) and take one momentum SGD step on mean cross-entropy + weight_decay * sum of
  l2_loss over conv/dense kernels (the original's 'DW' variables).
- As in the original, the attack runs on the model in training mode, so batch norm uses
  batch statistics and updates its moving averages during attack steps too
  (tf.contrib.layers.batch_norm with updates_collections=None).
- The optimizer is tf.compat.v1.train.MomentumOptimizer, the original's optimizer.
- The trained model is written to /output/model_tf/model.keras; the output adapter
  converts it to /output/model.pt.

Data follows the Landseer contract: /input/{data,labels,test_data,test_labels}.npy, images
NCHW float32 in [0, 1]. ``--epsilon`` and ``--step_size`` use the original's 0-255 pixel
scale. Defaults are the paper's CIFAR-10 setting (Sec. 5: 7 steps of size 2, epsilon 8;
the repository's config.json ships num_steps 10).
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
    parser = argparse.ArgumentParser(description="PGD adversarial training (Madry et al.) for Landseer")
    parser.add_argument("--input_dir", default=os.environ.get("INPUT_DIR", "/input"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--model_dir", default="model_tf",
                        help="Directory (relative to input/output) holding model.keras")
    parser.add_argument("--dataset", "-d", default="cifar10", help="Dataset name (logged only)")
    # Parameters of the original config.json.
    parser.add_argument("--tf_random_seed", type=int, default=451760341)
    parser.add_argument("--np_random_seed", type=int, default=216105420)
    parser.add_argument("--max_num_training_steps", type=int, default=80000)
    parser.add_argument("--num_output_steps", type=int, default=100)
    parser.add_argument("--training_batch_size", type=int, default=128)
    parser.add_argument("--step_size_schedule", nargs="+", type=pair, default=None,
                        help="Learning rate schedule as step,lr pairs, e.g. 0,0.1 40000,0.01 60000,0.001. "
                             "Default: 0.1, /10 at 50%% and 75%% of training (the original's "
                             "40000/60000 of 80000)")
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=8.0, help="L-inf radius on the 0-255 scale")
    parser.add_argument("--num_steps", type=int, default=7, help="PGD steps during training")
    parser.add_argument("--step_size", type=float, default=2.0, help="PGD step size on the 0-255 scale")
    parser.add_argument("--no_random_start", action="store_true", help="Start PGD at the clean input")
    parser.add_argument("--loss_func", choices=["xent", "cw"], default="xent")
    parser.add_argument("--num_eval_examples", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    # Landseer additions.
    parser.add_argument("--epochs", type=float, default=0,
                        help="If > 0, train for this many passes over the data instead of "
                             "--max_num_training_steps")
    parser.add_argument("--eval_num_steps", type=int, default=20,
                        help="PGD steps for the final robust accuracy in madry_metrics.json "
                             "(the paper's strongest PGD adversary; 0 = skip)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable random crop (pad 2) + horizontal flip of the original input pipeline")
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


class DataSubset:
    """cifar10_input.DataSubset.get_next_batch(multiple_passes=True): reshuffle when fewer than a batch remain."""

    def __init__(self, xs, ys):
        self.xs, self.ys, self.n = xs, ys, len(xs)
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size):
        if self.n - self.batch_start < batch_size:
            self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        idx = self.cur_order[self.batch_start:self.batch_start + batch_size]
        self.batch_start += batch_size
        return self.xs[idx], self.ys[idx]


def augment(batch: np.ndarray) -> np.ndarray:
    """cifar10_input.AugmentedCIFAR10Data: resize_image_with_crop_or_pad to +4 (2 per side), random crop, random flip."""
    n, _, h, w = batch.shape
    padded = np.pad(batch, ((0, 0), (0, 0), (2, 2), (2, 2)))
    tops = np.random.randint(0, 5, size=n)
    lefts = np.random.randint(0, 5, size=n)
    flips = np.random.rand(n) < 0.5
    out = np.empty_like(batch)
    for i in range(n):
        img = padded[i, :, tops[i]:tops[i] + h, lefts[i]:lefts[i] + w]
        out[i] = img[:, :, ::-1] if flips[i] else img
    return out


def make_attack(model, epsilon, num_steps, step_size, random_start, loss_func, training):
    """pgd_attack.LinfPGDAttack.perturb on [0, 1] images; ``training`` selects the batch-norm mode."""

    def loss_fn(x, y):
        logits = model(x, training=training)
        if loss_func == "xent":  # model.xent: summed cross-entropy
            return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        mask = tf.one_hot(y, tf.shape(logits)[1], dtype=logits.dtype)
        correct_logit = tf.reduce_sum(mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1 - mask) * logits - 1e4 * mask, axis=1)
        return tf.reduce_sum(-tf.nn.relu(correct_logit - wrong_logit + 50))

    @tf.function
    def perturb_steps(x_nat, x, y):
        for _ in tf.range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = loss_fn(x, y)
            x = x + step_size * tf.sign(tape.gradient(loss, x))
            x = tf.clip_by_value(x, x_nat - epsilon, x_nat + epsilon)
            x = tf.clip_by_value(x, 0.0, 1.0)
        return x

    def perturb(x_nat, y):
        x = x_nat
        if random_start:
            x = np.clip(x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape).astype(np.float32), 0.0, 1.0)
        return perturb_steps(tf.constant(x_nat), tf.constant(x), tf.constant(y)).numpy()

    return perturb


def accuracy(model, x, y, batch, attack=None) -> float:
    if len(x) == 0:
        return float("nan")
    correct = 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        if attack is not None:
            xb = attack(xb, yb)
        correct += int(np.sum(np.argmax(np.asarray(model(xb, training=False)), 1) == yb))
    return correct / len(x)


def main() -> int:
    args = parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output)
    tf.random.set_seed(args.tf_random_seed)
    np.random.seed(args.np_random_seed)

    model_path = input_dir / args.model_dir / "model.keras"
    if not model_path.exists():
        print(f"{model_path} not found. in_madry trains the pipeline's config_model.py architecture and needs "
              "the pytorch_to_tensorflow input adapter (configs/tools.yaml) to provide it.", file=sys.stderr)
        return 2
    model = tf.keras.models.load_model(model_path, compile=False)

    x_train, y_train = load_split(input_dir, "data.npy", "labels.npy", args.max_train_examples)
    x_test, y_test = load_split(input_dir, "test_data.npy", "test_labels.npy", args.num_eval_examples)
    batch_size = min(args.training_batch_size, len(x_train))
    train_steps = (max(1, int(round(args.epochs * len(x_train) / batch_size))) if args.epochs > 0
                   else args.max_num_training_steps)
    schedule = args.step_size_schedule or [[0, 0.1], [int(train_steps * 0.5), 0.01], [int(train_steps * 0.75), 0.001]]
    boundaries = [int(s[0]) for s in schedule][1:]
    values = [float(s[1]) for s in schedule]

    def lr_at(step):
        return values[sum(step > b for b in boundaries)]  # tf.train.piecewise_constant: values[i] up to and incl. b_i

    learning_rate = tf.Variable(lr_at(0), trainable=False, dtype=tf.float32, name="learning_rate")
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, args.momentum)
    # The original decays variables named 'DW': conv and fully connected kernels, not biases or BN parameters.
    decay_vars = [v for v in model.trainable_variables if "kernel" in v.name]

    epsilon, step_size = args.epsilon / 255.0, args.step_size / 255.0
    train_attack = make_attack(model, epsilon, args.num_steps, step_size, not args.no_random_start,
                               args.loss_func, training=True)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            mean_xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            decay = tf.add_n([tf.nn.l2_loss(v) for v in decay_vars]) if decay_vars else 0.0
            total_loss = mean_xent + args.weight_decay * decay
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    @tf.function
    def batch_accuracy(x, y):  # model.accuracy, evaluated with the training-mode model as in train.py
        logits = model(x, training=True)
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32))

    print(f"\n********** PGD adversarial training: epsilon={args.epsilon:.1f}, {args.num_steps} steps of "
          f"size {args.step_size:.1f}, loss={args.loss_func} **********\n")
    print(f"dataset: {args.dataset}  train samples: {len(x_train)}  test samples: {len(x_test)}  batch: {batch_size}  "
          f"train steps: {train_steps}  lr schedule: {schedule}  decayed kernels: {len(decay_vars)}", flush=True)

    data = DataSubset(x_train, y_train)
    start, attack_time = time.time(), 0.0
    for step in range(train_steps):
        x_batch, y_batch = data.get_next_batch(batch_size)
        if not args.no_augment:
            x_batch = augment(x_batch)
        t0 = time.time()
        x_batch_adv = train_attack(x_batch, y_batch)
        attack_time += time.time() - t0
        if step % args.num_output_steps == 0 or step == train_steps - 1:
            nat_acc = float(batch_accuracy(x_batch, y_batch))
            adv_acc = float(batch_accuracy(x_batch_adv, y_batch))
            print(f"Step {step}:    ({datetime.now()})  lr {lr_at(step):g}  training nat accuracy "
                  f"{nat_acc * 100:.4}%  training adv accuracy {adv_acc * 100:.4}%", flush=True)
        learning_rate.assign(lr_at(step))
        train_step(tf.constant(x_batch_adv), tf.constant(y_batch))

    train_time = time.time() - start
    natural_test = accuracy(model, x_test, y_test, args.eval_batch_size)
    natural_train = accuracy(model, x_train[:len(x_test)], y_train[:len(x_test)], args.eval_batch_size)
    pgd_test = None
    if args.eval_num_steps:
        eval_attack = make_attack(model, epsilon, args.eval_num_steps, step_size, True, "xent", training=False)
        pgd_test = accuracy(model, x_test, y_test, args.eval_batch_size, attack=eval_attack)
    print(f"natural accuracy: train {natural_train * 100:.2f}%  test {natural_test * 100:.2f}%  "
          f"PGD-{args.eval_num_steps} test {'n/a' if pgd_test is None else f'{pgd_test * 100:.2f}%'}  "
          f"({train_time:.1f}s training, {attack_time:.1f}s in the attack)", flush=True)

    model_out = output_dir / args.model_dir
    model_out.mkdir(parents=True, exist_ok=True)
    model.save(model_out / "model.keras")
    (output_dir / "madry_metrics.json").write_text(json.dumps({
        "tool": "in_madry",
        "framework": f"tensorflow=={tf.__version__}",
        "epsilon_255": args.epsilon,
        "num_steps": args.num_steps,
        "step_size_255": args.step_size,
        "random_start": not args.no_random_start,
        "loss_func": args.loss_func,
        "train_steps": train_steps,
        "train_batch_size": batch_size,
        "train_samples": int(len(x_train)),
        "lr_schedule": schedule,
        "natural_train_accuracy": natural_train,
        "natural_test_accuracy": natural_test,
        "eval_pgd_steps": args.eval_num_steps,
        "pgd_test_accuracy": pgd_test,
        "eval_examples": int(len(x_test)),
        "train_time_s": train_time,
    }, indent=2))
    print(f"saved {model_out / 'model.keras'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
