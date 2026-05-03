import os
import json
import argparse
import numpy as np
from PIL import Image

def load_trigger(trigger_path: str, trigger_size: int):
    trig = Image.open(trigger_path).convert("RGB")
    trig = trig.resize((trigger_size, trigger_size), Image.NEAREST)
    trig = np.asarray(trig).astype(np.float32) / 255.0  # (ts, ts, 3) in [0,1]
    trig = np.transpose(trig, (2, 0, 1))               # (3, ts, ts)
    return trig

def apply_trigger_batch(X, trig_chw, bottom_right=True):
    """
    X: (N,3,H,W) float32 in [0,1]
    trig_chw: (3,ts,ts)
    """
    X2 = X.copy()
    ts = trig_chw.shape[1]
    H, W = X.shape[2], X.shape[3]

    if ts > H or ts > W:
        raise ValueError(f"Trigger size {ts} is too large for image shape {(H, W)}")

    if bottom_right:
        r0 = H - ts
        c0 = W - ts
    else:
        r0 = 0
        c0 = 0

    X2[:, :, r0:r0+ts, c0:c0+ts] = trig_chw[None, :, :, :]
    return X2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="/data", help="dir containing data.npy/labels.npy/test_data.npy/test_labels.npy")
    p.add_argument("--output", type=str, default="/output", help="output directory")
    p.add_argument("--trigger_path", type=str, default="./trigger_white.png", help="path to trigger image")
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_label", type=int, default=0)
    p.add_argument("--wm_rate", type=float, default=0.1, help="fraction of TRAIN samples to poison")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--exclude_target_class", action="store_true")
    p.add_argument("--make_wm_test_all", action="store_true", default=True,
                   help="if true: wm_test is ALL triggered")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # load clean npy
    Xtr = np.load(os.path.join(args.input_dir, "data.npy")).astype(np.float32)
    ytr = np.load(os.path.join(args.input_dir, "labels.npy")).astype(np.int64)
    Xte = np.load(os.path.join(args.input_dir, "test_data.npy")).astype(np.float32)
    yte = np.load(os.path.join(args.input_dir, "test_labels.npy")).astype(np.int64)

    # generalized shape checks
    assert Xtr.ndim == 4 and Xte.ndim == 4
    assert Xtr.shape[1] == 3 and Xte.shape[1] == 3
    assert ytr.ndim == 1 and yte.ndim == 1
    assert Xtr.shape[2:] == Xte.shape[2:], f"Train/test image size mismatch: {Xtr.shape[2:]} vs {Xte.shape[2:]}"

    # generic sanity: labels should be non-negative integers
    if ytr.min() < 0 or yte.min() < 0:
        raise ValueError(f"Labels must be non-negative. Train [{ytr.min()},{ytr.max()}], Test [{yte.min()},{yte.max()}]")

    if args.trigger_label < 0:
        raise ValueError(f"trigger_label must be non-negative, got {args.trigger_label}")

    trig = load_trigger(args.trigger_path, args.trigger_size)

    rng = np.random.default_rng(args.seed)

    # select train indices to poison
    all_idx = np.arange(len(ytr))
    if args.exclude_target_class:
        all_idx = all_idx[ytr != args.trigger_label]

    k = int(len(all_idx) * args.wm_rate)
    k = max(k, 1) if args.wm_rate > 0 else 0
    poison_idx = rng.choice(all_idx, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)

    poison_mask_tr = np.zeros(len(ytr), dtype=np.uint8)
    poison_mask_tr[poison_idx] = 1

    # create watermarked train
    Xtr_wm = Xtr.copy()
    ytr_wm = ytr.copy()

    if k > 0:
        Xtr_wm[poison_idx] = apply_trigger_batch(Xtr_wm[poison_idx], trig)
        ytr_wm[poison_idx] = args.trigger_label

    # create watermarked test (ALL triggered) for wm accuracy
    if args.make_wm_test_all:
        Xte_wm = apply_trigger_batch(Xte, trig)
        yte_wm = np.full_like(yte, fill_value=args.trigger_label)
        poison_mask_te = np.ones(len(yte), dtype=np.uint8)
        test_mode = "all_triggered"
    else:
        Xte_wm = Xte.copy()
        yte_wm = yte.copy()
        poison_mask_te = np.zeros(len(yte), dtype=np.uint8)
        test_mode = "none"

    # save outputs
    np.save(os.path.join(args.output, "data.npy"), Xtr_wm)
    np.save(os.path.join(args.output, "labels.npy"), ytr_wm)

    np.save(os.path.join(args.output, "wm_test_data.npy"), Xte_wm)
    np.save(os.path.join(args.output, "wm_test_labels.npy"), yte_wm)

    print("Saved watermarked datasets to:", args.output)
    print("Train poisoned:", int(poison_mask_tr.sum()), "/", len(ytr))
    print("Test poisoned :", int(poison_mask_te.sum()), "/", len(yte))

if __name__ == "__main__":
    main()