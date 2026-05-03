#!/usr/bin/env python3
import os
import argparse
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt

from config_model import config

import numpy as np
import torch

def spatial_importance_from_shap(shap_vals: np.ndarray, class_idx: int) -> np.ndarray:
    """
    shap_vals: numpy array (B, C, H, W, K)
    returns: spatial importance map (B, H, W) using sum of abs across channels for class_idx
    """
    # (B, C, H, W)
    sv = shap_vals[..., class_idx]
    # importance (B, H, W)
    imp = np.abs(sv).sum(axis=1)
    return imp

def mask_top_p_percent(images: torch.Tensor, importance: np.ndarray, p: float, baseline: float = 0.0) -> torch.Tensor:
    """
    images: torch Tensor (B, C, H, W) on device
    importance: numpy array (B, H, W)
    p: fraction in [0,1], e.g. 0.1 for top 10%
    baseline: value to replace masked pixels with

    returns: masked_images (B, C, H, W) torch Tensor on same device
    """
    assert 0.0 < p <= 1.0
    B, C, H, W = images.shape
    k = max(1, int(round(p * H * W)))

    masked = images.clone()
    # We'll compute per-sample threshold by selecting top-k indices
    for i in range(B):
        flat = importance[i].reshape(-1)
        # indices of top-k importance pixels
        topk_idx = np.argpartition(flat, -k)[-k:]
        mask2d = np.zeros(H * W, dtype=bool)
        mask2d[topk_idx] = True
        mask2d = mask2d.reshape(H, W)

        # apply mask across all channels
        masked[i, :, mask2d] = baseline

    return masked

def faithfulness_drop_at_p(model, images: torch.Tensor, preds: torch.Tensor, importance: np.ndarray, p: float, baseline: float = 0.0) -> np.ndarray:
    """
    Drop@p% = logit(original pred class) - logit(masked pred class)
    Higher means explanation is more faithful (removing important pixels hurts prediction more).

    Returns: drop array (B,)
    """
    device = images.device
    B = images.size(0)

    with torch.no_grad():
        logits = model(images)  # (B, K)
        pred_logits = logits.gather(1, preds.view(-1, 1)).squeeze(1)  # (B,)

        masked_images = mask_top_p_percent(images, importance, p=p, baseline=baseline).to(device)
        masked_logits = model(masked_images)
        masked_pred_logits = masked_logits.gather(1, preds.view(-1, 1)).squeeze(1)

        drop = (pred_logits - masked_pred_logits).detach().cpu().numpy()

    return drop

def deletion_auc(model, images: torch.Tensor, preds: torch.Tensor, importance: np.ndarray,
                 steps: int = 20, baseline: float = 0.0) -> np.ndarray:
    """
    Deletion AUC: progressively mask more pixels (based on importance) and compute AUC
    of predicted-class logit over fraction removed.

    Lower AUC = prediction collapses faster = more faithful explanation.

    Returns: auc array (B,)
    """
    device = images.device
    B, C, H, W = images.shape
    total = H * W

    # precompute sorted pixel indices per sample (descending importance)
    order = np.argsort(importance.reshape(B, -1), axis=1)[:, ::-1]  # (B, total)

    # x-axis fractions removed
    fracs = np.linspace(0, 1, steps + 1)
    curves = np.zeros((B, steps + 1), dtype=np.float64)

    with torch.no_grad():
        # original point (0% removed)
        logits0 = model(images).gather(1, preds.view(-1, 1)).squeeze(1).detach().cpu().numpy()
        curves[:, 0] = logits0

        # progressively mask
        for s in range(1, steps + 1):
            k = int(round(fracs[s] * total))
            k = max(1, k)  # mask at least 1 pixel when s>0

            masked = images.clone()
            for i in range(B):
                idx = order[i, :k]  # top-k pixels
                mask2d = np.zeros(total, dtype=bool)
                mask2d[idx] = True
                mask2d = mask2d.reshape(H, W)
                masked[i, :, mask2d] = baseline

            logits_s = model(masked).gather(1, preds.view(-1, 1)).squeeze(1).detach().cpu().numpy()
            curves[:, s] = logits_s

    # trapezoidal AUC over fraction removed
    auc = np.trapezoid(curves, fracs, axis=1)
    return auc



def extract_class_shap(shap_values, class_idx: int) -> np.ndarray:
    """
    Returns numpy array of shape (B, C, H, W) for a given class.
    Handles common SHAP formats:
      - list of length K: shap_values[k] is (B, C, H, W)
      - numpy array (B, C, H, W, K)
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 5:
        return shap_values[..., class_idx]
    raise ValueError(f"Unexpected shap_values format/shape: {getattr(shap_values, 'shape', None)}")


def explanation_error_batch(logits: torch.Tensor,
                            preds: torch.Tensor,
                            shap_vals,
                            expected_value) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SHAP additivity error per sample (predicted class) in logit space.

    expected_value must be a 1D array of shape (K,) representing E[logits] over background.
    """
    logits_np = logits.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    expected_value = np.asarray(expected_value).reshape(-1)

    B = logits_np.shape[0]
    errors = np.zeros(B, dtype=np.float64)
    shap_sums = np.zeros(B, dtype=np.float64)
    logits_pred = np.zeros(B, dtype=np.float64)
    expected_vals = np.zeros(B, dtype=np.float64)

    for i in range(B):
        c = int(preds_np[i])
        if c < 0 or c >= expected_value.shape[0]:
            raise ValueError(f"Pred class {c} out of range for expected_value shape {expected_value.shape}")

        shap_pred = extract_class_shap(shap_vals, c)[i]  # (C,H,W)
        shap_sum = float(shap_pred.sum())
        logit_c = float(logits_np[i, c])

        ev_c = float(expected_value[c])

        errors[i] = abs((ev_c + shap_sum) - logit_c)
        shap_sums[i] = shap_sum
        logits_pred[i] = logit_c
        expected_vals[i] = ev_c

    return errors, shap_sums, logits_pred, expected_vals



def to_display_img(x_chw: torch.Tensor) -> np.ndarray:
    """
    For visualization only. Assumes inputs are already roughly displayable.
    Clips to [0, 1]. If your normalized inputs are not in [0,1], visuals may look odd.
    """
    img = x_chw.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)


def main():
    parser = argparse.ArgumentParser()

    # Paths / IO
    parser.add_argument("--input-dir", default="/data", help="Input directory containing .npy files and model.pt")
    parser.add_argument("--output", default="/output", help="Output directory")

    # Repro / SHAP controls
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default=None,
                        help="Optional fallback path to model weights if input-dir/model.pt missing")

    parser.add_argument("--nsamples", type=int, default=500, help="SHAP nsamples per batch")
    parser.add_argument("--background-size", type=int, default=50, help="Background samples for SHAP")
    parser.add_argument("--save-n", type=int, default=0, help="How many test explanations to save as images")
    parser.add_argument("--shap-batch", type=int, default=10, help="Batch size used for SHAP over test set")
    parser.add_argument("--save-csv", action="store_true", help="Also save a CSV with per-sample fields")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 1. Load model and weights
    model = config().to(device)

    weights_path = Path(args.input_dir) / "model.pt"
    if not weights_path.exists():
        if args.weights is None:
            raise FileNotFoundError(
                f"Could not find weights at {weights_path}. Provide --weights /path/to/model.pt"
            )
        weights_path = Path(args.weights)

    print(f"Loading weights from: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("Model loaded and ready.")

    # 2. Load data from .npy files (ALREADY NORMALIZED)

    print("Loading data from .npy files...")
    X_train = np.load(os.path.join(args.input_dir, "data.npy"))
    Y_train = np.load(os.path.join(args.input_dir, "labels.npy"))
    X_test = np.load(os.path.join(args.input_dir, "test_data.npy"))
    Y_test = np.load(os.path.join(args.input_dir, "test_labels.npy"))

    print("Loaded train range:", X_train.min(), X_train.max())
    print("Loaded test  range:", X_test.min(), X_test.max())


    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test).long()



    # If NHWC -> NCHW
    if X_train_tensor.dim() == 4 and X_train_tensor.shape[-1] == 3:
        X_train_tensor = X_train_tensor.permute(0, 3, 1, 2).contiguous()
        X_test_tensor = X_test_tensor.permute(0, 3, 1, 2).contiguous()

    trainset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    testset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.shap_batch, shuffle=False, num_workers=0)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


    # DEBUG quick check
    print("\n=== DEBUGGING ===")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}, min/max: {X_train.min():.6f}/{X_train.max():.6f}")
    print(f"X_test  shape: {X_test.shape},  dtype: {X_test.dtype},  min/max: {X_test.min():.6f}/{X_test.max():.6f}")
    print(f"Y_train unique: {np.unique(Y_train)[:20]} ...")
    print("=== END DEBUG ===\n")

    # 3. Prepare SHAP background
    batch_images = next(iter(train_loader))[0].to(device)
    background = batch_images[: args.background_size]


    # 4. Initialize SHAP explainer
    print("Initializing SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)
    # expected_value = explainer.expected_value
    # Compute expected value in logit space: E[f(x)] over background
    with torch.no_grad():
        bg_logits = model(background)              # (B_bg, K)
        expected_value = bg_logits.mean(dim=0)     # (K,)
    expected_value = expected_value.detach().cpu().numpy()


    # 5. Run SHAP over ALL test samples 
    N_test = len(testset)
    errors_all = np.zeros(N_test, dtype=np.float64)
    preds_all = np.zeros(N_test, dtype=np.int64)
    trues_all = np.zeros(N_test, dtype=np.int64)
    shap_sum_all = np.zeros(N_test, dtype=np.float64)
    logits_pred_all = np.zeros(N_test, dtype=np.float64)
    expected_val_all = np.zeros(N_test, dtype=np.float64)

    drop10_all = np.zeros(N_test, dtype=np.float64)
    delauc_all = np.zeros(N_test, dtype=np.float64)


    saved = 0
    seen = 0

    print(f"Running SHAP on all test samples: N={N_test}, batch={args.shap_batch}, nsamples={args.nsamples}")
    for batch_idx, (images, labels) in enumerate(test_loader):
        B = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

        # --- Sanity check: logit scale (batch) ---
        if batch_idx == 0:  # only print once
            lp = logits.gather(1, preds.view(-1, 1)).squeeze(1)  # predicted-class logit per sample
            print("[Sanity] logits (all classes) min/mean/max:",
                float(logits.min()), float(logits.mean()), float(logits.max()))
            print("[Sanity] predicted-class logit min/mean/max:",
                float(lp.min()), float(lp.mean()), float(lp.max()))


        # Compute SHAP
        shap_vals = explainer.shap_values(images, nsamples=args.nsamples)
        if isinstance(shap_vals, torch.Tensor):
            shap_vals = shap_vals.detach().cpu().numpy()

        # Faithfulness metrics (computed per batch after SHAP)
        preds_np = preds.detach().cpu().numpy()
        B, C, H, W = images.shape

        # importance map per sample based on its predicted class
        importance = np.zeros((B, H, W), dtype=np.float32)
        for i in range(B):
            c = int(preds_np[i])
            sv = shap_vals[i, :, :, :, c]          # (C,H,W) for predicted class
            importance[i] = np.abs(sv).sum(axis=0) # (H,W)

        # Faithfulness Drop@10% (higher is better)
        drop10 = faithfulness_drop_at_p(model, images, preds, importance, p=0.10, baseline=0.0)

        # Deletion AUC (lower is better)
        del_auc = deletion_auc(model, images, preds, importance, steps=20, baseline=0.0)

        # ---- SHAP output shape diagnostic (run once) ----
        if batch_idx == 0:
            if isinstance(shap_vals, list):
                print("[Diag] shap_vals is list, len =", len(shap_vals))
                print("[Diag] shap_vals[0] shape =", np.asarray(shap_vals[0]).shape)
            else:
                sv = np.asarray(shap_vals)
                print("[Diag] shap_vals is array, shape =", sv.shape)
        # ------------------------------------------------

        # Compute errors for this batch
        batch_errors, batch_shap_sums, batch_logits_pred, batch_ev = explanation_error_batch(
            logits, preds, shap_vals, expected_value
        )

        if batch_idx == 0:
            i = 0
            print("[Diag one] pred_c =", int(preds[i].item()))
            print("[Diag one] logit_c =", float(logits[i, int(preds[i].item())].item()))
            print("[Diag one] ev_c    =", float(batch_ev[i]))
            print("[Diag one] shap_sum=", float(batch_shap_sums[i]))
            print("[Diag one] ev+sum  =", float(batch_ev[i] + batch_shap_sums[i]))
            print("[Diag one] abs(err)=", float(batch_errors[i]))


        # Store into global arrays (keep strict order: shuffle=False)
        start = seen
        end = seen + B
        errors_all[start:end] = batch_errors
        preds_all[start:end] = preds.detach().cpu().numpy()
        trues_all[start:end] = labels.detach().cpu().numpy()
        shap_sum_all[start:end] = batch_shap_sums
        logits_pred_all[start:end] = batch_logits_pred
        expected_val_all[start:end] = batch_ev

        drop10_all[start:end] = drop10
        delauc_all[start:end] = del_auc


        # Save a few explanation visuals
        if saved < args.save_n:
            # Save some from this batch
            for i in range(B):
                if saved >= args.save_n:
                    break

                pred_c = int(preds_all[start + i])
                true_c = int(trues_all[start + i])

                # Per-sample shap tensor for predicted class -> (C,H,W)
                shap_pred = extract_class_shap(shap_vals, pred_c)[i]
                shap_spatial = shap_pred.sum(axis=0)

                vlim = float(np.max(np.abs(shap_spatial)) + 1e-8)
                img_disp = to_display_img(images[i])

                out_dir = out_root / f"sample_{saved:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                plt.imsave(out_dir / "original.png", img_disp)
                plt.imsave(out_dir / "shap_heatmap.png", shap_spatial, cmap="seismic", vmin=-vlim, vmax=vlim)

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img_disp)
                ax.imshow(shap_spatial, cmap="seismic", alpha=0.5, vmin=-vlim, vmax=vlim)
                ax.set_title(
                    f"True={classes[true_c] if 0 <= true_c < len(classes) else true_c} "
                    f"Pred={classes[pred_c] if 0 <= pred_c < len(classes) else pred_c}\n"
                    f"Err={errors_all[start+i]:.4g}",
                    fontsize=9
                )
                ax.axis("off")
                fig.savefig(out_dir / "overlay.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved += 1

        seen += B

        if (batch_idx + 1) % 50 == 0:
            print(f"  processed {seen}/{N_test} samples...")

    print(f"Done SHAP. processed={seen}, saved_examples={saved}")

    # 6. Save arrays for later summary
    np.save(out_root / "errors.npy", errors_all)
    np.save(out_root / "faith_drop10.npy", drop10_all)
    np.save(out_root / "faith_del_auc.npy", delauc_all)

    # np.save(out_root / "preds.npy", preds_all)
    # np.save(out_root / "trues.npy", trues_all)
    # np.save(out_root / "shap_sum_pred.npy", shap_sum_all)
    # np.save(out_root / "logits_pred.npy", logits_pred_all)
    # np.save(out_root / "expected_value_pred.npy", expected_val_all)

    # Also save a summary JSON right away
    summary = {
        "n_test": int(N_test),
        "nsamples": int(args.nsamples),
        "background_size": int(args.background_size),
        "shap_batch": int(args.shap_batch),
        "save_n": int(args.save_n),
        "mean_explanation_error": float(errors_all.mean()),
        "median_explanation_error": float(np.median(errors_all)),
        "p95_explanation_error": float(np.quantile(errors_all, 0.95)),
        "p99_explanation_error": float(np.quantile(errors_all, 0.99)),
        "max_explanation_error": float(errors_all.max()),
        "rmse_explanation_error": float(np.sqrt(np.mean(errors_all ** 2))),
    }

    summary.update({
        "faith_drop10_mean": float(drop10_all.mean()),
        "faith_drop10_median": float(np.median(drop10_all)),
        "faith_drop10_p95": float(np.quantile(drop10_all, 0.95)),
        "faith_del_auc_mean": float(delauc_all.mean()),
        "faith_del_auc_median": float(np.median(delauc_all)),
        "faith_del_auc_p95": float(np.quantile(delauc_all, 0.95)),
        "faithfulness_interpretation": {
            "drop10": "higher is better (bigger logit drop after masking top 10% SHAP-ranked pixels)",
            "del_auc": "lower is better (prediction collapses faster as SHAP-ranked pixels are deleted)"
        }
    })

    with open(out_root / "explanations_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_root/'errors.npy'}")
    print(f"Saved: {out_root/'explanations_summary.json'}")
    print(f"Saved: {out_root/'faith_drop10.npy'}")
    print(f"Saved: {out_root/'faith_del_auc.npy'}")


    # Optional CSV (bigger file)
    # if args.save_csv:
    #     df = pd.DataFrame({
    #         "index": np.arange(N_test),
    #         "true": trues_all,
    #         "pred": preds_all,
    #         "true_label": [classes[i] if 0 <= i < len(classes) else str(i) for i in trues_all],
    #         "pred_label": [classes[i] if 0 <= i < len(classes) else str(i) for i in preds_all],
    #         "logit_pred": logits_pred_all,
    #         "expected_value_pred": expected_val_all,
    #         "shap_sum_pred": shap_sum_all,
    #         "explanation_error": errors_all,
    #     })
    #     df.to_csv(out_root / "explanations_all_test.csv", index=False)
    #     print(f"Saved: {out_root/'explanations_all_test.csv'}")

    print("All done.")


if __name__ == "__main__":
    main()





