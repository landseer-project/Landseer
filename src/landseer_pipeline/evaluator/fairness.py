import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# pip install fairlearn
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def evaluate_fairness(model, test_X, test_y, sensitive_attrs, device, method="fairlearn"):
    """
    Pluggable fairness metric function.

    Args:
        model: Trained model.
        test_X: Test features (torch.Tensor).
        test_y: Test labels (torch.Tensor).
        sensitive_attrs: Sensitive attributes (array-like).
        device: 'cuda' or 'cpu'.
        method: 'fairlearn' or 'custom'.

    Returns:
        Tuple of (DP, DEO)
    """
    model.eval()
    preds = []

    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=64)

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.extend(outputs.argmax(1).cpu().numpy())

    y_true = test_y.cpu().numpy()
    sensitive_attrs = np.array(sensitive_attrs)

    if method == "fairlearn":
        dp = demographic_parity_difference(y_true=y_true, y_pred=preds, sensitive_features=sensitive_attrs)
        deo = equalized_odds_difference(y_true=y_true, y_pred=preds, sensitive_features=sensitive_attrs)
    elif method == "custom":
        # Custom DP
        dp = abs(
            np.mean([p for p, z in zip(preds, sensitive_attrs) if z == 1]) -
            np.mean([p for p, z in zip(preds, sensitive_attrs) if z == 0])
        )

        # Custom DEO
        tpr_z1 = np.mean([
            1 if p == 1 else 0 for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 1 and z == 1
        ]) if np.sum((y_true == 1) & (sensitive_attrs == 1)) > 0 else 0

        tpr_z0 = np.mean([
            1 if p == 1 else 0 for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 1 and z == 0
        ]) if np.sum((y_true == 1) & (sensitive_attrs == 0)) > 0 else 0

        fpr_z1 = np.mean([
            1 if p == 1 else 0 for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 0 and z == 1
        ]) if np.sum((y_true == 0) & (sensitive_attrs == 1)) > 0 else 0

        fpr_z0 = np.mean([
            1 if p == 1 else 0 for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 0 and z == 0
        ]) if np.sum((y_true == 0) & (sensitive_attrs == 0)) > 0 else 0

        deo = abs(tpr_z1 - tpr_z0) + abs(fpr_z1 - fpr_z0)
    else:
        raise ValueError(f"Unsupported method '{method}' for fairness evaluation")

    return dp, deo


#Example usage of fair eval: (Sensitive attributes should be provided for fairness evaluation)

    #    sensitive_path = os.path.join(output_dir, "test_sketch_sensitive.npy") if in_def == "fair" else None
    #     if sensitive_path and os.path.isfile(sensitive_path):
    #         sensitive_attrs = np.load(sensitive_path)
    #         # dp, deo = evaluate_fairness_with_fairlearn(model, test_X, test_y, sensitive_attrs, device)
    #         dp, deo = evaluate_fairness(model, test_X, test_y, sensitive_attrs, device, method="custom")
    #     else:
    #         dp, deo = 0.0, 0.0


#For regualr eval, sketches must be used, so instead of loading regular.npy, do below:
        #Customize loading of sketch images to test fair model
        # if in_def == 'fair':
        #     test_data_path = os.path.join(output_dir, "test_sketch_images.npy")
        #     test_labels_path = os.path.join(output_dir, "test_sketch_labels.npy")
        #     train_data_path = os.path.join(output_dir, "train_sketch_images.npy")
        #     train_labels_path = os.path.join(output_dir, "train_sketch_labels.npy")

    