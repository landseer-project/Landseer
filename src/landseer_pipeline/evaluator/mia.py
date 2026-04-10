import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Optional

@torch.no_grad()
def mia_loss_auc(
    model: torch.nn.Module,
    member_loader: torch.utils.data.DataLoader,
    nonmember_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Standard membership inference baseline:
      - score(x,y) = -CE(model(x), y)  (higher => more likely member)
      - metric = AUROC

    Returns:
      auc (float)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    model.eval()

    def collect_scores(loader):
        scores = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            # Use negative loss so "higher score => more likely member"
            s = -F.cross_entropy(logits, y, reduction="none")
            scores.append(s.detach().cpu().numpy())

        return np.concatenate(scores, axis=0) if scores else np.array([])

    m = collect_scores(member_loader)
    nm = collect_scores(nonmember_loader)

    if was_training:
        model.train()

    if m.size == 0 or nm.size == 0:
        return float("nan")

    y_true = np.concatenate([np.ones_like(m, dtype=int), np.zeros_like(nm, dtype=int)])
    y_score = np.concatenate([m, nm])

    return float(roc_auc_score(y_true, y_score))
