import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def loss_mingd(preds, target):
    return (preds.max(dim=1)[0] - preds[torch.arange(preds.shape[0]), target]).mean()

def mingd(model, X, y, target, alpha=0.01, num_iter=20):
    model.eval()
    delta = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        preds = model(X + delta)
        loss = -loss_mingd(preds, target)
        loss.backward()
        delta.data += alpha * delta.grad.sign()
        delta.data = torch.min(torch.max(delta, -X), 1 - X)
        delta.grad.zero_()
    return delta.detach()

def evaluate_fingerprinting_mingd(model, dataloader):
    model.cuda()
    model.eval()
    total = 0
    correct_clean = 0
    correct_mingd = 0

    for X, y in dataloader:
        X, y = X.cuda(), y.cuda()
        total += X.size(0)

        # Clean prediction
        pred_clean = model(X).max(1)[1]
        correct_clean += (pred_clean == y).sum().item()

        # MINGD attack: target is original clean prediction
        delta = mingd(model, X, y, pred_clean)
        pred_mingd = model(X + delta).max(1)[1]
        correct_mingd += (pred_mingd == y).sum().item()

    print(f"Clean Accuracy: {correct_clean / total:.4f}")
    return correct_mingd/total
