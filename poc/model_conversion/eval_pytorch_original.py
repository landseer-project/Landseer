import numpy as np
import torch
from config_model import config

images = np.load("/data/test_data.npy").astype("float32")
labels = np.load("/data/test_labels.npy").astype("int64")
norm = (images - 0.5) / 0.5
model = config()
model.load_state_dict(torch.load("/input/model.pt", map_location="cpu"), strict=True)


def acc(net, data, targets):
    correct = 0
    with torch.no_grad():
        for start in range(0, len(targets), 256):
            pred = net(torch.from_numpy(data[start : start + 256])).argmax(1).numpy()
            correct += int((pred == targets[start : start + 256]).sum())
    return correct / len(targets)


model.eval()
print("original forward eval", acc(model, norm, labels))
model.train()
print("original forward train-mode", acc(model, norm, labels))
