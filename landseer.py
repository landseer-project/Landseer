import os
import torch
import numpy as np
import itertools
import subprocess
from torch.utils.data import TensorDataset, DataLoader
from torchattacks import PGD
from sklearn.metrics import roc_auc_score
import json
import shutil
from config_model import config

with open("./config.json", "r") as file:
    defense_args = json.load(file)

def add_backdoor_trigger(images, target_label, trigger_value=1.0):
    images[:, :, -3:, -3:] = trigger_value
    labels = torch.full((images.size(0),), target_label, dtype=torch.long)
    return images, labels


def generate_ood_samples(images):
    return torch.rand_like(images)


def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_pgd(model, loader, device):
    model.eval()
    atk = PGD(model, eps=8/255, alpha=2/255, steps=10)
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        adv_X = atk(X, y)
        with torch.no_grad():
            pred = model(adv_X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_outlier(model, clean_loader, ood_loader, device):
    model.eval()
    scores, labels = [], []
    for loader, label in [(clean_loader, 0), (ood_loader, 1)]:
        for X, _ in loader:
            X = X.to(device)
            with torch.no_grad():
                out = torch.softmax(model(X), dim=1)
                max_conf = out.max(1)[0]
            scores.extend(1 - max_conf.cpu().numpy())
            labels.extend([label] * X.size(0))
    return roc_auc_score(labels, scores)


def evaluate_backdoor(model, poisoned_loader, target_class, device):
    model.eval()
    total, target_hits = 0, 0
    with torch.no_grad():
        for X, _ in poisoned_loader:
            X = X.to(device)
            pred = model(X).argmax(1)
            target_hits += (pred == target_class).sum().item()
            total += X.size(0)
    return target_hits / total


def run_containerized_pipeline(pre_def, in_def, post_def, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Pre-training
    pre_script = "defense.py" if pre_def != "noop" else "normal.py"
    # pre_script = "feature_squeeze_celebA.py" if pre_def != "noop" else "normal.py"
    image = list(defense_args["pre"].keys())[0]
    args = defense_args["pre"].get(pre_def, [])
    subprocess.run([
        "docker", "run", "--rm",
        # "-v", f"{base_data_path}:/data",
        "-v", f"{output_path}:/output",
        # "-v", "/share/landseer/img_align_celeba:/app/data/celeba/img_align_celeba",
        # "-v", "/share/landseer/Improving-Fairness-in-Image-Classification-via-Sketching/face_image_classification(CelebA)/dataset/list_attr_celeba.csv:/app/data/celeba/list_attr_celeba.csv",
        f"pre_{image}",
        "python3", pre_script
    ] + args)

    # In-training
    in_script = "defense.py" if in_def != "noop" else "normal.py"
    image = list(defense_args["in"].keys())[0]
    args = defense_args["in"].get(in_def, [])
    subprocess.run([
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{output_path}:/output",
        "-v", "./config_model.py:/app/config_model.py",
        f"in_{image}",
        "python3", in_script
    ] + args)

    # Post-training
    image = list(defense_args["post"].keys())[0]
    if post_def != "noop":
        post_script = "defense.py"
        args = defense_args["post"].get(post_def, [])
        subprocess.run([
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{output_path}:/output",
            "-v", "./config_model.py:/app/config_model.py",
            f"post_{image}",
            "python3", post_script
        ] + args)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combinations = list(itertools.product(['noop', 'squeeze'], ['noop', 'trades'], ['noop', 'fineprune']))
    final_results = {}

    for pre_def, in_def, post_def in combinations:
        pipeline_id = f"pre_{pre_def}_in_{in_def}_post_{post_def}"
        output_dir = os.path.join("./output", pipeline_id)

        print(f"\n=== Running Pipeline: {pipeline_id} ===")
        run_containerized_pipeline(pre_def, in_def, post_def, output_dir)

        model_path = os.path.join(output_dir, "model.pt")
        test_data_path = os.path.join(output_dir, "test_data.npy")
        test_labels_path = os.path.join(output_dir, "test_labels.npy")
        train_data_path = os.path.join(output_dir, "data.npy")
        train_labels_path = os.path.join(output_dir, "labels.npy")

        if not all(os.path.exists(p) for p in [model_path, test_data_path, test_labels_path, train_data_path, train_labels_path]):
            print(f"Skipping {pipeline_id}, missing files.")
            continue

        model = config().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        train_X = torch.from_numpy(np.load(train_data_path)).float()
        train_y = torch.from_numpy(np.load(train_labels_path)).long()
        test_X = torch.from_numpy(np.load(test_data_path)).float()
        test_y = torch.from_numpy(np.load(test_labels_path)).long()

        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64)
        test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=64)

        clean_acc_train = evaluate_clean(model, train_loader, device)
        clean_acc_test = evaluate_clean(model, test_loader, device)
        pgd_acc = evaluate_pgd(model, test_loader, device)
        outlier_score = evaluate_outlier(model, test_loader, DataLoader(TensorDataset(generate_ood_samples(test_X), test_y), batch_size=64), device)
        backdoor_success = evaluate_backdoor(model, DataLoader(TensorDataset(*add_backdoor_trigger(test_X.clone(), 0)), batch_size=64), 0, device)

        result = {
            'Clean Train Accuracy': clean_acc_train,
            'Clean Test Accuracy': clean_acc_test,
            'PGD Accuracy': pgd_acc,
            'Outlier AUROC': outlier_score,
            'Backdoor ASR': backdoor_success
        }

        final_results[pipeline_id] = result

        print(f"Train Acc: {clean_acc_train*100:.2f}% | Test Acc: {clean_acc_test*100:.2f}% | PGD Acc: {pgd_acc*100:.2f}% | AUROC: {outlier_score:.4f} | ASR: {backdoor_success*100:.2f}%")

        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(result, f, indent=2)

    with open("./output/summary.csv", 'w') as f:
        headers = ["Pipeline", "Clean Train Accuracy", "Clean Test Accuracy", "PGD Accuracy", "Outlier AUROC", "Backdoor ASR"]
        f.write(",".join(headers) + "\n")
        for pipeline, metrics in final_results.items():
            row = [
                pipeline,
                f"{metrics['Clean Train Accuracy']:.4f}",
                f"{metrics['Clean Test Accuracy']:.4f}",
                f"{metrics['PGD Accuracy']:.4f}",
                f"{metrics['Outlier AUROC']:.4f}",
                f"{metrics['Backdoor ASR']:.4f}"
            ]
            f.write(",".join(row) + "\n")
