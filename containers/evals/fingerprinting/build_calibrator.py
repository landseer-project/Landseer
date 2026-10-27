#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"common"))
from model_loader import load_torch_model_for_eval

from fingerprinting import build_fingerprinting_calibrator


def make_loader(data_path,labels_path,batch_size=100):
    x=np.load(data_path,mmap_mode="r")
    y=np.load(labels_path,mmap_mode="r")

    dataset=TensorDataset(
        torch.from_numpy(np.asarray(x)).float(),
        torch.from_numpy(np.asarray(y)).long(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def main():
    p=argparse.ArgumentParser()

    p.add_argument("--input-dir",required=True)
    p.add_argument("--model-config",required=True)
    p.add_argument("--output",required=True)

    p.add_argument("--dataset",default="unknown")
    p.add_argument("--model-name",default="unknown")

    p.add_argument("--mean",nargs="+",type=float)
    p.add_argument("--std",nargs="+",type=float)

    p.add_argument("--max-samples",type=int,default=1000)
    p.add_argument("--num-iter",type=int,default=500)
    p.add_argument("--split-index",type=int,default=500)
    p.add_argument("--pval-samples",type=int,default=50)
    p.add_argument("--attack-batch-size",type=int,default=100)
    p.add_argument("--loader-batch-size",type=int,default=100)
    p.add_argument("--device",default=None)

    args=p.parse_args()

    input_dir=Path(args.input_dir)
    model_path=input_dir/"model.pt"

    required=[
        model_path,
        input_dir/"data.npy",
        input_dir/"labels.npy",
        input_dir/"test_data.npy",
        input_dir/"test_labels.npy",
    ]

    for f in required:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config_path=Path(args.model_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    # model_loader expects input/config_model.py.
    # Load the architecture directly from the supplied config instead.
    import importlib.util

    spec=importlib.util.spec_from_file_location(
        "fingerprinting_model_config",
        config_path,
    )
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module,"config"):
        raise AttributeError(
            f"{config_path} does not define config()"
        )

    model=module.config().to(device)

    state=torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(state,dict) and "state_dict" in state:
        state=state["state_dict"]

    if isinstance(state,dict) and "model_state_dict" in state:
        state=state["model_state_dict"]

    if not isinstance(state,dict):
        raise ValueError(
            f"Expected model.pt to contain a state_dict, got {type(state)}"
        )

    clean_state={}
    for k,v in state.items():
        if k.startswith("module."):
            k=k[len("module."):]
        clean_state[k]=v

    model.load_state_dict(clean_state,strict=False)
    model.eval()

    train_loader=make_loader(
        input_dir/"data.npy",
        input_dir/"labels.npy",
        args.loader_batch_size,
    )

    public_loader=make_loader(
        input_dir/"test_data.npy",
        input_dir/"test_labels.npy",
        args.loader_batch_size,
    )

    normalization=None

    if args.mean is not None or args.std is not None:
        if args.mean is None or args.std is None:
            raise ValueError(
                "--mean and --std must be supplied together"
            )

        if len(args.mean)!=len(args.std):
            raise ValueError(
                "--mean and --std must have the same length"
            )

        normalization={
            "mean":args.mean,
            "std":args.std,
        }

    print(f"Device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Normalization: {normalization}")
    print(f"Output: {args.output}")

    build_fingerprinting_calibrator(
        model=model,
        train_loader=train_loader,
        public_loader=public_loader,
        save_path=args.output,
        device=device,
        dataset=args.dataset,
        model_name=args.model_name,
        normalization=normalization,
        max_samples=args.max_samples,
        num_iter=args.num_iter,
        split_index=args.split_index,
        pval_samples=args.pval_samples,
        attack_batch_size=args.attack_batch_size,
    )


if __name__=="__main__":
    main()