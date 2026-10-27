import math
from pathlib import Path

import torch
import torch.nn as nn
from scipy.stats import ttest_ind


class PixelSpaceModel(nn.Module):
    """
    Wrap a classifier so MinGD always operates in canonical pixel [0,1] space.

    normalization=None:
        model already expects [0,1]

    normalization={"mean":[...], "std":[...]}:
        model expects (pixel - mean) / std
    """
    def __init__(self, model, normalization=None):
        super().__init__()
        self.model=model
        self.normalization=normalization

        if normalization is not None:
            mean=torch.tensor(normalization["mean"],dtype=torch.float32).view(1,-1,1,1)
            std=torch.tensor(normalization["std"],dtype=torch.float32).view(1,-1,1,1)
            self.register_buffer("mean",mean)
            self.register_buffer("std",std)

    def forward(self,x):
        if self.normalization is None:
            return self.model(x)
        return self.model((x-self.mean)/self.std)


def to_pixel_space(x,normalization=None):
    """Convert loader/model-space tensors into canonical [0,1] pixel space."""
    if normalization is None:
        return x
    mean=torch.tensor(
        normalization["mean"],device=x.device,dtype=x.dtype
    ).view(1,-1,1,1)
    std=torch.tensor(
        normalization["std"],device=x.device,dtype=x.dtype
    ).view(1,-1,1,1)
    return x*std+mean


def _norm_linf(x):
    return x.flatten(1).abs().amax(dim=1)


def _norm_l2(x):
    return x.flatten(1).norm(p=2,dim=1)


def _norm_l1(x):
    return x.flatten(1).norm(p=1,dim=1)


def _l2_normalize(x):
    return x/_norm_l2(x).clamp_min(1e-12).view(-1,1,1,1)


def _l1_topk_direction(grad,delta,x,gap=0.001,k=100):
    current=x+delta
    g=grad.flatten(1).clone()
    cur=current.flatten(1)

    blocked=((g<0)&(cur<=gap))|((g>0)&(cur>=1.0-gap))
    g[blocked]=0.0

    k=min(k,g.shape[1])
    idx=g.abs().topk(k,dim=1).indices
    direction=torch.zeros_like(g)
    direction.scatter_(1,idx,g.gather(1,idx).sign())

    return direction.view_as(grad)


def loss_mingd(preds,target):
    idx=torch.arange(preds.shape[0],device=preds.device)
    return (preds.max(dim=1).values-preds[idx,target]).mean()


def mingd(
    model,
    x,
    target,
    distance="linf",
    alpha_l1=1.0,
    alpha_l2=0.01,
    alpha_linf=0.001,
    num_iter=500,
    gap=0.001,
    k=100,
):
    """Targeted MinGD boundary-distance estimation in pixel [0,1] space."""
    was_training=model.training
    model.eval()
    delta=torch.zeros_like(x)

    for _ in range(num_iter):
        with torch.no_grad():
            remaining=model(x+delta).argmax(1).ne(target)

        if not remaining.any():
            break

        xr=x[remaining]
        dr=delta[remaining].detach().requires_grad_(True)
        tr=target[remaining]

        loss=-loss_mingd(model(xr+dr),tr)
        grad=torch.autograd.grad(loss,dr)[0]

        if distance=="linf":
            dr=dr+alpha_linf*grad.sign()
        elif distance=="l2":
            dr=dr+alpha_l2*_l2_normalize(grad)
        elif distance=="l1":
            direction=_l1_topk_direction(
                grad,dr.detach(),xr,gap=gap,k=k
            )
            dr=dr+(alpha_l1/k)*direction
        else:
            raise ValueError(f"Unknown distance: {distance}")

        # MinGD always operates in canonical pixel [0,1] space.
        dr=torch.minimum(torch.maximum(dr.detach(),-xr),1.0-xr)
        delta[remaining]=dr

    if was_training:
        model.train()

    return delta.detach()


def extract_mingd_features(
    model,
    loader,
    device=None,
    max_samples=1000,
    num_iter=500,
    normalization=None,
    attack_batch_size=100,
):
    """
    Generate Dataset Inference MinGD features.

    Returns:
        Tensor [N, num_classes, 3]

    Last dimension:
        0 = Linf
        1 = L2
        2 = L1
    """
    if device is None:
        device=next(model.parameters()).device

    device=torch.device(device)
    base_model=model.to(device).eval()
    attack_model=PixelSpaceModel(base_model,normalization).to(device).eval()

    all_features=[]
    seen=0

    for xb,_ in loader:
        if seen>=max_samples:
            break

        xb=xb.float()

        for start in range(0,xb.shape[0],attack_batch_size):
            if seen>=max_samples:
                break

            x=xb[start:start+attack_batch_size].to(device)
            x=x[:max_samples-seen]
            x=to_pixel_space(x,normalization)

            if x.min()<-1e-4 or x.max()>1.0001:
                raise ValueError(
                    "Could not convert inputs to pixel [0,1] space: "
                    f"min={x.min().item():.6f}, max={x.max().item():.6f}"
                )

            x=x.clamp(0.0,1.0)

            with torch.no_grad():
                output=attack_model(x[:1])

            if output.ndim!=2:
                raise ValueError(
                    f"Expected classifier output [N,C], got {tuple(output.shape)}"
                )

            num_classes=output.shape[1]
            norm_features=[]

            for distance in ("linf","l2","l1"):
                class_features=[]

                for target_class in range(num_classes):
                    target=torch.full(
                        (x.shape[0],),
                        target_class,
                        dtype=torch.long,
                        device=device,
                    )

                    delta=mingd(
                        attack_model,
                        x,
                        target,
                        distance=distance,
                        num_iter=num_iter,
                    )

                    if distance=="linf":
                        dist=_norm_linf(delta)
                    elif distance=="l2":
                        dist=_norm_l2(delta)
                    else:
                        dist=_norm_l1(delta)

                    class_features.append(dist.unsqueeze(1))

                norm_features.append(
                    torch.cat(class_features,dim=1).unsqueeze(2)
                )

            all_features.append(
                torch.cat(norm_features,dim=2).cpu()
            )

            seen+=x.shape[0]

    if not all_features:
        raise RuntimeError("No MinGD features were generated.")

    return torch.cat(all_features,dim=0)


class DIClassifier(nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(feature_dim,100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.Tanh(),
        )

    def forward(self,x):
        return self.net(x)


def _prepare_features(features,mean,std):
    # Dataset Inference is invariant to class ordering.
    features=features.sort(dim=1).values
    features=(features-mean.view(1,1,-1))/std.view(1,1,-1)
    return features.reshape(features.shape[0],-1)


def fit_di_calibrator(
    member_features,
    public_features,
    split_index=500,
    epochs=1000,
    lr=0.1,
    seed=0,
):
    """Train the fixed member-vs-public Dataset Inference classifier."""
    torch.manual_seed(seed)

    n=min(len(member_features),len(public_features))
    member_features=member_features[:n]
    public_features=public_features[:n]

    if split_index>=n:
        raise ValueError(
            f"split_index={split_index} must be smaller than n={n}"
        )

    mean=member_features.mean(dim=(0,1))
    std=member_features.std(dim=(0,1)).clamp_min(1e-8)

    member_x=_prepare_features(member_features,mean,std)
    public_x=_prepare_features(public_features,mean,std)

    x=torch.cat(
        [member_x[:split_index],public_x[:split_index]],dim=0
    )
    y=torch.cat(
        [torch.zeros(split_index),torch.ones(split_index)]
    )

    perm=torch.randperm(len(y))
    x=x[perm]
    y=y[perm]

    classifier=DIClassifier(x.shape[1])
    optimizer=torch.optim.SGD(classifier.parameters(),lr=lr)
    classifier.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        output=classifier(x).squeeze(1)
        loss=-((2.0*y-1.0)*output).mean()
        loss.backward()
        optimizer.step()

    return {
        "state_dict":{
            k:v.detach().cpu()
            for k,v in classifier.state_dict().items()
        },
        "mean":mean.cpu(),
        "std":std.cpu(),
        "feature_dim":x.shape[1],
        "split_index":split_index,
        "num_samples":n,
    }


def evaluate_di_features(
    member_features,
    public_features,
    calibrator,
    pval_samples=50,
):
    """Apply the fixed DI classifier and compute the one-sided p-value."""
    mean=calibrator["mean"]
    std=calibrator["std"]
    split=int(calibrator["split_index"])

    member_x=_prepare_features(
        member_features,mean,std
    )[split:split+pval_samples]

    public_x=_prepare_features(
        public_features,mean,std
    )[split:split+pval_samples]

    if len(member_x)<pval_samples or len(public_x)<pval_samples:
        raise ValueError(
            f"Need {pval_samples} samples per group for DI p-value."
        )

    classifier=DIClassifier(calibrator["feature_dim"])
    classifier.load_state_dict(calibrator["state_dict"])
    classifier.eval()

    with torch.no_grad():
        member_scores=classifier(member_x).squeeze(1).cpu().numpy()
        public_scores=classifier(public_x).squeeze(1).cpu().numpy()

    print(
        "DI scores | "
        f"member mean={member_scores.mean():.8f} std={member_scores.std():.8f} "
        f"min={member_scores.min():.8f} max={member_scores.max():.8f} | "
        f"public mean={public_scores.mean():.8f} std={public_scores.std():.8f} "
        f"min={public_scores.min():.8f} max={public_scores.max():.8f}",
        flush=True,
    )

    # t_stat,p_value=ttest_ind(
    #     public_scores,
    #     member_scores,
    #     alternative="greater",
    #     equal_var=False,
    # )

    member_mean=float(member_scores.mean())
    public_mean=float(public_scores.mean())
    mean_difference=public_mean-member_mean
    member_std=float(member_scores.std())
    public_std=float(public_scores.std())
    eps=1e-12

    if member_std<eps and public_std<eps:
        if abs(mean_difference)<eps:
            # Both score distributions are effectively identical constants.
            t_stat=0.0
            p_value=1.0
        elif mean_difference>0:
            # Public scores are uniformly greater than member scores.
            t_stat=1e300
            p_value=1e-300
        else:
            # Direction is opposite to the one-sided DI hypothesis.
            t_stat=-1e300
            p_value=1.0
    else:
        t_stat,p_value=ttest_ind(
            public_scores,
            member_scores,
            alternative="greater",
            equal_var=False,
        )
        t_stat=float(t_stat)
        p_value=float(p_value)

        if not math.isfinite(t_stat) or not math.isfinite(p_value):
            print(
                f"Non-finite Welch result: "
                f"effect={mean_difference:.8e}, "
                f"member_std={member_std:.8e}, public_std={public_std:.8e}",
                flush=True,
            )

            if abs(mean_difference)<eps:
                t_stat=0.0
                p_value=1.0
            elif mean_difference>0:
                t_stat=1e300
                p_value=1e-300
            else:
                t_stat=-1e300
                p_value=1.0

    evidence_score=-math.log10(max(p_value,1e-300))


    return {
        "p_value":p_value,
        "mean_difference":mean_difference,
        "t_statistic":float(t_stat),
        "evidence_score":float(evidence_score),
    }


def build_fingerprinting_calibrator(
    model,
    train_loader,
    public_loader,
    save_path,
    device=None,
    dataset="unknown",
    model_name="unknown",
    normalization=None,
    max_samples=1000,
    num_iter=500,
    split_index=500,
    pval_samples=50,
    attack_batch_size=100,
):
    """
    Run once on the fixed owner/reference model.

    The resulting artifact is reused for all candidate pipelines derived
    from the same baseline dataset/model.
    """
    print("Generating owner TRAIN MinGD features...",flush=True)

    member=extract_mingd_features(
        model,
        train_loader,
        device=device,
        max_samples=max_samples,
        num_iter=num_iter,
        normalization=normalization,
        attack_batch_size=attack_batch_size,
    )

    print("Generating PUBLIC MinGD features...",flush=True)

    public=extract_mingd_features(
        model,
        public_loader,
        device=device,
        max_samples=max_samples,
        num_iter=num_iter,
        normalization=normalization,
        attack_batch_size=attack_batch_size,
    )

    print("Training Dataset Inference calibrator...",flush=True)

    calibrator=fit_di_calibrator(
        member,
        public,
        split_index=split_index,
    )

    baseline=evaluate_di_features(
        member,
        public,
        calibrator,
        pval_samples=pval_samples,
    )

    calibrator.update({
        "schema_version":1,
        "method":"dataset_inference_mingd",
        "dataset":dataset,
        "model":model_name,
        "normalization":normalization,
        "max_samples":max_samples,
        "num_iter":num_iter,
        "pval_samples":pval_samples,
        "attack_batch_size":attack_batch_size,
        "baseline_result":baseline,

        # Keep the original features for reproducibility/debugging.
        "member_features":member.cpu(),
        "public_features":public.cpu(),
    })

    save_path=Path(save_path)
    save_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(calibrator,save_path)

    print(f"Saved fingerprinting calibrator: {save_path}")
    print(
        f"Baseline DI result: "
        f"p={baseline['p_value']:.6e}, "
        f"effect={baseline['mean_difference']:.6f}, "
        f"evidence={baseline['evidence_score']:.6f}"
    )

    return calibrator


def evaluate_fingerprinting(
    model,
    train_loader,
    public_loader,
    calibrator_path,
    device=None,
):
    """
    Evaluate a candidate model against a fixed fingerprinting calibrator.
    """
    calibrator=torch.load(
        calibrator_path,
        map_location="cpu",
        weights_only=False,
    )

    normalization=calibrator.get("normalization")
    max_samples=(
        int(calibrator["split_index"])
        +int(calibrator.get("pval_samples",50))
    )
    num_iter=int(calibrator.get("num_iter",500))
    pval_samples=int(calibrator.get("pval_samples",50))
    attack_batch_size=int(
        calibrator.get("attack_batch_size",100)
    )

    member=extract_mingd_features(
        model,
        train_loader,
        device=device,
        max_samples=max_samples,
        num_iter=num_iter,
        normalization=normalization,
        attack_batch_size=attack_batch_size,
    )

    public=extract_mingd_features(
        model,
        public_loader,
        device=device,
        max_samples=max_samples,
        num_iter=num_iter,
        normalization=normalization,
        attack_batch_size=attack_batch_size,
    )

    return evaluate_di_features(
        member,
        public,
        calibrator,
        pval_samples=pval_samples,
    )


# Backward-compatible alias while transitioning LANDSEER code.
evaluate_fingerprinting_mingd=evaluate_fingerprinting