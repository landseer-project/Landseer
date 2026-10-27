import os
import torch
import torch.nn as nn

# from improved_diffusion.script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
# )

from .improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


class DDSProtectedModel(nn.Module):
    def __init__(self, classifier, sigma=0.25, classifier_expects_normalized=True,
                 diffusion_checkpoint=None):
        super().__init__()
        self.classifier = classifier
        self.sigma = sigma
        self.classifier_expects_normalized = classifier_expects_normalized

        cfg = model_and_diffusion_defaults()
        cfg.update({
            "image_size": 32,
            "num_channels": 128,
            "num_res_blocks": 3,
            "num_heads": 4,
            "num_heads_upsample": -1,
            "attention_resolutions": "16,8",
            "dropout": 0.3,
            "learn_sigma": True,
            "sigma_small": False,
            "class_cond": False,
            "diffusion_steps": 4000,
            "noise_schedule": "cosine",
            "timestep_respacing": "",
            "use_kl": False,
            "predict_xstart": False,
            "rescale_timesteps": True,
            "rescale_learned_sigmas": True,
            "use_checkpoint": False,
            "use_scale_shift_norm": True,
        })

        self.diffusion_model, self.diffusion = create_model_and_diffusion(**cfg)

        if diffusion_checkpoint is None:
            diffusion_checkpoint = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "cifar10_uncond_50M_500K.pt",
            )

        state = torch.load(diffusion_checkpoint, map_location="cpu")
        self.diffusion_model.load_state_dict(state)
        self.diffusion_model.eval()

        for p in self.diffusion_model.parameters():
            p.requires_grad = False

        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        mean = torch.tensor(CIFAR_MEAN, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(CIFAR_STD, dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.timestep = self._sigma_to_timestep(sigma)

    def _sigma_to_timestep(self, sigma):
        target = sigma * 2.0
        real_sigma, t = 0.0, 0

        while real_sigma < target:
            t += 1
            a = self.diffusion.sqrt_alphas_cumprod[t]
            b = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a

        print(f"DDS sigma={sigma} -> timestep={t}, diffusion_sigma={real_sigma}")
        return t

    def denoise(self, x):
        t = torch.full(
            (x.shape[0],),
            self.timestep,
            device=x.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)

        out = self.diffusion.p_sample(
            self.diffusion_model,
            x_t,
            t,
            clip_denoised=True,
        )

        return out["pred_xstart"]

    def forward(self, x):
        """
        Input:
            raw image tensor in [0,1], shape [B,3,32,32]

        Output:
            classifier logits
        """

        x = x * 2.0 - 1.0
        x = self.denoise(x)

        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)

        if self.classifier_expects_normalized:
            x = (x - self.mean) / self.std

        return self.classifier(x)