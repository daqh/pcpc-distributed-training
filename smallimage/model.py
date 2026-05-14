import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from diffusers import DDPMScheduler


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2

        exponent = -math.log(10000) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)

        emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time_emb = self.time_proj(F.silu(time_emb))
        x = x + time_emb[:, :, None, None]

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)

        skip = x
        x = self.downsample(x)

        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res1 = ResBlock(
            out_channels + skip_channels,
            out_channels,
            time_emb_dim,
        )
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="nearest",
            )

        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)

        return x


class SmallUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.down1 = DownBlock(
            base_channels,
            base_channels,
            time_emb_dim,
        )

        self.down2 = DownBlock(
            base_channels,
            base_channels * 2,
            time_emb_dim,
        )

        self.down3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
            time_emb_dim,
        )

        self.mid1 = ResBlock(
            base_channels * 4,
            base_channels * 4,
            time_emb_dim,
        )
        self.mid2 = ResBlock(
            base_channels * 4,
            base_channels * 4,
            time_emb_dim,
        )

        self.up3 = UpBlock(
            base_channels * 4,
            base_channels * 4,
            base_channels * 2,
            time_emb_dim,
        )

        self.up2 = UpBlock(
            base_channels * 2,
            base_channels * 2,
            base_channels,
            time_emb_dim,
        )

        self.up1 = UpBlock(
            base_channels,
            base_channels,
            base_channels,
            time_emb_dim,
        )

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)

        x = self.input_conv(x)

        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)
        x, skip3 = self.down3(x, time_emb)

        x = self.mid1(x, time_emb)
        x = self.mid2(x, time_emb)

        x = self.up3(x, skip3, time_emb)
        x = self.up2(x, skip2, time_emb)
        x = self.up1(x, skip1, time_emb)

        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)

        return x


class LightningSmallDiffusionModel(L.LightningModule):
    def __init__(
        self,
        image_size: int = 64,
        image_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        num_train_timesteps: int = 1000,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.image_channels = image_channels
        self.lr = lr

        self.model = SmallUNet(
            image_channels=image_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

    def training_step(self, batch, batch_idx):
        clean_images = batch["pixel_values"]

        noise = torch.randn_like(clean_images)

        batch_size = clean_images.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=clean_images.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps,
        )

        noise_pred = self.model(
            noisy_images,
            timesteps,
        )

        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        self.eval()

        if device is None:
            device = self.device

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)

        images = torch.randn(
            batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for timestep in self.noise_scheduler.timesteps:
            timestep_batch = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )

            noise_pred = self.model(
                images,
                timestep_batch,
            )

            images = self.noise_scheduler.step(
                noise_pred,
                timestep,
                images,
            ).prev_sample

        images = images.clamp(-1, 1)
        images = (images + 1) / 2

        return images

    def save_model(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "hyper_parameters": dict(self.hparams),
            },
            output_dir / "small_diffusion_model.pt",
        )