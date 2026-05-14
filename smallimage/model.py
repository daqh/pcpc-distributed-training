import torch
import torch.nn.functional as F
import lightning as L

from diffusers import StableDiffusionPipeline, DDPMScheduler


class LightningStableDiffusionFineTuner(L.LightningModule):
    def __init__(
        self,
        model_name: str = "hf-internal-testing/tiny-stable-diffusion-pipe",
        lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet

        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Fine-tune only the UNet.
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.vae.eval()
        self.text_encoder.eval()

        self.lr = lr

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            encoder_hidden_states = self.text_encoder(input_ids)[0]

        noise = torch.randn_like(latents)

        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
        )

    def save_pipeline(self, output_dir: str):
        self.pipe.unet = self.unet
        self.pipe.save_pretrained(output_dir)