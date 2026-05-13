import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from datamodule import MusicGenDataModule
from model import LitMusicGen

from datetime import timedelta

if __name__ == "__main__":
    model_id = "facebook/musicgen-small"

    data = MusicGenDataModule(
        model_id=model_id,
        dataset_name="sanchit-gandhi/gtzan",
        split="train",
        batch_size=1,
        num_workers=1,
        seconds=8,
    )

    lit_model = LitMusicGen(
        model_id=model_id,
        lr=1e-5,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        precision="32-true",
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                filename="best-checkpoint",
                save_last=True,
                train_time_interval=timedelta(minutes=1),
            )
        ]
    )

    trainer.fit(lit_model, datamodule=data)

    lit_model.model.save_pretrained("musicgen-finetuned")
    data.processor.save_pretrained("musicgen-finetuned")
