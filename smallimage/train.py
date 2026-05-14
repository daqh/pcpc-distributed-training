import argparse
import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from model import LightningSmallDiffusionModel
from datamodules import ImageDataModule


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="hf-internal-testing/tiny-stable-diffusion-pipe",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="zh-plus/tiny-imagenet",
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Optional local imagefolder dataset path.",
    )
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")

    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="finetuned-tiny-sd")

    return parser.parse_args()

def main():
    args = parse_args()

    L.seed_everything(42)

    model = LightningSmallDiffusionModel(
        image_size=args.resolution,
        image_channels=3,
        base_channels=128,
        time_emb_dim = 512,
        lr=args.lr,
    )

    datamodule = ImageDataModule(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        train_data_dir=args.train_data_dir,
        image_column=args.image_column,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="tiny-sd-{epoch:02d}-{train/loss:.4f}",
        save_top_k=1,
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            checkpoint_callback,
        ],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule)

    model.save_pipeline(args.output_dir)
    print(f"Saved fine-tuned pipeline to: {args.output_dir}")


if __name__ == "__main__":
    main()
