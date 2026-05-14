from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import lightning as L


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        image_column: str = "image",
        caption_column: str = "text",
        resolution: int = 64,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item[self.image_column]
        caption = item[self.caption_column]

        if isinstance(caption, list):
            caption = caption[0]

        image = image.convert("RGB")
        pixel_values = self.image_transforms(image)

        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }

class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: Optional[str] = "lambdalabs/naruto-blip-captions",
        train_data_dir: Optional[str] = None,
        image_column: str = "image",
        resolution: int = 64,
        batch_size: int = 2,
        num_workers: int = 4,
        max_train_samples: Optional[int] = 128,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.train_data_dir = train_data_dir
        self.image_column = image_column
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples

    def setup(self, stage=None):
        if self.train_data_dir is not None:
            dataset = load_dataset(
                "imagefolder",
                data_dir=self.train_data_dir,
                split="train",
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                split="train",
            )

        if self.max_train_samples is not None:
            dataset = dataset.shuffle(seed=42).select(
                range(min(self.max_train_samples, len(dataset)))
            )

        self.train_dataset = ImageDataset(
            dataset=dataset,
            image_column=self.image_column,
            resolution=self.resolution,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
from typing import Optional

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import lightning as L


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        image_column: str = "image",
        resolution: int = 64,
    ):
        self.dataset = dataset
        self.image_column = image_column

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item[self.image_column]

        # Some datasets can contain grayscale images.
        image = image.convert("RGB")

        pixel_values = self.image_transforms(image)

        return {
            "pixel_values": pixel_values,
        }


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "zh-plus/tiny-imagenet",
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        train_data_dir: Optional[str] = None,
        image_column: str = "image",
        resolution: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        max_train_samples: Optional[int] = 10_000,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.dataset_split = dataset_split
        self.train_data_dir = train_data_dir
        self.image_column = image_column
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples

    def setup(self, stage=None):
        if self.train_data_dir is not None:
            dataset = load_dataset(
                "imagefolder",
                data_dir=self.train_data_dir,
                split=self.dataset_split,
            )
        else:
            if self.dataset_config_name is not None:
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config_name,
                    split=self.dataset_split,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.dataset_split,
                )

        if self.max_train_samples is not None:
            num_samples = min(self.max_train_samples, len(dataset))
            dataset = dataset.shuffle(seed=42).select(range(num_samples))

        self.train_dataset = ImageDataset(
            dataset=dataset,
            image_column=self.image_column,
            resolution=self.resolution,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
