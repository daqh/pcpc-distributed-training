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

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
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
        image = image.convert("RGB")

        pixel_values = self.image_transforms(image)

        return {
            "pixel_values": pixel_values,
        }


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: Optional[str] = "lambdalabs/naruto-blip-captions",
        custom_data_dir: Optional[str] = None,
        image_column: str = "image",
        resolution: int = 64,
        batch_size: int = 8,
        num_workers: int = 4,
        max_train_samples: Optional[int] = None,
        max_hf_samples: Optional[int] = None,
        max_custom_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.custom_data_dir = custom_data_dir
        self.image_column = image_column
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_hf_samples = max_hf_samples
        self.max_custom_samples = max_custom_samples
        self.shuffle_seed = shuffle_seed

    def setup(self, stage=None):
        datasets_to_combine = []

        if self.dataset_name is not None:
            hf_dataset = load_dataset(
                self.dataset_name,
                split="train",
            )

            if self.max_hf_samples is not None:
                hf_dataset = hf_dataset.shuffle(seed=self.shuffle_seed).select(
                    range(min(self.max_hf_samples, len(hf_dataset)))
                )

            datasets_to_combine.append(hf_dataset)

        if self.custom_data_dir is not None:
            custom_dataset = load_dataset(
                "imagefolder",
                data_dir=self.custom_data_dir,
                split="train",
            )

            if self.max_custom_samples is not None:
                custom_dataset = custom_dataset.shuffle(seed=self.shuffle_seed).select(
                    range(min(self.max_custom_samples, len(custom_dataset)))
                )

            datasets_to_combine.append(custom_dataset)

        if not datasets_to_combine:
            raise ValueError(
                "You must provide at least one of: dataset_name or custom_data_dir."
            )

        if len(datasets_to_combine) == 1:
            train_dataset = datasets_to_combine[0]
        else:
            train_dataset = concatenate_datasets(datasets_to_combine)

        train_dataset = train_dataset.shuffle(seed=self.shuffle_seed)

        if self.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(min(self.max_train_samples, len(train_dataset)))
            )

        self.train_dataset = ImageDataset(
            dataset=train_dataset,
            image_column=self.image_column,
            resolution=self.resolution,
        )

        print(f"Loaded {len(self.train_dataset)} total training images.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
