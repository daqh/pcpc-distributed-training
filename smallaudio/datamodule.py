import torch
import os
from typing import Dict, Any, List

from transformers import AutoProcessor
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import lightning as L

class MusicGenDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_id: str = "facebook/musicgen-small",
        dataset_name: str = "sanchit-gandhi/gtzan",
        split: str = "train",
        batch_size: int = 1,
        num_workers: int = 4,
        seconds: int = 8,
    ):
        super().__init__()
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seconds = seconds

        self.processor = AutoProcessor.from_pretrained(model_id)

        # MusicGen uses 32 kHz audio.
        self.target_sampling_rate = 32_000

    def setup(self, stage=None):
        ds = load_dataset(self.dataset_name, split=self.split)

        # Ensure audio is decoded and resampled to MusicGen's expected rate.
        ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))

        # For GTZAN, the genre label can be converted into a crude caption.
        # For your own dataset, replace this with your metadata caption column.
        if "genre" in ds.features:
            genre_names = ds.features["genre"].names

            def add_caption(example):
                genre = genre_names[example["genre"]]
                example["text"] = f"a {genre} music track"
                return example

            ds = ds.map(add_caption)

        self.dataset = ds

    def _crop_or_pad(self, wav: torch.Tensor, target_len: int) -> torch.Tensor:
        if wav.numel() > target_len:
            return wav[:target_len]
        if wav.numel() < target_len:
            pad = target_len - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in examples]

        target_len = self.seconds * self.target_sampling_rate
        audios = []

        for ex in examples:
            audio_array = ex["audio"]["array"]
            wav = torch.tensor(audio_array, dtype=torch.float32)

            # Convert stereo/multichannel to mono if needed.
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)

            wav = self._crop_or_pad(wav, target_len)
            audios.append(wav.numpy())

        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.target_sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )