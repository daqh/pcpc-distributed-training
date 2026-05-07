from os import cpu_count

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

from models import LitTinyGPT

# -------------------------
# Dataset
# -------------------------

TOKENIZER_NAME = "gpt2"
DATASET_NAME = "cchoi1022/openwebtext_20"
DATASET_SPLIT = "train"


class HFDataset(Dataset):
    def __init__(
        self,
        block_size: int = 32,
        max_examples: int = 10_000,
        tokenizer_name: str = TOKENIZER_NAME,
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        texts = dataset.select(range(max_examples))["text"]

        token_ids = []
        for text in texts:
            if not text:
                continue
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_ids.extend(encoded + [tokenizer.eos_token_id])

        if len(token_ids) <= block_size:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for block_size={block_size}."
            )

        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_name = tokenizer_name
        self.dataset_name = DATASET_NAME
        self.dataset_split = DATASET_SPLIT
        self.max_examples = max_examples
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

# -------------------------
# Train locally
# -------------------------

def main():
    block_size = 128

    dataset = HFDataset(block_size=block_size)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
    )

    model = LitTinyGPT(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
    )
    
    logger = TensorBoardLogger(
        save_dir=".",
        name="lightning_logs",
    )

    trainer = L.Trainer(
        accelerator="cpu",
        max_steps=5000,
        devices=3,      # Numero di cpu da utilizzare
        num_nodes=3,    # Numero di nodi nel cluster
        precision="bf16-mixed",
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="tiny-gpt-{step}",
                save_top_k=3,
                monitor="train_loss",
                every_n_train_steps=10,
                save_last="link",
            ),
            # Save the first checkpoint immediately to have a reference for the initial model state
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="tiny-gpt-initial",
                save_top_k=1,
                monitor="step",
                save_last="link",
                every_n_train_steps=1,
            ),
        ],
        logger=logger,
        enable_checkpointing=True,
        # strategy=DDPStrategy(process_group_backend="mpi"),
        strategy=DDPStrategy(process_group_backend="gloo"),
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
