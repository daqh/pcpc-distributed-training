import lightning as L
import torch
import torch.nn.functional as F

from components import TinyGPT

class LitTinyGPT(L.LightningModule):
    def __init__(self, vocab_size, block_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.model = TinyGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size

        positions = torch.arange(0, T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.lm_head(x)

    def forward(self, idx):
        return self.model(idx)
