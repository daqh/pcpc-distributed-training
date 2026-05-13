import torch
from transformers import MusicgenForConditionalGeneration
import lightning as L

class LitMusicGen(L.LightningModule):
    def __init__(
        self,
        model_id: str = "facebook/musicgen-small",
        lr: float = 1e-5,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)

        self.model.config.decoder_start_token_id = self.model.generation_config.decoder_start_token_id
        self.model.config.pad_token_id = self.model.generation_config.pad_token_id
        self.model.config.bos_token_id = self.model.generation_config.bos_token_id

        self.model.config.decoder.decoder_start_token_id = self.model.generation_config.decoder_start_token_id
        self.model.config.decoder.pad_token_id = self.model.generation_config.pad_token_id
        self.model.config.decoder.bos_token_id = self.model.generation_config.bos_token_id

        # Usually you fine-tune only the autoregressive decoder first.
        # This is cheaper and more stable than updating everything.
        if freeze_text_encoder:
            for p in self.model.text_encoder.parameters():
                p.requires_grad = False

        if freeze_audio_encoder:
            for p in self.model.audio_encoder.parameters():
                p.requires_grad = False

    def _audio_to_labels(self, input_values: torch.Tensor, padding_mask=None):
        """
        Convert waveform audio to MusicGen/Encodec discrete token labels.

        input_values usually has shape:
            [batch, sequence_length]

        MusicGen expects labels roughly as:
            [batch, sequence_length_in_codes, num_codebooks]
        """
        with torch.no_grad():
            if input_values.ndim == 2:
                # EnCodec expects [batch, channels, time].
                input_values = input_values.unsqueeze(1)

            encoded = self.model.audio_encoder.encode(
                input_values,
                padding_mask=padding_mask,
            )

            # HF EnCodec-style outputs expose audio_codes.
            # Shape can be model/version dependent, commonly:
            # [num_frames, batch, num_codebooks, frame_len]
            audio_codes = encoded.audio_codes

            if audio_codes.ndim == 4:
                # [frames, batch, codebooks, frame_len]
                audio_codes = audio_codes.permute(1, 0, 3, 2)
                audio_codes = audio_codes.reshape(
                    audio_codes.shape[0],
                    -1,
                    audio_codes.shape[-1],
                )
            elif audio_codes.ndim == 3:
                # Possible already [batch, codebooks, seq]
                audio_codes = audio_codes.permute(0, 2, 1)
            else:
                raise ValueError(f"Unexpected audio_codes shape: {audio_codes.shape}")

        return audio_codes.long()

    def training_step(self, batch, batch_idx):
        input_values = batch.pop("input_values")
        padding_mask = batch.pop("padding_mask", None)

        labels = self._audio_to_labels(input_values, padding_mask)

        # Remove audio inputs after converting them to labels.
        batch.pop("audio_values", None)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=labels,
        )

        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=input_values.shape[0],
        )
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.hparams.lr)