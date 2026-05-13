import os
import glob
import threading
from pathlib import Path
from datetime import datetime

import torch
import scipy.io.wavfile
import gradio as gr
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from model import LitMusicGen


BASE_MODEL_ID = "facebook/musicgen-small"
FINETUNED_MODEL_DIR = "musicgen-finetuned"

# Where Lightning usually saves checkpoints if you did not specify dirpath.
CHECKPOINT_SEARCH_DIRS = [
    "lightning_logs",
    "checkpoints",
    ".",
]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_lock = threading.Lock()

processor = None
model = None
loaded_model_info = None


def find_latest_checkpoint():
    """
    Finds the newest .ckpt file from common Lightning checkpoint locations.

    Your ModelCheckpoint callback likely saves to something like:
        lightning_logs/version_0/checkpoints/best-checkpoint.ckpt
        lightning_logs/version_0/checkpoints/last.ckpt
    """
    checkpoint_paths = []

    for search_dir in CHECKPOINT_SEARCH_DIRS:
        checkpoint_paths.extend(
            glob.glob(
                os.path.join(search_dir, "**", "*.ckpt"),
                recursive=True,
            )
        )

    checkpoint_paths = [Path(p) for p in checkpoint_paths if Path(p).is_file()]

    if not checkpoint_paths:
        return None

    latest_checkpoint = max(
        checkpoint_paths,
        key=lambda p: p.stat().st_mtime,
    )

    return latest_checkpoint


def patch_musicgen_config(musicgen_model):
    """
    Makes sure training/generation config fields are populated.

    Useful when loading from either HF pretrained weights or a Lightning ckpt.
    """
    generation_config = musicgen_model.generation_config

    start_token_id = generation_config.decoder_start_token_id
    pad_token_id = generation_config.pad_token_id
    bos_token_id = generation_config.bos_token_id

    musicgen_model.config.decoder_start_token_id = start_token_id
    musicgen_model.config.pad_token_id = pad_token_id
    musicgen_model.config.bos_token_id = bos_token_id

    musicgen_model.config.decoder.decoder_start_token_id = start_token_id
    musicgen_model.config.decoder.pad_token_id = pad_token_id
    musicgen_model.config.decoder.bos_token_id = bos_token_id

    return musicgen_model


def load_base_model():
    global processor, model, loaded_model_info

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    loaded = MusicgenForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    loaded = patch_musicgen_config(loaded)
    loaded.to(device)
    loaded.eval()

    model = loaded
    loaded_model_info = f"Loaded base model: {BASE_MODEL_ID}"

    return loaded_model_info


def load_model_from_checkpoint(checkpoint_path: Path):
    global processor, model, loaded_model_info

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    lit_model = LitMusicGen.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location=device,
    )

    loaded = lit_model.model
    loaded = patch_musicgen_config(loaded)
    loaded.to(device)
    loaded.eval()

    model = loaded

    modified_time = datetime.fromtimestamp(
        checkpoint_path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    loaded_model_info = (
        f"Loaded fine-tuned checkpoint:\n"
        f"{checkpoint_path}\n"
        f"Modified: {modified_time}"
    )

    return loaded_model_info


def load_best_available_model():
    """
    Priority:
    1. Latest Lightning .ckpt file.
    2. Saved Hugging Face-style fine-tuned directory.
    3. Base facebook/musicgen-small model.
    """
    global processor, model, loaded_model_info

    with model_lock:
        latest_checkpoint = find_latest_checkpoint()

        if latest_checkpoint is not None:
            return load_model_from_checkpoint(latest_checkpoint)

        if Path(FINETUNED_MODEL_DIR).is_dir():
            processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_DIR)

            loaded = MusicgenForConditionalGeneration.from_pretrained(
                FINETUNED_MODEL_DIR
            )
            loaded = patch_musicgen_config(loaded)
            loaded.to(device)
            loaded.eval()

            model = loaded
            loaded_model_info = f"Loaded fine-tuned model directory: {FINETUNED_MODEL_DIR}"
            return loaded_model_info

        return load_base_model()


def refresh_model():
    """
    Called by the Gradio refresh button.
    """
    try:
        status = load_best_available_model()
        return status
    except Exception as exc:
        raise gr.Error(f"Failed to refresh model: {exc}")


def generate_music(
    prompt: str,
    guidance_scale: float,
    max_new_tokens: int,
    do_sample: bool,
):
    global processor, model

    if not prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    if processor is None or model is None:
        load_best_available_model()

    with model_lock:
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio = model.generate(
                **inputs,
                do_sample=do_sample,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
            )

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_array = audio[0, 0].detach().cpu().float().numpy()

    os.makedirs("gradio_outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"gradio_outputs/musicgen_{timestamp}.wav"

    scipy.io.wavfile.write(
        output_path,
        rate=sampling_rate,
        data=audio_array,
    )

    return output_path


# Initial load at startup.
initial_status = load_best_available_model()


with gr.Blocks(title="MusicGen Text-to-Music Generator") as demo:
    gr.Markdown("# MusicGen Text-to-Music Generator")
    gr.Markdown(
        "Generate music from a text prompt. "
        "The app automatically uses the latest fine-tuned checkpoint when available."
    )

    model_status = gr.Textbox(
        label="Current Model",
        value=initial_status,
        lines=4,
        interactive=False,
    )

    refresh_button = gr.Button("Refresh model from latest checkpoint")

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="lo-fi hip hop beat with warm vinyl crackle and soft piano",
            lines=3,
        )

    with gr.Row():
        guidance_scale = gr.Slider(
            minimum=0.5,
            maximum=10.0,
            value=1.0,
            step=0.5,
            label="Guidance Scale",
        )

        max_new_tokens = gr.Slider(
            minimum=64,
            maximum=4096,
            value=1024,
            step=64,
            label="Max New Tokens",
        )

        do_sample = gr.Checkbox(
            value=True,
            label="Do Sample",
        )

    generate_button = gr.Button("Generate Music")

    output_audio = gr.Audio(
        label="Generated Music",
        type="filepath",
    )

    refresh_button.click(
        fn=refresh_model,
        inputs=[],
        outputs=[model_status],
    )

    generate_button.click(
        fn=generate_music,
        inputs=[
            prompt,
            guidance_scale,
            max_new_tokens,
            do_sample,
        ],
        outputs=[output_audio],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
