import argparse
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

from model import LightningStableDiffusionFineTuner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

loaded_pipe: Optional[StableDiffusionPipeline] = None
loaded_ckpt_path: Optional[str] = None

def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    checkpoint_root = Path(checkpoint_dir)

    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_root.resolve()}"
        )

    checkpoint_paths = list(checkpoint_root.rglob("*.ckpt"))

    if not checkpoint_paths:
        searched_path = checkpoint_root.resolve()
        raise FileNotFoundError(
            f"No .ckpt files found under: {searched_path}\n\n"
            "Try running:\n"
            "  find . -name '*.ckpt'\n\n"
            "Then start Gradio with:\n"
            "  python run-gradio.py --checkpoint_dir /path/to/checkpoint/folder"
        )

    return max(checkpoint_paths, key=lambda path: path.stat().st_mtime)


def load_latest_checkpoint(
    checkpoint_dir: str,
    model_name: str,
) -> str:
    global loaded_pipe
    global loaded_ckpt_path

    ckpt_path = find_latest_checkpoint(checkpoint_dir)

    lightning_model = LightningStableDiffusionFineTuner.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_name=model_name,
        map_location=DEVICE,
    )

    lightning_model.eval()
    lightning_model.to(DEVICE)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.unet = lightning_model.unet
    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    loaded_pipe = pipe
    loaded_ckpt_path = str(ckpt_path)

    return f"Loaded checkpoint: {ckpt_path}"


@torch.inference_mode()
def generate_image(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> Tuple[object, str]:
    if loaded_pipe is None:
        return None, "No checkpoint loaded yet. Click 'Load latest checkpoint' first."

    if not prompt.strip():
        return None, "Please enter a prompt."

    generator = torch.Generator(device=DEVICE)

    if seed >= 0:
        generator = generator.manual_seed(seed)

    image = loaded_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image, f"Generated with: {loaded_ckpt_path}"


def build_app(
    checkpoint_dir: str,
    model_name: str,
):
    with gr.Blocks(title="Tiny Stable Diffusion Checkpoint Tester") as demo:
        gr.Markdown("# Tiny Stable Diffusion Checkpoint Tester")

        gr.Markdown(
            "Click the button to load the latest Lightning `.ckpt`, then test prompts."
        )

        with gr.Row():
            load_button = gr.Button("Load latest checkpoint", variant="primary")
            load_status = gr.Textbox(
                label="Checkpoint status",
                interactive=False,
            )

        prompt = gr.Textbox(
            label="Prompt",
            value="a watercolor painting of a castle",
        )

        negative_prompt = gr.Textbox(
            label="Negative prompt",
            value="",
        )

        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Inference steps",
                minimum=1,
                maximum=100,
                value=10,
                step=1,
            )

            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=15.0,
                value=1.0,
                step=0.5,
            )

            seed = gr.Number(
                label="Seed, use -1 for random",
                value=42,
                precision=0,
            )

        generate_button = gr.Button("Generate image")

        output_image = gr.Image(label="Generated image")
        output_status = gr.Textbox(label="Generation status", interactive=False)

        load_button.click(
            fn=lambda: load_latest_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model_name=model_name,
            ),
            inputs=None,
            outputs=load_status,
        )

        generate_button.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                seed,
            ],
            outputs=[
                output_image,
                output_status,
            ],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="segmind/tiny-sd",
    )

    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
    )

    parser.add_argument(
        "--share",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = build_app(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )

    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
