import argparse
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image

from model import LightningSmallDiffusionModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model: Optional[LightningSmallDiffusionModel] = None
loaded_ckpt_path: Optional[str] = None


def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    checkpoint_root = Path(checkpoint_dir)

    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_root.resolve()}"
        )

    checkpoint_paths = list(checkpoint_root.rglob("*.ckpt"))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .ckpt files found under: {checkpoint_root.resolve()}"
        )

    return max(checkpoint_paths, key=lambda path: path.stat().st_mtime)


def load_latest_checkpoint(checkpoint_dir: str) -> str:
    global loaded_model
    global loaded_ckpt_path

    try:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)

        model = LightningSmallDiffusionModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            map_location=DEVICE,
        )

        model.eval()
        model.to(DEVICE)

        loaded_model = model
        loaded_ckpt_path = str(ckpt_path)

        return f"Loaded checkpoint: {ckpt_path}"

    except Exception as error:
        loaded_model = None
        loaded_ckpt_path = None
        return f"Failed to load checkpoint:\n{error}"


@torch.inference_mode()
def generate_image(
    num_inference_steps: int,
    seed: int,
) -> Tuple[object, str]:
    if loaded_model is None:
        return None, "No checkpoint loaded yet. Click 'Load latest checkpoint' first."

    try:
        images = loaded_model.sample(
            batch_size=1,
            num_inference_steps=num_inference_steps,
            seed=seed,
            device=DEVICE,
        )

        image_tensor = images[0].detach().cpu()
        image = to_pil_image(image_tensor)

        return image, f"Generated with: {loaded_ckpt_path}"

    except Exception as error:
        return None, f"Generation failed:\n{error}"


def build_app(checkpoint_dir: str):
    with gr.Blocks(title="Small Diffusion Checkpoint Tester") as demo:
        gr.Markdown("# Small Diffusion Checkpoint Tester")

        gr.Markdown(
            "Click the button to load the latest Lightning `.ckpt`, then generate unconditional samples."
        )

        with gr.Row():
            load_button = gr.Button("Load latest checkpoint", variant="primary")
            load_status = gr.Textbox(
                label="Checkpoint status",
                interactive=False,
            )

        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Inference steps",
                minimum=1,
                maximum=500,
                value=50,
                step=1,
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
            fn=lambda: load_latest_checkpoint(checkpoint_dir=checkpoint_dir),
            inputs=None,
            outputs=load_status,
        )

        generate_button.click(
            fn=generate_image,
            inputs=[
                num_inference_steps,
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
        default=False,
    )

    parser.add_argument(
        "--no-share",
        action="store_false",
        dest="share",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = build_app(
        checkpoint_dir=args.checkpoint_dir,
    )

    app.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )
