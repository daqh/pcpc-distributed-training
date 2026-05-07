import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models import LitTinyGPT

TOKENIZER_NAME = "gpt2"


# -------------------------
# Text generation
# -------------------------

@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    device,
    max_new_tokens=300,
    temperature=1.0,
    top_k=40,
):
    model.eval()

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(token_ids) == 0:
        token_ids = [tokenizer.eos_token_id]

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    block_size = model.hparams.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)


# -------------------------
# Chat loop
# -------------------------

def main():

    checkpoint_path = "checkpoints/last.ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = LitTinyGPT.load_from_checkpoint(
        checkpoint_path,
        vocab_size=vocab_size,
        block_size=128,
    )

    model.to(device)
    model.eval()

    print("TinyGPT loaded.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_text = input("You: ")

        if user_text.lower().strip() in {"exit", "quit"}:
            break

        prompt = f"User: {user_text}\nAssistant:"

        output = generate(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=300,
            temperature=1.0,
            top_k=40,
        )

        answer = output[len(prompt):]

        print(f"Model:{answer}\n")


if __name__ == "__main__":
    main()
