"""
Run this FIRST, before writing any graph code.

Confirms both local models load and run on your machine (CPU, Intel Iris Xe iGPU
-> no CUDA, so this deliberately forces CPU) and prints timing so we know the
generation model is fast enough to demo live in the video.

Usage:
    python scripts/smoke_test.py
"""
import time

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cpu"  # Intel Iris Xe has no CUDA support; forcing CPU explicitly


def time_block(label):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *a):
            print(f"[{label}] {time.time() - self.t0:.2f}s")

    return _Timer()


def main():
    print(f"torch version: {torch.__version__}")
    print(f"device: {DEVICE}\n")

    # ---- Embedding model ----
    with time_block("embedding model load"):
        embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    with time_block("embedding inference (1 sentence)"):
        vec = embedder.encode(["Scheduled exports stopped after timezone change."])
    print(f"embedding dim: {vec.shape}\n")

    # ---- Generation model ----
    with time_block("generation model load"):
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME, torch_dtype=torch.float32
        ).to(DEVICE)

    prompt = (
        "You are a support assistant. Answer only from the context below.\n"
        "Context: Scheduled exports run in the workspace timezone. Changing the "
        "timezone shifts the next run time but does not retroactively fix missed runs.\n"
        "Question: My exports stopped after I changed the timezone. What should I check?\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True forces a consistent BatchEncoding output across transformers
    # versions (some recent versions silently switch the return type otherwise).
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)
    input_len = encoded["input_ids"].shape[1]

    with time_block("generation inference (~150 new tokens)"):
        output = model.generate(
            **encoded, max_new_tokens=150, do_sample=False, temperature=None, top_p=None
        )

    text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    print(f"\nSample output:\n{text}\n")
    print("Record the four timings above in your README's model section.")


if __name__ == "__main__":
    main()