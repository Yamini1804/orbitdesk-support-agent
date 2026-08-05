"""
Shared local-model singletons.

Both triage (constrained classification) and generate (answer drafting) need
the same local LLM. Loading it once at graph-build time and reusing it across
nodes/questions is the difference between ~4s load time paid once vs paid on
every single node call - loading per-call would make the whole graph
unusably slow and would misrepresent the real latency numbers in the README.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"  # Intel Iris Xe iGPU has no CUDA support; forced CPU throughout


class LocalLLM:
    """Thin wrapper around tokenizer + model so nodes call llm.generate(prompt)
    instead of re-deriving the chat-template/generate boilerplate each time."""

    def __init__(self, model_name: str = GEN_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).to(DEVICE)

    def generate(self, prompt: str, max_new_tokens: int = 100, do_sample: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(DEVICE)
        input_len = encoded["input_ids"].shape[1]

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if not do_sample:
            gen_kwargs.update(temperature=None, top_p=None)  # avoid HF warning on greedy decode

        output = self.model.generate(**encoded, **gen_kwargs)
        return self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


_llm_instance: LocalLLM | None = None


def get_llm() -> LocalLLM:
    """Lazy singleton - first call loads the model (~4s), every call after
    reuses it. Call this from graph.py once at startup to pay that cost
    up front rather than on the first user question."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance