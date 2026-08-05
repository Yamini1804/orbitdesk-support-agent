"""
Stubs out torch / transformers / sentence-transformers before anything in
`src` gets imported, so the routing tests in this directory can import
src.graph, src.models, src.retrieval, and every node module without
requiring those (large, slow-to-install) packages to be present, and
without ever downloading or loading real model weights.

This is safe because the routing tests never call LocalLLM(), get_llm(), or
Retriever() - they inject FakeLLM / fake retrieved-passage lists directly
into node functions instead. If a stubbed class *is* accidentally
instantiated, it raises immediately rather than silently pretending to
work, so a test that should be using a fake will fail loudly, not quietly
pass against a stub.

If the real packages are already installed (e.g. running inside the
project's actual venv), this file defers to them instead of overriding -
so these tests also pass unmodified in the full environment.
"""
from __future__ import annotations

import sys
import types


def _stub(name: str, build):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = build()


def _build_torch():
    mod = types.ModuleType("torch")
    mod.float32 = "float32"
    return mod


def _build_transformers():
    mod = types.ModuleType("transformers")

    class _StubAutoClass:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise RuntimeError(
                "transformers is stubbed out in tests - inject a FakeLLM "
                "instead of instantiating LocalLLM/AutoModelForCausalLM."
            )

    mod.AutoModelForCausalLM = _StubAutoClass
    mod.AutoTokenizer = _StubAutoClass
    return mod


def _build_sentence_transformers():
    mod = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "sentence_transformers is stubbed out in tests - pass a "
                "hand-built `retrieved` list instead of using a real Retriever."
            )

    mod.SentenceTransformer = _StubSentenceTransformer
    return mod


_stub("torch", _build_torch)
_stub("transformers", _build_transformers)
_stub("sentence_transformers", _build_sentence_transformers)