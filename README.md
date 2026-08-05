# OrbitDesk Support Agent

A local-first support agent network built with LangGraph. Answers support
questions about the fictional OrbitDesk product using a supplied knowledge
base and resolved-case history — no remote LLM APIs involved anywhere in
the pipeline.

## Architecture

```
retrieve -> triage -> generate -> verify
                                     |
                     [conditional: passed / retry / failed]
                          |            |            |
                        END       generate        END
```

- **retrieve** (`src/nodes/retrieve.py`) — runs first, unconditionally.
  Pulls the top-k passages from the knowledge base + resolved cases via
  cosine similarity over `all-MiniLM-L6-v2` embeddings, so triage has real
  evidence to reason over instead of just the raw question.
- **triage** (`src/nodes/triage.py`) — classifies the request as
  `answerable`, `requires_clarification`, `requires_escalation`, or
  `out_of_scope`. Deterministic rules catch clear cases (no evidence above
  threshold → clarification; prompt-injection patterns → out_of_scope);
  the local LLM handles the rest.
- **generate** (`src/nodes/generate.py`) — drafts a `SupportResponse` for
  `answerable`/`requires_clarification`/`requires_escalation` routes using
  the local LLM constrained to the retrieved passages; templated responses
  are used for deterministic routes like `out_of_scope`.
- **verify** (`src/nodes/verify.py`) — checks the draft against the
  retrieved evidence: schema validity, presence of a source citation, and
  no unsupported/invented instructions. On failure it either routes back
  to `generate` (one retry only) or returns a `safe_failure` response with
  `requires_human: true`.

Shared state (`src/state.py`) is a single typed `GraphState` (TypedDict).
Nodes read a subset and return partial updates — LangGraph merges these,
so `triage` never needs to know how `retrieve` works internally, only
which fields it fills in.

**Loop protection:** `state["retry_count"]` vs `MAX_RETRIES = 1`, enforced
inside `verify_node` itself. The conditional edge in `graph.py` only ever
sees two outcomes — `"done"` or `"retry"` — so it structurally cannot spin
indefinitely; `verify_node` stops emitting `"retry"` once the ceiling is
hit and returns a `safe_failure` response instead.

Diagram: see `diagrams/` for the exported graph image.

## Models Used

| Purpose | Model | Revision | Device |
|---|---|---|---|
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | CPU |
| Response generation | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | CPU |

The assignment requires an *exact* revision, not just the model name — get
it from your local HF cache after the first download, no network call
needed:

```bash
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/
ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/
```

Each folder name under `snapshots/` *is* the commit hash HF resolved
`main` to at download time. Paste those into the table above.

Both loaded once at graph-build time and shared across all requests (see
`src/models.py`, `src/retrieval.py`) — reloading per call would misrepresent
real latency and make the graph unusably slow.

Run without CUDA — forced to CPU (see `DEVICE = "cpu"` in `models.py`);
target hardware had no CUDA-capable GPU available.

**Hardware used for this run:**

- Laptop: HP Pavilion Laptop 14-dv2xxx
- CPU: Intel Core i5-1235U (12th Gen, 10 cores / 12 threads)
- RAM: 16 GB
- Operating System: Windows 11 Home Single Language (64-bit)
- GPU: CPU-only inference (CUDA unavailable)

## Load Time & Latency

### Measured Model Performance

| Stage | Time |
|---|---:|
| Embedding model load | 7.85 s |
| Embedding inference (1 sentence) | 0.05 s |
| Generation model load | 4.09 s |
| Generation inference (~150 new tokens) | 16.17 s |

Both models are loaded once during graph initialization and reused for all subsequent requests, avoiding repeated model loading overhead.

| Question | Route | Latency (s) |
|---|---|---:|
| Q-001 (answerable) | answerable → verify pass on retry | 41.39 |
| Q-002 (answerable) | answerable → verify pass on retry | 18.98 |
| Q-003 (clarification) | requires_clarification (templated, no LLM) | 0.02 |
| Q-004 (answerable) | answerable → verify pass on retry | 25.09 |
| Q-005 (out of scope) | out_of_scope (templated, no LLM) | 0.06 |

Deterministic routes (`requires_clarification` and `out_of_scope`) complete in under 0.1 seconds because they bypass the language model and return predefined responses. LLM-backed routes require approximately 19–41 seconds on CPU using the 0.5B Qwen model. When verification triggers a retry, the overall latency increases significantly because the generation and verification stages are executed a second time.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.graph   # runs the 5 sample questions end to end
```

After the first run (models cached locally by Hugging Face), network
access can be disabled and the graph still runs.

## Tests

```bash
pytest tests/
```

`tests/test_graph_routing.py` asserts on `state["route"]`, `verification_passed`,
and response fields rather than on exact LLM-generated wording, so it stays
valid regardless of what the local model happens to phrase — this includes
routes the 5 sample questions never naturally exercise (`requires_escalation`
is untested by the sample data itself; see `test_generate_escalation_branch_produces_valid_response`).
`scripts/manual_smoke_check.py` is a separate, non-pytest script for eyeballing
real-model behavior; it's not run as part of the automated suite.

## AI Assistant Disclosure

This project was developed with assistance from Claude AI and ChatGPT. These tools were used for discussing the architecture, debugging LangGraph workflow issues, refining prompts, improving documentation, and reviewing code quality. The final implementation, testing, and verification were completed locally before submission.

## Design Trade-offs

- **In-memory numpy cosine similarity instead of a vector DB.** The
  corpus is ~10 short docs + 8 cases — a managed vector database would be
  over-engineering at this scale; a flat matrix is fast enough and keeps
  the dependency surface small.
- **Section-level chunking for KB docs, one chunk per resolved case.**
  KB docs mix multiple sub-topics under one file, so chunking by `##`
  heading gives tighter retrieval than whole-document chunks. Resolved
  cases are short enough that per-case chunking loses nothing.
- **Deterministic rules ahead of the LLM for triage on clear-cut cases**
  (no-evidence → clarification, injection pattern → out_of_scope). Keeps
  the safety-relevant routing decisions out of the hands of a 0.5B model
  that can't be fully trusted to always get it right, and keeps those
  paths near-instant.

## Known Limitations

- The 0.5B local generation model doesn't reliably include a
  `[SOURCE_ID]`-style citation on its first attempt, even when relevant
  evidence was retrieved and supplied in the prompt — this triggered the
  retry path on 3 of 5 sample questions in testing. `generate.py`
  deterministically prepends the top retrieved source's citation tag if
  it's still missing on the retry attempt (not the first, so the natural
  failure/retry signal is preserved), which keeps genuinely-answerable
  questions from being kicked to `safe_failure` over a formatting slip
  alone.
- Verification checks that any citation tag present points to a source
  that was actually retrieved, but it does not check that every claim in
  the answer is supported by the cited passage — a partially unsupported
  detail alongside a valid citation can still pass. Observed once in
  testing (Q-004's answer mentioned an "Incident Management system" not
  present in the retrieved evidence).
- Retrieval uses a fixed `top_k` and `min_score` threshold rather than
  anything adaptive to query type.

## What I'd Improve With More Time

- A real faithfulness check in `verify.py` — e.g. confirming each
  sentence's key claims appear in the cited passage, not just that the
  citation tag itself is valid — to catch cases like the Q-004 example
  above.
- Swap in a slightly larger instruction-tuned model (e.g. a 1.5B–3B class
  model) to see whether citation-following improves enough on the first
  attempt that the retry path is needed less often.