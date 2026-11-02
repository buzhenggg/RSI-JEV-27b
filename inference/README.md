# Inference and reproduction

RSI-JEV-27b is the release name of the evaluated `mixed27b-step343` engine.
The Python package/module names and the inference source are retained unchanged
from v0.2.0. `model="RSI-JEV-27b"` changes only the returned display label.
The archived original results retain their original engine/model names.

## Installation

The tested environment is Linux, Python 3.13.16, CUDA 12.8, PyTorch 2.8.0+cu128,
Transformers 5.20.0, PEFT 0.21.2, FLA 0.3.2 and Triton 3.4.0. The lock pins
all runtime dependencies. Use one BF16-capable CUDA GPU with sufficient memory;
the evaluated hardware was A800-SXM4-80GB. No CPU-only inference is advertised.

From the downloaded model repository or cloned GitHub code repository:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps \
  "decision-index @ git+https://github.com/apolinario/decision-index.git@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d" .
```

The HF model repository also includes the exact SDK source under `vendor/decision-index`;
that directory can replace the Git URL in the installation command.

For a GitHub checkout, download the **complete model** from our HF repository:

```sh
.venv/bin/python scripts/fetch_weights.py --local-dir . --revision 3a126489958a67c08233711a1c5db31c57919229
```

This immutable revision matches GitHub `release.json`.
A complete HF repository snapshot already includes all weights; skip this step
when working from that snapshot. There is no required upstream Qwen download.

```text
base/                              # all 18 original base shards and tokenizer/processor/config
bundle/default/                    # parent LoRA, step 3000
bundle/mixed/                      # additional LoRA, step 343
bundle/decision_head.safetensors    # decision head
```

`base/` is a byte-identical copy of `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Both adapters remain unmerged.
Use **all** components together with the supplied Engine. Do not load just
`base/`, a single standard PEFT adapter, or a stock causal-LM generation pipeline.
The Engine loads the base and processor with `local_files_only=True`; after
installing dependencies and downloading the package, model inference requires
no connection to Hugging Face. Optionally check file integrity before loading:

```sh
.venv/bin/mixed27b verify --bundle ./bundle --base ./base
```

## Python usage

```python
from mixed27b.engine import Mixed27BEngine

engine = Mixed27BEngine(
    base="./base", bundle="./bundle", device="cuda:0", model="RSI-JEV-27b"
)
try:
    response, raw = engine(
        state={},
        questions={"answer": {
            "type": "choice",
            "instructions": "Which object is closer to the camera?",
            "criteria": {"A": "The chair", "B": "The table"}
        }},
        images=["example.jpg"]
    )
    print(response)
finally:
    engine.close()
```

`example.jpg` is a user-provided image, not bundled benchmark data. Images may be
paths, bytes, raw base64 or data URLs accepted by the official SDK. Order is preserved.
The returned `answers.answer` has `type`, `choice` and a probability for every option.
The selected choice is the argmax of a temperature-1 softmax over the decision
head logits for the supplied options. No probability calibration is applied.

## Official Vision runner

Build and verify the licensed Vision suite using the official SDK documentation.
Then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/decision-index vision pipeline \
  --dir /your/verified/vision-suite \
  --engine mixed27b.engine:Mixed27BEngine \
  --option base=/absolute/path/to/base \
  --option bundle=/absolute/path/to/bundle \
  --option model=RSI-JEV-27b \
  --workers 1 --out runs/rsi-jev-27b-vision
```

The published run used eight independent single-GPU official runner processes,
batch size 1 and workers=1 per process. The original rows were partitioned by
index modulo 8. `scripts/official_8gpu.py` records that orchestration. Original
result lines were merged without rewriting and scored by the unchanged official
SDK. All shard records and environments are published. Absolute paths in the
archived records describe the original server; use your own paths in these commands.

## Fixed inference settings and capacity

- BF16 backbone; FP32 parent/mixed LoRA and decision head; BF16 autocast; SDPA.
- Last-token readout, one forward pass per question, no autoregressive generation.
- 16 full-attention layers are bidirectional; 48 recurrent/linear layers remain causal.
- Both `default` and `mixed` adapters are active, unmerged. The decision head is the frozen parent head.
- Library-default FLA/Triton autotuning; no custom kernel pinning or probability-parity threshold.
- Native resolution capped at 1,638,400 pixels **per image**. No shared image budget,
  length-triggered shrinking, input truncation or option filtering.
- Choice questions with 2–26 options. The public suite contains 1–4 images per request;
  there is no added four-image rejection. Native context/memory limits still apply.
- `prompt.readout` is a generative-engine hint and is unused by this decision-head model.
  One prompt renderer is used for every benchmark; the engine receives no gold labels or IDs.
- No `noul` or `score` interface or Text leaderboard result is claimed in this release.

Model loading errors remain errors; neither adapter is silently omitted. The
optional `mixed27b verify` command is an offline file-integrity utility and is
not called during inference. No HTTP service is required for this custom Engine route.
