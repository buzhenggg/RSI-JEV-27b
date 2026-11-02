# RSI-JEV-27b

RSI-JEV-27b is a 27B multimodal decision model based on **Qwen3.8-27B**.
It answers supplied multiple-choice questions over images, returning a selected
option and probabilities for all options in one forward pass per question.

## Public Decision Index 0.3

| Panel | Public index | Correct / requests | Complete |
| --- | ---: | ---: | --- |
| [Vision](https://huggingface.co/datasets/k2021211255/RSI-JEV-27b-Results/blob/main/runs/rsi-jev-27b-vision/vision-scores.json) | **78.4096** | **9,940 / 12,210** | Yes |

Overall micro accuracy: **81.4087%**. This is the local public Vision index, not
the private/full leaderboard score. No Text benchmark result is claimed.

[Detailed evaluation results](https://huggingface.co/datasets/k2021211255/RSI-JEV-27b-Results) · [Inference code](https://github.com/buzhenggg/RSI-JEV-27b)

## Usage

Follow [inference/README.md](inference/README.md) for installation, a complete
model download, Python examples and the official runner command. The
[HF model repository](https://huggingface.co/k2021211255/RSI-JEV-27b)
contains **all 18 base-weight shards, tokenizer/processor files, both LoRA
adapters and the decision head**, plus the exact evaluated inference source.
No separate download from the upstream Qwen repository is required.

The complete model is stored as `base/` plus `bundle/`. Both adapters are active
and unmerged, exactly as in the evaluated model; use the supplied custom Engine.
The base weights alone are not RSI-JEV-27b. The GitHub repository contains code
and download instructions; the large weights are hosted on Hugging Face.

## Model composition

| Component | Source |
| --- | --- |
| Base (included in `base/`) | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Parent adapter | Multitask SFT step-3000, `default` LoRA, rank 16 / alpha 32 |
| Additional adapter | Additional fine-tuning step-343, `mixed` LoRA, rank 8 / alpha 16 |
| Decision head | Frozen parent 26 × 5120 readout |

## Per-benchmark results

| Benchmark | Accuracy | Chance-corrected score |
| --- | ---: | ---: |
| CV-Bench | 88.47% | 80.98 |
| BLINK | 85.22% | 77.12 |
| RealWorldQA | 84.21% | 74.20 |
| CharXiv | 84.59% | 79.45 |
| InfographicVQA | 98.36% | 97.81 |
| Mind2Web | 94.55% | 92.74 |
| Winoground | 92.62% | 85.25 |
| KIE (CORD+FUNSD) | 97.69% | 96.19 |
| Moderation (Hateful Memes) | 76.42% | 52.84 |
| R-Bench-M | 46.47% | 35.75 |
| MMMU-Pro vision | 47.80% | 40.13 |

The public index uses the official chance correction and weights, rather than a
simple mean of the raw accuracies. R-Bench-M has weight 0.5 and MMMU-Pro vision
0.25; the other nine benchmarks have weight 1.

## Inference hardware and latency

The complete public run used eight A800-SXM4-80GB GPUs, each running a separate
batch-1 model instance. Recorded per-request wall times: mean **777.36 ms**,
median **520.24 ms**, P80 **893.07 ms**, P95 **1,685.89 ms**. Image processing and
in-call compilation/autotuning costs are included; model initialization is excluded.
These are A800 measurements, not the maintainers' RTX PRO 6000 latency certification.

## License and availability

Model adapters and project code are released under Apache-2.0. The Qwen base and
runtime dependencies retain their own licenses; the vendored Decision Index SDK
is MIT-licensed. Benchmark image datasets are not included. This repository is
preparation for a Vision submission; no submission PR has been opened for this release.
