Submit **RSI-JEV-27b** to the **Vision** leaderboard with complete public Decision Index 0.3 results. The proposed change adds an entry to `submissions/README.md`. There is no Text submission or private/full score claim.

RSI-JEV-27b is a 27B multimodal decision model based on Qwen3.8-27B. It supports `choice` questions with 2–26 supplied options, returning an answer and all option probabilities in one forward pass per question.

## Model and results

- Complete model (base weights, both adapters, decision head) and inference source: [RSI-JEV-27b](https://huggingface.co/k2021211255/RSI-JEV-27b/tree/3a126489958a67c08233711a1c5db31c57919229), revision `3a126489958a67c08233711a1c5db31c57919229`.
- Results: [RSI-JEV-27b-Results](https://huggingface.co/datasets/k2021211255/RSI-JEV-27b-Results/tree/d92a0cdc2c59cf27e50074c5e1c4a047f8051577), revision `d92a0cdc2c59cf27e50074c5e1c4a047f8051577`.
- Evaluated inference implementation (unchanged): [mixed27b source](https://github.com/buzhenggg/RSI-JEV-27b/tree/13447a13b2df2e3bed6bf7145117eb045e3faf62/mixed27b), source commit `13447a13b2df2e3bed6bf7145117eb045e3faf62`.

| Public suite | Public index | Answered requests | Correct | Complete |
| --- | ---: | ---: | ---: | --- |
| [Vision 0.3](https://huggingface.co/datasets/k2021211255/RSI-JEV-27b-Results/blob/d92a0cdc2c59cf27e50074c5e1c4a047f8051577/runs/rsi-jev-27b-vision/vision-scores.json) | **78.4096464583** | 12,210 | 9,940 | Yes |

Predictions, option probabilities, scores, all shard environments and SHA256 manifests are available in the results repository. All 12,210 rows have status `ok`; `vision-scores.json` says `complete: true`. Original result records retain the internal label `mixed27b-step343`; RSI-JEV-27b is the release name. Inference source, weights and original result lines are unchanged for publication.

## Inference and hardware

- Use the bundled `mixed27b.engine:Mixed27BEngine` and [installation / reproduction instructions](https://huggingface.co/k2021211255/RSI-JEV-27b/blob/3a126489958a67c08233711a1c5db31c57919229/inference/README.md).
- Load `base/` and `bundle/` from the same pinned model snapshot: the included Qwen base, parent `default` LoRA, additional `mixed` LoRA and frozen parent decision head. Both adapters are explicitly active and remain unmerged.
- Official SDK commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`. Eight independent processes call the original official runner on disjoint shards (original row index modulo 8), merge unchanged JSONL lines and invoke the original official scorer. The suite passed the official verifier first; orchestration and shard records are included.
- Hardware: eight A800-SXM4-80GB GPUs; one complete model per GPU, batch=1 and workers=1 per process.
- Settings: BF16 backbone, FP32 adapters/head, BF16 autocast, SDPA, last-token readout, temperature 1, no generated chain of thought or permutation averaging. Sixteen full-attention layers are bidirectional; 48 recurrent/linear layers are causal. Normal FLA/Triton autotuning, no kernel pinning or probability-parity gate.
- Environment: Python 3.13.16, CUDA 12.8, PyTorch 2.8.0+cu128, Transformers 5.20.0, PEFT 0.21.2, FLA 0.3.2 and Triton 3.4.0. Remaining pins are in `requirements.lock`.

## Images and capacity

Native resolution is capped at **1,638,400 pixels per image**, using the official image helper and processor. No shared image budget, length-triggered shrinking, input truncation or option filtering is used. The public suite has 1–4 images per request; there is no added four-image rejection. The decision head supports 2–26 options; native context and GPU memory limits apply. Only `choice` is implemented; no `noul` or `score` interface is claimed. The generative `prompt.readout` hint is unused by this decision-head architecture.

## Latency

Recorded per-request wall time across all 12,210 rows: mean 777.36 ms, median 520.24 ms, P80 893.07 ms, P95 1,685.89 ms. Image processing and in-call compilation/autotuning are included; model loading is excluded. These A800 measurements under eight independent processes are **not** the maintainers' RTX PRO 6000 latency certification. No slow results were dropped.

## Training disclosure

The parent was trained for one epoch on 119,826 cleaned multitask questions; step-3000 was selected using internal validation. Additional training used 1,200 synthetic questions each from camera motion, counting ≥3, counting ≥5 and hateful memes, plus 4,800 clean replay questions: 9,600 examples in one epoch. Only the new mixed adapter was trained; the base, parent adapter and head stayed frozen. Mixed step-343 was selected using synthetic validation with internal retention constraints; DI was not used for checkpoint selection.

The synthetic families were designed after analysis of public benchmark weaknesses. This is targeted adaptation, not a blind benchmark experiment. Image-SHA audits found no overlap in the audited training/evaluation splits; no claim is made about unknown base pretraining data or private tests.

## Reproduction

Follow [inference/README.md](https://huggingface.co/k2021211255/RSI-JEV-27b/blob/3a126489958a67c08233711a1c5db31c57919229/inference/README.md) using the pinned release revisions above. Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The model, code and result repositories are public. All 18 base shards and tokenizer/processor files are included under `base/`; no upstream Qwen download is required. Offline loading and a synthetic-image prediction passed with both adapters active and zero outbound network attempts.
