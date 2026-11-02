"""Official Engine contract. No dataset IDs, gold labels or result lookup."""
import hashlib
import json
import platform
import threading
from pathlib import Path

from decision_index.engines import Engine, Unsupported, validate
from decision_index.vision.images import pil

CAP = 1_638_400
CONTEXT = 262_144


def render_question(state, question):
    if question.get('type') != 'choice':
        raise Unsupported('Only choice questions are supported')
    criteria = question.get('criteria')
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
        raise Unsupported('The decision head supports 2 through 26 options')
    keys = sorted(criteria)
    text = question.get('instructions', '')
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, separators=(',', ':'))
    if state not in (None, '', {}, []):
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(',', ':'))
        text = 'Context:\n' + context + '\n\n' + text
    values = [criteria[k] for k in keys]
    prompt = text + '\n' + '\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(values)) + '\nAnswer with the option letter only.'
    return keys, prompt


class Mixed27BEngine(Engine):
    name = 'mixed27b-step343'
    latency = 'Synchronized single-GPU serial wall time: decode/resize, prompt, processor, H2D, forward and CPU probabilities; excludes model initialization.'

    def __init__(self, bundle=None, base=None, image_root=None, model=None, device='cuda:0', **options):
        if bundle is None or base is None:
            raise ValueError('Explicit bundle and base directories are required')
        import torch
        import transformers
        import peft
        from torch import nn
        from transformers import AutoProcessor, Qwen3_5Model
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from peft import PeftModel
        from safetensors.torch import load_file
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from .attention import enable_bidirectional_full_attention

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('This release requires one CUDA GPU with BF16 support')
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        torch.manual_seed(2026110201)
        torch.cuda.manual_seed_all(2026110201)
        self._lock = threading.Lock()
        self.image_root = image_root
        self.model_name = model or self.name
        bundle, base = Path(bundle).resolve(), Path(base).resolve()

        def delta(q, k, v, g, beta, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
            return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                initial_state=initial_state, output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens, head_first=False)

        qwen.torch_chunk_gated_delta_rule = delta
        backbone = Qwen3_5Model.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': self.device},
            attn_implementation='sdpa')
        attention = enable_bidirectional_full_attention(backbone.language_model)
        backbone.language_model = PeftModel.from_pretrained(backbone.language_model,
            bundle/'default', adapter_name='default', is_trainable=False)
        backbone.language_model.load_adapter(bundle/'mixed', adapter_name='mixed', is_trainable=False)
        backbone.language_model.base_model.set_adapter(['default', 'mixed'])
        head = nn.Linear(5120, 26, bias=False, dtype=torch.float32, device=self.device)
        head.load_state_dict(load_file(str(bundle/'decision_head.safetensors')), strict=True)

        class DecisionModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__(); self.backbone = backbone; self.decision_head = head
            def forward(self, inputs, n):
                hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
                return self.decision_head(hidden).float()[:, :n]

        self.model = DecisionModel(backbone, head)
        self.model.requires_grad_(False); self.model.eval()
        self._check_activation()
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        self.provenance = {
            'base_model': 'Qwen/Qwen3.8-27B',
            'base_revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'active_adapters': ['default', 'mixed'],
            'parent_step': 3000, 'mixed_step': 343, 'decision_head': 'frozen parent head',
            'attention': attention, 'max_pixels': CAP,
            'kernel_policy': 'Library default FLA/Triton autotuning; no kernel override',
            'image_resize': 'Official capped/pil, native up to 1638400 per image; no shared budget or length-triggered shrink',
            'limits': {'max_options': 26, 'max_sequence_tokens': CONTEXT, 'question_types': ['choice']},
            'processor_image_budget': {'max_pixels': CAP, 'min_pixels': self.processor.image_processor.size['shortest_edge']},
            'precision': 'BF16 backbone, FP32 adapters/head, BF16 autocast',
            'readout': 'single forward decision head; autoregressive prompt.readout is not used',
            'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
        }

    def _check_activation(self):
        if self.model.backbone.language_model.active_adapters != ['default', 'mixed']:
            raise RuntimeError('Both default and mixed LoRA must be active')

    def __call__(self, state, questions, images=(), prompt=None):
        torch = self.torch
        if not isinstance(questions, dict) or not questions:
            raise Unsupported('At least one typed question is required')
        # Validate every question before doing any inference; never drop options.
        rendered = {key: render_question(state, q) for key, q in questions.items()}
        with self._lock, torch.no_grad():
            self.synchronize()
            pictures = [pil(ref, root=self.image_root, max_pixels=CAP).convert('RGB') for ref in images]
            image_pixels = [im.width * im.height for im in pictures]
            answers, details = {}, {}
            for key, (keys, text) in rendered.items():
                content = [{'type': 'image', 'image': im} for im in pictures] + [{'type': 'text', 'text': text}]
                inp = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors='pt', processor_kwargs={'max_pixels': CAP, 'truncation': False})
                tokens = int(inp['input_ids'].shape[1])
                grid = inp['image_grid_thw'].tolist() if 'image_grid_thw' in inp else []
                values = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inp.items() if isinstance(v, torch.Tensor)}
                values = {k: v.to(self.device) for k, v in values.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(values, len(keys))
                probabilities = logits.softmax(-1)[0].tolist()
                best = max(range(len(keys)), key=probabilities.__getitem__)
                answers[key] = {'type': 'choice', 'choice': keys[best], 'probabilities': dict(zip(keys, probabilities))}
                details[key] = {'sequence_tokens': tokens, 'image_grid_thw': grid,
                    'image_tokens': sum(t*h*w//4 for t, h, w in grid),
                    'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
            self.synchronize()
            response = {'model': self.model_name, 'answers': answers}
            validate(questions, response)
            return response, {'image_pixels': image_pixels,
                'image_tokens': max(v['image_tokens'] for v in details.values()),
                'questions': details, 'active_adapters': ['default', 'mixed']}

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def runtime(self):
        return {'python': platform.python_version(), 'torch': self.torch.__version__,
            'cuda': self.torch.version.cuda, 'gpu': self.torch.cuda.get_device_name(self.device),
            'device': str(self.device), 'batch_size': 1}

    def close(self):
        self.synchronize()
        del self.model
        self.torch.cuda.empty_cache()
