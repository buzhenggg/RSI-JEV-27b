"""Bidirectional full-attention for Qwen3.5 classification; recurrent layers stay causal.

Apply this hook during both training and inference. The multimodal backbone computes
positions and inserts image features first; only its language model's masks change.
No cached decoding, parameter changes, or label-dependent input is introduced.
"""
import torch
from transformers.masking_utils import create_recurrent_attention_mask


def enable_bidirectional_full_attention(text_model):
    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("This implementation is verified for SDPA only")
    if getattr(text_model, "_decision_bidir_hook", None) is not None:
        raise ValueError("Bidirectional hook is already installed")

    def masks(module, args, kwargs):
        if args:
            raise ValueError("Use keyword inputs for the language backbone")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Bidirectional classification requires use_cache=False")
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            embeds = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.bool)
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError("Expected a two-dimensional padding mask")
        if tuple(padding.shape) != tuple(embeds.shape[:2]):
            raise ValueError("Padding mask and embeddings have different shapes")
        positions = kwargs.get("position_ids")
        text_positions = positions[0] if positions is not None and positions.ndim == 3 and positions.shape[0] == 4 else positions if positions is not None and positions.ndim == 2 else None
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeds, attention_mask=padding,
                past_key_values=None, position_ids=text_positions),
        }
        return args, kwargs

    text_model._decision_bidir_hook = text_model.register_forward_pre_hook(masks, with_kwargs=True)
    text_model.config.decision_attention_mode = "noncausal_full_attention"
    return {"full_attention_layers": text_model.config.layer_types.count("full_attention"),
            "causal_recurrent_layers": text_model.config.layer_types.count("linear_attention"),
            "parameters_added": 0}
