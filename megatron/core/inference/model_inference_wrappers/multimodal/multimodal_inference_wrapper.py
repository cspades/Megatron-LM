# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import ClassVar, Optional

import torch

from megatron.core.inference.config import MultimodalPromptConfig


class MultimodalInferenceWrapperMixin:
    """Shared media-token behavior for multimodal inference wrappers."""

    multimodal_prompt_config: ClassVar[Optional[MultimodalPromptConfig]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configured_prompt_contract = getattr(
            self.inference_context.config, "multimodal_prompt_config", None
        )
        if configured_prompt_contract is not None:
            self.multimodal_prompt_config = configured_prompt_contract

    def resolve_media_token_id(self, tokenizer, modality: str) -> int:
        """Resolve a compact media marker to one nonnegative tokenizer ID."""
        prompt_config = self.multimodal_prompt_config
        if prompt_config is None:
            raise ValueError(f"{type(self).__name__} does not define a multimodal prompt contract.")
        spec = prompt_config.get_spec(modality)
        if not spec.model_token:
            raise ValueError(
                f"{type(self).__name__} does not define a model token for {modality} inputs."
            )

        candidates = [
            getattr(getattr(tokenizer, "_tokenizer", None), "tokenizer", None),
            getattr(tokenizer, "_tokenizer", None),
            getattr(tokenizer, "tokenizer", None),
            tokenizer,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            resolved_id = None
            if hasattr(candidate, "convert_tokens_to_ids"):
                try:
                    resolved_id = candidate.convert_tokens_to_ids(spec.model_token)
                except (TypeError, ValueError):
                    resolved_id = candidate.convert_tokens_to_ids([spec.model_token])
                if isinstance(resolved_id, (list, tuple)):
                    resolved_id = resolved_id[0] if len(resolved_id) == 1 else None
                if resolved_id == getattr(candidate, "unk_token_id", None):
                    resolved_id = None
            if resolved_id is None and hasattr(candidate, "encode"):
                try:
                    encoded = candidate.encode(spec.model_token, add_special_tokens=False)
                except TypeError:
                    encoded = candidate.encode(spec.model_token)
                if torch.is_tensor(encoded):
                    encoded = encoded.reshape(-1).tolist()
                if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
                    resolved_id = encoded[0]
            if resolved_id is None and hasattr(candidate, "tokenize"):
                encoded = candidate.tokenize(spec.model_token)
                if torch.is_tensor(encoded):
                    encoded = encoded.reshape(-1).tolist()
                if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
                    resolved_id = encoded[0]
            try:
                resolved_id = int(resolved_id) if resolved_id is not None else None
            except (TypeError, ValueError):
                resolved_id = None
            if resolved_id is not None and resolved_id >= 0:
                return resolved_id

        raise ValueError(
            f"Tokenizer does not define {spec.model_token!r} as one nonnegative token."
        )

    def get_preexpanded_media_token_id(self, modality: str) -> int:
        """Return the model-internal marker used by already-expanded prompts."""
        raise NotImplementedError(
            f"{type(self).__name__} must define its pre-expanded {modality} token id."
        )

    def build_preexpanded_media_token_mask(
        self, prompt_tokens: torch.Tensor, modality: str
    ) -> torch.Tensor:
        """Map pre-expanded media-token positions to sequential embedding indices."""
        media_token_id = self.get_preexpanded_media_token_id(modality)
        media_positions = prompt_tokens == int(media_token_id)
        media_indices = torch.nonzero(media_positions, as_tuple=False).flatten()
        mask = torch.full_like(prompt_tokens, -1, dtype=torch.int64)
        mask[media_indices] = torch.arange(
            media_indices.numel(), dtype=torch.int64, device=prompt_tokens.device
        )
        return mask
