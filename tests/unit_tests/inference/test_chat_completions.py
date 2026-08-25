# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _add_tokenized_response_data,
    _extract_multimodal_from_messages,
    _media_slots_through_message,
    _previous_turn_token_ids,
    _tokenize_with_media_slots,
)


class _ChatTokenizer:
    unk_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return []

    def convert_tokens_to_ids(self, token):
        return 77 if token == "<image>" else self.unk_token_id


def _llava_image_prompt_config():
    spec = MediaPromptSpec(model_token="<image>")
    return MultimodalPromptConfig(image_spec=spec, video_spec=spec)


def test_adjacent_media_slots_use_real_tokenizer_ids_and_stay_distinct():
    tokenizer = _ChatTokenizer()
    media_slots = [("__MCORE_MEDIA_SLOT_0__", "image", 0), ("__MCORE_MEDIA_SLOT_1__", "image", 0)]

    tokens = _tokenize_with_media_slots(
        tokenizer,
        [{"role": "user", "content": "__MCORE_MEDIA_SLOT_0____MCORE_MEDIA_SLOT_1__"}],
        media_slots,
        _llava_image_prompt_config(),
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [77, 77]


def test_compact_prompt_tokens_preserve_adjacent_placeholders_for_stitching():
    message = {
        "prompt_token_ids": [10, -1, -1, -1, -1, 20],
        "compact_prompt_token_ids": [10, 77, 77, 20],
        "generation_token_ids": [30, 31],
    }

    assert _previous_turn_token_ids(message, _ChatTokenizer(), _llava_image_prompt_config()) == [
        10,
        77,
        77,
        20,
        30,
        31,
    ]


def test_previous_turn_tokens_fall_back_for_old_chat_responses():
    message = {"prompt_token_ids": [10, -1, -1, 20], "generation_token_ids": [30]}

    assert _previous_turn_token_ids(message, _ChatTokenizer(), _llava_image_prompt_config()) == [
        10,
        77,
        20,
        30,
    ]


def test_tokenized_response_returns_expanded_and_compact_prompt_ids():
    message = {"role": "assistant", "content": "ok"}
    _add_tokenized_response_data(
        message,
        {
            "prompt_tokens": [10, -1, -1, 20],
            "compact_prompt_tokens": [10, 77, 20],
            "generated_tokens": [30],
        },
    )

    assert message["prompt_token_ids"] == [10, -1, -1, 20]
    assert message["compact_prompt_token_ids"] == [10, 77, 20]
    assert message["generation_token_ids"] == [30]


def test_media_slots_track_source_message_index_structurally():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
        },
        {"role": "assistant", "content": "seen"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}}],
        },
    ]

    _, _, _, media_slots = _extract_multimodal_from_messages(messages, MultimodalPromptConfig())

    assert [slot[2] for slot in media_slots] == [0, 2]
    assert _media_slots_through_message(media_slots, 1) == [media_slots[0]]


def test_llava_expands_positive_tokenizer_id_directly():
    wrapper = object.__new__(VLMInferenceWrapper)
    model = SimpleNamespace(image_token_index=-200, img_seq_len=2)
    wrapper.model = model
    compact_tokens = [[10, 77, 77, 20]]

    expanded_tokens, masks = wrapper.expand_image_tokens(
        compact_tokens, image_token_id=77, num_tiles=torch.tensor([1, 1])
    )

    assert compact_tokens == [[10, 77, 77, 20]]
    assert expanded_tokens == [[10, -1, -1, -1, -1, 20]]
    assert masks == [[None, 0, 1, 2, 3, None]]


def test_llava_prompt_config_uses_compact_image_token():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(image_token_index=-200)

    config = wrapper.get_multimodal_prompt_config()

    assert config.image_spec.model_token == "<image>"
    assert config.video_spec.model_token == "<image>"


def test_llava_resolves_compact_token_from_tokenizer():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(image_token_index=-200)

    assert wrapper.resolve_media_token_id(_ChatTokenizer(), "image") == 77


def test_media_token_resolution_requires_configured_model_token():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.get_multimodal_prompt_config = MultimodalPromptConfig

    with pytest.raises(ValueError, match="does not define a model token for image"):
        wrapper.resolve_media_token_id(_ChatTokenizer(), "image")
