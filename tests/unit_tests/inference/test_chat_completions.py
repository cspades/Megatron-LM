# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import base64
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    compute_media_cache_key,
    serialize_multimodal_data,
)
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _extract_media_url_bytes,
    _extract_multimodal_from_messages,
    _has_previous_turn_tokens,
    _last_assistant_message,
    _replace_prefix_tokens_metadata,
    _sanitize_messages_for_template,
    _tokenize_with_media_slots_sync,
)


def test_extract_media_data_url_accepts_payload_at_limit():
    payload = b"four"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    assert _extract_media_url_bytes(url, max_bytes=len(payload)) == payload


def test_extract_media_data_url_rejects_decoded_payload_over_limit():
    # Four- and five-byte payloads both occupy eight base64 characters, so
    # this exercises the decoded-size check in addition to the encoded bound.
    payload = b"five!"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    with pytest.raises(ValueError, match="data:video/mp4;base64 payload exceeds 4 byte limit"):
        _extract_media_url_bytes(url, max_bytes=4)


def test_replace_prefix_tokens_metadata_ships_the_rendered_prefix_and_eos():
    eos = 99
    template_prefix = (1, 99, 2, 99)
    offload_params = {"ng_capture": {"staging_chain": ["k1"]}}

    out = _replace_prefix_tokens_metadata(eos, template_prefix, offload_params)

    assert out[PREFIX_TEMPLATE_TOKEN_IDS_FIELD] == [1, 99, 2, 99]
    assert out[PREFIX_EOS_TOKEN_ID_FIELD] == 99
    assert out["ng_capture"] == {"staging_chain": ["k1"]}
    assert offload_params == {"ng_capture": {"staging_chain": ["k1"]}}  # input not mutated


_USER = {"role": "user", "content": "hi"}
_ASSISTANT_TEXT = {"role": "assistant", "content": "hello"}
_ASSISTANT_WITH_TOKENS = {
    "role": "assistant",
    "content": "hello",
    "prompt_token_ids": [1, 2],
    "compact_prompt_token_ids": [1, 2],
    "generation_token_ids": [3, 99],
}
_ENGINE_METADATA = {"ng_capture": {"staging_chain": ["k1"]}}


def test_has_previous_turn_tokens():
    assert _has_previous_turn_tokens(None) is False
    assert _has_previous_turn_tokens(_ASSISTANT_TEXT) is False  # dataset-provided history
    assert _has_previous_turn_tokens(_ASSISTANT_WITH_TOKENS) is True


def test_last_assistant_message_returns_the_last_assistant_turn():
    assert _last_assistant_message([_USER]) == (None, None)
    assert _last_assistant_message([_USER, _ASSISTANT_TEXT, _USER]) == (1, _ASSISTANT_TEXT)
    messages = [_USER, _ASSISTANT_WITH_TOKENS, _USER, _ASSISTANT_TEXT, _USER]
    assert _last_assistant_message(messages) == (3, _ASSISTANT_TEXT)


def test_media_slot_uses_tokenizer_id_when_model_id_is_unspecified():
    class _Tokenizer:
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return "__MEDIA__"

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

    spec = MediaPromptSpec(model_token="<image>")
    prompt_config = MultimodalPromptConfig(image_spec=spec, video_spec=spec)

    tokens = _tokenize_with_media_slots_sync(
        _Tokenizer(),
        messages=[],
        media_slots=[("__MEDIA__", "image", 0)],
        prompt_config=prompt_config,
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [99]


def test_temporal_video_slot_uses_the_configured_compact_wrapper():
    class _Tokenizer:
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return "__VIDEO__"

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7] if text else []

    prompt_config = MultimodalPromptConfig(
        video_spec=MediaPromptSpec(
            model_token="<image>",
            prefix="<img>",
            suffix="</img>",
            expansion_mode="temporal_patch",
            include_frame_timestamps_for_nemotron_vl=True,
        )
    )

    tokens = _tokenize_with_media_slots_sync(
        _Tokenizer(),
        messages=[],
        media_slots=[("__VIDEO__", "video", 0)],
        prompt_config=prompt_config,
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [7, 99, 7]


def test_media_content_uses_the_configured_part_separator():
    prompt_config = MultimodalPromptConfig(video_spec=MediaPromptSpec(content_part_separator="\n"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "text", "text": "__VIDEO__"},
            ],
        }
    ]

    sanitized = _sanitize_messages_for_template(
        messages, media_slots=[("__VIDEO__", "video", 0)], prompt_config=prompt_config
    )

    assert sanitized[0]["content"] == "question\n__VIDEO__"


def test_media_first_content_order_matches_structured_hf_rendering():
    prompt_config = MultimodalPromptConfig(
        image_spec=MediaPromptSpec(content_part_separator="\n"), content_part_order="media_first"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "text", "text": "__IMAGE_0__"},
                {"type": "text", "text": "Image 1:"},
                {"type": "text", "text": "__IMAGE_1__"},
                {"type": "text", "text": "Image 2:"},
            ],
        }
    ]

    sanitized = _sanitize_messages_for_template(
        messages,
        media_slots=[("__IMAGE_0__", "image", 0), ("__IMAGE_1__", "image", 0)],
        prompt_config=prompt_config,
    )

    assert sanitized[0]["content"] == ("__IMAGE_0__\n__IMAGE_1__\nquestion\nImage 1:\nImage 2:")


_MEDIA_TAG_PATTERN = re.compile(r"(<img>|<image>|</img>)")


class _SegmentTokenizer:
    """Emits each run of plain text as one token so expected prompts stay readable."""

    unk_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<image>" else self.unk_token_id

    def tokenize(self, text):
        return [
            99 if part == "<image>" else part for part in _MEDIA_TAG_PATTERN.split(text) if part
        ]

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.tokenize(text)


def _media_block(modality, index):
    payload = base64.b64encode(f"{modality}-{index}".encode()).decode()
    if modality == "image":
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{payload}"}}


def _omni_prompt_config(content_part_order, frame_timestamps):
    defaults = NemotronOmniInferenceWrapper.multimodal_prompt_config
    return replace(
        defaults,
        content_part_order=content_part_order,
        video_spec=replace(
            defaults.video_spec, include_frame_timestamps_for_nemotron_vl=frame_timestamps
        ),
    )


def _endpoint_prompt_tokens(messages, prompt_config):
    messages, _images, _videos, media_slots = _extract_multimodal_from_messages(
        messages, prompt_config
    )
    template_messages = _sanitize_messages_for_template(messages, media_slots, prompt_config)
    return _tokenize_with_media_slots_sync(
        _SegmentTokenizer(),
        template_messages,
        media_slots,
        prompt_config,
        tools=None,
        chat_template_kwargs={},
    )


@pytest.mark.parametrize(
    "modality, frame_timestamps",
    [("image", False), ("video", False), ("video", True)],
    ids=["image", "video", "video_with_timestamps"],
)
@pytest.mark.parametrize(
    "content_part_order, expected_tokens",
    [
        (
            "preserve",
            [
                "<system>Be brief.<user>Compare these.\n",
                *("<img>", 99, "</img>"),
                "\nFirst.\n",
                *("<img>", 99, "</img>"),
                "\nSecond.",
            ],
        ),
        (
            "media_first",
            [
                "<system>Be brief.<user>",
                *("<img>", 99, "</img>"),
                "\n",
                *("<img>", 99, "</img>"),
                "\nCompare these.\nFirst.\nSecond.",
            ],
        ),
    ],
)
def test_endpoint_places_media_by_content_part_order(
    modality, frame_timestamps, content_part_order, expected_tokens
):
    """Frame timestamps are rendered later by the model wrapper, so the endpoint's
    compact prompt depends only on the content-part order."""
    messages = [
        # A list-content message without media must be left as-is under either order.
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these."},
                _media_block(modality, 0),
                {"type": "text", "text": "First."},
                _media_block(modality, 1),
                {"type": "text", "text": "Second."},
            ],
        },
    ]

    tokens = _endpoint_prompt_tokens(
        messages, _omni_prompt_config(content_part_order, frame_timestamps)
    )

    assert tokens == expected_tokens


@pytest.mark.parametrize(
    "frame_timestamps, expanded_video_tokens",
    [
        (
            True,
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: ",
                *("<img>", -1, "</img>"),
                "\nFrame 3 sampled at 2.00 seconds and frame 4 sampled at 3.00 seconds: ",
                *("<img>", -1, "</img>"),
            ],
        ),
        (False, [*("<img>", -1, "</img>"), "\n", *("<img>", -1, "</img>")]),
    ],
    ids=["with_timestamps", "without_timestamps"],
)
def test_omni_expands_media_first_video_prompt_ahead_of_text(
    frame_timestamps, expanded_video_tokens
):
    assert NemotronOmniInferenceWrapper.multimodal_prompt_config.content_part_order == (
        "media_first"
    )
    prompt_config = _omni_prompt_config("media_first", frame_timestamps)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe it."}, _media_block("video", 0)],
        }
    ]

    tokens = _endpoint_prompt_tokens(messages, prompt_config)
    assert tokens == ["<user>", "<img>", 99, "</img>", "\nDescribe it."]

    wrapper = object.__new__(NemotronOmniInferenceWrapper)
    wrapper.multimodal_prompt_config = prompt_config
    wrapper.model = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        patch_dim=16,
        vision_model=SimpleNamespace(temporal_patch_dim=2),
    )
    # 32x32 frames give one embedding per frame after pixel shuffle, so each
    # two-frame tubelet expands to a single -1 placeholder.
    expanded, _masks = wrapper.expand_image_tokens(
        [tokens],
        imgs_sizes=torch.tensor([[32, 32]] * 4),
        num_frames=torch.tensor([4]),
        image_token_id=99,
        tokenizer=_SegmentTokenizer(),
        video_frame_indices=[[0, 10, 20, 30]],
        video_fps=[10.0],
    )

    assert expanded == [["<user>", *expanded_video_tokens, "\nDescribe it."]]


def test_media_tokenization_is_synchronous_so_it_can_be_offloaded_whole():
    """Lowering media slots must not be a coroutine.

    Rendering is only the first of 3N+1 tokenizer calls for N slots. While this
    was async, awaiting the render put the remaining encodes back on the event
    loop, where they stall every other request the replica owns. Being a plain
    function is what lets the endpoint hand the whole thing to the tokenize
    executor in one hop, on the thread that owns the private tokenizer copy.
    """
    import inspect

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    assert not inspect.iscoroutinefunction(chat_completions._tokenize_with_media_slots_sync)
    # And the endpoint must not have kept a direct call that skips the executor.
    src = inspect.getsource(chat_completions.chat_completions)
    assert "_tokenize_with_media_slots_sync" in src
    for line in src.splitlines():
        if "_tokenize_with_media_slots_sync" in line:
            assert "await" not in line, f"must be dispatched via the executor, got: {line.strip()}"


@pytest.mark.asyncio
async def test_n_choices_prepare_and_serialize_shared_media_once():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    class _Tokenizer:
        chat_template = "test-template"
        unk_token_id = 0
        eod = None

        def apply_chat_template(self, messages, **_kwargs):
            return "".join(message["content"] for message in messages)

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

        def detokenize(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class _Client:
        def __init__(self):
            self.serialized_media = []

        def add_request_with_id(
            self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
        ):
            wire = serialize_multimodal_data(multi_modal_data)
            self.serialized_media.append(wire)
            request_id = len(self.serialized_media)
            future = asyncio.get_running_loop().create_future()
            future.set_result(
                {
                    "uid": f"choice-{request_id}",
                    "status": "COMPLETED",
                    "generated_tokens": [request_id],
                    "prompt_length": len(prompt_tokens),
                    "prompt_tokens": prompt_tokens,
                    "compact_prompt_tokens": prompt_tokens,
                    "num_cached_tokens": 0,
                    "sampling_params": sampling_params.serialize(),
                    "routing_indices": None,
                }
            )
            return request_id, future

        def abort_request(self, _request_id):
            raise AssertionError("Successful choices must not be aborted")

    tokenizer = _Tokenizer()
    client = _Client()
    spec = MediaPromptSpec(model_token="<image>")
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=tokenizer,
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(image_spec=spec, video_spec=spec),
        default_temperature=1.0,
        default_top_p=1.0,
        default_top_k=0,
        eval_mode=False,
    )
    app.register_blueprint(chat_completions.bp)
    image = b"shared-image"
    image_url = f"data:image/png;base64,{base64.b64encode(image).decode()}"

    with mock.patch(
        "megatron.core.inference.inference_request.compute_media_cache_key",
        wraps=compute_media_cache_key,
    ) as compute_key:
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    }
                ],
                "n": 3,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    assert len((await response.get_json())["choices"]) == 3
    assert len(client.serialized_media) == 3
    assert all(wire == client.serialized_media[0] for wire in client.serialized_media)
    assert all(wire is not client.serialized_media[0] for wire in client.serialized_media[1:])
    assert compute_key.call_count == 1
