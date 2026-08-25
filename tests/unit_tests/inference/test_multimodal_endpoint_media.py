# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import base64

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
    chat_completions,
)


def test_data_urls_preserve_media_mime_formats():
    payload = b"encoded media"
    encoded = base64.b64encode(payload).decode()

    assert (
        chat_completions._extract_media_url_bytes(
            f"data:image/webp;charset=utf-8;base64,{encoded}", "image", allow_http=True
        )
        == payload
    )
    assert (
        chat_completions._extract_media_url_bytes(
            f"data:video/mp4;base64,{encoded}", "video"
        )
        == payload
    )


def test_image_and_video_base64_have_separate_limits(monkeypatch):
    encoded = base64.b64encode(b"1234").decode()
    monkeypatch.setattr(chat_completions, "_MAX_IMAGE_BYTES", 3)
    monkeypatch.setattr(chat_completions, "_MAX_VIDEO_BYTES", 4)

    with pytest.raises(ValueError, match="Image exceeds 3 byte limit"):
        chat_completions._decode_base64_media(encoded, "image")

    assert chat_completions._decode_base64_media(encoded, "video") == b"1234"


def test_oversized_base64_is_rejected_before_decode(monkeypatch):
    monkeypatch.setattr(chat_completions, "_MAX_IMAGE_BYTES", 2)

    def unexpected_decode(*_args, **_kwargs):
        raise AssertionError("oversized input reached base64 decode")

    monkeypatch.setattr(chat_completions.base64, "b64decode", unexpected_decode)

    with pytest.raises(ValueError, match="Image exceeds 2 byte limit"):
        chat_completions._decode_base64_media("AAAAAAAA", "image")


@pytest.mark.parametrize("value", ["not base64!", "data:image/png;base64"])
def test_invalid_base64_and_malformed_data_urls_are_rejected(value):
    if value.startswith("data:"):
        with pytest.raises(ValueError, match="Malformed image data URL"):
            chat_completions._extract_media_url_bytes(value, "image", allow_http=True)
    else:
        with pytest.raises(ValueError, match="Invalid base64 image data"):
            chat_completions._decode_base64_media(value, "image")
