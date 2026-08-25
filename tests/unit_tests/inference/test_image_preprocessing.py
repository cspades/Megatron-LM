# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    image_preprocessing,
)


def _image_config(**kwargs):
    values = {
        "patch_dim": 16,
        "dynamic_resolution": True,
        "dynamic_resolution_min_patches": 1,
        "dynamic_resolution_max_patches": 128,
    }
    values.update(kwargs)
    return ImageProcessingConfig(**values)


def test_video_target_resolution_uses_dynamic_grid_constraints(monkeypatch):
    monkeypatch.setattr(
        image_preprocessing,
        "_load_frame_sequence_manifest",
        lambda *_args: [Image.new("RGB", (300, 100))],
    )
    target_sizes = []

    def fake_preprocess(_frame, _config, target_hw=None, device=None):
        target_sizes.append(target_hw)
        return torch.zeros((1, 1, 1), device=device), torch.tensor(
            [target_hw], dtype=torch.int32, device=device
        )

    monkeypatch.setattr(image_preprocessing, "preprocess_image", fake_preprocess)
    image_config = _image_config(
        pixel_shuffle=True,
        spatial_merge_size=4,
        dynamic_resolution_min_patches=16,
        dynamic_resolution_max_patches=64,
    )
    config = VideoProcessingConfig(
        image_config=image_config, num_frames=1, frame_manifest_magic=b"frames:"
    )

    image_preprocessing.preprocess_video_bytes_list([b"video"], config)
    height, width = target_sizes[0]
    grid_height, grid_width = height // image_config.patch_dim, width // image_config.patch_dim

    assert 16 <= grid_height * grid_width <= 64
    assert grid_height % 4 == 0
    assert grid_width % 4 == 0


def test_frame_manifest_reads_trusted_rl_paths(tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(frame_path)
    magic = b"frames:"
    payload = magic + json.dumps({"frame_paths": [str(frame_path)]}).encode()

    frames = image_preprocessing._load_frame_sequence_manifest(payload, magic)
    assert len(frames) == 1
    assert frames[0].size == (8, 8)


@pytest.mark.parametrize("declared_frames", [10, 0])
def test_pyav_sampling_converts_only_bounded_frames(monkeypatch, declared_frames):
    converted = []

    class FakeFrame:
        def __init__(self, index):
            self.index = index

        def to_image(self):
            converted.append(self.index)
            return Image.new("RGB", (16, 16))

    class FakeContainer:
        def __init__(self):
            self.stream = SimpleNamespace(frames=declared_frames)
            self.streams = SimpleNamespace(video=[self.stream])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, stream):
            assert stream is self.stream
            yield from (FakeFrame(index) for index in range(10))

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _payload: FakeContainer()))
    config = VideoProcessingConfig(image_config=_image_config(), num_frames=3)

    frames = image_preprocessing._decode_sampled_video_frames(b"video", config)

    assert len(frames) == 3
    assert len(converted) == 3
    if declared_frames:
        assert converted == [0, 4, 9]


def test_pyav_num_frames_minus_one_decodes_all_frames(monkeypatch):
    converted = []

    class FakeFrame:
        def __init__(self, index):
            self.index = index

        def to_image(self):
            converted.append(self.index)
            return Image.new("RGB", (16, 16))

    class FakeContainer:
        streams = SimpleNamespace(video=[SimpleNamespace(frames=5)])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            yield from (FakeFrame(index) for index in range(5))

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _payload: FakeContainer()))
    config = VideoProcessingConfig(image_config=_image_config(), num_frames=-1)

    frames = image_preprocessing._decode_sampled_video_frames(b"video", config)

    assert len(frames) == 5
    assert converted == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("num_frames", [0, -2])
def test_video_num_frames_rejects_invalid_nonpositive_values(num_frames):
    config = VideoProcessingConfig(image_config=_image_config(), num_frames=num_frames)

    with pytest.raises(ValueError, match="must be positive or -1"):
        image_preprocessing.preprocess_video_bytes_list([b"video"], config)


def test_video_reference_resolution_is_computed_per_video(monkeypatch):
    first = Image.new("RGB", (32, 16))
    second = Image.new("RGB", (16, 48))
    decoded = iter([([first], True), ([second], True)])
    target_sizes = []

    monkeypatch.setattr(
        image_preprocessing, "_load_frame_sequence_manifest", lambda *_args: next(decoded)[0]
    )

    def fake_preprocess(_frame, _config, target_hw=None, device=None):
        target_sizes.append(target_hw)
        return torch.zeros((1, 1, 1), device=device), torch.tensor(
            [target_hw], dtype=torch.int32, device=device
        )

    monkeypatch.setattr(image_preprocessing, "preprocess_image", fake_preprocess)
    config = VideoProcessingConfig(
        image_config=_image_config(), num_frames=1, frame_manifest_magic=b"frames:"
    )

    result = image_preprocessing.preprocess_video_bytes_list([b"first", b"second"], config)

    assert target_sizes == [(16, 32), (48, 16)]
    assert result["num_frames"].tolist() == [1, 1]
