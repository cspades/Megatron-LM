# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal Nemotron Omni text+image/video inference through MegatronAsyncLLM.

It loads the Megatron-Bridge Nemotron Omni model, creates configurable mock media
in memory, and submits correctness or benchmark requests through the high-level
dynamic-inference API.

Launch with any number of GPUs. TP and EP both default to one:

    torchrun --standalone --nproc-per-node=<num-gpus> \
      examples/inference/omni_multimodal_infer.py \
      --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --graph-mode decode

Use ``--graph-mode off`` as the eager baseline and ``--graph-mode all`` to
include multimodal prefill steps in CUDA-graph selection.

Requires pip installation of Megatron-Bridge for the model provider.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

from megatron.bridge import AutoBridge
from megatron.core.inference.apis import MegatronAsyncLLM, MegatronLLM
from megatron.core.inference.apis.serve_config import ServeConfig
from megatron.core.inference.config import (
    ImageProcessingConfig,
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingCoordinatorPolicy,
    PrefixCachingEvictionPolicy,
    VideoProcessingConfig,
)
from megatron.core.inference.inference_request import DynamicInferenceEventType
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        help="HF model ID or local HF checkpoint directory.",
    )
    parser.add_argument(
        "--megatron-checkpoint",
        default=None,
        help="Optional converted Megatron checkpoint. Without it, Bridge converts HF weights.",
    )
    parser.add_argument(
        "--api",
        choices=("async", "sync", "completions"),
        default="completions",
        help="Submit directly or through the OpenAI-compatible completions endpoint.",
    )
    parser.add_argument(
        "--tp",
        "--tensor-model-parallel-size",
        dest="tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor-model-parallel size.",
    )
    parser.add_argument(
        "--ep",
        "--expert-model-parallel-size",
        dest="expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert-model-parallel size.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("off", "decode", "all"),
        default="decode",
        help="CUDA graphs off, decode-only, or decode+prefill.",
    )
    parser.add_argument(
        "--cuda-graph-scope",
        choices=("layer", "block", "none"),
        default="block",
        help="Megatron local CUDA-graph capture granularity.",
    )
    parser.add_argument("--num-cuda-graphs", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-gb", type=int, default=4)
    parser.add_argument("--mamba-cache-gb", type=float, default=20.0)
    parser.add_argument("--vision-cache-gb", type=float, default=0.5)
    parser.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable chunked prefill.",
    )
    parser.add_argument(
        "--inference-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the inference-optimized transformer implementation.",
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable prefix caching.",
    )
    parser.add_argument(
        "--prefix-caching-eviction-policy",
        choices=("ref_zero", "lru"),
        default="lru",
    )
    parser.add_argument(
        "--prefix-caching-coordinator-policy",
        choices=("longest_prefix", "first_prefix_block", "load_balanced"),
        default="longest_prefix",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ignore EOS/EOD and always decode --max-new-tokens tokens.",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable reasoning with the model's /think and /no_think controls.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Number of sequential requests to submit.",
    )
    parser.add_argument(
        "--chain-requests",
        action="store_true",
        help="Append each response and another user turn to the next request for prefix reuse.",
    )
    parser.add_argument(
        "--media",
        choices=("image", "video"),
        default="image",
        help="Send either the generated mock PNG or mock MP4.",
    )
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--image-max-patches", type=int, default=128)
    parser.add_argument("--video-width", type=int, default=384)
    parser.add_argument("--video-height", type=int, default=256)
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=8,
        help="Number of frames generated and sampled for mock video input.",
    )
    parser.add_argument("--coordinator-port", type=int, default=50055)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument(
        "--prompt",
        default=(
            # ~ 50 Tokens
            "Carefully analyze the visual content. Describe the main subjects, objects, colors, "
            "shapes, text, spatial relationships, background details, lighting, and overall "
            "composition. For video, also explain notable actions, motion, scene changes, and "
            "temporal relationships. Be precise, factual, and concise."
        ),
    )
    return parser.parse_args()


def _validate_media_dimensions(width: int, height: int, media_name: str) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"--{media_name}-width and --{media_name}-height must be positive.")


def validate_parallelism(args: argparse.Namespace) -> None:
    """Validate TP and EP against the torchrun world size."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    for name, size in (
        ("--tp", args.tensor_model_parallel_size),
        ("--ep", args.expert_model_parallel_size),
    ):
        if size <= 0:
            raise ValueError(f"{name} must be positive.")
        if world_size % size != 0:
            raise ValueError(
                f"WORLD_SIZE={world_size} must be divisible by {name}={size}."
            )


def make_mock_png(width: int, height: int, image_index: int = 0) -> bytes:
    """Create a deterministic image without downloading a dataset."""
    _validate_media_dimensions(width, height, "image")
    image = Image.new("RGB", (width, height), color=(235, 240, 248))
    draw = ImageDraw.Draw(image)
    line_width = max(1, min(width, height) // 64)
    draw.rectangle(
        (width * 0.08, height * 0.14, width * 0.44, height * 0.72),
        fill=(32, 112, 220),
        outline="black",
        width=line_width,
    )
    draw.ellipse(
        (width * 0.56, height * 0.18, width * 0.91, height * 0.70),
        fill=(242, 92, 84),
        outline="black",
        width=line_width,
    )
    draw.text(
        (width * 0.20, height * 0.84),
        f"NEMOTRON OMNI {image_index + 1}",
        fill=(10, 10, 10),
    )
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_mock_mp4(num_frames: int, width: int, height: int) -> bytes:
    """Create a deterministic moving-shape video without downloading a dataset."""
    if num_frames <= 0:
        raise ValueError("--video-num-frames must be positive.")
    _validate_media_dimensions(width, height, "video")
    if width % 2 or height % 2:
        raise ValueError("--video-width and --video-height must be even for yuv420p.")

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Mock video generation requires PyAV (`pip install av`).") from error

    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    video_stream = container.add_stream("mpeg4", rate=4)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "yuv420p"

    for frame_index in range(num_frames):
        image = Image.new("RGB", (width, height), color=(235, 240, 248))
        draw = ImageDraw.Draw(image)
        progress = frame_index / max(num_frames - 1, 1)
        box_width = max(1, round(width * 0.26))
        box_height = max(1, round(height * 0.39))
        x = round(width * (0.07 + progress * 0.60))
        y = round(height * 0.21)
        draw.rectangle(
            (x, y, min(x + box_width, width - 1), min(y + box_height, height - 1)),
            fill=(32, 112, 220),
            outline="black",
            width=max(1, min(width, height) // 64),
        )
        draw.text(
            (width * 0.05, height * 0.06),
            f"NEMOTRON OMNI - FRAME {frame_index + 1}",
            fill=(10, 10, 10),
        )
        frame = av.VideoFrame.from_image(image)
        for packet in video_stream.encode(frame):
            container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def configure_provider(provider: Any, args: argparse.Namespace) -> None:
    """Apply the requested parallelism and graph mode."""
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = args.tensor_model_parallel_size > 1
    provider.pipeline_dtype = torch.bfloat16
    provider.dynamic_resolution = True
    provider.temporal_patch_dim = 1
    provider.separate_video_embedder = False
    provider.temporal_ckpt_compat = False
    provider.vision_class_token_len = 10
    provider.transformer_impl = (
        "inference_optimized" if args.inference_optimized else "transformer_engine"
    )
    if args.inference_optimized:
        num_groups = getattr(provider, "moe_router_num_groups", None)
        group_topk = getattr(provider, "moe_router_group_topk", None)
        if num_groups not in (None, 1) or group_topk not in (None, 1):
            raise ValueError(
                "The inference-optimized router does not support group-limited routing; "
                f"got moe_router_num_groups={num_groups} and "
                f"moe_router_group_topk={group_topk}."
            )
        # One group selects from every expert regardless of EP size, so removing
        # this redundant restriction preserves the Nano model's routing.
        provider.moe_router_num_groups = None
        provider.moe_router_group_topk = None
    provider.cuda_graph_impl = "none" if args.graph_mode == "off" else "local"
    provider.inference_cuda_graph_scope = (
        "none" if args.graph_mode == "off" else args.cuda_graph_scope
    )
    provider.moe_pad_experts_for_cuda_graph_inference = (
        args.graph_mode != "off"
        and not args.inference_optimized
        and args.expert_model_parallel_size > 1
    )


def model_overrides(args: argparse.Namespace) -> dict:
    """Return overrides needed when loading an already converted checkpoint."""
    return {
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": args.tensor_model_parallel_size > 1,
        "pipeline_dtype": torch.bfloat16,
        "dynamic_resolution": True,
        "temporal_patch_dim": 1,
        "separate_video_embedder": False,
        "temporal_ckpt_compat": False,
        "vision_class_token_len": 10,
        "transformer_impl": (
            "inference_optimized" if args.inference_optimized else "transformer_engine"
        ),
        "moe_router_num_groups": None if args.inference_optimized else 1,
        "moe_router_group_topk": None if args.inference_optimized else 1,
        "cuda_graph_impl": "none" if args.graph_mode == "off" else "local",
        "inference_cuda_graph_scope": (
            "none" if args.graph_mode == "off" else args.cuda_graph_scope
        ),
        "moe_pad_experts_for_cuda_graph_inference": (
            args.graph_mode != "off"
            and not args.inference_optimized
            and args.expert_model_parallel_size > 1
        ),
    }


def load_model(args: argparse.Namespace):
    """Load/convert and return the canonical Bridge NemotronOmniModel."""
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=args.megatron_checkpoint is None)
    configure_provider(provider, args)
    provider.initialize_model_parallel(
        seed=1234,
        seed_kwargs={"inference_rng_tracker": True},
    )

    if args.megatron_checkpoint:
        distributed_models = bridge.load_megatron_model(
            args.megatron_checkpoint,
            mp_overrides=model_overrides(args),
            wrap_with_ddp=False,
        )
    else:
        provider.finalize()
        distributed_models = provider.provide_distributed_model(wrap_with_ddp=False)

    if len(distributed_models) != 1:
        raise RuntimeError(
            "This example requires pipeline_model_parallel_size=1 and one local model chunk."
        )

    model = distributed_models[0].cuda().bfloat16().eval()
    model = model.module if hasattr(model, "module") else model

    # Training checkpoints may retain a bound optimizer loss scaler.
    model.config.grad_scale_func = None
    model.language_model.config.grad_scale_func = None
    return model


def build_tokenizer(model_path: str) -> MegatronTokenizerText:
    return MegatronTokenizerText(
        model_path,
        {"library": "huggingface"},
        trust_remote_code=True,
        use_fast=True,
        include_special_tokens=True,
    )


def build_initial_conversation(args: argparse.Namespace) -> list[dict[str, str]]:
    """Build the first request, including one marker for each media item."""
    num_media_items = args.num_images if args.media == "image" else 1
    if num_media_items <= 0:
        raise ValueError("--num-images must be positive.")
    media_markers = "\n".join("<img><image></img>" for _ in range(num_media_items))
    return [
        {"role": "system", "content": "/think" if args.reasoning else "/no_think"},
        {"role": "user", "content": f"{media_markers}\n{args.prompt}"},
    ]


def build_prompt_tokens(tokenizer, model, conversation: list[dict[str, str]]) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer.tokenize(prompt)
    expected_markers = sum(
        message["content"].count("<img><image></img>") for message in conversation
    )
    marker_count = tokens.count(int(model.image_token_index))
    if marker_count != expected_markers:
        raise RuntimeError(
            f"Expected {expected_markers} image marker tokens ({model.image_token_index}), "
            f"got {marker_count}. "
            "Check that the HF tokenizer matches the checkpoint."
        )
    return tokens


def build_sampling_params(tokenizer, args: argparse.Namespace) -> SamplingParams:
    """Build deterministic sampling parameters for one request."""
    return SamplingParams(
        temperature=1.0,
        top_k=1,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=-1 if args.benchmark else tokenizer.eod,
        skip_prompt_log_probs=True,
    )


def build_media_data(args: argparse.Namespace) -> dict[str, Any]:
    """Generate the requested mock image(s) or video."""
    if args.media == "video":
        return {
            "video": make_mock_mp4(
                args.video_num_frames,
                args.video_width,
                args.video_height,
            )
        }
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")
    return {
        "image": [
            make_mock_png(args.image_width, args.image_height, image_index)
            for image_index in range(args.num_images)
        ]
    }


def append_conversation_turn(
    conversation: list[dict[str, str]],
    generated_text: str,
    user_prompt: str,
) -> None:
    """Extend a conversation so the next request reuses its complete prefix."""
    conversation.extend(
        (
            {"role": "assistant", "content": generated_text},
            {"role": "user", "content": user_prompt},
        )
    )


def build_inference_config(model, args: argparse.Namespace) -> InferenceConfig:
    if args.mamba_cache_gb < 0:
        raise ValueError("--mamba-cache-gb must be non-negative.")
    if args.vision_cache_gb < 0:
        raise ValueError("--vision-cache-gb must be non-negative.")
    image_config = ImageProcessingConfig(
        patch_dim=int(model.patch_dim),
        dynamic_resolution=True,
        use_tiling=False,
        pixel_shuffle=True,
        spatial_merge_size=1,
        dynamic_resolution_min_patches=16,
        dynamic_resolution_max_patches=args.image_max_patches,
        vision_model_type="radio",
    )
    return InferenceConfig(
        block_size_tokens=256,
        buffer_size_gb=args.kv_cache_gb,
        max_requests=2,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_sequence_length,
        enable_chunked_prefill=args.chunked_prefill,
        enable_prefix_caching=args.prefix_caching,
        prefix_caching_mamba_gb=args.mamba_cache_gb if args.prefix_caching else None,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy(
            args.prefix_caching_eviction_policy
        ),
        prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(
            args.prefix_caching_coordinator_policy
        ),
        vision_embedding_cache_max_bytes=int(args.vision_cache_gb * 1024**3),
        track_generated_token_events=True,
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(
            model.language_model
        ),
        pg_collection=model.pg_collection,
        num_cuda_graphs=(
            None if args.graph_mode == "off" else args.num_cuda_graphs
        ),
        use_cuda_graphs_for_non_decode_steps=args.graph_mode == "all",
        cuda_graph_max_tokens=min(args.max_tokens, 512),
        image_preprocessing_config=image_config,
        video_preprocessing_config=VideoProcessingConfig(
            image_config=image_config,
            num_frames=args.video_num_frames,
            temporal_patch_size=int(
                getattr(model.vision_model, "temporal_patch_dim", 1)
            ),
        ),
    )


@dataclass
class RequestMetrics:
    """Approximate latency breakdown for one generation request, in milliseconds."""

    prefill_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    itl_ms: float | None
    decode_ms: float | None
    e2e_ms: float
    generated_tokens: int


def metrics_from_result(result, wall_e2e_ms: float) -> RequestMetrics:
    """Build engine-side metrics from request lifecycle events."""
    events_by_type = {
        event_type: [
            event.timestamp for event in result.events if event.type == event_type
        ]
        for event_type in (
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.GENERATED_TOKEN,
            DynamicInferenceEventType.FINISH,
        )
    }
    add_engine = events_by_type[DynamicInferenceEventType.ADD_ENGINE]
    add_context = events_by_type[DynamicInferenceEventType.ADD_CONTEXT]
    token_times = events_by_type[DynamicInferenceEventType.GENERATED_TOKEN]
    finish = events_by_type[DynamicInferenceEventType.FINISH]

    e2e_ms = (
        (finish[-1] - add_engine[0]) * 1e3
        if add_engine and finish
        else wall_e2e_ms
    )
    ttft_ms = (
        (token_times[0] - add_engine[0]) * 1e3
        if add_engine and token_times
        else None
    )
    prefill_ms = (
        (token_times[0] - add_context[0]) * 1e3
        if add_context and token_times
        else None
    )
    decode_ms = max(e2e_ms - ttft_ms, 0.0) if ttft_ms is not None else None
    itl_ms = (
        sum(second - first for first, second in zip(token_times, token_times[1:]))
        * 1e3
        / (len(token_times) - 1)
        if len(token_times) > 1
        else None
    )
    generated_tokens = len(result.generated_tokens or [])
    tpot_ms = (
        decode_ms / (generated_tokens - 1)
        if decode_ms is not None and generated_tokens > 1
        else None
    )
    return RequestMetrics(
        prefill_ms=prefill_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        itl_ms=itl_ms,
        decode_ms=decode_ms,
        e2e_ms=e2e_ms,
        generated_tokens=generated_tokens,
    )


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_metrics(metrics: RequestMetrics, prefill_estimated: bool = False) -> None:
    if dist.get_rank() != 0:
        return
    prefill_label = "Prefill ms (est.)" if prefill_estimated else "Prefill ms"
    print(
        f"{prefill_label}={_format_ms(metrics.prefill_ms)}, "
        f"TTFT ms={_format_ms(metrics.ttft_ms)}, "
        f"TPOT ms={_format_ms(metrics.tpot_ms)}, "
        f"ITL ms={_format_ms(metrics.itl_ms)}, "
        f"Decode ms={_format_ms(metrics.decode_ms)}, "
        f"E2E ms={metrics.e2e_ms:.2f}, "
        f"output_tokens={metrics.generated_tokens}"
    )


def print_configuration(
    args: argparse.Namespace,
    prompt_token_count: int,
) -> None:
    """Print the workload and performance settings controlled by this script."""
    if dist.get_rank() != 0:
        return
    if args.media == "image":
        media_config = (
            f"image={args.image_width}x{args.image_height}, "
            f"num_images={args.num_images}, max_patches={args.image_max_patches}"
        )
    else:
        media_config = (
            f"video_frames={args.video_width}x{args.video_height}, "
            f"num_frames={args.video_num_frames}"
        )
    print(
        f"Workload: api={args.api}, {media_config}, prompt_tokens={prompt_token_count}, "
        f"max_new_tokens={args.max_new_tokens}, max_sequence_length={args.max_sequence_length}, "
        f"requests={args.num_requests}, chained={args.chain_requests}, "
        f"reasoning={args.reasoning}, benchmark={args.benchmark}"
    )
    print(
        f"Runtime: tp={args.tensor_model_parallel_size}, ep={args.expert_model_parallel_size}, "
        f"inference_optimized={args.inference_optimized}, "
        f"chunked_prefill={args.chunked_prefill}, graph={args.graph_mode}/{args.cuda_graph_scope}, "
        f"kv_cache_gb={args.kv_cache_gb}, mamba_cache_gb={args.mamba_cache_gb}, "
        f"vision_cache_gb={args.vision_cache_gb}, prefix_caching={args.prefix_caching}, "
        f"prefix_policies={args.prefix_caching_eviction_policy}/"
        f"{args.prefix_caching_coordinator_policy}"
    )


def print_result(
    result,
    request_index: int,
    metrics: RequestMetrics,
    args: argparse.Namespace,
    prompt_token_count: int,
) -> None:
    print_generated_text(result.generated_text, request_index)
    print_metrics(metrics)
    print_configuration(args, prompt_token_count)


def print_generated_text(text: str, request_index: int) -> None:
    if dist.get_rank() != 0:
        return
    print(f"\n======== NEMOTRON OMNI OUTPUT {request_index + 1} ========")
    print(text)
    print("======================================")


async def run_async(args, model, tokenizer, config, conversation, media_data) -> None:
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=NemotronOmniInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            for request_index in range(args.num_requests):
                prompt_tokens = build_prompt_tokens(tokenizer, model, conversation)
                request_start = time.perf_counter()
                result = await llm.generate(
                    prompt_tokens,
                    build_sampling_params(tokenizer, args),
                    multi_modal_data=media_data,
                )
                torch.cuda.synchronize()
                wall_e2e_ms = (time.perf_counter() - request_start) * 1e3
                print_result(
                    result,
                    request_index,
                    metrics_from_result(result, wall_e2e_ms),
                    args,
                    len(prompt_tokens),
                )
                if args.chain_requests:
                    append_conversation_turn(conversation, result.generated_text, args.prompt)


def post_completion(args, prompt_tokens, media_data) -> tuple[str, RequestMetrics]:
    """Stream one completion and collect client-observed token timings."""
    encoded_media = {}
    for modality, value in media_data.items():
        if isinstance(value, list):
            encoded_media[modality] = [
                base64.b64encode(media_bytes).decode("ascii") for media_bytes in value
            ]
        else:
            encoded_media[modality] = base64.b64encode(value).decode("ascii")
    payload = json.dumps(
        {
            "prompt": prompt_tokens,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "max_tokens": args.max_new_tokens,
            "ignore_eos": args.benchmark,
            "stream": True,
            "streaming_interval": 1,
            "multi_modal_data": encoded_media,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://{args.http_host}:{args.http_port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # The listening socket is bound before serve() returns, but the frontend
    # worker may still be importing dependencies when the first POST arrives.
    for attempt in range(50):
        try:
            request_start = time.perf_counter()
            token_times = []
            generated_text = ""
            generated_tokens = 0
            with urllib.request.urlopen(request, timeout=300) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line.removeprefix("data: ")
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError(
                            f"Completions endpoint failed: {chunk['error']['message']}"
                        )
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is None:
                        token_times.append(time.perf_counter())
                    else:
                        generated_text = choice.get("generated_text", generated_text)
                        generated_tokens = choice.get("generated_length", generated_tokens)

            request_end = time.perf_counter()
            e2e_ms = (request_end - request_start) * 1e3
            ttft_ms = (
                (token_times[0] - request_start) * 1e3 if token_times else None
            )
            decode_ms = (
                max(e2e_ms - ttft_ms, 0.0) if ttft_ms is not None else None
            )
            itl_ms = (
                sum(
                    second - first
                    for first, second in zip(token_times, token_times[1:])
                )
                * 1e3
                / (len(token_times) - 1)
                if len(token_times) > 1
                else None
            )
            tpot_ms = (
                decode_ms / (generated_tokens - 1)
                if decode_ms is not None and generated_tokens > 1
                else None
            )
            # Client streaming does not expose the end of prefill. Subtracting one
            # average token interval removes an estimate of first-token decode time.
            prefill_ms = (
                max(ttft_ms - itl_ms, 0.0)
                if ttft_ms is not None and itl_ms is not None
                else ttft_ms
            )
            return generated_text, RequestMetrics(
                prefill_ms=prefill_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                itl_ms=itl_ms,
                decode_ms=decode_ms,
                e2e_ms=e2e_ms,
                generated_tokens=generated_tokens,
            )
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Completions endpoint returned HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError:
            if attempt == 49:
                raise
            time.sleep(0.1)
    raise RuntimeError("Completions endpoint did not become ready.")


async def run_completions(
    args, model, tokenizer, config, conversation, media_data
) -> None:
    # Lifecycle synchronization must not use the default NCCL group while the
    # background engine is running. A worker blocked in an NCCL barrier cannot
    # participate in the TP/EP collectives required by rank 0's HTTP request.
    lifecycle_group = dist.new_group(backend="gloo")
    try:
        async with MegatronAsyncLLM(
            model=model,
            tokenizer=tokenizer,
            inference_config=config,
            use_coordinator=True,
            coordinator_port=args.coordinator_port,
            inference_wrapper_cls=NemotronOmniInferenceWrapper,
        ) as llm:
            await llm.serve(
                ServeConfig(
                    host=args.http_host,
                    port=args.http_port,
                    frontend_replicas=1,
                ),
                blocking=False,
            )
            dist.barrier(group=lifecycle_group)
            if llm.is_primary_rank:
                for request_index in range(args.num_requests):
                    prompt_tokens = build_prompt_tokens(tokenizer, model, conversation)
                    generated_text, metrics = await asyncio.to_thread(
                        post_completion, args, prompt_tokens, media_data
                    )
                    print_generated_text(generated_text, request_index)
                    print_metrics(metrics, prefill_estimated=True)
                    print_configuration(args, len(prompt_tokens))
                    if args.chain_requests:
                        append_conversation_turn(conversation, generated_text, args.prompt)
            dist.barrier(group=lifecycle_group)
    finally:
        dist.destroy_process_group(lifecycle_group)


def run_sync(args, model, tokenizer, config, conversation, media_data) -> None:
    with MegatronLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=config,
        use_coordinator=True,
        coordinator_port=args.coordinator_port,
        inference_wrapper_cls=NemotronOmniInferenceWrapper,
    ) as llm:
        if llm.is_primary_rank:
            for request_index in range(args.num_requests):
                prompt_tokens = build_prompt_tokens(tokenizer, model, conversation)
                request_start = time.perf_counter()
                result = llm.generate(
                    prompt_tokens,
                    build_sampling_params(tokenizer, args),
                    multi_modal_data=media_data,
                )[0]
                torch.cuda.synchronize()
                wall_e2e_ms = (time.perf_counter() - request_start) * 1e3
                print_result(
                    result,
                    request_index,
                    metrics_from_result(result, wall_e2e_ms),
                    args,
                    len(prompt_tokens),
                )
                if args.chain_requests:
                    append_conversation_turn(conversation, result.generated_text, args.prompt)


def main() -> None:
    args = parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive.")
    validate_parallelism(args)
    logging.basicConfig(level=logging.INFO)
    model = load_model(args)
    tokenizer = build_tokenizer(args.hf_model)
    conversation = build_initial_conversation(args)
    media_data = build_media_data(args)
    config = build_inference_config(model, args)

    if args.api == "async":
        asyncio.run(run_async(args, model, tokenizer, config, conversation, media_data))
    elif args.api == "completions":
        asyncio.run(
            run_completions(args, model, tokenizer, config, conversation, media_data)
        )
    else:
        run_sync(args, model, tokenizer, config, conversation, media_data)


if __name__ == "__main__":
    main()
