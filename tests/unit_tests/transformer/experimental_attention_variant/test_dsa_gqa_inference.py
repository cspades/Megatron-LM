# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.inference_layers import (
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cute_kernels,
    dsa_gqa_flashinfer,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    dynamic_sparse_attention,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HAVE_SSA = importlib.util.find_spec("simplified_sparse_attention") is not None

requires_ssa = pytest.mark.skipif(
    not HAVE_SSA, reason="The simplified-sparse-attention wheel is required"
)

HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 256
INDEX_DIM = 128
TOPK = 512


def _config_kwargs(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=1,
        kv_channels=HEAD_DIM,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_min_memory_backend="cute",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=INDEX_DIM,
        dsa_indexer_topk=TOPK,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module", autouse=True)
def initialize_decode_test_distributed():
    Utils.initialize_distributed()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(dsa_indexer_topk=256), "top-K"),
        (dict(num_query_groups=2), "MQA1"),
        (dict(dsa_indexer_scoring_relu=True), "unweighted"),
        (dict(dsa_simplified_use_learned_k=False), "learned-Q/K"),
        (dict(experimental_attention_variant=None), "experimental_attention_variant='dsa'"),
        (dict(dsa_min_memory_backend="reference"), "only implemented by"),
    ],
)
def test_config_rejects_configs_outside_the_cute_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        TransformerConfig(**_config_kwargs(**overrides))


def test_missing_sparse_attention_package_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "simplified_sparse_attention", None)
    with pytest.raises(ImportError, match="simplified_sparse_attention"):
        dsa_cute_kernels.run_sparse_attention_inference(
            query=None,
            key=None,
            value=None,
            q_indexer=None,
            k_indexer=None,
            key_offset=0,
            indexer_topk=TOPK,
        )


def test_speculative_decode_steps_are_rejected():
    context = SimpleNamespace(num_speculative_tokens=2)
    with pytest.raises(NotImplementedError, match="MTP speculative decoding"):
        dynamic_sparse_attention(
            query=None,
            key_cache=None,
            value_cache=None,
            index_query=None,
            index_key_cache=None,
            block_table=None,
            inference_context=context,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )


def _reference_scores(index_q, index_keys, table, row_requests, row_lengths, page_size):
    """Float64 scores of each row over its request's keys [0, length)."""
    scores = []
    for row in range(index_q.size(0)):
        length = int(row_lengths[row])
        positions = torch.arange(length, device=index_q.device)
        pages = table[max(int(row_requests[row]), 0), positions // page_size].long()
        keys = index_keys[pages, positions % page_size].double()
        scores.append(keys @ index_q[row].double())
    return scores


def _reference_attention(query, keys, values, table, row_requests, row_lengths, indices, scale):
    """Per-row attention over its valid selected keys; probabilities rounded to BF16 like PV."""
    page_size, groups = keys.size(1), keys.size(2)
    heads_per_group = query.size(1) // groups
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row in range(query.size(0)):
        selected = indices[row]
        selected = selected[(selected >= 0) & (selected < row_lengths[row])].long()
        if selected.numel() == 0:
            continue
        pages = table[int(row_requests[row]), selected // page_size].long()
        for head in range(query.size(1)):
            group = head // heads_per_group
            k = keys[pages, selected % page_size, group].float()
            v = values[pages, selected % page_size, group].float()
            probs = (query[row, head].float() @ k.T * scale).softmax(-1)
            out[row, head] = probs @ v
    return out


def _flashinfer_supported(topk=TOPK):
    device = torch.device("cuda", torch.cuda.current_device())
    return dsa_gqa_flashinfer.flashinfer_dsa_supported(device, topk, INDEX_DIM, 256, NUM_HEADS)


def test_flashinfer_step_plan():
    """Leading single-token requests take the decode path; the rest fill 128-row tiles whose
    8-row groups are consecutive, gap-free union-core batches; sizes follow the padded batch."""
    # Two decode requests, a 300-token chunk, a single-token request after it (prefill path),
    # padded to 400 rows and 8 requests.
    plan = dsa_gqa_flashinfer.build_step_plan(
        [1, 1, 300, 1],
        [50, 9000, 1300, 77],
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(400, 8, False)
    assert sizes == {"decode_rows": 8, "tiles": 4 + 8, "prefill_requests": 8}
    assert plan["decode_lengths"].tolist() == [50, 9000] + [0] * 6
    assert plan["decode_requests"][:2].tolist() == [0, 1]
    assert len(plan["query_rows"]) == 12 * 128 and len(plan["batch_starts"]) == 12 * 16 + 1
    rows = plan["query_rows"][plan["query_rows"] >= 0]
    assert rows.tolist() == list(range(2, 303))
    # Request 2's rows end at keys 1001 .. 1300; request 3's single row at key 77.
    ends = plan["row_ends"][plan["query_rows"] >= 0]
    assert ends.tolist() == list(range(1001, 1301)) + [77]
    assert plan["tile_requests"][:4].tolist() == [2, 2, 2, 3]
    starts = plan["batch_starts"]
    assert (np.diff(starts) >= 0).all() and starts[0] == 2 and starts[-1] == 303
    assert plan["sample_ends"][:2].tolist() == [1300 // 128 * 3, 77 // 128 * 3]
    decode_plan = dsa_gqa_flashinfer.build_step_plan(
        [1] * 3, [10, 20, 30], padded_rows=4, padded_requests=4, decode_only=True, heads=NUM_HEADS
    )
    assert decode_plan["decode_lengths"].tolist() == [10, 20, 30, 0]
    assert decode_plan["query_rows"].size == 0


def _rows_of(query_lengths, kv_lengths):
    """Each step row's request and causal key count."""
    requests, lengths = [], []
    for request, (q_len, kv_len) in enumerate(zip(query_lengths, kv_lengths)):
        requests += [request] * q_len
        lengths += list(range(kv_len - q_len + 1, kv_len + 1))
    return requests, lengths


def _reference_rows(query, index_query, keys, values, index_keys, table, requests, lengths):
    """Exact float64 top-k selection, then attention over the selected keys."""
    device = query.device
    row_requests = torch.tensor(requests, device=device)
    row_lengths = torch.tensor(lengths, device=device)
    scores = _reference_scores(
        index_query, index_keys, table, row_requests, row_lengths, index_keys.size(1)
    )
    indices = torch.full((len(requests), TOPK), -1, dtype=torch.int64, device=device)
    for row, row_scores in enumerate(scores):
        top = row_scores.topk(min(TOPK, row_scores.numel())).indices
        indices[row, : top.numel()] = top
    return _reference_attention(
        query, keys, values, table, row_requests, row_lengths, indices, HEAD_DIM**-0.5
    )


@pytest.mark.parametrize("decode_only", [True, False])
def test_flashinfer_replays_in_cuda_graph(decode_only):
    """A captured layer, replayed for new batches of one padded size (new keys, queries, tables
    and lengths, plan buffers refilled in place), matches an exact PyTorch reference, and
    padding rows stay zero."""
    if not _flashinfer_supported():
        pytest.skip("FlashInfer's DSA kernels are not installed or not supported here")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(71)
    page_size, bound = 256, 9216
    if decode_only:
        padded_rows = padded_requests = 4
        batches = [([1, 1, 1], [700, 9000, 2600]), ([1, 1, 1, 1], [9000, 513, 4000, 1])]
    else:
        padded_rows, padded_requests = 640, 4
        batches = [([1, 1, 300, 37], [700, 9000, 3000, 37]), ([1, 520], [8000, 9000])]
    max_pages = bound // page_size
    num_pages = padded_requests * max_pages
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.zeros(padded_requests, max_pages, device=device, dtype=torch.int32)
    query = torch.zeros(padded_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(padded_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, decode_only)
    empty = dsa_gqa_flashinfer.build_step_plan(
        [1],
        [1],
        padded_rows=padded_rows,
        padded_requests=padded_requests,
        decode_only=decode_only,
        heads=NUM_HEADS,
    )
    buffers = {
        name: torch.zeros(len(v), device=device, dtype=torch.int32) for name, v in empty.items()
    }
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"], num_tiles=sizes["tiles"], max_context=bound, **buffers
    )

    def fill(query_lengths, kv_lengths):
        arrays = dsa_gqa_flashinfer.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_rows,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=NUM_HEADS,
        )
        for name, values_np in arrays.items():
            buffers[name].copy_(torch.from_numpy(values_np))
        real = sum(query_lengths)
        query.zero_()
        index_query.zero_()
        query[:real] = torch.randn(real, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        index_query[:real] = torch.randn(real, INDEX_DIM, device=device, dtype=torch.bfloat16)
        keys.copy_(torch.randn_like(keys))
        index_keys.copy_(torch.randn_like(index_keys))
        table.copy_(torch.randperm(num_pages, device=device).view(padded_requests, max_pages))
        return real

    def attend():
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    fill(*batches[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attend()  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = attend()

    for query_lengths, kv_lengths in batches:
        real = fill(query_lengths, kv_lengths)
        graph.replay()
        requests, lengths = _rows_of(query_lengths, kv_lengths)
        expected = _reference_rows(
            query[:real], index_query[:real], keys, values, index_keys, table, requests, lengths
        )
        torch.testing.assert_close(graph_output[:real].float(), expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[real:]) == 0


@requires_ssa
@pytest.mark.parametrize("key_offset", [0, 1, 300, 777])
def test_rectangular_prefill_matches_full_sequence_rows(key_offset):
    """A chunk over its cached prefix must equal the matching rows of a full prefill."""
    torch.manual_seed(5)
    device = torch.device("cuda", torch.cuda.current_device())
    total, chunk = key_offset + 400, 400

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)

    query = randn(total, NUM_HEADS, HEAD_DIM)
    key, value = randn(total, 1, HEAD_DIM), randn(total, 1, HEAD_DIM)
    q_idx, k_idx = randn(total, INDEX_DIM), randn(total, INDEX_DIM)
    full = dsa_cute_kernels.run_sparse_attention_inference(
        query=query,
        key=key,
        value=value,
        q_indexer=q_idx,
        k_indexer=k_idx,
        key_offset=0,
        indexer_topk=TOPK,
    )
    tail = dsa_cute_kernels.run_sparse_attention_inference(
        query=query[key_offset:],
        key=key,
        value=value,
        q_indexer=q_idx[key_offset:],
        k_indexer=k_idx,
        key_offset=key_offset,
        indexer_topk=TOPK,
    )
    assert tail.shape == (chunk, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(tail, full[key_offset:], atol=2e-2, rtol=2e-2)


@requires_ssa
class TestDynamicSparseAttention:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _config(self, transformer_impl: str) -> TransformerConfig:
        return TransformerConfig(**_config_kwargs(transformer_impl=transformer_impl))

    def _attention(self, config: TransformerConfig) -> DSGroupedSelfAttention:
        inference_layers = config.transformer_impl == "inference_optimized"
        return DSGroupedSelfAttention(
            config=config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=(
                    InferenceLayerNormColumnParallelLinear
                    if inference_layers
                    else TELayerNormColumnParallelLinear
                ),
                core_attention=TEDotProductAttention,
                linear_proj=InferenceRowParallelLinear if inference_layers else TERowParallelLinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).eval()

    @pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
    def test_indexer_reads_the_fused_qkv_norm_output(self, transformer_impl):
        """The QKV projection returns its RMSNorm output, which feeds the indexer."""
        attention = self._attention(self._config(transformer_impl))
        assert set(attention.state_dict()) >= {
            "core_attention.indexer.linear_q.weight",
            "core_attention.indexer.linear_k.weight",
        }
        assert not any(
            key.startswith("core_attention.") and ".indexer." not in key
            for key in attention.state_dict()
        )
        hidden = torch.randn(32, 1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            output, bias, normalized = attention.linear_qkv(hidden)
        assert bias is None
        weight = attention.linear_qkv.layer_norm_weight.float()
        expected = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + attention.config.layernorm_epsilon
        )
        torch.testing.assert_close(normalized.float(), expected * weight, atol=2e-2, rtol=2e-2)
        assert output.shape[-1] == (NUM_HEADS + 2) * HEAD_DIM

    def test_static_batching_context_is_rejected(self):
        attention = self._attention(self._config("transformer_engine"))
        context = SimpleNamespace(is_dynamic_batching=lambda: False)
        with pytest.raises(NotImplementedError, match="dynamic batching only"):
            attention(hidden_states=None, attention_mask=None, inference_context=context)

    @pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
    def test_mixed_prefill_and_decode_match_training_forward(self, transformer_impl):
        """Cached decode and mixed batches must match uncached attention.

        Prompts longer than the routing top-k make sparse selection matter.
        """
        if not _flashinfer_supported():
            pytest.skip("FlashInfer's DSA kernels are not installed or not supported here")
        config = self._config(transformer_impl)
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=2048,
                max_requests=4,
                max_tokens=4096,
                block_size_tokens=64,
                buffer_size_gb=1,
                num_cuda_graphs=None,
                unified_memory_level=0,
            ),
        )
        assert context.cache_dsa_indexer_keys
        attention = self._attention(config)

        def add_request(request_id, prompt_length):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(prompt_length, dtype=torch.long),
                    sampling_params=SamplingParams(num_tokens_to_generate=8),
                )
            )

        def step():
            hidden = torch.randn(
                context.padded_active_token_count,
                1,
                HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.inference_mode(), InferenceMode.active():
                output, _ = attention(
                    hidden_states=hidden, attention_mask=None, inference_context=context
                )
            return output, hidden

        histories = {}
        prompt_lengths = {0: 700, 1: 1100}
        for request_id, length in prompt_lengths.items():
            add_request(request_id, length)
        context.initialize_attention_state()
        output, hidden = step()
        start = 0
        for request_id, length in prompt_lengths.items():
            histories[request_id] = hidden[start : start + length]
            with torch.inference_mode():
                expected, _ = attention(hidden_states=histories[request_id], attention_mask=None)
            torch.testing.assert_close(
                output[start : start + length], expected, atol=2e-2, rtol=2e-2
            )
            start += length

        for decode_step in range(3):
            active = sorted(histories)
            context.update_requests(
                active_requests_mask=torch.ones(len(active), dtype=torch.int32),
                new_tokens=torch.full((len(active),), 7, dtype=torch.long),
            )
            if decode_step == 1:
                # Mix a fresh prefill into a decode step.
                add_request(2, 37)
            context.initialize_attention_state()
            output, hidden = step()
            for row, request_id in enumerate(active):
                histories[request_id] = torch.cat((histories[request_id], hidden[row : row + 1]))
                with torch.inference_mode():
                    expected, _ = attention(
                        hidden_states=histories[request_id], attention_mask=None
                    )
                torch.testing.assert_close(output[row], expected[-1], atol=2e-2, rtol=2e-2)
            if decode_step == 1:
                histories[2] = hidden[len(active) : len(active) + 37]
                with torch.inference_mode():
                    expected, _ = attention(hidden_states=histories[2], attention_mask=None)
                torch.testing.assert_close(
                    output[len(active) : len(active) + 37], expected, atol=2e-2, rtol=2e-2
                )
