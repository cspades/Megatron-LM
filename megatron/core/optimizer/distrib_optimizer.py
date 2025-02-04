# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


import itertools
import warnings
from dataclasses import replace
from logging import getLogger
from typing import Callable, Dict, List, Optional, Tuple

import torch

HAVE_APEX_OR_TE = True
try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
    except ImportError:
        from torch.optim import Adam

        HAVE_APEX_OR_TE = False

from .. import tensor_parallel
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing import ShardedTensor
from ..dist_checkpointing.dict_utils import nested_values
from ..dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedStateDict,
    ShardedTensorFactory,
)
from ..dist_checkpointing.utils import extract_sharded_tensors_and_factories
from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from ..transformer.module import MegatronModule
from ..utils import is_float8tensor
from .grad_scaler import MegatronGradScaler
from .optimizer import (
    MixedPrecisionOptimizer,
    _multi_tensor_copy_this_to_that,
    _zero_grad_group_helper,
)
from .optimizer_config import OptimizerConfig
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

try:
    # This will be used when "--fp8-param-gather" is enabled.
    # When BF16/FP16 parameters don't exist, we need to cast the FP32 main parameters to
    # FP8 directly in the optimizer.
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
except:
    pass

logger = getLogger(__name__)


class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.

    Args:
        start (int): Start index.
        end (int): End index.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0):
        """Shift start/end indexes to start at new start index.

        Both start and end indexes will be shifted by [new start] - [old start].

        Args:
            start (int): New start index.
        """
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start


class DistributedOptimizer(MixedPrecisionOptimizer):
    """Distributed optimizer, for all data types (fp16, bf16, and fp32).

    See __init__() below for argument details.
    """

    @classmethod
    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        gbuf_world_range: Range,
        bucket_offset: int,
    ):
        """
        Build mapping from param reference to grad buffer shard ranges.

        This method builds a mapping from parameter references to grad
        buffer shard ranges, specific to each data-parallel (DP) rank's
        set of 'owned' parameters. Each grad buffer (padded to be an even
        multiple of DP-world-size) is conceptually divided into DP-world-size
        contiguous regions, where each DP rank 'owns' a contiguous region.
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        """

        # Param range map.
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = torch.distributed.get_rank(param_and_grad_buffer.data_parallel_group)
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {"param_map": param_range_map}

        return data

    @classmethod
    def _build_gbuf_range_map(cls, param_and_grad_buffer: _ParamAndGradBuffer):
        """
        Build mapping between params and their grad buffers. These mappings are
        partitioned according to data type.

        Iterate through all buckets of grad buffer to construct param ranges
        that this rank "owns" (the dp_rank'th shard of each bucket, where each
        shard is 1/dp_world_size of the bucket).

        Args:
            param_and_grad_buffer (_ParamAndGradBuffer): buffer to build mapping for.
        """
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }

    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """
        Create a reverse of the gbuf_ranges, for referencing in opposite direction.
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        assert param not in param_gbuf_map, (
                            "Param should not be in param_gbuf_map; each param only belongs "
                            "to a single bucket."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        """

        # Param group map.
        # World param group map.
        # - Store a mapping of <model_parameter:group_index> for all parameters
        #   across all DP ranks. This is necessary because it is our first
        #   cross reference between the DDP mappings and the optimizer group
        #   parameters. This mapping only for use in the next step of building
        #   the local mapping over this DP rank's parameters.
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # Optimizer group ranges & param-group mapping.
        # - Build a mapping from groups to their contained parameters, and also
        #   from parameters to their containing group index and order within
        #   the group. The group index and order are particularly important for
        #   saving and loading checkpoints.
        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            for model_param in group_range["params"]:

                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start : param_range.end
                    ]

                    # If we use FP8 params to initialize FP32 main params (compared to using the
                    # bf16/fp16 params to initialize the main params), there will be a loss of
                    # precision at the beginning of training (this problem will not occur if the
                    # training is long enough or if the main params are loaded from a checkpoint).
                    if is_float8tensor(model_param) and hasattr(
                        model_param, 'get_high_precision_init_val'
                    ):
                        shard_main_param = (
                            model_param.get_high_precision_init_val()
                            .view(-1)[param_range.start : param_range.end]
                            .clone()
                            .to(shard_model_param.device)
                            .float()
                        )
                        model_param.clear_high_precision_init_val()
                    else:
                        shard_main_param = shard_model_param.clone().float()

                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        model_chunks: List[MegatronModule],
        per_model_buffers: Dict[int, List[_ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: torch.distributed.ProcessGroup,
        data_parallel_group_idx: int,
    ):
        """
        Distributed optimizer, for all data types (fp16, bf16, and fp32).

        The steps in this method create the core mapping between param and grad buffers,
        parameters, and parameter shard ranges, that is needed for converting between model
        param indexes and main parameter shard indexes. This method also updates the optimizer
        parameter groups with the newly created shards.

        Args:
            optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
            config (OptimizerConfig): configuration object for optimizer.
            grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
                this can be None. This case happens when `bf16 = True` and we don't
                use any loss scale. Note that for `bf16 = True`, we can have
                a constant gradient scaler. Also for `bf16 = False`, we
                always require a grad scaler.
            init_state_fn (Callable, optional): function to initialize state in the optimizer.
            model_chunks (List[MegatronModule]): list of model chunks.
            per_model_buffers (Dict[int, List[ParamAndGradBuffer]]): the implementation of the
                distributed optimizer is centered on using a contiguous buffer for
                communicating grads & params between the model state and the optimizer state.
                You can find a more detailed description in
                https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md.
            data_parallel_group (torch.distributed.ProcessGroup): data-parallel group to use to
                all-gather params after optimizer.step().
            data_parallel_group_gloo (torch.distributed.ProcessGroup): gloo data-parallel group
                (used in checkpoint loading and saving).
            data_parallel_group_idx (int): index in data-parallel group (used by
                distributed checkpointing logic).
        """

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        assert (
            HAVE_APEX_OR_TE
        ), f'Please install Apex or Transformer Engine to use DistributedOptimizer.'

        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        self.model_chunks = model_chunks
        self.ddp_config = self.model_chunks[0].ddp_config
        for model_chunk in self.model_chunks:
            assert self.ddp_config == model_chunk.ddp_config

        assert isinstance(
            optimizer, Adam
        ), "Only Adam currently supported, due to checkpointing requirements."

        # Model grad buffer ranges.
        assert per_model_buffers is not None, "per_model_buffers must be provided"
        self.buffers = list(itertools.chain(*per_model_buffers.values()))
        self.per_model_buffers = per_model_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.data_parallel_group_idx = data_parallel_group_idx

        self.gbuf_idx_to_model_idx_map = {}
        gbuf_idx = 0
        for model_idx, buffers in self.per_model_buffers.items():
            for _ in buffers:
                self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
                gbuf_idx += 1

        if not self.ddp_config.with_megatron_fsdp_code_path:
            self.per_model_bucket_groups = {}
            for model_idx, buffers in self.per_model_buffers.items():
                self.per_model_bucket_groups[model_idx] = partition_buckets(buffers)

            self.gbuf_ranges = []
            self.per_bucket_numel = []
            self.per_bucket_numel_unpadded = []
            for buffer in self.buffers:

                self.per_bucket_numel.append(
                    {
                        (buffer.param_dtype, buffer.grad_dtype): [
                            bucket.grad_data.numel() for bucket in buffer.buckets
                        ]
                    }
                )
                self.per_bucket_numel_unpadded.append(
                    {
                        (buffer.param_dtype, buffer.grad_dtype): [
                            bucket.numel_unpadded for bucket in buffer.buckets
                        ]
                    }
                )
                self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))
            self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

            # Optimizer ranges.
            (self.model_param_group_index_map, self.opt_group_ranges) = (
                self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)
            )

            # Allocate main param shards.
            (
                self.model_float16_groups,
                self.model_fp32_groups,
                self.shard_float16_groups,
                self.shard_fp32_groups,
                self.shard_fp32_from_float16_groups,
            ) = self._build_model_and_main_param_groups(
                self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
            )

            # Update optimizer groups.
            # - Also, leverage state_dict() and load_state_dict() to
            #   recast preexisting per-param state tensors.
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def enable_pre_hook(self):
        """
        Enable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        warnings.warn(
            "`DistributedOptimizer.enable_pre_hook` will be deprecated in a future release. "
            "Use `DistributedDataParallel.enable_forward_pre_hook` directly."
        )
        for model_chunk in self.model_chunks:
            model_chunk.enable_forward_pre_hook()

    def disable_pre_hook(self):
        """
        Disable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        warnings.warn(
            "`DistributedOptimizer.disable_pre_hook` will be deprecated in a future release. "
            "Use `DistributedDataParallel.disable_forward_pre_hook` directly."
        )
        for model_chunk in self.model_chunks:
            model_chunk.disable_forward_pre_hook()

    def _get_model_param_range_map(self, param: torch.nn.Parameter):
        """
        Given a model param, get the index sub-range of the param that this
        data-parallel rank owns.
        """
        gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[param]
        gbuf_range_map = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """
        With the distributed optimizer, the model parallel group is the
        entire world.
        """
        return None

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """

        inner_state_dict = self.optimizer.state_dict()
        state_dict = {}

        # Extract 'step', for non-Apex/TE support.
        if not HAVE_APEX_OR_TE:
            steps = list(set([s["step"].item() for s in inner_state_dict["state"].values()]))
            assert len(steps) == 1
            step = steps[0]

        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {k: v for k, v in inner_state_dict.items() if k != "state"}
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]
            if not HAVE_APEX_OR_TE:
                # Native PyTorch param group requires step (i.e., iteration).
                param_group["step"] = step

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer ordering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [
            {**group, "params": list(inner_state_dict["param_groups"][idx]["params"])}
            for idx, group in enumerate(state_dict["optimizer"]["param_groups"])
        ]

        # Allocate or retrieve optimizer state (i.e., tensors).
        if len(self.optimizer.state) == 0 and not self.ddp_config.with_megatron_fsdp_code_path:
            # Allocate empty optimizer state if not previously initialized.
            # - If len(self.optimizer.state) == 0, this means that the optimizer
            #   state has not been previously initialized. Once it has been
            #   initialized, we skip this code block to avoid reallocating
            #   empty tensors (i.e., torch.empty), which in turn reduces memory
            #   fragmentation.
            # - Real data is overwritten during load_parameter_state().
            state_dict_state = []
            for gbuf_range_maps in self.gbuf_ranges:
                for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                    for gbuf_range_map in gbuf_range_map_for_all_buckets:
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():

                            # Get parameter ordering information (see method docstring
                            # for details).
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            state_order = inner_state_dict["param_groups"][group_index]["params"][
                                group_order
                            ]

                            # Allocate dummy tensors.
                            numel = len(param_range_map["gbuf_world"])
                            init_shard = lambda: torch.empty(
                                (numel,), dtype=torch.float32, device=torch.cuda.current_device()
                            )

                            state_dict_state.append(
                                (state_order, {"exp_avg": init_shard(), "exp_avg_sq": init_shard()})
                            )

            # Sort by state order (see method docstring for details).
            state_dict_state.sort(key=lambda s: s[0])
            state_dict_state = {s[0]: s[1] for s in state_dict_state}

        else:
            # Retrieve existing optimizer state.
            state_dict_state = inner_state_dict["state"]

        # Extract 'step', for non-Apex/TE support.
        if not HAVE_APEX_OR_TE:
            steps = list(set([g["step"] for g in state_dict["optimizer"]["param_groups"]]))
            assert len(steps) == 1
            step = torch.tensor(steps[0], dtype=torch.float)

            for s in state_dict_state.values():
                # Native PyTorch state dict requires step (i.e., iteration).
                s["step"] = step

        # Optimizer.
        self.optimizer.load_state_dict(
            {"state": state_dict_state, "param_groups": state_dict_param_groups}
        )

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        if 'param_state' in state_dict:
            assert 'param_state_sharding_type' in state_dict, state_dict.keys()
            param_state = state_dict['param_state']
            sharding_type = state_dict['param_state_sharding_type']
            if self.ddp_config.with_megatron_fsdp_code_path:
                assert sharding_type == "fully_sharded_model_space", "Only fully sharded model space is supported"
            logger.info(f'Loading distributed optimizer sharded state of type {sharding_type}')
            if sharding_type == 'dp_zero_gather_scatter':
                self.load_parameter_state_from_dp_zero(param_state)
            elif sharding_type == 'fully_sharded_bucket_space':
                self.load_parameter_state_from_fs_bucket_space(param_state)
            elif sharding_type == 'fully_sharded_model_space':
                self.load_parameter_state_from_fs_model_space(param_state)
            else:
                raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

    def get_parameter_state_fs_bucket_space(self):
        """Get internal representation of parameter state without any copies and modifications.

        This is referred to as "fully sharded bucket space" because the optimizer state is
        fully sharded (e.g. no gather involved) and bucket-centric (the state
        follows the internal structure of the Distributed Optimizer buckets)
        as opposed to model-centric (typical structure of PyT optimizers)
        """
        state = {
            "per_bucket_numel": self.per_bucket_numel,
            "per_bucket_numel_unpadded": self.per_bucket_numel_unpadded,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buckets_state = []
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = []
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                            "gbuf_local_start": param_range_map["gbuf_local"].start,
                            "gbuf_local_end": param_range_map["gbuf_local"].end,
                        }
                        bucket_state.append(tensors)
                    buckets_state.append(bucket_state)
                dtype_state[dtype] = buckets_state
            state[gbuf_idx] = dtype_state
        return state

    def get_parameter_state_dp_zero(self):
        """Get parameter state (i.e., parameter & optimizer tensors).

        This method performs two steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Collect param states.
        state = {"buckets_coalesced": True}
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                # Create coalesced tensors for all state related to parameters in this buffer.
                world_tensors = {}
                if data_parallel_rank == 0:
                    world_tensors = {
                        key: torch.zeros(
                            (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                        )
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }
                    world_tensors["numel_unpadded"] = buffer_numel_unpadded
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel

                    local_shards = {
                        key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {"param": main_param, **optim_state}

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                                tensors[key].detach().cpu()
                            )

                    # Gather contiguous shards on DP rank 0.
                    for key, send_tensor in local_shards.items():

                        # Gather tensor list.
                        if data_parallel_rank == 0:
                            recv_tensors = [
                                torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        # Gather.
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Concatenate.
                        if data_parallel_rank == 0:
                            recv_tensors_concatenated = torch.cat(recv_tensors)
                            # Copy this bucket's collected all-gather tensors into the right place
                            # in the tensor for the buffer. The tensor for the buffer gets rid of
                            # the padding between buckets.
                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            world_tensors[key][start:end].copy_(
                                recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                            )

                    offset_in_world_tensors += gbuf_world_numel_unpadded

                # Collect world state.
                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter state on DP rank 0.

        Args:
            filename (str): path to save parameter state to.
        """

        state_dict = self.get_parameter_state_dp_zero()
        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            torch.save(state_dict, filename)

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        sharding_type: str = 'fully_sharded_model_space',
    ):
        """
        Chooses between 3 param state sharding implementations as requested by `sharding_type`.

        Regular state dict parameters are saved on DP rank 0 and loaded on all ranks.
        """
        if not is_loading and sharding_type == 'fully_sharded_bucket_space':
            logger.warning(
                '`fully_sharded_bucket_space` sharding for DistributedOptimizer'
                ' checkpoint is deprecated and will be removed in the future.'
                ' Please switch to `full_sharded_model_space`.'
            )

        if self.ddp_config.with_megatron_fsdp_code_path:
            assert sharding_type == 'fully_sharded_model_space', (
                f'For FSDP, only `fully_sharded_model_space` is supported. '
                f'Got: {sharding_type}'
            )

        state_dict = self.state_dict()
        if sharding_type != 'fully_sharded_model_space':
            # State dict differs between different model parallel groups
            state_dict = {
                k: ShardedObject(
                    f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.{k}',
                    v,
                    (1,),
                    (0,),
                    replica_id=torch.distributed.get_rank(self.data_parallel_group),
                )
                for k, v in state_dict.items()
            }

        if is_loading:
            # Call the distributed optimizer's specialized load_state_dict(),
            # which conditionally skips re-allocating the optimizer's state if
            # already initialized, which in turn reduces memory fragmentation.
            self.load_state_dict(self.state_dict())
        if sharding_type == 'fully_sharded_bucket_space':
            param_state = self.sharded_param_state_fs_bucket_space(
                model_sharded_state_dict, is_loading
            )

        elif sharding_type == 'dp_zero_gather_scatter':
            param_state = self.sharded_param_state_dp_zero(model_sharded_state_dict, is_loading)
        elif sharding_type == 'fully_sharded_model_space':
            param_state = self.sharded_param_state_fs_model_space(
                model_sharded_state_dict, is_loading
            )
        else:
            raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

        state_dict['param_state'] = param_state
        state_dict['param_state_sharding_type'] = sharding_type
        return state_dict

    def sharded_param_state_dp_zero(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        """Naive implementation which reuses gather/scatter from the legacy ckpt format.

        During saving, gathers the parameters state on DP rank 0 and saves a ShardedObject
        with fixed TPxPP structure. During loading, loads the saved data on DP rank 0
        (None on other ranks). Relies on the parameters scatter done in load_state_dict.
        """
        if is_loading:
            param_state_data = None
        else:
            # Gather on rank 0
            param_state_data = self.get_parameter_state_dp_zero()

        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            # Fixed TPxPP. Save on DP rank 0 only
            param_state = ShardedObject(
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.param_state',
                param_state_data,
                (1,),
                (0,),
            )
        else:
            # DP ranks > 0 don't save. During loading, the param_state needs to be None.
            param_state = LocalNonpersistentObject(None)

        return param_state

    def sharded_param_state_fs_bucket_space(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        """Sharded state dict where each noncontiguous buffer is a separate ShardedTensor.

        Results in fully parallel save and load without any inter-process
        communication or intermediate buffers/copies.
        """
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group)
        data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)

        state = self.get_parameter_state_fs_bucket_space()
        # per_bucket_numel metadata is saved separately for each TPxPP domain.
        for per_bucket_key in ('per_bucket_numel', 'per_bucket_numel_unpadded'):
            key = (
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
                f'.{per_bucket_key}'
            )
            state[per_bucket_key] = ShardedObject(
                key, state[per_bucket_key], (1,), (0,), replica_id=data_parallel_rank
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in state[gbuf_idx].items():
                for bucket_idx, bucket_state in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    sharded_bucket_key = (
                        f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
                        f'.gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket_idx}'
                    )

                    # The global ckpt tensors must be fully covered.
                    # We add extra empty padding if necessary
                    assert bucket_state, 'empty bucket encountered'

                    # Insert padding between parameter tensors to ensure full coverage as needed.
                    all_pad_tensors = {}
                    for i in range(len(bucket_state) - 1):
                        next_param_start = bucket_state[i + 1]['gbuf_local_start']
                        cur_param_end = bucket_state[i]['gbuf_local_end']
                        if next_param_start != cur_param_end:
                            pad_tensors = {
                                k: torch.empty(
                                    next_param_start - cur_param_end, dtype=v.dtype, device=v.device
                                )
                                for k, v in bucket_state[i].items()
                                if isinstance(v, torch.Tensor)
                            }
                            all_pad_tensors[i + 1] = {
                                **pad_tensors,
                                'gbuf_local_start': cur_param_end,
                                'gbuf_local_end': next_param_start,
                                'padding': True,
                            }

                    # Insert from end so that insertion positions are still correct.
                    indices_to_insert = sorted(list(all_pad_tensors.keys()))
                    for index_to_insert in reversed(indices_to_insert):
                        bucket_state.insert(index_to_insert, all_pad_tensors[index_to_insert])

                    if bucket_state[-1]['gbuf_local_end'] != gbuf_local_numel:
                        pad_tensors = {
                            k: torch.empty(
                                gbuf_local_numel - bucket_state[-1]['gbuf_local_end'],
                                dtype=v.dtype,
                                device=v.device,
                            )
                            for k, v in bucket_state[-1].items()
                            if isinstance(v, torch.Tensor)
                        }
                        bucket_state.append(
                            {
                                **pad_tensors,
                                'gbuf_local_start': bucket_state[-1]['gbuf_local_end'],
                                'gbuf_local_end': gbuf_local_numel,
                                'padding': True,
                            }
                        )

                    # Each tensor is mapped to a slice (`flattened_range`)
                    # of a DP-local shard of size `gbuf_local_numel`.
                    for bucket_params_idx in range(len(bucket_state)):
                        tensors = bucket_state[bucket_params_idx]
                        gbuf_local_start = tensors.pop('gbuf_local_start')
                        gbuf_local_end = tensors.pop('gbuf_local_end')
                        if 'padding' not in tensors:
                            tensors['padding'] = False

                        for key in tensors:
                            if key == 'padding':
                                tensors[key] = LocalNonpersistentObject(tensors[key])
                                continue
                            assert tensors[key].shape == (gbuf_local_end - gbuf_local_start,), (
                                tensors[key].shape,
                                gbuf_local_start,
                                gbuf_local_end,
                            )

                            tensors[key] = ShardedTensor(
                                f'{sharded_bucket_key}.{key}',
                                tensors[key],
                                tensors[key].dtype,
                                (gbuf_local_numel,),
                                (data_parallel_world_size * gbuf_local_numel,),
                                (data_parallel_rank * gbuf_local_numel,),
                                axis_fragmentations=(data_parallel_world_size,),
                                flattened_range=slice(gbuf_local_start, gbuf_local_end),
                                allow_shape_mismatch=True,
                            )
        return state

    def sharded_param_state_fs_model_space(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        """Sharded state dict where each buffer is mapped to corresponding model param.

        In this approach the optimizer state tensors are directly related to model parameters
        by linking them with metadata from `model_sharded_state_dict`.
        This will allow changing TP and PP while using DistOpt (as with other optimizers).
        """

        param_to_sharded_metadata = {}
        model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(
            model_sharded_state_dict
        )
        for sh_base in nested_values(model_sharded_state_dict):
            param_to_sharded_metadata[sh_base.data] = sh_base

        prefix = 'optimizer.state'
        state = {}

        # Not stored in the checkpoint, used only to identify params in
        # `sharded_param_state_fs_model_space`.
        def _get_param_state_sharded_tensors(model_param, main_param, item_slice):
            optim_state = self.optimizer.state[main_param]
            if self.ddp_config.with_megatron_fsdp_code_path and is_loading:
                # Pre-populate optimizer state for dist-ckpt loading.
                assert isinstance(self.optimizer, Adam)
                optim_state['exp_avg'] = torch.empty_like(main_param)
                optim_state['exp_avg_sq'] = torch.empty_like(main_param)

            tensors = {"fp32_param": main_param, **optim_state}
            # Match optimizer parameter with model ShardedTensor (or
            # ShardedTensorFactory).
            try:
                sharded_metadata = param_to_sharded_metadata[model_param]
            except KeyError as e:
                raise ValueError(
                    f'Model param {model_param} not in model_sharded_state_dict'
                ) from e

            # Set DP corresponding replica_id coordinate to 0.
            assert (
                len(sharded_metadata.replica_id) == 3
            ), f'Expected replica_id format (PP, TP, DP), got: {sharded_metadata}'
            replica_id = (*sharded_metadata.replica_id[:2], 0)

            # Instantiate ShardedTensor (or ShardedTensorFactory) for optimizer
            # params.
            for state_key, state_ten in tensors.items():
                replace_kwargs = dict(
                    key=f'{prefix}.{state_key}.{sharded_metadata.key}',
                    data=state_ten,
                    dtype=state_ten.dtype,
                    flattened_range=slice(*item_slice),
                    replica_id=replica_id,
                )
                if isinstance(sharded_metadata, ShardedTensorFactory):
                    replace_kwargs.pop('dtype')
                tensors[state_key] = replace(sharded_metadata, **replace_kwargs)
                tensors[state_key].validate_metadata_integrity()
            return tensors

        if self.ddp_config.with_megatron_fsdp_code_path:
            for pg_buffer in self.buffers:
                for pg in pg_buffer.parameter_groups:
                    gbuf = pg.main_grad_buffer
                    wbuf = pg.model_weight_buffer
                    for model_param in gbuf.params:
                        item_id = gbuf.param_idx[model_param]
                        param_name = pg_buffer.param_to_name[model_param]
                        main_param = dict(pg_buffer.optimizer_named_parameters)[param_name]

                        item_slice = gbuf._get_item_slice_in_shard(item_id)
                        if item_slice[0] == item_slice[1]:
                            # This param is not in this shard.
                            continue

                        if wbuf.is_data_distributed:
                            state[param_name] = _get_param_state_sharded_tensors(
                                model_param.fully_shard_param_local_shard, main_param, item_slice
                            )
                        else:
                            state[param_name] = _get_param_state_sharded_tensors(
                                model_param, main_param, item_slice)
        else:
            param_idx = 0
            for gbuf_range_maps in self.gbuf_ranges:
                for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                    for gbuf_range_map in gbuf_range_map_for_all_buckets:
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            param_range = param_range_map['param']
                            item_slice = (param_range.start, param_range.end)

                            main_param = self.optimizer.param_groups[group_index]["params"][group_order]

                            state[param_idx] = _get_param_state_sharded_tensors(
                                model_param, main_param, item_slice)
                            param_idx += 1
        return state

    def load_parameter_state_from_fs_bucket_space(self, state_dict):
        """Loads the parameter state from an internal representation.

        Inverse of the `get_parameter_state_fs_bucket_space` method.
        """
        logger.warning(
            '`fully_sharded_bucket_space` sharding for DistributedOptimizer'
            'checkpoint is deprecated. Please switch to `full_sharded_model_space`'
        )

        if state_dict is not None and "per_bucket_numel_unpadded" in state_dict:
            per_bucket_numel_unpadded_in_checkpoint = state_dict["per_bucket_numel_unpadded"]
            assert self.per_bucket_numel_unpadded == per_bucket_numel_unpadded_in_checkpoint, (
                f"Number of unpadded elements in each bucket need to be the same in current run "
                f"({self.per_bucket_numel_unpadded}) and checkpoint "
                f"({per_bucket_numel_unpadded_in_checkpoint})"
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = state_dict[gbuf_idx][dtype][bucket_idx]
                    bucket_state = [
                        bucket_state_elem
                        for bucket_state_elem in bucket_state
                        if not bucket_state_elem['padding']
                    ]

                    assert len(bucket_state) == len(gbuf_range_map["param_map"]), (
                        len(bucket_state),
                        len(gbuf_range_map["param_map"]),
                    )
                    for src_tensors, (model_param, param_range_map) in zip(
                        bucket_state, gbuf_range_map["param_map"].items()
                    ):
                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        dst_tensors = {"param": main_param, **optim_state}
                        for key in dst_tensors:
                            dst_tensors[key].copy_(src_tensors[key])

    @torch.no_grad()
    def load_parameter_state_from_fs_model_space(self, state_dict):
        """Loads the parameter state from a "model space" representation.

        Inverse of the `sharded_param_state_fs_model_space` method.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            for pg_buffer in self.buffers:
                for model_param in pg_buffer.params:
                    param_name = pg_buffer.param_to_name[model_param]
                    main_param = dict(pg_buffer.optimizer_named_parameters)[param_name]
                    optim_state = self.optimizer.state[main_param]

                    if param_name not in state_dict:
                        continue

                    src_tensors = state_dict[param_name]
                    main_param.copy_(src_tensors["fp32_param"])
                    for key in src_tensors:
                        if key != "fp32_param":
                            optim_state[key] = src_tensors[key]
        else:
            param_idx = 0  # matching order with `sharded_param_state_fs_model_space`
            for gbuf_range_maps in self.gbuf_ranges:
                for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                    for gbuf_range_map in gbuf_range_map_for_all_buckets:
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                            optim_state = self.optimizer.state[main_param]

                            src_tensors = state_dict[param_idx]
                            dst_tensors = {"fp32_param": main_param, **optim_state}
                            for key in dst_tensors:
                                dst_tensors[key].copy_(src_tensors[key])

                            param_idx += 1

    @classmethod
    def _update_legacy_world_tensors(cls, old_tensors, new_numels):
        '''Reshard buckets (where each bucket is a tensor) to new target
        numels, where the total numel remains the same.'''

        old_total = sum([t.numel() for t in old_tensors])
        new_total = sum(new_numels)

        assert old_total == new_total

        unified_tensor = torch.cat(old_tensors, dim=0)

        new_tensors = []
        start_idx = 0
        for new_numel in new_numels:
            new_tensors.append(unified_tensor[start_idx : (start_idx + new_numel)])
            start_idx += new_numel

        return new_tensors

    def load_parameter_state_from_dp_zero_legacy(self, state_dict):
        """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
        using the legacy checkpoint format as described below.

        The difference between this method and `load_parameter_state_from_dp_zero_modern()`
        is that this method is used for updating the format of checkpoints that
        were saved using code from before Feb 13, 2024. Starting on this date, a
        new format was used (i.e., different format for the parameter mapping and
        bucket sharding).

        Use arg `--ckpt-convert-update-legacy-dist-opt-format` to call this
        method, along with `--ckpt-convert-format` and `--ckpt-convert-save` to
        update a legacy-format checkpoint to the modern format.
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Scatter tensors to all DP ranks.
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                if data_parallel_rank == 0:
                    buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                    model_numels = [b.numel_unpadded for b in self.buffers[gbuf_idx].buckets]
                    checkpoint_numels = [
                        t.numel() for t in state_dict[gbuf_idx][torch.float32]["param"]
                    ]
                    assert sum(model_numels) == sum(checkpoint_numels)
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    legacy_world_tensors = self._update_legacy_world_tensors(
                        state_dict[gbuf_idx][torch.float32][key],
                        [
                            self.buffers[gbuf_idx].buckets[bi].numel_unpadded
                            for bi in range(len(gbuf_range_map_for_all_buckets))
                        ],
                    )
                    offset_in_world_tensors = 0
                    for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                        # Compute local DP contiguous shard's size.
                        gbuf_world_numel = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                        )
                        assert gbuf_world_numel % data_parallel_world_size == 0
                        gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                        gbuf_world_numel_unpadded = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                        )
                        assert gbuf_world_numel_unpadded <= gbuf_world_numel

                        # Contiguous local shards (received from DP rank 0).
                        recv_tensor = torch.empty(
                            (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                        )

                        # Scatter tensor list.
                        if data_parallel_rank == 0:

                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded

                            world_tensor = legacy_world_tensors[bucket_idx]
                            assert (
                                world_tensor.numel() == gbuf_world_numel_unpadded
                            ), "%d vs. %d." % (world_tensor.numel(), gbuf_world_numel_unpadded)
                            offset_in_world_tensors += gbuf_world_numel_unpadded

                            # Pad world_tensor to gbuf_world_numel. Don't pad at the front,
                            # pad at the back.
                            world_tensor = torch.nn.functional.pad(
                                world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                            )
                            assert world_tensor.numel() == gbuf_world_numel
                            gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                            send_tensors = [
                                world_tensor[i : (i + gbuf_local_numel)] for i in gbuf_start_idxs
                            ]
                        else:
                            send_tensors = None

                        # Scatter.
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Copy local contiguous shards to param/optim shards.
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():

                            # Main param & optimizer states.
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            main_param = self.optimizer.param_groups[group_index]["params"][
                                group_order
                            ]
                            if key == "param":
                                tensor_to_copy_into = main_param
                            else:
                                optim_state = self.optimizer.state[main_param]
                                tensor_to_copy_into = optim_state[key]

                            # Copy states into contiguous shard.
                            gbuf_local_start = param_range_map["gbuf_local"].start
                            gbuf_local_end = param_range_map["gbuf_local"].end
                            tensor_to_copy_into.data.copy_(
                                recv_tensor[gbuf_local_start:gbuf_local_end]
                            )

    def load_parameter_state_from_dp_zero(self, state_dict, *, update_legacy_format=False):
        """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
        using the new checkpoint format with coalesced state across buckets.

        This method performs the reverse of get_parameter_state_dp_zero():
        - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
          rank receives its relevant subset of the world buffers).
        - For each DP rank, copy param & optimizer shards from contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """

        # Selectively load from a legacy checkpoint. The legacy format was used
        # prior to Feb 13, 2024.
        if update_legacy_format:
            return self.load_parameter_state_from_dp_zero_legacy(state_dict)

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        if data_parallel_rank == 0:
            # Do nothing if "--fp8-param-gather" is not used.
            self.split_state_dict_if_needed(state_dict)

        # Scatter tensors to all DP ranks.
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                if data_parallel_rank == 0:
                    buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                    checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                    assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                        f"Number of unpadded elements must be same in current run "
                        f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                    )
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    offset_in_world_tensors = 0
                    for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                        # Compute local DP contiguous shard's size.
                        gbuf_world_numel = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                        )
                        assert gbuf_world_numel % data_parallel_world_size == 0
                        gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                        gbuf_world_numel_unpadded = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                        )
                        assert gbuf_world_numel_unpadded <= gbuf_world_numel

                        # Contiguous local shards (received from DP rank 0).
                        recv_tensor = torch.zeros(
                            (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                        )

                        # Scatter tensor list.
                        if data_parallel_rank == 0:
                            world_tensors = state_dict[gbuf_idx][dtype][key]

                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            assert 0 <= start < end <= world_tensors.numel()
                            world_tensor = world_tensors[start:end]
                            offset_in_world_tensors += gbuf_world_numel_unpadded

                            # Pad world_tensor to gbuf_world_numel. Don't pad at the front,
                            # pad at the back.
                            world_tensor = torch.nn.functional.pad(
                                world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                            )
                            assert world_tensor.numel() == gbuf_world_numel
                            gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                            send_tensors = [
                                world_tensor[i : (i + gbuf_local_numel)] for i in gbuf_start_idxs
                            ]
                        else:
                            send_tensors = None

                        # Scatter.
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Copy local contiguous shards to param/optim shards.
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():

                            # Main param & optimizer states.
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            main_param = self.optimizer.param_groups[group_index]["params"][
                                group_order
                            ]
                            if key == "param":
                                tensor_to_copy_into = main_param
                            else:
                                optim_state = self.optimizer.state[main_param]
                                tensor_to_copy_into = optim_state[key]

                            # Copy states into contiguous shard.
                            gbuf_local_start = param_range_map["gbuf_local"].start
                            gbuf_local_end = param_range_map["gbuf_local"].end
                            tensor_to_copy_into.data.copy_(
                                recv_tensor[gbuf_local_start:gbuf_local_end]
                            )

    def split_state_dict_if_needed(self, state_dict):
        """
        When "--fp8-param-gather" is disabled, weights and biases are stored in the same
        `ParamAndGradBuffer`. So, when saving a checkpoint, the optimizer's main parameters are
        saved in a single continuous tensor (this also applies to "exp_avg" and "exp_avg_sq").

        However, when "--fp8-param-gather" is enabled, weights(in fp8 dtype) and biases(in bf16/fp16
        dtype) are stored in separate `ParamAndGradBuffer`. Therefore, when we enabled
        "--fp8-param-gather", and want to load a checkpoint saved without "--fp8-param-gather", we
        need to split the weights(fp8) and biases(bf16/fp16) in the static_dict into two separate
        tensors.
        """
        # Skip if there is no fp8 buffers.
        fp8_gbuf_indices = []
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, _ in gbuf_range_maps.items():
                if is_float8tensor(self.buffers[gbuf_idx].params[0]):
                    fp8_gbuf_indices.append(gbuf_idx)
        if len(fp8_gbuf_indices) == 0:
            return

        dtype_to_gbuf_idx = {}
        for key in state_dict.keys():
            if key != 'buckets_coalesced':
                for dtype in state_dict[key].keys():
                    assert dtype not in dtype_to_gbuf_idx
                    if dtype[0] == torch.uint8:
                        # If the `state_dict`` already contains a torch.uint8 buffer, we assumed
                        # that the fp8 weights and fp16/bf16 biases in the checkpoint are already
                        # separated. In this case, no action is required, so we can return directly.
                        return
                    dtype_to_gbuf_idx[dtype] = key

        # 1. Replace the gbuf_idx in the checkpoint with the new gbuf_idx.
        # 2. Copy the non-tensor data (i.e., the "buckets_coalesced") to `new_state_dict`.
        new_state_dict = {'buckets_coalesced': state_dict['buckets_coalesced']}
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, _ in gbuf_range_maps.items():
                if not is_float8tensor(self.buffers[gbuf_idx].params[0]):
                    new_state_dict[gbuf_idx] = state_dict[dtype_to_gbuf_idx[dtype]]

        for fp8_gbuf_idx in fp8_gbuf_indices:
            # Note that `self.buffers[fp8_gbuf_idx].params[0].dtype` is the dummy dtype of
            # `Float8Tensor`, not torch.uint8.
            non_fp8_param_and_grad_dtype = (
                self.buffers[fp8_gbuf_idx].params[0].dtype,
                self.buffers[fp8_gbuf_idx].grad_dtype,
            )

            # Iterate through all buffers to find the one that needs to be split.
            non_fp8_gbuf_idx = None
            for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
                for dtype, _ in gbuf_range_maps.items():
                    if dtype == non_fp8_param_and_grad_dtype:
                        non_fp8_gbuf_idx = gbuf_idx
            assert non_fp8_gbuf_idx is not None

            # We need the fp8_flags to determine the order of weight (fp8) and bias (fp16/bf16) in
            # the buffer.
            index_to_fp8_map = {}
            for index in self.buffers[fp8_gbuf_idx].param_indices:
                assert index not in index_to_fp8_map
                index_to_fp8_map[index] = True
            for index in self.buffers[non_fp8_gbuf_idx].param_indices:
                assert index not in index_to_fp8_map
                index_to_fp8_map[index] = False
            param_indices = (
                self.buffers[fp8_gbuf_idx].param_indices
                + self.buffers[non_fp8_gbuf_idx].param_indices
            )
            assert min(param_indices) == 0
            assert max(param_indices) == len(param_indices) - 1
            fp8_flags = []
            for i in range(len(param_indices)):
                fp8_flag.append(index_to_fp8_map[i])

            fp8_buffer = self.buffers[fp8_gbuf_idx]
            non_fp8_buffer = self.buffers[non_fp8_gbuf_idx]

            fp8_idx = len(fp8_buffer.params) - 1
            non_fp8_idx = len(non_fp8_buffer.params) - 1
            offsets, fp8_offsets, non_fp8_offsets = [0], [0], [0]

            # Because the parameters in `ParamAndGradBuffer` are traversed in reverse order, the
            # flag here also needs to be traversed in reverse order.
            for fp8_flag in fp8_flags[::-1]:
                if fp8_flag:
                    numel = fp8_buffer.params[fp8_idx].nelement()
                    fp8_idx -= 1
                    offsets.append(offsets[-1] + numel)
                    fp8_offsets.append(fp8_offsets[-1] + numel)
                else:
                    numel = non_fp8_buffer.params[non_fp8_idx].nelement()
                    non_fp8_idx -= 1
                    offsets.append(offsets[-1] + numel)
                    non_fp8_offsets.append(non_fp8_offsets[-1] + numel)

            # Split the target buffer into two separate buffers.
            fp8_state_dict, non_fp8_state_dict = {}, {}
            for key in ['param', 'exp_avg', 'exp_avg_sq']:
                tensor = state_dict[non_fp8_gbuf_idx][non_fp8_param_and_grad_dtype][key]
                fp8_tensor = torch.empty([fp8_offsets[-1]], dtype=tensor.dtype)
                non_fp8_tensor = torch.empty([non_fp8_offsets[-1]], dtype=tensor.dtype)

                fp8_idx, non_fp8_idx = 0, 0
                for i in range(len(offsets) - 1):
                    if fp8_flags[-(i + 1)]:
                        fp8_tensor[fp8_offsets[fp8_idx] : fp8_offsets[fp8_idx + 1]].copy_(
                            tensor[offsets[i] : offsets[i + 1]]
                        )
                        fp8_idx += 1
                    else:
                        non_fp8_tensor[
                            non_fp8_offsets[non_fp8_idx] : non_fp8_offsets[non_fp8_idx + 1]
                        ].copy_(tensor[offsets[i] : offsets[i + 1]])
                        non_fp8_idx += 1

                fp8_state_dict[key] = fp8_tensor
                non_fp8_state_dict[key] = non_fp8_tensor

            fp8_state_dict['numel_unpadded'] = fp8_offsets[-1]
            non_fp8_state_dict['numel_unpadded'] = non_fp8_offsets[-1]

            # Add the two separate buffers into `new_state_dict`.
            new_state_dict[fp8_gbuf_idx] = {}
            new_state_dict[fp8_gbuf_idx][(torch.uint8, fp8_buffer.grad_dtype)] = fp8_state_dict
            new_state_dict[non_fp8_gbuf_idx][non_fp8_param_and_grad_dtype] = non_fp8_state_dict

        # Inplace update state_dict
        state_dict.clear()
        for key, value in new_state_dict.items():
            state_dict[key] = value

    def load_parameter_state(self, filename: str, *, update_legacy_format=False):
        """Load the distributed parameter state from disk.

        Args:
            filename (str): path to load parameter state from.
        """
        state_dict = None
        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            state_dict = torch.load(filename)

        self.load_parameter_state_from_dp_zero(
            state_dict, update_legacy_format=update_legacy_format
        )

    def zero_grad(self, set_to_none: bool = True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.

        Args:
            set_to_none (bool): if true, set grads to None.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            for param_and_grad_buffer in self.buffers:
                param_and_grad_buffer.zero_grad()
            return

        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_fp32_from_float16_groups,
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)

    def _collect_main_grad_data_for_unscaling(self):
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but written differently, so the two should be combined.
        """
        return [
            param.grad.data for group in self.optimizer.param_groups for param in group["params"]
        ]

    def _get_model_and_main_params_data_float16(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.shard_float16_groups, self.shard_fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        if self.ddp_config.with_megatron_fsdp_code_path:
            # FSDP will handle the grad copy, so we can return early.
            return

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        if self.ddp_config.with_megatron_fsdp_code_path:
            for param_and_grad_buffer in self.buffers:
                param_and_grad_buffer.update_model_weights()
            return

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]

                    if is_float8tensor(model_param):
                        # 1. When "--fp8-param-gather" is disabled, the main param is first cast to
                        #    BF16/FP16, and then cast to FP8, so the amax_history is calculated
                        #    using BF16/FP16 param.
                        # 2. When "--fp8-param-gather" is enabled, we can cast the FP32 main param
                        #    to FP8 directly, which results in slightly different results with
                        #    higher speed. In theory, this does not affect convergence.
                        # TODO: The following code maintains the logic of the point-1 above. It can
                        # be deleted if it is not necessary.
                        shard_main_param = shard_main_param.to(model_param.dtype)

                        cast_to_fp8(
                            shard_main_param.view(1, -1),
                            model_param._fp8_meta['scaling_fwd'],
                            model_param._fp8_meta_index,
                            model_param._fp8_dtype,
                            out=shard_model_param.view(1, -1),
                        )
                    else:
                        shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        if self.ddp_config.with_megatron_fsdp_code_path:
            for param_and_grad_buffer in self.buffers:
                param_and_grad_buffer.copy_model_weight_to_master_weight()
            return

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

    def _update_fp8_scale_inv_and_amax(self):
        """
        If detect FP8 parameters, update their `_scale_inv` and do reduce-max for their
        `amax_history`.
        """
        amaxes = []
        scales = []
        scale_invs = []
        # Iterate over all parameters inside this optimizer to find FP8 parameters.
        for buffer in self.buffers:
            for param in buffer.params:
                if is_float8tensor(param):
                    fp8_meta = param._fp8_meta['scaling_fwd']
                    fp8_meta_index = param._fp8_meta_index
                    amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
                    scales.append(fp8_meta.scale[fp8_meta_index].view(1))
                    scale_invs.append(param._scale_inv.view(1))
                    # Reset transpose cache
                    param._reset_caches()

        # If there is no FP8 parameters, skip all operations.
        if len(scales) > 0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

            # Update scaling factors.
            packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
            packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
            _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
            torch.reciprocal(packed_scales, out=packed_scales)
            _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

            # Reduce amaxes.
            # Note: Assume each param has a separate amax.
            packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
            packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
            _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
            torch.distributed.all_reduce(
                packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=self.data_parallel_group
            )
            _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        update_successful = super().step_with_ready_grads()

        # If there is no FP8 parameters, this will do nothing.
        self._update_fp8_scale_inv_and_amax()

        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)

        if self.ddp_config.with_megatron_fsdp_code_path:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        else:
            # If not overlapping all-gather for parameters, launch synchronous all-gather
            # communication calls here. If overlapping all-gather for parameters, the following
            # the first all-gather is launched asynchronously in the next optimizer.zero_grad()
            # call and subsequent all-gathers are launched in the forward pre-hook.
            if not self.ddp_config.overlap_param_gather:
                for model_chunk in self.model_chunks:
                    model_chunk.start_param_sync()
        if timers is not None:
            timers('params-all-gather').stop()

        return update_successful
