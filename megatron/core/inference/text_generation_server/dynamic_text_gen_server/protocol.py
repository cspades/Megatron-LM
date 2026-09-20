# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transport constants shared without importing endpoint modules."""

REQUEST_TIMEOUT_HEADER = "X-Megatron-Request-Timeout-Seconds"
RETRYABLE_HEADER = "X-Nemo-Retryable"
ERROR_CODE_HEADER = "X-Nemo-Error-Code"
GENERATION_ABORTED_ERROR_CODE = "generation_aborted"
