# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in request-lifecycle logging for dynamic inference debugging."""

import logging
import os
from typing import Any

_TRACE_ENV_VAR = "MCORE_INFERENCE_REQUEST_TRACE"


def trace_request(stage: str, **fields: Any) -> None:
    """Emit a request-lifecycle record when debugging is explicitly enabled.

    The dynamic HTTP frontend, coordinator, and client run in separate processes.
    This intentionally logs only process-local or protocol identifiers, never a
    prompt or generated text, so a hang can be located without exposing payloads.
    """
    if os.environ.get(_TRACE_ENV_VAR) != "1":
        return

    def format_value(value: Any) -> str:
        return value.hex() if isinstance(value, bytes) else str(value)

    details = " ".join(
        f"{key}={format_value(value)}" for key, value in sorted(fields.items())
    )
    logging.info(
        "MCORE_REQUEST_TRACE stage=%s pid=%s%s",
        stage,
        os.getpid(),
        f" {details}" if details else "",
    )
