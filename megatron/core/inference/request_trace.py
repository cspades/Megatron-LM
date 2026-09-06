# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in request-lifecycle logging for dynamic inference debugging."""

import os
from typing import Any

_TRACE_ENV_VAR = "MCORE_INFERENCE_REQUEST_TRACE"
_TRACE_FILE_ENV_VAR = "MCORE_INFERENCE_REQUEST_TRACE_FILE"


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
    # Frontend replicas are spawned with a fresh interpreter. They do not
    # necessarily inherit Ray's logging configuration, so logging.info() can be
    # silently filtered there. stdout is the useful default, but Ray can merge
    # or coalesce output from an actor's background engine thread. A per-PID
    # append-only file is therefore available for a lossless hang trace.
    message = f"MCORE_REQUEST_TRACE stage={stage} pid={os.getpid()}"
    if details:
        message += f" {details}"

    trace_file = os.environ.get(_TRACE_FILE_ENV_VAR)
    if trace_file:
        try:
            fd = os.open(
                f"{trace_file}.{os.getpid()}",
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, f"{message}\n".encode())
            finally:
                os.close(fd)
            return
        except OSError:
            # Fall through to stdout: diagnostics must not affect serving.
            pass

    print(message, flush=True)
