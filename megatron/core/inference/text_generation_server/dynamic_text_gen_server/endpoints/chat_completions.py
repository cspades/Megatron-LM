# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import base64
import contextlib
import ipaddress
import json
import logging
import socket
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from functools import partial

_MEDIA_FETCH_TIMEOUT_S = 5.0
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
_MAX_VIDEO_BYTES = 256 * 1024 * 1024  # 256 MiB
_MEDIA_FETCH_USER_AGENT = "megatron-inference"

from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    prepare_multimodal_data,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    GENERATION_ABORTED_ERROR_CODE,
    frontend_request_finished,
    frontend_request_phase,
    frontend_request_started,
    generation_config_sampling_defaults,
    log_sampling_defaults_once,
    non_retryable_error_headers,
    remaining_request_timeout,
    resolve_sampling_default,
)
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

from ..incremental_detokenizer import HuggingFaceFastIncrementalDetokenizer
from ..openai_streaming import (
    StreamingChatParser,
    json_safe_logprobs,
    json_safe_top_n_logprobs,
    openai_stream,
    stream_with_deadline,
)
from .common import abort_requests

logger = logging.getLogger(__name__)

# pylint: disable=line-too-long


def _request_tag(request_id: int) -> str:
    """Letters-only request tag so Ray does not deduplicate lifecycle logs."""
    value = int(request_id)
    chars = []
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord("a") + remainder))
        if value == 0:
            return "".join(reversed(chars))
        value -= 1


async def _log_pending_requests(
    client,
    request_ids: list[int],
    *,
    http_started_at: float,
    prompt_length: int,
    max_tokens: int,
) -> None:
    """Expose coordinator/admission/decode waits hidden behind one HTTP await."""
    next_tick = time.perf_counter() + 30.0
    while True:
        await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))
        woke_at = time.perf_counter()
        event_loop_lag = max(0.0, woke_at - next_tick)
        next_tick += 30.0
        snapshot = client.pending_request_diagnostics(request_ids)
        if not snapshot["pending"]:
            return
        logger.warning(
            "chat request pending: tag=%s ids=%s http_age=%.1fs "
            "engine_pending=%d/%d oldest_engine=%.1fs prompt_length=%d "
            "max_new_tokens=%d event_loop_lag=%.3fs listener_done=%s "
            "reply_frames=%d reply_idle=%s socket_events=%d "
            "monitor_events=%s last_monitor=%s oldest=%s",
            _request_tag(request_ids[0]),
            request_ids,
            time.perf_counter() - http_started_at,
            snapshot["pending"],
            snapshot["requested"],
            snapshot["oldest_seconds"],
            prompt_length,
            max_tokens,
            event_loop_lag,
            snapshot["listener_done"],
            snapshot["reply_frames_received"],
            (
                f"{snapshot['seconds_since_last_reply']:.1f}s"
                if snapshot["seconds_since_last_reply"] is not None
                else "never"
            ),
            snapshot["socket_events"],
            snapshot["monitor_event_counts"],
            snapshot["last_monitor_event"],
            snapshot["oldest"],
        )


_TOKEN_ID_FIELDS_TO_REDACT = {
    "prompt_tokens",
    "remaining_prompt_tokens",
    "generated_tokens",
    "prompt_token_ids",
    "generation_token_ids",
}

_INDEX_FIELDS_TO_REDACT = {"routing_indices", "moe_topk_indices", "prompt_moe_topk_indices"}

_HASH_FIELDS_TO_REDACT = {"precomputed_block_hashes"}

_NUMERIC_SERIES_FIELDS_TO_REDACT = {"tpot"}


def _is_int_list_like(value):
    """Return True for integer lists, including nested integer lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, int) or _is_int_list_like(item) for item in value)


def _is_numeric_list_like(value):
    """Return True for numeric lists, including nested numeric lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, (int, float)) or _is_numeric_list_like(item) for item in value)


def _redact_token_id_lists_for_logging(value):
    """Redact verbose token-id arrays from logs."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if (
                key in _TOKEN_ID_FIELDS_TO_REDACT
                or key in _INDEX_FIELDS_TO_REDACT
                or key in _HASH_FIELDS_TO_REDACT
                or key.endswith("_token_ids")
                or key.endswith("_topk_indices")
                or key.endswith("_hashes")
            ) and _is_int_list_like(item):
                redacted[key] = "...truncated..."
            elif key in _NUMERIC_SERIES_FIELDS_TO_REDACT and _is_numeric_list_like(item):
                redacted[key] = "...truncated..."
            else:
                redacted[key] = _redact_token_id_lists_for_logging(item)
        return redacted
    if isinstance(value, list):
        return [_redact_token_id_lists_for_logging(item) for item in value]
    return value


def _get_field(obj, key, default=None):
    """Read a field from dict-like or object-like values."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_non_none(obj, key, default):
    """Returns the value from the object or default if the key is missing or None."""
    val = obj.get(key)
    return default if val is None else val


def _try_parse_jsonish(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _extract_declared_types(schema):
    """Recursively extract declared JSON-schema type names."""
    declared = set()
    if not isinstance(schema, dict):
        return declared

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        declared.add(schema_type.strip().lower())
    elif isinstance(schema_type, list):
        for item in schema_type:
            if isinstance(item, str):
                declared.add(item.strip().lower())

    for combinator in ("anyOf", "oneOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for option in options:
                declared.update(_extract_declared_types(option))
    return declared


def _get_tool_argument_schemas(tools):
    """Build function-name to argument-schema mapping from request tools."""
    schemas = {}
    if not isinstance(tools, list):
        return schemas

    for tool in tools:
        function = _get_field(tool, "function", {}) or {}
        function_name = _get_field(function, "name")
        params = _get_field(function, "parameters", {})
        if not isinstance(function_name, str) or not isinstance(params, dict):
            continue
        if isinstance(params.get("properties"), dict):
            schemas[function_name] = params.get("properties")
        else:
            schemas[function_name] = params
    return schemas


def _normalize_structured_tool_arguments(arguments, function_name, tool_argument_schemas):
    """Coerce structured (array/object) args from JSON strings to native types."""
    if not isinstance(arguments, dict):
        return arguments

    function_schema = tool_argument_schemas.get(function_name, {})
    if not isinstance(function_schema, dict):
        return arguments

    normalized = dict(arguments)
    for key in normalized:
        param_schema = function_schema.get(key)
        declared_types = _extract_declared_types(param_schema)
        if not (declared_types & {"array", "arr", "object", "dict", "list"}):
            continue
        parsed = _try_parse_jsonish(normalized[key])
        if isinstance(parsed, (dict, list)):
            normalized[key] = parsed
    return normalized


def _normalize_tool_calls(tool_calls, tools=None):
    """Normalize tool calls to OpenAI-compatible JSON primitives."""
    tool_argument_schemas = _get_tool_argument_schemas(tools)
    normalized = []
    for call in tool_calls or []:
        fn = _get_field(call, "function", {}) or {}
        fn_name = _get_field(fn, "name")
        fn_args = _get_field(fn, "arguments", "")
        if fn_name is None:
            continue
        if isinstance(fn_args, str):
            try:
                parsed_args = json.loads(fn_args)
            except (TypeError, ValueError):
                parsed_args = None
            if isinstance(parsed_args, dict):
                fn_args = json.dumps(
                    _normalize_structured_tool_arguments(
                        parsed_args, fn_name, tool_argument_schemas
                    ),
                    ensure_ascii=False,
                )
        elif isinstance(fn_args, dict):
            fn_args = json.dumps(
                _normalize_structured_tool_arguments(fn_args, fn_name, tool_argument_schemas),
                ensure_ascii=False,
            )
        else:
            try:
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            except TypeError:
                fn_args = str(fn_args)
        normalized.append(
            {
                "id": str(_get_field(call, "id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {"name": str(fn_name), "arguments": fn_args},
            }
        )
    return normalized


def _maybe_filter_parallel_tool_calls(tool_calls, parallel_tool_calls):
    """Filter to first tool call only when parallel_tool_calls is False.

    Matches vLLM's maybe_filter_parallel_tool_calls behavior.
    """
    if parallel_tool_calls:
        return tool_calls
    if tool_calls:
        return tool_calls[:1]
    return tool_calls


def _coerce_arguments_mapping(arguments):
    """Coerce function.arguments to a mapping for HF/Jinja chat templates.

    Examples:
    - {"x": 1} -> {"x": 1}
    - '{"x": 1}' -> {"x": 1}
    - "[1, 2]" -> {}  # JSON parses, but not a mapping
    - "not-json" -> {}
    - None -> {}
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTP redirects so a 3xx to a private address can't bypass the
    pre-fetch allowlist check."""

    def http_error_301(self, req, fp, code, msg, headers):
        """Turn a 3xx redirect into an HTTPError so the fetch fails closed."""
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects disabled for image_url fetches", headers, fp
        )

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


_no_redirect_opener = urllib.request.build_opener(_NoRedirectHandler())


def _extract_media_url_bytes(url: str, *, max_bytes: int) -> bytes:
    """Extract size-bounded bytes from an OpenAI-style media URL.

    Supports base64-encoded data URLs (``data:image/...;base64,<b64>``) and
    plain ``http(s)://`` URLs.
    """
    if url.startswith("data:"):
        # Base64 encodes each three input bytes as four characters. Bound the
        # complete request before splitting or decoding it; 256 characters is
        # ample for the data-URL metadata preceding the comma.
        max_encoded_chars = 4 * ((max_bytes + 2) // 3)
        if len(url) > max_encoded_chars + 256:
            raise ValueError(f"Media data URL exceeds {max_bytes} byte limit")
        try:
            metadata, b64_data = url.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed media data URL: {url[:40]!r}") from exc
        if len(b64_data) > max_encoded_chars:
            raise ValueError(f"{metadata} payload exceeds {max_bytes} byte limit")
        data = base64.b64decode(b64_data)
        if len(data) > max_bytes:
            raise ValueError(f"{metadata} payload exceeds {max_bytes} byte limit")
        return data
    if url.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname:
            raise ValueError(f"Invalid media URL: {url[:40]!r}")
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        except (socket.gaierror, ValueError) as exc:
            raise ValueError(f"Cannot resolve media URL host: {parsed.hostname}") from exc
        # Refuse SSRF-prone destinations (loopback, RFC1918, link-local,
        # multicast, reserved, unspecified). Public addresses only.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Refusing to fetch media from non-public address: {parsed.hostname}")
        req = urllib.request.Request(url, headers={"User-Agent": _MEDIA_FETCH_USER_AGENT})
        with _no_redirect_opener.open(req, timeout=_MEDIA_FETCH_TIMEOUT_S) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Media at {parsed.hostname} exceeds {max_bytes} byte limit")
        return data
    raise ValueError(f"Unsupported media URL scheme: {url[:40]!r}")


def _extract_multimodal_from_messages(messages, prompt_config: MultimodalPromptConfig):
    """Extract media bytes and replace structured blocks with internal slots.

    Remote image fetching is blocking, so callers must run this function off
    the event loop.
    """
    if not isinstance(messages, list):
        return messages, [], [], []

    rewritten = []
    image_bytes_list: list[bytes] = []
    video_bytes_list: list[bytes] = []
    media_slots = []

    def add_slot(modality, message_index):
        """Preserve a media block's position while the chat template renders text.

        The actual bytes travel separately. After rendering,
        _tokenize_with_media_slots_sync replaces this sentinel with the
        model-specific MediaPromptSpec tokens.
        """
        sentinel = f"__MCORE_MEDIA_SLOT_{len(media_slots)}__"
        media_slots.append((sentinel, modality, message_index))
        return {"type": "text", "text": sentinel}

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            rewritten.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue

        new_chunks = []
        found_modalities = set()
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "image_url":
                url = chunk.get("image_url", {}).get("url", "")
                if not url:
                    continue
                try:
                    image_bytes_list.append(
                        _extract_media_url_bytes(url, max_bytes=_MAX_IMAGE_BYTES)
                    )
                except Exception as e:
                    # Dropping the image would answer the request as if it were
                    # text-only, handing the client a confident answer about an
                    # image the model never saw. Surface it as a 400 instead.
                    raise ValueError(f"Failed to load image_url: {e}") from e
                new_chunks.append(add_slot("image", message_index))
                found_modalities.add("image")
            elif isinstance(chunk, dict) and chunk.get("type") in {"video_url", "input_video"}:
                video_value = chunk.get("video_url") or chunk.get("video")
                url = video_value.get("url", "") if isinstance(video_value, dict) else video_value
                if not isinstance(url, str) or not url.startswith("data:"):
                    raise ValueError("Megatron chat video inputs must be base64 data URLs.")
                try:
                    video_bytes_list.append(
                        _extract_media_url_bytes(url, max_bytes=_MAX_VIDEO_BYTES)
                    )
                except Exception as e:
                    raise ValueError(f"Failed to load video_url: {e}") from e
                new_chunks.append(add_slot("video", message_index))
                found_modalities.add("video")
            else:
                new_chunks.append(chunk)

        if found_modalities:
            input_markers = {
                prompt_config.get_spec(modality).input_marker for modality in found_modalities
            } - {None}
            for index, chunk in enumerate(new_chunks):
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    chunk = dict(chunk)
                    for marker in input_markers:
                        chunk["text"] = str(chunk.get("text", "")).replace(marker, "")
                    new_chunks[index] = chunk
            msg_copy = dict(message)
            msg_copy["content"] = new_chunks
            rewritten.append(msg_copy)
        else:
            rewritten.append(message)

    if image_bytes_list and video_bytes_list:
        raise ValueError("Mixing image and video blocks in one request is not supported.")
    return rewritten, image_bytes_list, video_bytes_list, media_slots


def _sanitize_messages_for_template(messages, media_slots=(), prompt_config=None):
    """Prepare messages so tokenizer chat templates can safely consume them.

    This only normalizes tool-call argument payloads inside each message:
    - messages[*].tool_calls[*].function.arguments is coerced to a dict.

    Example transformation:
    Input:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]}]
    Output:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}]

    Another example:
    - arguments: "[1,2,3]" -> arguments: {}
    """
    if not isinstance(messages, list):
        return messages
    sanitized = []
    media_modalities_by_message = {}
    for _sentinel, modality, message_index in media_slots:
        media_modalities_by_message.setdefault(message_index, set()).add(modality)

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        msg_copy = dict(message)
        content = msg_copy.get("content")
        # OpenAI-style multimodal/text content may arrive as a list of blocks.
        # HF/Jinja chat templates used by this server expect plain strings.
        if isinstance(content, list):
            text_chunks = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text":
                        text_chunks.append(str(chunk.get("text", "")))
                    elif "text" in chunk:
                        text_chunks.append(str(chunk.get("text", "")))
                elif isinstance(chunk, str):
                    text_chunks.append(chunk)
            separator = ""
            message_modalities = media_modalities_by_message.get(message_index, set())
            if message_modalities:
                if prompt_config is None:
                    raise ValueError("Media content normalization requires a prompt config.")
                separators = {
                    prompt_config.get_spec(modality).content_part_separator
                    for modality in message_modalities
                }
                if len(separators) != 1:
                    raise ValueError(
                        "Media types in one message must use the same content-part separator."
                    )
                separator = separators.pop()
            msg_copy["content"] = separator.join(chunk for chunk in text_chunks if chunk)
        elif isinstance(content, dict):
            msg_copy["content"] = str(content.get("text", ""))
        elif content is None:
            msg_copy["content"] = ""
        elif not isinstance(content, str):
            msg_copy["content"] = str(content)

        tool_calls = msg_copy.get("tool_calls")
        if isinstance(tool_calls, list):
            sanitized_tool_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    sanitized_tool_calls.append(call)
                    continue
                call_copy = dict(call)
                function = call_copy.get("function")
                if isinstance(function, dict):
                    function_copy = dict(function)
                    function_copy["arguments"] = _coerce_arguments_mapping(
                        function_copy.get("arguments", {})
                    )
                    call_copy["function"] = function_copy
                sanitized_tool_calls.append(call_copy)
            msg_copy["tool_calls"] = sanitized_tool_calls
        sanitized.append(msg_copy)
    return sanitized


def _sanitize_tools_for_template(tools):
    """Ensure tools payload is template-safe and has mapping parameters.

    Example transformations:
    - {"function": {"name": "f", "parameters": "not-a-dict"}}
      -> {"function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    - non-dict tool entries are dropped.
    - non-list input returns None.
    """
    if not isinstance(tools, list):
        return None

    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_copy = dict(tool)
        function = tool_copy.get("function")
        if isinstance(function, dict):
            function_copy = dict(function)
            if not isinstance(function_copy.get("parameters"), dict):
                function_copy["parameters"] = {"type": "object", "properties": {}}
            tool_copy["function"] = function_copy
        sanitized.append(tool_copy)
    return sanitized


# Keys a client may never supply via `chat_template_kwargs`.
#
# `chat_template` replaces the Jinja template that the server renders for the
# request. The template is server/tokenizer configuration (`--chat-template` or
# the tokenizer's own template), never request data. Honoring a caller-supplied
# template lets an unauthenticated POST hand us arbitrary Jinja that we then
# compile and render synchronously: the sandbox blocks attribute escapes, but it
# does not bound work, so nested `range()` loops or unbounded string
# multiplication pin the worker and stall every other in-flight request on it.
_DISALLOWED_CHAT_TEMPLATE_KWARGS = frozenset({"chat_template"})


def _sanitize_chat_template_kwargs(raw_kwargs):
    """Drop request-supplied `chat_template_kwargs` entries that are not caller-controllable.

    Returns a new dict; the caller's object is never mutated. Non-dict input
    (including `None`) yields an empty dict.
    """
    if not isinstance(raw_kwargs, dict):
        if raw_kwargs is not None:
            logger.warning("Ignoring non-dict chat_template_kwargs: %s", type(raw_kwargs).__name__)
        return {}

    sanitized = {k: v for k, v in raw_kwargs.items() if k not in _DISALLOWED_CHAT_TEMPLATE_KWARGS}
    rejected = sorted(set(raw_kwargs) - set(sanitized))
    if rejected:
        logger.warning(
            "Ignoring disallowed chat_template_kwargs key(s) from request: %s. "
            "The chat template is server configuration; use --chat-template to set it.",
            ", ".join(rejected),
        )
    return sanitized


def _replace_prefix_tokens(
    eos_token_id,
    previous_turn_token_ids,
    retokeenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Replace the token ids that are associated with the previous turn with the actual tokens
    from the previous generation (rather than the ones from the chat template application)."""

    # Strip the EOS from the previous turn token ids if it exists
    if previous_turn_token_ids and previous_turn_token_ids[-1] == eos_token_id:
        previous_turn_token_ids = previous_turn_token_ids[:-1]

    # Find the last EOS token id in the previous turn token ids
    last_eos_token_id_index = len(retokeenized_previous_turn_token_ids) - 1
    # Note that the current conversation stat may be shorter than the previous conversation state.
    scan_len = min(len(retokeenized_previous_turn_token_ids), len(current_turn_token_ids))
    for i in reversed(range(scan_len)):
        if current_turn_token_ids[i] == eos_token_id:
            last_eos_token_id_index = i
            break

    # Replace the current turn token ids with the tokens from the previous generation
    current_turn_additional_token_ids = current_turn_token_ids[last_eos_token_id_index:]

    # Return the previous turn token ids + the current turn token ids
    return previous_turn_token_ids + current_turn_additional_token_ids


def _apply_chat_template_sync(
    tokenizer, messages, tools, chat_template_kwargs, add_generation_prompt=True
):
    """Apply the chat template and coerce to `list[int]`, for use in a worker thread.

    The coercion runs here too: it walks every token, so leaving it on the event loop
    would keep part of the stall this offload exists to remove.
    """
    return _coerce_to_token_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **chat_template_kwargs,
        )
    )


def _coerce_to_token_id_list(result):
    """Convert the return value of `tokenizer.apply_chat_template` to `list[int]`.

    transformers >= 5.x.x sometimes returns a `BatchEncoding` object instead of a `list[int]`.
    """
    # BatchEncoding / dict-like with input_ids
    if isinstance(result, dict) or hasattr(result, "input_ids"):
        ids = result["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    # Fast-tokenizer Encoding object
    if hasattr(result, "ids"):
        ids = result.ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    # Raw tensor / ndarray
    if hasattr(result, "tolist"):
        ids = result.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids
    # Plain list
    return list(result)


def _tokenize_with_media_slots_sync(
    chat_tok,
    messages,
    media_slots,
    prompt_config,
    *,
    tools,
    chat_template_kwargs,
    add_generation_prompt=True,
):
    """Render a chat template and lower internal media slots to model tokens.

    Synchronous, and run whole on the frontend's tokenize executor rather than
    piecewise. Rendering is only the first of 3N+1 tokenizer calls for N media
    slots -- each slot encodes the text before it, then the model token's prefix
    and suffix -- and awaiting just the render left the rest on the event loop,
    where they stall every other request this replica owns.

    One hop to the executor also means one thread touches the tokenizer for the
    whole request. HF tokenizers are not thread-safe, and this runs on the
    executor's private copy.
    """
    rendered = chat_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("Multimodal chat template rendering must return a string.")

    positioned_slots = []
    for sentinel, modality, _message_index in media_slots:
        if rendered.count(sentinel) != 1:
            raise ValueError(f"Chat template did not preserve media slot {sentinel}.")
        positioned_slots.append((rendered.index(sentinel), sentinel, modality))

    prompt_tokens = []
    cursor = 0
    for position, sentinel, modality in sorted(positioned_slots):
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(rendered[cursor:position], add_special_tokens=False))
        )
        spec = prompt_config.get_spec(modality)
        resolved_id = (
            chat_tok.convert_tokens_to_ids(spec.model_token)
            if hasattr(chat_tok, "convert_tokens_to_ids")
            else None
        )
        if resolved_id == getattr(chat_tok, "unk_token_id", None):
            resolved_id = None
        if resolved_id is None:
            raise ValueError(f"Tokenizer does not define media token {spec.model_token!r}.")
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(spec.prefix, add_special_tokens=False))
        )
        prompt_tokens.append(int(resolved_id))
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(spec.suffix, add_special_tokens=False))
        )
        cursor = position + len(sentinel)

    prompt_tokens.extend(
        _coerce_to_token_id_list(chat_tok(rendered[cursor:], add_special_tokens=False))
    )
    return prompt_tokens


try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, g, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    @bp.after_request
    async def _finish_frontend_request(response):
        trace = getattr(g, "nemo_frontend_trace", None)
        if trace is not None:
            frontend_request_phase(trace, "handler_returned", status=response.status_code)
            frontend_request_finished(trace, f"http-{response.status_code}")
        return response

    @bp.teardown_request
    async def _finish_failed_frontend_request(error):
        if error is None:
            return
        trace = getattr(g, "nemo_frontend_trace", None)
        if trace is not None:
            frontend_request_finished(trace, f"exception-{type(error).__name__}")

    def apply_parsers(
        message_text, tools, parsers_list, tools_requested, chat_template_kwargs=None
    ):
        """Runs CPU-intensive text parsing."""
        for parser in parsers_list:
            if parser not in PARSER_MAPPING:
                raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")

        implicit_reasoning_end_markers = (
            tuple(
                marker
                for parser_name in parsers_list
                for marker in getattr(
                    PARSER_MAPPING[parser_name], "implicit_reasoning_end_markers", ()
                )
            )
            if tools_requested
            else ()
        )

        meta = {}
        for parser in parsers_list:
            prev_text = message_text
            parsed_text, new_info = PARSER_MAPPING[parser].parse(
                message_text,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                implicit_reasoning_end_markers=implicit_reasoning_end_markers,
            )
            if "tool_calls" in new_info:
                new_info["tool_calls"] = _normalize_tool_calls(
                    new_info.get("tool_calls", []), tools=tools
                )
                if not tools_requested:
                    # Ignore incidental tool-call syntax in plain chat mode.
                    parsed_text = prev_text
                    new_info.pop("tool_calls", None)
            message_text = parsed_text

            assert not (
                meta.keys() & new_info.keys()
            ), "Multiple parsers found the same information."
            meta.update(new_info)

        return message_text, meta

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        http_started_at = time.perf_counter()
        frontend_trace = frontend_request_started("chat_completions", request.headers)
        g.nemo_frontend_trace = frontend_trace
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']

        req = await request.get_json()
        json_parsed_at = time.perf_counter()
        frontend_request_phase(
            frontend_trace,
            "extract_multimodal",
            json_keys=sorted(req.keys()) if isinstance(req, dict) else [],
        )
        prevent_retokenization = req.get(
            "prevent_retokenization", not current_app.config.get('eval_mode', False)
        )
        tools = req.get("tools", None)
        tool_choice = req.get("tool_choice", None)
        parallel_tool_calls = req.get("parallel_tool_calls", True)
        tools_requested = bool(tools) and tool_choice != "none"
        messages = req.get("messages")
        chat_template_kwargs = _sanitize_chat_template_kwargs(req.get("chat_template_kwargs"))
        # --- 1. Parse Messages ---
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)
        prompt_config = current_app.config['multimodal_prompt_config']
        # Extract structured media before template sanitization. Remote image
        # fetches block, so keep this work off the event loop.
        try:
            messages, image_bytes_list, video_bytes_list, media_slots = await asyncio.to_thread(
                _extract_multimodal_from_messages, messages, prompt_config
            )
        except ValueError as error:
            return Response(str(error), status=400)
        media_extracted_at = time.perf_counter()
        frontend_request_phase(
            frontend_trace,
            "tokenize",
            images=len(image_bytes_list),
            videos=len(video_bytes_list),
        )
        multi_modal_data = None
        if image_bytes_list:
            multi_modal_data = {"image": image_bytes_list}
        elif video_bytes_list:
            multi_modal_data = {"video": video_bytes_list}
        template_messages = _sanitize_messages_for_template(
            messages, media_slots, prompt_config
        )
        template_tools = _sanitize_tools_for_template(tools)

        # Inject the server-configured chat template (e.g. pretraining.jinja for
        # VLM checkpoints). Loaded once at server startup from --chat-template
        # into app.config. This is the only path by which a template reaches the
        # renderer -- requests cannot override it (see
        # _sanitize_chat_template_kwargs).
        server_chat_template = current_app.config.get('chat_template', None)
        if server_chat_template:
            chat_template_kwargs['chat_template'] = server_chat_template

        # Prefer the underlying HF tokenizer for chat-template application when
        # reachable. The vision tokenizer's apply_chat_template is a stub, and
        # the text wrapper just forwards anyway; reaching the HF tokenizer lets
        # us pass tokenize/add_generation_prompt/chat_template directly.
        hf_tok = getattr(getattr(tokenizer, '_tokenizer', None), 'tokenizer', None)
        chat_tok = hf_tok if hf_tok is not None else tokenizer

        # Same resolution against the private copy the worker thread applies the
        # template with; `chat_tok` above is only read for the capability checks.
        # Falls back to the shared tokenizer when no private copy is registered:
        # callers that build the app config directly do not set one, and the copy
        # is a thread-safety isolation measure rather than a correctness one.
        tokenize_tok = current_app.config.get('tokenizer_copy', tokenizer)
        tokenize_hf_tok = getattr(getattr(tokenize_tok, '_tokenizer', None), 'tokenizer', None)
        tokenize_chat_tok = tokenize_hf_tok if tokenize_hf_tok is not None else tokenize_tok

        try:
            if hasattr(chat_tok, 'apply_chat_template') and (
                getattr(chat_tok, "chat_template", None) is not None
                or chat_template_kwargs.get('chat_template') is not None
            ):
                # Template render + tokenization is synchronous and CPU-bound, so
                # run it off the event loop. This is a latency mitigation, not a
                # DoS defense: Jinja rendering is pure Python and holds the GIL,
                # so an expensive template still degrades the loop badly (it just
                # no longer hangs it outright). Request-supplied templates are
                # rejected in _sanitize_chat_template_kwargs -- that is the actual
                # fix. The real win here is the tokenizer half, which drops into
                # Rust and releases the GIL for long conversations.
                #
                # Both paths go through the dedicated single-worker executor
                # rather than asyncio.to_thread: the default executor is shared
                # process-wide, so tokenization would queue behind unrelated work
                # and vice versa, and its worker count would admit many concurrent
                # Jinja renders that contend for the GIL with the very loop this
                # offload protects. The executor also owns a private tokenizer
                # copy, since HF tokenizers are not thread-safe.
                if media_slots:
                    prompt_tokens = await asyncio.get_running_loop().run_in_executor(
                        current_app.config.get('tokenize_executor'),
                        partial(
                            _tokenize_with_media_slots_sync,
                            tokenize_chat_tok,
                            template_messages,
                            media_slots,
                            prompt_config,
                            tools=template_tools,
                            chat_template_kwargs=chat_template_kwargs,
                            add_generation_prompt=True,
                        ),
                    )
                else:
                    prompt_tokens = await asyncio.get_running_loop().run_in_executor(
                        current_app.config.get('tokenize_executor'),
                        partial(
                            _apply_chat_template_sync,
                            tokenize_chat_tok,
                            template_messages,
                            template_tools,
                            chat_template_kwargs,
                        ),
                    )

                if prevent_retokenization:
                    # If we are avoiding retokenization, we need to replace some prompt tokens with the prompt/generation tokens from the previous generation
                    # This improves prefix cache hits and reduces logprob variation between training and inference.

                    # Find the last assistant message
                    last_assistant_message_idx = None
                    for i in reversed(range(len(template_messages))):
                        if template_messages[i]["role"] == "assistant":
                            last_assistant_message_idx = i
                            break

                    last_assistant_message = (
                        template_messages[last_assistant_message_idx]
                        if last_assistant_message_idx is not None
                        else None
                    )

                    # Only proceed if the last assistant message has the token IDs from a previous generation.
                    # Dataset-provided conversation history won't have these fields.
                    if (
                        last_assistant_message is not None
                        and isinstance(last_assistant_message.get("prompt_token_ids"), list)
                        and isinstance(last_assistant_message.get("generation_token_ids"), list)
                    ):
                        messages_to_last_assistant_message = template_messages[
                            : last_assistant_message_idx + 1
                        ]
                        previous_media_slots = [
                            slot for slot in media_slots if slot[2] <= last_assistant_message_idx
                        ]
                        previous_prompt_token_ids = last_assistant_message.get(
                            "compact_prompt_token_ids"
                        )
                        if not isinstance(previous_prompt_token_ids, list):
                            if previous_media_slots:
                                raise ValueError(
                                    "Prefix stitching requires compact_prompt_token_ids "
                                    "from the previous Megatron-Inference response."
                                )
                            previous_prompt_token_ids = last_assistant_message["prompt_token_ids"]
                        eos_token_id = tokenizer.eos_id
                        assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

                        warnings.warn(
                            "Avoiding prefix retokenization."
                            " This is a patch that ensures subsequent generations are not retokenized differently than the previous generation."
                            " This may cause unexpected behavior if messages (including system messages) are altered between generations."
                        )

                        # Get the templated tokenization of just the previous generation.
                        if previous_media_slots:
                            retokenized_previous_turn_token_ids = (
                                await asyncio.get_running_loop().run_in_executor(
                                    current_app.config.get('tokenize_executor'),
                                    partial(
                                        _tokenize_with_media_slots_sync,
                                        tokenize_chat_tok,
                                        messages_to_last_assistant_message,
                                        previous_media_slots,
                                        prompt_config,
                                        tools=template_tools,
                                        chat_template_kwargs=chat_template_kwargs,
                                        add_generation_prompt=False,
                                    ),
                                )
                            )
                        else:
                            retokenized_previous_turn_token_ids = (
                                await asyncio.get_running_loop().run_in_executor(
                                    current_app.config.get('tokenize_executor'),
                                    partial(
                                        _apply_chat_template_sync,
                                        tokenize_chat_tok,
                                        messages_to_last_assistant_message,
                                        template_tools,
                                        chat_template_kwargs,
                                        add_generation_prompt=False,
                                    ),
                                )
                            )

                        previous_turn_token_ids = (
                            previous_prompt_token_ids
                            + last_assistant_message["generation_token_ids"]
                        )
                        prompt_tokens = _replace_prefix_tokens(
                            eos_token_id,
                            previous_turn_token_ids,
                            retokenized_previous_turn_token_ids,
                            prompt_tokens,
                        )

            else:
                if media_slots:
                    raise ValueError("Multimodal chat requests require a chat template.")
                warnings.warn(
                    "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
                )
                prompt_tokens = tokenizer.tokenize(
                    "\n".join([message["content"] for message in messages])
                )
        except ValueError as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Invalid 'messages': {e}", status=400)
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        frontend_request_phase(frontend_trace, "sampling_parameters")
        try:
            # For a field the request omits: an explicitly configured server default
            # wins, then the model's generation_config.json, then the previous
            # hardcoded fallback.
            gen_defaults = generation_config_sampling_defaults(tokenizer)
            cfg = current_app.config
            temperature = float(
                _get_non_none(
                    req,
                    "temperature",
                    resolve_sampling_default(
                        cfg, gen_defaults, "temperature", 'default_temperature', 1.0
                    ),
                )
            )
            top_p = float(
                _get_non_none(
                    req,
                    "top_p",
                    resolve_sampling_default(cfg, gen_defaults, "top_p", 'default_top_p', 1.0),
                )
            )
            top_k = int(
                _get_non_none(
                    req,
                    "top_k",
                    resolve_sampling_default(cfg, gen_defaults, "top_k", 'default_top_k', 0),
                )
            )
            log_sampling_defaults_once(
                tokenizer, {"temperature": temperature, "top_p": top_p, "top_k": top_k}
            )
            n = int(_get_non_none(req, "n", 1))  # Number of choices to generate

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0

            # Check for 'logprobs' (bool) and 'top_logprobs' (int)
            return_log_probs = bool(_get_non_none(req, "logprobs", False))
            top_n_logprobs = int(_get_non_none(req, "top_logprobs", 0)) if return_log_probs else 0
            skip_prompt_log_probs = bool(_get_non_none(req, "skip_prompt_log_probs", True))
            add_BOS = bool(_get_non_none(req, "add_BOS", False))

            # The engine only handles add_BOS for string prompts, not pre-tokenized
            # input. Since we pre-tokenize via apply_chat_template, we must handle
            # BOS ourselves, matching the logic in tokenize_prompt().
            if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
                start_idx = 0
                while start_idx < len(prompt_tokens) and prompt_tokens[start_idx] == tokenizer.bos:
                    start_idx += 1
                if start_idx > 0:
                    prompt_tokens = prompt_tokens[start_idx:]

                if add_BOS:
                    prompt_tokens = [tokenizer.bos] + prompt_tokens

            max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)
            ignore_eos = bool(req.get("ignore_eos", False))

            # Does the client want the prompt tokens echoed back? Only then does the
            # engine need to keep the prompt_tokens tensor on the response payload.
            # return_tokenized_data (implied by prevent_retokenization) needs the ids;
            # return_raw_text needs the ids to detokenize the prompt into raw_text.
            return_tokenized_data = (
                req.get("return_tokenized_data", False) or prevent_retokenization
            )
            return_raw_text = req.get("return_raw_text", False)
            return_prompt_tokens = return_tokenized_data or return_raw_text

            # OpenAI-style "stop" may be a string or list of strings; normalize.
            stop = req.get("stop", None)
            if isinstance(stop, str):
                stop = [stop]

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(int(max_tokens) if max_tokens is not None else None),
                stop_words=stop,
                skip_prompt_log_probs=skip_prompt_log_probs,
                add_BOS=add_BOS,
                termination_id=-1 if ignore_eos else None,
                return_prompt_tokens=return_prompt_tokens,
                streaming_interval=int(_get_non_none(req, "streaming_interval", 1)),
                # This frontend detokenizes its own output below. Keeping it off the
                # coordinator matters because that is one process shared by all DP ranks.
                detokenize_generations=False,
            )
        except ValueError as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        # Hash and serialize shared media once before fanning one prompt out to
        # multiple independently sampled choices. Each request still carries
        # its own media payload, while coordinator affinity keeps equivalent
        # requests on the engine that owns the cached vision embedding.
        frontend_request_phase(frontend_trace, "prepare_multimodal")
        media_prepare_started_at = time.perf_counter()
        prepared_multimodal_data = prepare_multimodal_data(multi_modal_data)
        media_prepared_at = time.perf_counter()
        stream_requested = bool(req.get("stream", False))
        if stream_requested:
            # Streaming currently supports only Hugging Face fast tokenizers.
            try:
                incremental_detokenizers = [
                    HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_tokens)
                    for _ in range(n)
                ]
            except ValueError as error:
                return Response(str(error), status=400)

            streams = []
            try:
                for _ in range(n):
                    streams.append(
                        client.add_request_streaming(
                            prompt_tokens,
                            sampling_params,
                            multi_modal_data=prepared_multimodal_data,
                        )
                    )
            except Exception as error:
                for stream in streams:
                    await stream.aclose()
                logger.exception("Error submitting streaming chat request")
                return Response(f"Error submitting request: {error}", status=500)
            chat_parsers = None
            if parsers:
                marker_prefixes = (
                    tuple(
                        marker
                        for parser_name in parsers
                        for marker in getattr(PARSER_MAPPING[parser_name], "streaming_markers", ())
                    )
                    if tools_requested
                    else ()
                )

                def parse_streaming_text(text):
                    parsed_text, metadata = apply_parsers(
                        text,
                        tools,
                        parsers,
                        tools_requested,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    metadata["tool_calls"] = _maybe_filter_parallel_tool_calls(
                        metadata.get("tool_calls", []), parallel_tool_calls
                    )
                    return parsed_text, metadata

                is_named_tool_choice = isinstance(tool_choice, dict) and "function" in tool_choice
                chat_parsers = [
                    StreamingChatParser(
                        parse_streaming_text,
                        marker_prefixes=marker_prefixes,
                        named_tool_choice=is_named_tool_choice,
                    )
                    for _ in range(n)
                ]
            include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
            chunks = openai_stream(
                streams,
                tokenizer,
                incremental_detokenizers,
                chat=True,
                return_log_probs=return_log_probs,
                include_usage=include_usage,
                chat_parsers=chat_parsers,
            )
            response = Response(
                stream_with_deadline(
                    chunks,
                    remaining_request_timeout(request.headers, http_started_at),
                ),
                content_type="text/event-stream",
            )
            response.timeout = None
            return response

        # add_request_with_id, not add_request: a non-streaming response writes
        # nothing to the socket while generating, so a disconnect is never
        # discovered as a broken pipe. Aborting needs the request ids.
        #
        # Submitted one at a time rather than in a comprehension so a failure on
        # admission k -- a zmq send error, or multimodal serialization on a
        # malformed payload -- can abort the k-1 already in flight. Left to
        # escape they would generate to their token limit holding batch slots,
        # the same leak the abort below the gather closes.
        request_ids = []
        tasks = []
        submission_started_at = time.perf_counter()
        frontend_request_phase(frontend_trace, "engine_submit", choices=n)
        try:
            for _ in range(n):
                request_id, future = client.add_request_with_id(
                    prompt_tokens, sampling_params, multi_modal_data=prepared_multimodal_data
                )
                request_ids.append(request_id)
                tasks.append(future)
        except Exception as e:
            abort_requests(client, request_ids, f"submission failed: {e}")
            logger.error(f"Error submitting request: {e}")
            return Response(f"Error submitting request: {e}", status=500)
        submission_finished_at = time.perf_counter()
        logger.warning(
            "chat frontend submitted: router_id=%s zmq_client=%s tag=%s ids=%s "
            "json_ms=%.1f media_extract_ms=%.1f "
            "tokenize_ms=%.1f media_prepare_ms=%.1f client_pack_send_ms=%.1f "
            "http_to_submit_ms=%.1f prompt_tokens=%d media_items=%d",
            request.headers.get("X-Nemo-Router-Request-Id", "direct"),
            getattr(client, "client_trace_id", "unknown"),
            _request_tag(request_ids[0]),
            request_ids,
            (json_parsed_at - http_started_at) * 1000,
            (media_extracted_at - json_parsed_at) * 1000,
            (media_prepare_started_at - media_extracted_at) * 1000,
            (media_prepared_at - media_prepare_started_at) * 1000,
            (submission_finished_at - submission_started_at) * 1000,
            (submission_finished_at - http_started_at) * 1000,
            len(prompt_tokens),
            len(image_bytes_list) + len(video_bytes_list),
        )
        frontend_request_phase(
            frontend_trace,
            "engine_await",
            request_ids=request_ids,
            prompt_tokens=len(prompt_tokens),
            media_items=len(image_bytes_list) + len(video_bytes_list),
        )

        if current_app.config['verbose']:
            start_time = time.perf_counter()

        pending_watchdog = asyncio.create_task(
            _log_pending_requests(
                client,
                request_ids,
                http_started_at=http_started_at,
                prompt_length=len(prompt_tokens),
                max_tokens=sampling_params.num_tokens_to_generate,
            )
        )
        try:
            remaining_timeout = remaining_request_timeout(request.headers, http_started_at)
            if remaining_timeout is None:
                batch_results = await asyncio.gather(*tasks)
            else:
                async with asyncio.timeout(remaining_timeout):
                    batch_results = await asyncio.gather(*tasks)
        except TimeoutError:
            snapshot = client.pending_request_diagnostics(request_ids)
            abort_requests(client, request_ids, "frontend request deadline")
            logger.error(
                "FRONTEND GENERATION TIMEOUT: endpoint=chat_completions "
                "router_id=%s zmq_client=%s tag=%s ids=%s http_age=%.1fs "
                "classification=local_deadline_exceeded "
                "engine_exception_observed=false pending=%s",
                request.headers.get("X-Nemo-Router-Request-Id", "direct"),
                getattr(client, "client_trace_id", "unknown"),
                _request_tag(request_ids[0]),
                request_ids,
                time.perf_counter() - http_started_at,
                snapshot,
            )
            response = jsonify(
                {
                    "error": {
                        "code": GENERATION_ABORTED_ERROR_CODE,
                        "message": "Inference request deadline expired and was aborted",
                        "retryable": False,
                    }
                }
            )
            response.status_code = 500
            response.headers.update(
                non_retryable_error_headers(GENERATION_ABORTED_ERROR_CODE)
            )
            return response
        except asyncio.CancelledError:
            # Quart cancels this handler when the peer goes away (its ASGI
            # connection races handle_messages against handle_request and
            # cancels the loser). Without this the engine would keep generating
            # for a client that is gone, holding a slot until it hit the token
            # limit -- orphans then accumulate faster than they retire and the
            # batch saturates.
            abort_requests(client, request_ids, "client disconnected")
            raise
        except Exception as e:
            snapshot = client.pending_request_diagnostics(request_ids)
            abort_requests(client, request_ids, "inference failed")
            logger.exception(
                "chat inference failed: tag=%s ids=%s http_age=%.1fs "
                "error=%s: %r pending=%s",
                _request_tag(request_ids[0]),
                request_ids,
                time.perf_counter() - http_started_at,
                type(e).__name__,
                e,
                snapshot,
            )
            return Response(
                f"Error during inference: {type(e).__name__}: {e!r}", status=500
            )
        finally:
            pending_watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_watchdog

        frontend_request_phase(frontend_trace, "format_response", request_ids=request_ids)
        formatting_started_at = time.perf_counter()
        http_elapsed = time.perf_counter() - http_started_at
        if http_elapsed >= 30.0:
            generation_config = getattr(tokenizer, "generation_config", None)
            configured_eos = (
                generation_config.get("eos_token_id")
                if isinstance(generation_config, dict)
                else None
            )
            model_eos_ids = {
                int(token_id)
                for token_id in (
                    configured_eos
                    if isinstance(configured_eos, (list, tuple))
                    else [configured_eos]
                )
                if isinstance(token_id, int) and not isinstance(token_id, bool)
            }
            finish_diagnostics = []
            for result in batch_results:
                generated_tokens = result.get("generated_tokens") or []
                result_sampling_params = result.get("sampling_params") or {}
                token_limit = result_sampling_params.get("num_tokens_to_generate")
                termination_id = result_sampling_params.get("termination_id")
                if token_limit is not None and len(generated_tokens) >= token_limit:
                    finish_cause = "length"
                elif (
                    termination_id is not None
                    and termination_id >= 0
                    and termination_id in generated_tokens
                ):
                    finish_cause = "termination_id"
                elif generated_tokens and generated_tokens[-1] in model_eos_ids:
                    finish_cause = "model_eos"
                else:
                    finish_cause = "stop_word_or_other"
                finish_diagnostics.append(
                    {
                        "cause": finish_cause,
                        "limit": token_limit,
                        "termination_id": termination_id,
                        "model_eos_ids": sorted(model_eos_ids),
                        "termination_seen": termination_id in generated_tokens
                        if termination_id is not None and termination_id >= 0
                        else False,
                        "last_token_ids": generated_tokens[-8:],
                    }
                )
            logger.warning(
                "chat request done: tag=%s ids=%s http_elapsed=%.1fs "
                "engine_latencies=%s output_tokens=%s eos_token_id=%s finishes=%s",
                _request_tag(request_ids[0]),
                request_ids,
                http_elapsed,
                [result.get("latency") for result in batch_results],
                [len(result.get("generated_tokens") or []) for result in batch_results],
                getattr(tokenizer, "eos_id", None),
                finish_diagnostics,
            )

        if current_app.config['verbose']:
            logging.info(
                f"Batch of {len(tasks)} requests (n={n}) processed in "
                f"{time.perf_counter() - start_time:.2f}s"
            )

        # --- 4. Check for failed requests ---
        failed_errors = []
        has_nontransient_error = False
        for i, record in enumerate(batch_results):
            if record.get("status") == "FAILED":
                events = record.get("events", [])
                error_events = [
                    e for e in events if e.get("type") in ("ERROR_NONTRANSIENT", "ERROR_TRANSIENT")
                ]
                if any(e.get("type") == "ERROR_NONTRANSIENT" for e in error_events):
                    has_nontransient_error = True
                error_msg = (
                    str(error_events[-1].get("payload", "Unknown error"))
                    if error_events
                    else "Unknown error"
                )
                failed_errors.append(f"Request {i}: {error_msg}")

        if failed_errors:
            error_detail = "; ".join(failed_errors)
            status = 400 if has_nontransient_error else 500
            logger.error(f"Inference request(s) failed: {error_detail}")

            # NOTE: This exact string is required for compatibility with Nemo-RL, DO NOT MODIFY.
            if "MaxSequenceLengthOverflowError" in error_detail:
                error_msg = (
                    f"This model's maximum context length was exceeded. "
                    f"Your messages resulted in {len(prompt_tokens)} tokens. "
                    f"Please reduce the length of the messages. {error_detail}"
                )
                return Response(error_msg, status=400)

            return Response(f"Inference request(s) failed: {error_detail}", status=status)

        # --- 5. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_tokens_counts = []
        cached_tokens_counts = []

        # return_tokenized_data / return_raw_text / return_prompt_tokens were computed
        # at submit time (above) and drive both the response shape here and whether the
        # engine kept the prompt_tokens tensor on the payload.
        request_idx = 0
        response_uid = None
        for result_item in batch_results:
            result = unwrap_serialized_tensors(result_item)
            if response_uid is None:
                response_uid = result["uid"]

            text_output = TextGenerationController.detokenize(
                tokenizer,
                result["generated_tokens"],
                remove_EOD=not sampling_params.detokenize_stop_sequence,
            )
            # The engine always reports prompt_length (for usage), but drops the
            # prompt_tokens tensor unless return_prompt_tokens was set.
            prompt_tokens_count = result.get("prompt_length")
            if prompt_tokens_count is None:
                prompt_tokens_out = result["prompt_tokens"]
                prompt_tokens_count = len(prompt_tokens_out) if prompt_tokens_out is not None else 0
            prompt_tokens_counts.append(prompt_tokens_count)
            cached_tokens_counts.append(result.get("num_cached_tokens", 0))

            logprobs_content = None
            if sampling_params.return_log_probs:
                token_logprobs = json_safe_logprobs(result.get('log_probs') or [])

                tokens_to_decode = [[tok] for tok in result["generated_tokens"]]
                tokens = list(map(tokenizer.detokenize, tokens_to_decode))

                # Get top_n_logprobs if available
                generated_top_n_logprobs = json_safe_top_n_logprobs(
                    result.get('generated_top_n_logprobs') or []
                )

                logprobs_content = []
                for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
                    # Build top_logprobs list for this token position
                    top_logprobs_list = []
                    if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                        top_n_dict = generated_top_n_logprobs[i]
                        for token_str, logprob in top_n_dict.items():
                            top_logprobs_list.append(
                                {
                                    "token": token_str,
                                    "logprob": logprob,
                                    "bytes": list(token_str.encode("utf-8")),
                                }
                            )

                    logprobs_content.append(
                        {
                            "token": tok,
                            "logprob": lp,
                            "bytes": list(tok.encode("utf-8")),
                            "top_logprobs": top_logprobs_list,
                        }
                    )

            metadata = {}
            message_text = text_output

            if parsers:
                message_text, metadata = apply_parsers(
                    message_text,
                    tools,
                    parsers,
                    tools_requested,
                    chat_template_kwargs=chat_template_kwargs,
                )

            normalized_tool_calls = metadata.get("tool_calls", [])

            # Apply parallel_tool_calls filtering (matches vLLM behavior)
            normalized_tool_calls = _maybe_filter_parallel_tool_calls(
                normalized_tool_calls, parallel_tool_calls
            )

            # Determine content based on tool_choice (matches vLLM behavior):
            # - Named tool choice or "required": content is empty string
            # - Otherwise: content is the parsed message text
            is_named_tool_choice = isinstance(tool_choice, dict) and "function" in tool_choice
            if normalized_tool_calls and (is_named_tool_choice or tool_choice == "required"):
                content = ""
            else:
                content = message_text if message_text is not None else ""

            message = {"role": "assistant", "content": content}
            if normalized_tool_calls:
                message["tool_calls"] = normalized_tool_calls
            if "reasoning" in metadata:
                message["reasoning_content"] = metadata["reasoning"]

            if return_tokenized_data:
                # Wire contract matches vLLM: prompt_token_ids are model-input tokens
                # (post vision/video expansion). Preserve the exact compact form
                # separately for lossless multi-turn prefix stitching.
                message["prompt_token_ids"] = result["prompt_tokens"]
                message["compact_prompt_token_ids"] = (
                    result.get("compact_prompt_tokens") or result["prompt_tokens"]
                )
                message["generation_token_ids"] = result["generated_tokens"]
            if return_raw_text:
                prompt_str = tokenizer.detokenize(result["prompt_tokens"])
                message["raw_text"] = prompt_str + text_output
            # Preserve small diagnostic scalars returned by the inference engine.
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            return_log_probs = sampling_params.return_log_probs

            # Determine finish_reason following vLLM conventions:
            # - "tool_calls" for auto or required tool choice when tools are called
            # - "stop" for named tool choice (even when tools are called)
            # - "length" when max tokens is reached
            if (
                len(result["generated_tokens"])
                >= result["sampling_params"]["num_tokens_to_generate"]
            ):
                finish_reason = "length"
            elif normalized_tool_calls and not is_named_tool_choice:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

            # Choice-level prompt/generation_token_ids, generation_log_probs and
            # raw_text were duplicates of message-level data (or reconstructable);
            # dropped to match vLLM's response shape and cut payload size.
            choice_data = {
                "index": request_idx,
                "message": message,
                # 'logprobs' in chat API is an object containing 'content'
                "logprobs": {"content": logprobs_content} if return_log_probs else None,
                "finish_reason": finish_reason,
            }
            if current_app.config['verbose']:
                logging.info(_redact_token_id_lists_for_logging(result))

            if result["routing_indices"] is not None:
                choice_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    choice_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]

            choices.append(choice_data)
            if result.get("generated_log_probs") is None:
                logger.warning(
                    "Generation log probs is None for request:\n%s",
                    json.dumps(_redact_token_id_lists_for_logging(result), indent=4),
                )
            total_completion_tokens += len(result["generated_tokens"])
            request_idx += 1

        prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
        cached_token_count = max(cached_tokens_counts) if cached_tokens_counts else 0
        response = {
            "id": response_uid,
            "created": int(time.time()),
            "model": "EMPTY",
            "object": "chat.completion",
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_token_count + total_completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_token_count},
            },
        }

        logger.warning(
            "chat frontend response ready: router_id=%s tag=%s ids=%s "
            "format_ms=%.1f total_ms=%.1f",
            request.headers.get("X-Nemo-Router-Request-Id", "direct"),
            _request_tag(request_ids[0]),
            request_ids,
            (time.perf_counter() - formatting_started_at) * 1000,
            (time.perf_counter() - http_started_at) * 1000,
        )
        frontend_request_phase(
            frontend_trace,
            "serialize_response",
            choices=len(choices),
            completion_tokens=total_completion_tokens,
        )
        if HAVE_ORJSON:
            # Use orjson for faster serialization
            return Response(orjson.dumps(response), mimetype="application/json")
        else:
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
