# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import copy
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

try:
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from quart import Quart

    HAS_BACKEND = True
except ImportError as e:
    HAS_BACKEND = False

import megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints as endpoints
from megatron.core.inference.config import MultimodalPromptConfig, PrefixCachingCoordinatorPolicy
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    apply_optional_sampling_default,
)
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.utils import trace_async_exceptions

logger = logging.getLogger(__name__)

# Global reference to manage the background server processes
_SERVER_PROCESSES: List[mp.Process] = []
_SERVER_PROCESS_LOCK = threading.Lock()
_SERVER_SUPERVISOR_STOP = threading.Event()
_SERVER_SUPERVISOR_THREAD: Optional[threading.Thread] = None
_SERVER_SUPERVISOR_INTERVAL_SECONDS = 1.0
_SERVER_SUPERVISOR_HEARTBEAT_SECONDS = 30.0
_SERVER_STARTUP_TIMEOUT_SECONDS = 120.0
# The policy worker is a live Ray/CUDA process with background threads by the
# time it starts HTTP replicas. Forking it copies locks and runtime state
# without the threads that own them, which can leave a child alive but unable
# to make progress. Every frontend must therefore start from a clean interpreter.
_SERVER_PROCESS_CONTEXT = mp.get_context("spawn")


class _ResponseDeliveryTelemetry:
    """Log the terminal ASGI body send after an endpoint handler returns."""

    _MONITORED_PATHS = frozenset(
        {
            "/chat/completions",
            "/v1/chat/completions",
            "/completions",
            "/v1/completions",
        }
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in self._MONITORED_PATHS:
            await self.app(scope, receive, send)
            return

        started_at = asyncio.get_running_loop().time()
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        router_id = headers.get("x-nemo-router-request-id", "direct")
        status = None
        body_bytes = 0
        chunks = 0
        complete = False
        failed = False

        async def telemetry_send(message):
            nonlocal status, body_bytes, chunks, complete
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                body_bytes += len(body)
                chunks += 1
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                complete = True
                logger.warning(
                    "frontend ASGI response complete: router_id=%s path=%s "
                    "status=%s chunks=%d body_bytes=%d total_ms=%.1f",
                    router_id,
                    scope.get("path"),
                    status,
                    chunks,
                    body_bytes,
                    (asyncio.get_running_loop().time() - started_at) * 1000,
                )

        try:
            await self.app(scope, receive, telemetry_send)
        except BaseException as error:
            failed = True
            logger.warning(
                "frontend ASGI response failed: router_id=%s path=%s status=%s "
                "chunks=%d body_bytes=%d complete=%s total_ms=%.1f error=%s: %r",
                router_id,
                scope.get("path"),
                status,
                chunks,
                body_bytes,
                complete,
                (asyncio.get_running_loop().time() - started_at) * 1000,
                type(error).__name__,
                error,
            )
            raise
        finally:
            if status is not None and not complete and not failed:
                logger.warning(
                    "frontend ASGI response incomplete: router_id=%s path=%s "
                    "status=%s chunks=%d body_bytes=%d total_ms=%.1f",
                    router_id,
                    scope.get("path"),
                    status,
                    chunks,
                    body_bytes,
                    (asyncio.get_running_loop().time() - started_at) * 1000,
                )


@contextmanager
def temp_log_level(level, logger=None):
    """Enables temporarily overriding the logging level."""
    logger = logger or logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


@trace_async_exceptions
async def _run_text_gen_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    hostname: Optional[str] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
    default_temperature: Optional[float] = None,
    default_top_p: Optional[float] = None,
    default_top_k: Optional[int] = None,
    eval_mode: bool = False,
    block_size_tokens: Optional[int] = None,
    prefix_caching_coordinator_policy: Optional[PrefixCachingCoordinatorPolicy] = None,
    replica_index: Optional[int] = None,
    ready_event=None,
):
    """
    Initializes and runs the async web server. Automatically starts and
    manages its own InferenceClient connected to the provided coordinator address.
    """
    if not HAS_BACKEND:
        raise RuntimeError(f"Web backend framework (Quart) not available")

    # Create and start the client locally inside this process
    # The client hashes prompts for prefix-affinity routing so the coordinator
    # does not have to on its single serial loop. It is the only place holding
    # both the tokens and, for multimodal, the media key that salts them.
    inference_client = InferenceClient(
        coordinator_addr,
        deserialize=False,
        block_size_tokens=block_size_tokens,
        prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
    )
    inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")

    try:
        # Bind what the caller asked for -- None means every interface, which is
        # not the single address gethostname() resolves to. The resolved name is
        # for the log line only.
        bind_host = hostname
        if hostname is None:
            try:
                hostname = socket.gethostname()
            except Exception as e:
                logger.warning(f"Could not get hostname: {e}")
                hostname = "0.0.0.0"

        app = Quart(__name__)

        # Quart native way to handle max body size (1 GB; needed for large prompts)
        app.config['MAX_CONTENT_LENGTH'] = 2**30

        # Store client and tokenizer in app config for Blueprints to use
        app.config['client'] = inference_client
        app.config['tokenizer'] = tokenizer
        app.config['parsers'] = parsers
        app.config['verbose'] = verbose
        app.config['chat_template'] = chat_template
        app.config['multimodal_prompt_config'] = (
            multimodal_prompt_config or MultimodalPromptConfig()
        )
        # Only set when the operator actually configured a value -- see
        # apply_optional_sampling_default's docstring for why unconditional
        # assignment here would break resolve_sampling_default's precedence.
        apply_optional_sampling_default(app.config, 'default_temperature', default_temperature)
        apply_optional_sampling_default(app.config, 'default_top_p', default_top_p)
        apply_optional_sampling_default(app.config, 'default_top_k', default_top_k)
        app.config['eval_mode'] = eval_mode

        # Applying the chat template is synchronous and O(prompt); on the event loop it
        # stalls every other request this replica owns, including delivery of responses
        # that already finished. One worker is enough - the point is the yield, not
        # throughput. The copy is required: HF tokenizers are not thread-safe.
        app.config['tokenize_executor'] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenize"
        )
        app.config['tokenizer_copy'] = copy.deepcopy(tokenizer)

        # Register all blueprints from the 'endpoints' package
        for endpoint in endpoints.__all__:
            app.register_blueprint(endpoint)

        if ready_event is not None:

            @app.before_serving
            async def _signal_replica_ready():
                # Quart invokes this from the serving event loop after ASGI
                # startup. The parent waits for every replica's own event; a
                # shared-port probe cannot prove that every SO_REUSEPORT member
                # is able to serve.
                ready_event.set()
                logger.warning(
                    "FRONTEND REPLICA READY: pid=%d rank=%d replica=%s port=%d "
                    "zmq_client=%s",
                    os.getpid(),
                    rank,
                    replica_index,
                    server_port,
                    inference_client.client_trace_id,
                )

        config = Config()
        config.keep_alive_timeout = 30.0  # Keep connection alive between long-running requests.
        config.backlog = 2**14  # Expect high load; ensure we do not drop connections.
        config.h2_max_concurrent_streams = (
            2**14
        )  # Allow many concurrent streams for HTTP/2 clients.

        # Held for this worker's lifetime; closing it would drop the listener.
        own_socket = _bind_reuseport_socket(server_port, bind_host)
        config.bind = [f"fd://{own_socket.fileno()}"]

        with temp_log_level(logging.INFO, logger):
            logger.info(f"Starting text generation server on http://{hostname}:{server_port}")
            logger.info(f"Using tokenizer: {type(tokenizer)}")
            logger.info(f"Using parsers: {parsers}")
            logger.info(
                f"Default sampling: temperature={default_temperature}, "
                f"top_p={default_top_p}, top_k={default_top_k}"
            )
            logger.info(f"Evaluation mode: {eval_mode}")

        try:
            # Quart is natively ASGI, so we can serve the app directly
            await serve(_ResponseDeliveryTelemetry(app), config)
        finally:
            own_socket.close()

    finally:
        # Gracefully shut down the client when the server stops
        inference_client.stop()
        logger.info(f"Rank {rank}: Web server and client shut down.")


def _server_process_worker(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    hostname: Optional[str] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
    default_temperature: Optional[float] = None,
    default_top_p: Optional[float] = None,
    default_top_k: Optional[int] = None,
    eval_mode: bool = False,
    block_size_tokens: Optional[int] = None,
    prefix_caching_coordinator_policy: Optional[PrefixCachingCoordinatorPolicy] = None,
    replica_index: Optional[int] = None,
    ready_event=None,
):
    """Synchronous worker function that sets up a new event loop for the separate process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _log_unhandled_loop_exception(_loop, context):
        error = context.get("exception")
        logger.error(
            "FRONTEND ASYNC TASK FAILED: pid=%d rank=%d port=%d message=%s "
            "error_type=%s error=%r task=%r",
            os.getpid(),
            rank,
            server_port,
            context.get("message", "no message"),
            type(error).__name__ if error is not None else "None",
            error,
            context.get("task") or context.get("future"),
            exc_info=(
                (type(error), error, error.__traceback__)
                if error is not None
                else None
            ),
        )

    loop.set_exception_handler(_log_unhandled_loop_exception)
    logger.info(
        "FRONTEND REPLICA STARTING: pid=%d parent_pid=%d rank=%d port=%d",
        os.getpid(),
        os.getppid(),
        rank,
        server_port,
    )
    try:
        loop.run_until_complete(
            _run_text_gen_server(
                coordinator_addr,
                tokenizer,
                rank,
                server_port,
                parsers,
                verbose,
                hostname,
                chat_template,
                multimodal_prompt_config,
                default_temperature,
                default_top_p,
                default_top_k,
                eval_mode,
                block_size_tokens,
                prefix_caching_coordinator_policy,
                replica_index,
                ready_event,
            )
        )
    except KeyboardInterrupt:
        logger.info(
            "FRONTEND REPLICA INTERRUPTED: pid=%d rank=%d port=%d",
            os.getpid(),
            rank,
            server_port,
        )
    except BaseException as error:
        # This is the process boundary. Exceptions hidden by an ASGI task or by
        # trace_async_exceptions (which raises SystemExit) must be visible before
        # multiprocessing reduces them to an integer exit status.
        logger.critical(
            "FRONTEND REPLICA FATAL: pid=%d parent_pid=%d rank=%d port=%d "
            "error_type=%s error=%r",
            os.getpid(),
            os.getppid(),
            rank,
            server_port,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        raise
    finally:
        pending = asyncio.all_tasks(loop)
        if pending:
            logger.warning(
                "FRONTEND REPLICA CLEANUP: pid=%d rank=%d port=%d pending_tasks=%d",
                os.getpid(),
                rank,
                server_port,
                len(pending),
            )
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        logger.info(
            "FRONTEND REPLICA STOPPED: pid=%d rank=%d port=%d",
            os.getpid(),
            rank,
            server_port,
        )


def _bind_reuseport_socket(server_port: int, hostname: Optional[str]) -> socket.socket:
    """Bind this worker's own socket on the shared port, with SO_REUSEPORT.

    Unlike inheriting one fd, this gives every worker its own accept queue and
    lets the kernel hash each connection's 4-tuple across them. It is a large
    improvement on sharing (measured 604x -> 2x spread at 32 replicas) but is
    still hashing, not balancing: it cannot see that a replica is already busy,
    so the spread is statistical rather than exact.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Required on every socket sharing the port; without it the second bind fails.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((hostname if hostname is not None else "0.0.0.0", server_port))
    sock.setblocking(False)
    return sock


def _reserve_port(hostname: Optional[str]) -> int:
    """Pick a free port for the replicas to bind individually.

    Replicas each bind the port themselves, so the parent cannot hold the socket
    and hand out its fd; it binds only long enough to learn a free port. The gap
    before the replicas bind is a small race with unrelated processes, which is
    why an explicit port is preferred when one is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((hostname if hostname is not None else "0.0.0.0", 0))
        return probe.getsockname()[1]


def _spawn_server_process(
    args: Tuple[Any, ...], replica_index: int, ready_event
) -> mp.Process:
    """Start one named frontend replica in a clean interpreter."""
    process = _SERVER_PROCESS_CONTEXT.Process(
        target=_server_process_worker,
        args=(*args, replica_index, ready_event),
        daemon=True,
    )
    process.name = f"megatron-http-frontend-{replica_index}"
    process.start()
    return process


def _format_exit_status(exitcode: Optional[int]) -> str:
    """Render a multiprocessing exit code without losing terminating signals."""
    if exitcode is None:
        return "running"
    if exitcode < 0:
        try:
            signal_name = signal.Signals(-exitcode).name
        except (ValueError, AttributeError):
            signal_name = "UNKNOWN"
        return f"signal={signal_name}({-exitcode})"
    return f"exitcode={exitcode}"


def _process_resource_snapshot(process: mp.Process) -> str:
    """Read cheap Linux process counters used to diagnose frontend exhaustion."""
    pid = process.pid
    if pid is None:
        return "pid=unknown"
    values = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
            for line in status_file:
                key, _, value = line.partition(":")
                if key in {"State", "VmRSS", "VmPeak", "Threads", "FDSize"}:
                    values[key] = value.strip()
    except OSError as error:
        return f"pid={pid} status_unavailable={error!r}"
    try:
        fd_count = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        fd_count = "unknown"
    return (
        f"pid={pid} state={values.get('State', 'unknown')} "
        f"rss={values.get('VmRSS', 'unknown')} "
        f"peak={values.get('VmPeak', 'unknown')} "
        f"threads={values.get('Threads', 'unknown')} "
        f"fds={fd_count} fd_capacity={values.get('FDSize', 'unknown')}"
    )


def _supervise_server_processes(rank: int, server_port: int) -> None:
    """Fail the owning rank immediately if any frontend child exits.

    A dead HTTP child must never remain silently advertised as a healthy backend.
    Replicas use the explicit ``spawn`` context because the owner is a live,
    multithreaded CUDA/Ray process. We still fail the backend instead of
    restarting an individual replica: dropping one member of a SO_REUSEPORT
    group can strand accepted requests, and backend-level recovery is the only
    operation that can atomically remove the advertised endpoint.
    """
    next_heartbeat_at = time.monotonic()
    while not _SERVER_SUPERVISOR_STOP.wait(_SERVER_SUPERVISOR_INTERVAL_SECONDS):
        now = time.monotonic()
        child_exited = False
        snapshots = []
        with _SERVER_PROCESS_LOCK:
            for replica_index, process in enumerate(_SERVER_PROCESSES):
                exitcode = process.exitcode
                if exitcode is None:
                    if now >= next_heartbeat_at:
                        snapshots.append(_process_resource_snapshot(process))
                    continue

                process.join()
                logger.critical(
                    "FRONTEND REPLICA EXITED: rank=%d replica=%d/%d pid=%s port=%d "
                    "status=%s action=terminate_owner",
                    rank,
                    replica_index + 1,
                    len(_SERVER_PROCESSES),
                    process.pid,
                    server_port,
                    _format_exit_status(exitcode),
                )
                child_exited = True

        if snapshots:
            logger.warning(
                "FRONTEND SUPERVISOR HEARTBEAT: rank=%d port=%d alive=%d/%d "
                "processes=%s",
                rank,
                server_port,
                len(snapshots),
                len(_SERVER_PROCESSES),
                snapshots,
            )
            next_heartbeat_at = now + _SERVER_SUPERVISOR_HEARTBEAT_SECONDS

        if child_exited:
            # Raising in this daemon thread would leave the Ray actor and engine
            # alive. Terminating the owner makes the backend failure immediate
            # and observable to the distributed launcher.
            logger.critical(
                "FRONTEND SUPERVISOR TERMINATING OWNER: rank=%d pid=%d port=%d",
                rank,
                os.getpid(),
                server_port,
            )
            os.kill(os.getpid(), signal.SIGTERM)
            return


def start_text_gen_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    num_replicas: int = 4,
    hostname: Optional[str] = None,
    sock: Optional[socket.socket] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
    default_temperature: Optional[float] = None,
    default_top_p: Optional[float] = None,
    default_top_k: Optional[int] = None,
    eval_mode: bool = False,
    block_size_tokens: Optional[int] = None,
    prefix_caching_coordinator_policy: Optional[PrefixCachingCoordinatorPolicy] = None,
) -> Optional[str]:
    """Start the text generation server.

    Every replica binds its own socket on ``server_port`` with SO_REUSEPORT, so
    each gets its own accept queue and the kernel spreads connections across
    them. Sharing one inherited socket does not balance -- replicas race to
    accept from a single queue and whichever is already running keeps winning,
    which concentrates most traffic on a handful of them as replica count grows.

    Call this on every rank that should host a frontend. Frontend work (chat
    template, detokenize, parsers, JSON) is CPU-bound, so hosting on a single
    rank confines it to that rank's CPU allocation and leaves the rest of the
    job's cores unused. Each caller gets its own URL back; collecting them and
    spreading requests over the result is the caller's business.

    Args:
        server_port: Port to listen on. Overridden by ``sock`` when given; 0
            asks the OS to choose a free one.
        sock: A socket the caller already bound, used only to fix the port.
            Replicas bind that port themselves, so it is closed here rather than
            shared with them.
        chat_template: Chat template to apply, as a file path or an inline
            template string. None falls back to the tokenizer's own template.

    Returns:
        The base URL this rank serves on, or None if the server was already
        running.
    """
    global _SERVER_PROCESSES
    global _SERVER_SUPERVISOR_THREAD

    if _SERVER_PROCESSES:
        logger.warning("Text gen server processes are already running.")
        return None
    if num_replicas < 1:
        raise ValueError(f"num_replicas must be at least 1, got {num_replicas}")

    if sock is not None:
        # Take the port and release the socket: replicas each bind their own with
        # SO_REUSEPORT, which one shared socket cannot provide.
        server_port = sock.getsockname()[1]
        if server_port == 0:
            raise ValueError(
                "socket must be bound to a real port before being passed to start_text_gen_server"
            )
        sock.close()
    elif server_port == 0:
        server_port = _reserve_port(hostname)

    worker_args = (
        coordinator_addr,
        tokenizer,
        rank,
        server_port,
        parsers,
        verbose,
        hostname,
        chat_template,
        multimodal_prompt_config,
        default_temperature,
        default_top_p,
        default_top_k,
        eval_mode,
        block_size_tokens,
        prefix_caching_coordinator_policy,
    )
    _SERVER_SUPERVISOR_STOP.clear()
    ready_events = []

    try:
        for i in range(num_replicas):
            ready_event = _SERVER_PROCESS_CONTEXT.Event()
            p = _spawn_server_process(worker_args, i, ready_event)
            _SERVER_PROCESSES.append(p)
            ready_events.append(ready_event)
            logger.info(
                "Started text gen frontend replica %d/%d on port %d (PID: %s)",
                i + 1,
                num_replicas,
                server_port,
                p.pid,
            )
    except BaseException:
        logger.exception(
            "Failed to start text gen frontend replicas: rank=%d port=%d "
            "started=%d requested=%d",
            rank,
            server_port,
            len(_SERVER_PROCESSES),
            num_replicas,
        )
        _terminate(_SERVER_PROCESSES, "partially started Text Gen frontend")
        _SERVER_PROCESSES = []
        raise

    # Unit-test process doubles do not expose a multiprocessing sentinel. Real
    # processes always do; avoid starting a background thread around test doubles.
    if _SERVER_PROCESSES and hasattr(_SERVER_PROCESSES[0], "sentinel"):
        try:
            startup_deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_SECONDS
            pending_replicas = set(range(len(_SERVER_PROCESSES)))
            while pending_replicas:
                for replica_index in tuple(pending_replicas):
                    process = _SERVER_PROCESSES[replica_index]
                    if process.exitcode is not None:
                        raise RuntimeError(
                            "Text gen frontend replica "
                            f"{replica_index + 1}/{len(_SERVER_PROCESSES)} exited during "
                            f"startup ({_format_exit_status(process.exitcode)})"
                        )
                    if ready_events[replica_index].is_set():
                        pending_replicas.remove(replica_index)
                if not pending_replicas:
                    break
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError(
                        "Timed out waiting for text gen frontend replicas to become ready: "
                        f"rank={rank} port={server_port} "
                        f"pending={[index + 1 for index in sorted(pending_replicas)]} "
                        f"timeout_s={_SERVER_STARTUP_TIMEOUT_SECONDS}"
                    )
                time.sleep(0.05)

            logger.warning(
                "TEXT GEN FRONTEND READY: rank=%d port=%d replicas=%d/%d start_method=%s",
                rank,
                server_port,
                len(_SERVER_PROCESSES),
                num_replicas,
                _SERVER_PROCESS_CONTEXT.get_start_method(),
            )
            _SERVER_SUPERVISOR_THREAD = threading.Thread(
                target=_supervise_server_processes,
                args=(rank, server_port),
                name=f"megatron-http-supervisor-rank-{rank}",
                daemon=True,
            )
            _SERVER_SUPERVISOR_THREAD.start()
        except BaseException:
            logger.exception(
                "Text gen frontend readiness failed: rank=%d port=%d ready=%d/%d",
                rank,
                server_port,
                len(_SERVER_PROCESSES) - len(pending_replicas),
                len(_SERVER_PROCESSES),
            )
            _terminate(_SERVER_PROCESSES, "unready Text Gen frontend")
            _SERVER_PROCESSES = []
            raise

    return f"http://{hostname or socket.gethostname()}:{server_port}"


def _terminate(processes: List[mp.Process], what: str):
    """Terminate a group of worker processes, escalating to kill if needed."""
    if not processes:
        return
    logger.info(f"Terminating {len(processes)} {what} processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join()


def stop_text_gen_server():
    """Stop this rank's frontend replica processes."""
    global _SERVER_PROCESSES
    global _SERVER_SUPERVISOR_THREAD

    if not _SERVER_PROCESSES:
        return

    _SERVER_SUPERVISOR_STOP.set()
    if _SERVER_SUPERVISOR_THREAD is not None:
        _SERVER_SUPERVISOR_THREAD.join(timeout=3)
        if _SERVER_SUPERVISOR_THREAD.is_alive():
            logger.error("Text Gen frontend supervisor did not stop within 3 seconds.")
        _SERVER_SUPERVISOR_THREAD = None

    with _SERVER_PROCESS_LOCK:
        _terminate(_SERVER_PROCESSES, "Text Gen frontend")
        _SERVER_PROCESSES = []
    logger.info("All text gen frontend processes terminated.")
