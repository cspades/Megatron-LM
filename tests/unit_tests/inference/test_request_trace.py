# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

from megatron.core.inference.request_trace import trace_request


def test_request_trace_is_disabled_by_default(monkeypatch, capsys):
    monkeypatch.delenv("MCORE_INFERENCE_REQUEST_TRACE", raising=False)

    trace_request("frontend_submitted", client_request_id=3)

    assert not capsys.readouterr().out


def test_request_trace_logs_identifiers_but_not_payload(monkeypatch, capsys):
    monkeypatch.setenv("MCORE_INFERENCE_REQUEST_TRACE", "1")

    trace_request("coordinator_admitted", client_identity=b"client", request_id=7)

    output = capsys.readouterr().out
    assert "MCORE_REQUEST_TRACE stage=coordinator_admitted" in output
    assert "client_identity=636c69656e74" in output
    assert "request_id=7" in output


def test_request_trace_writes_to_pid_scoped_file(monkeypatch, tmp_path, capsys):
    trace_file = tmp_path / "mcore-request-trace"
    monkeypatch.setenv("MCORE_INFERENCE_REQUEST_TRACE", "1")
    monkeypatch.setenv("MCORE_INFERENCE_REQUEST_TRACE_FILE", str(trace_file))

    trace_request("engine_received", request_id=7)

    output = capsys.readouterr().out
    assert not output
    trace_output = trace_file.with_name(f"{trace_file.name}.{os.getpid()}").read_text()
    assert "MCORE_REQUEST_TRACE stage=engine_received" in trace_output
    assert "request_id=7" in trace_output
