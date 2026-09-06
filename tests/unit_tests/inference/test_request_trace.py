# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging

from megatron.core.inference.request_trace import trace_request


def test_request_trace_is_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("MCORE_INFERENCE_REQUEST_TRACE", raising=False)

    with caplog.at_level(logging.INFO):
        trace_request("frontend_submitted", client_request_id=3)

    assert not caplog.messages


def test_request_trace_logs_identifiers_but_not_payload(monkeypatch, caplog):
    monkeypatch.setenv("MCORE_INFERENCE_REQUEST_TRACE", "1")

    with caplog.at_level(logging.INFO):
        trace_request("coordinator_admitted", client_identity=b"client", request_id=7)

    assert len(caplog.messages) == 1
    assert "MCORE_REQUEST_TRACE stage=coordinator_admitted" in caplog.messages[0]
    assert "client_identity=636c69656e74" in caplog.messages[0]
    assert "request_id=7" in caplog.messages[0]
