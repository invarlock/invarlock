"""Outbound controls for trusted pinned SDKs running local model callbacks.

These Python audit checks are not operating-system containment against malicious
SDK code. The model transport uses AF_UNIX. Only Promptfoo may use loopback TCP
for its local Node callback bridge; its Node process has a narrower endpoint rule.
"""

from __future__ import annotations

import os
import socket
import sys

LOOPBACK = {"127.0.0.1", "::1"}
ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "RAGAS_DO_NOT_TRACK": "true",
    "OPIK_TRACK_DISABLE": "true",
    "OTEL_SDK_DISABLED": "true",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "AZURE_TELEMETRY_DISABLED": "1",
    "DO_NOT_TRACK": "1",
    "PROMPTFOO_DISABLE_TELEMETRY": "1",
}


def audit(evaluator, event, args):
    """Reject outbound internet sockets before their operation executes."""
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        if evaluator != "promptfoo" or args[0] not in LOOPBACK:
            raise RuntimeError("live SDK capture forbids external name resolution")
        return
    if event not in {
        "socket.connect",
        "socket.sendto",
        "socket.sendmsg",
        "socket.bind",
    }:
        return
    channel, address = args[:2]
    if channel.family == socket.AF_UNIX:
        return
    if address is None and event == "socket.sendmsg":
        address = channel.getpeername()
    if (
        evaluator == "promptfoo"
        and channel.family in {socket.AF_INET, socket.AF_INET6}
        and isinstance(address, tuple)
        and address[0] in LOOPBACK
    ):
        return
    raise RuntimeError("live SDK capture permits only its local callback transport")


def configure(evaluator):
    """Install permanent process-local checks before importing optional SDKs."""
    os.environ.update(ENVIRONMENT)
    sys.addaudithook(lambda event, args: audit(evaluator, event, args))
