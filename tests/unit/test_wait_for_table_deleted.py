"""Unit tests for wait_for_table_deleted and the wait option of delete_table.

The Rust waiter talks to a small local HTTP server that fakes DynamoDB, so the
polling loop runs for real without Docker or AWS credentials.
"""

from __future__ import annotations

import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydynox import DynamoDBClient
from pydynox.exceptions import AccessDeniedException
from pydynox.testing import MemoryBackend

_PREFIX = "com.amazonaws.dynamodb.v20120810#"
_TABLE = {"TableName": "users", "TableStatus": "DELETING"}


class FakeDynamoDB:
    """Fake endpoint. Describe answers come from a script, then repeat the last one."""

    def __init__(self, describe_script: list[str]) -> None:
        self.describe_script = describe_script
        self.calls: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                operation = self.headers["X-Amz-Target"].split(".")[-1]
                fake.calls.append(operation)
                if operation == "DeleteTable":
                    self._reply(200, {"TableDescription": _TABLE})
                    return
                index = fake.calls.count("DescribeTable") - 1
                outcome = fake.describe_script[min(index, len(fake.describe_script) - 1)]
                if outcome == "exists":
                    self._reply(200, {"Table": _TABLE})
                else:
                    self._reply(400, {"__type": _PREFIX + outcome, "message": outcome})

            def _reply(self, status: int, body: dict) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/x-amz-json-1.0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def client(self) -> DynamoDBClient:
        return DynamoDBClient(
            region="us-east-1",
            access_key="test",
            secret_key="test",
            endpoint_url=self.url,
            max_retries=0,
        )

    def __enter__(self) -> FakeDynamoDB:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()


# ========== Waiter ==========


async def test_wait_returns_when_table_already_absent() -> None:
    with FakeDynamoDB(["ResourceNotFoundException"]) as server:
        await server.client().wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


def test_sync_wait_returns_when_table_already_absent() -> None:
    with FakeDynamoDB(["ResourceNotFoundException"]) as server:
        server.client().sync_wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


async def test_wait_polls_until_table_is_absent() -> None:
    script = ["exists", "exists", "ResourceNotFoundException"]
    with FakeDynamoDB(script) as server:
        await server.client().wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 3


def test_sync_wait_polls_until_table_is_absent() -> None:
    script = ["exists", "exists", "ResourceNotFoundException"]
    with FakeDynamoDB(script) as server:
        server.client().sync_wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 3


async def test_wait_times_out_while_table_exists() -> None:
    with FakeDynamoDB(["exists"]) as server:
        with pytest.raises(TimeoutError, match="users"):
            await server.client().wait_for_table_deleted("users", timeout_seconds=1)
    assert len(server.calls) >= 2


def test_sync_wait_times_out_while_table_exists() -> None:
    with FakeDynamoDB(["exists"]) as server:
        with pytest.raises(TimeoutError, match="users"):
            server.client().sync_wait_for_table_deleted("users", timeout_seconds=1)


async def test_wait_propagates_service_errors() -> None:
    with FakeDynamoDB(["AccessDeniedException"]) as server:
        with pytest.raises(AccessDeniedException):
            await server.client().wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


def test_sync_wait_propagates_service_errors() -> None:
    with FakeDynamoDB(["AccessDeniedException"]) as server:
        with pytest.raises(AccessDeniedException):
            server.client().sync_wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


async def test_wait_propagates_service_error_after_polling() -> None:
    with FakeDynamoDB(["exists", "AccessDeniedException"]) as server:
        with pytest.raises(AccessDeniedException):
            await server.client().wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 2


# ========== delete_table ==========


async def test_delete_table_does_not_wait_by_default() -> None:
    with FakeDynamoDB(["exists"]) as server:
        await server.client().delete_table("users")
    assert server.calls == ["DeleteTable"]


def test_sync_delete_table_does_not_wait_by_default() -> None:
    with FakeDynamoDB(["exists"]) as server:
        server.client().sync_delete_table("users")
    assert server.calls == ["DeleteTable"]


async def test_delete_table_wait_polls_until_absent() -> None:
    script = ["exists", "ResourceNotFoundException"]
    with FakeDynamoDB(script) as server:
        await server.client().delete_table("users", wait=True, timeout_seconds=10)
    assert server.calls == ["DeleteTable", "DescribeTable", "DescribeTable"]


def test_sync_delete_table_wait_polls_until_absent() -> None:
    script = ["exists", "ResourceNotFoundException"]
    with FakeDynamoDB(script) as server:
        server.client().sync_delete_table("users", wait=True, timeout_seconds=10)
    assert server.calls == ["DeleteTable", "DescribeTable", "DescribeTable"]


async def test_delete_table_wait_times_out() -> None:
    with FakeDynamoDB(["exists"]) as server:
        with pytest.raises(TimeoutError):
            await server.client().delete_table("users", wait=True, timeout_seconds=1)


def test_sync_delete_table_wait_times_out() -> None:
    with FakeDynamoDB(["exists"]) as server:
        with pytest.raises(TimeoutError):
            server.client().sync_delete_table("users", wait=True, timeout_seconds=1)


def test_delete_table_wait_options_are_keyword_only() -> None:
    with FakeDynamoDB(["exists"]) as server:
        with pytest.raises(TypeError):
            server.client().sync_delete_table("users", True)  # type: ignore[misc]


# ========== Python wrapper forwarding ==========


def _client_with_mock(inner: MagicMock) -> DynamoDBClient:
    client = DynamoDBClient.__new__(DynamoDBClient)
    client._client = inner
    return client


async def test_wrapper_forwards_async_wait_for_table_deleted() -> None:
    inner = MagicMock()
    inner.wait_for_table_deleted = AsyncMock()
    await _client_with_mock(inner).wait_for_table_deleted("users", timeout_seconds=5)
    inner.wait_for_table_deleted.assert_called_once_with("users", timeout_seconds=5)


def test_wrapper_forwards_sync_wait_for_table_deleted() -> None:
    inner = MagicMock()
    _client_with_mock(inner).sync_wait_for_table_deleted("users", timeout_seconds=5)
    inner.sync_wait_for_table_deleted.assert_called_once_with("users", timeout_seconds=5)


def test_wrapper_wait_for_table_deleted_defaults_timeout_to_none() -> None:
    inner = MagicMock()
    _client_with_mock(inner).sync_wait_for_table_deleted("users")
    inner.sync_wait_for_table_deleted.assert_called_once_with("users", timeout_seconds=None)


async def test_wrapper_forwards_async_delete_options() -> None:
    inner = MagicMock()
    inner.delete_table = AsyncMock()
    await _client_with_mock(inner).delete_table("users", wait=True, timeout_seconds=5)
    inner.delete_table.assert_called_once_with("users", wait=True, timeout_seconds=5)


def test_wrapper_forwards_sync_delete_options() -> None:
    inner = MagicMock()
    _client_with_mock(inner).sync_delete_table("users", wait=True, timeout_seconds=5)
    inner.sync_delete_table.assert_called_once_with("users", wait=True, timeout_seconds=5)


def test_wrapper_delete_defaults_to_no_wait() -> None:
    inner = MagicMock()
    _client_with_mock(inner).sync_delete_table("users")
    inner.sync_delete_table.assert_called_once_with("users", wait=False, timeout_seconds=None)


# ========== Memory backend ==========


async def test_memory_backend_async_delete_and_wait() -> None:
    with MemoryBackend() as backend:
        backend.client.sync_create_table("users", partition_key=("pk", "S"))
        await backend.client.delete_table("users", wait=True, timeout_seconds=1)
        assert not backend.client.sync_table_exists("users")
        await backend.client.wait_for_table_deleted("users")


def test_memory_backend_sync_delete_and_wait() -> None:
    with MemoryBackend() as backend:
        backend.client.sync_create_table("users", partition_key=("pk", "S"))
        backend.client.sync_delete_table("users", wait=True, timeout_seconds=1)
        assert not backend.client.sync_table_exists("users")
        backend.client.sync_wait_for_table_deleted("users")


def test_memory_backend_wait_for_absent_table_succeeds() -> None:
    with MemoryBackend() as backend:
        backend.client.sync_wait_for_table_deleted("never_created")


# ========== Type stubs ==========


def _stub_methods() -> dict[str, ast.FunctionDef]:
    stub = Path(__file__).parents[2] / "python" / "pydynox" / "pydynox_core.pyi"
    tree = ast.parse(stub.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DynamoDBClient":
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError("DynamoDBClient not found in stub")


@pytest.mark.parametrize("name", ["wait_for_table_deleted", "sync_wait_for_table_deleted"])
def test_stub_declares_wait_for_table_deleted(name: str) -> None:
    args = _stub_methods()[name].args
    assert [a.arg for a in args.args] == ["self", "table_name", "timeout_seconds"]


@pytest.mark.parametrize("name", ["delete_table", "sync_delete_table"])
def test_stub_declares_delete_table_wait_options(name: str) -> None:
    args = _stub_methods()[name].args
    assert [a.arg for a in args.kwonlyargs] == ["wait", "timeout_seconds"]
