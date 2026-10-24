"""Unit tests for wait_for_table_deleted and the wait option of delete_table.

The Rust waiter talks to a small local HTTP server that fakes DynamoDB, so the
polling loop runs for real without Docker or AWS credentials.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydynox import DynamoDBClient
from pydynox.exceptions import AccessDeniedException
from pydynox.testing import MemoryBackend

_EXISTS = {"Table": {"TableName": "users", "TableStatus": "DELETING"}}
_DELETED = {"TableDescription": {"TableName": "users", "TableStatus": "DELETING"}}
_ABSENT = "ResourceNotFoundException"
_DENIED = "AccessDeniedException"


def _client(fake) -> DynamoDBClient:
    return DynamoDBClient(
        region="us-east-1",
        access_key="test",
        secret_key="test",
        endpoint_url=fake.endpoint_url,
        max_retries=0,
    )


# ========== Waiter ==========


async def test_wait_returns_when_table_already_absent(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_ABSENT]})
    await _client(server).wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


def test_sync_wait_returns_when_table_already_absent(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_ABSENT]})
    _client(server).sync_wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


async def test_wait_polls_until_table_is_absent(fake_dynamodb) -> None:
    script = [_EXISTS, _EXISTS, _ABSENT]
    server = fake_dynamodb({"DescribeTable": script})
    await _client(server).wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 3


def test_sync_wait_polls_until_table_is_absent(fake_dynamodb) -> None:
    script = [_EXISTS, _EXISTS, _ABSENT]
    server = fake_dynamodb({"DescribeTable": script})
    _client(server).sync_wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 3


async def test_wait_times_out_while_table_exists(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_EXISTS]})
    with pytest.raises(TimeoutError, match="users"):
        await _client(server).wait_for_table_deleted("users", timeout_seconds=1)
    assert len(server.calls) >= 2


def test_sync_wait_times_out_while_table_exists(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_EXISTS]})
    with pytest.raises(TimeoutError, match="users"):
        _client(server).sync_wait_for_table_deleted("users", timeout_seconds=1)


async def test_wait_propagates_service_errors(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_DENIED]})
    with pytest.raises(AccessDeniedException):
        await _client(server).wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


def test_sync_wait_propagates_service_errors(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_DENIED]})
    with pytest.raises(AccessDeniedException):
        _client(server).sync_wait_for_table_deleted("users")
    assert server.calls == ["DescribeTable"]


async def test_wait_propagates_service_error_after_polling(fake_dynamodb) -> None:
    server = fake_dynamodb({"DescribeTable": [_EXISTS, _DENIED]})
    with pytest.raises(AccessDeniedException):
        await _client(server).wait_for_table_deleted("users", timeout_seconds=10)
    assert server.calls == ["DescribeTable"] * 2


# ========== delete_table ==========


async def test_delete_table_does_not_wait_by_default(fake_dynamodb) -> None:
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": [_EXISTS]})
    await _client(server).delete_table("users")
    assert server.calls == ["DeleteTable"]


def test_sync_delete_table_does_not_wait_by_default(fake_dynamodb) -> None:
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": [_EXISTS]})
    _client(server).sync_delete_table("users")
    assert server.calls == ["DeleteTable"]


async def test_delete_table_wait_polls_until_absent(fake_dynamodb) -> None:
    script = [_EXISTS, _ABSENT]
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": script})
    await _client(server).delete_table("users", wait=True, timeout_seconds=10)
    assert server.calls == ["DeleteTable", "DescribeTable", "DescribeTable"]


def test_sync_delete_table_wait_polls_until_absent(fake_dynamodb) -> None:
    script = [_EXISTS, _ABSENT]
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": script})
    _client(server).sync_delete_table("users", wait=True, timeout_seconds=10)
    assert server.calls == ["DeleteTable", "DescribeTable", "DescribeTable"]


async def test_delete_table_wait_times_out(fake_dynamodb) -> None:
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": [_EXISTS]})
    with pytest.raises(TimeoutError):
        await _client(server).delete_table("users", wait=True, timeout_seconds=1)


def test_sync_delete_table_wait_times_out(fake_dynamodb) -> None:
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": [_EXISTS]})
    with pytest.raises(TimeoutError):
        _client(server).sync_delete_table("users", wait=True, timeout_seconds=1)


def test_delete_table_wait_options_are_keyword_only(fake_dynamodb) -> None:
    server = fake_dynamodb({"DeleteTable": [_DELETED], "DescribeTable": [_EXISTS]})
    with pytest.raises(TypeError):
        _client(server).sync_delete_table("users", True)  # type: ignore[misc]


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
