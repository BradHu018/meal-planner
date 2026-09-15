"""A process-safe stdio runner for the official MCP server object.

The SDK's server and protocol types remain authoritative. This runner only owns
the OS pipe bridge, keeping diagnostics off stdout and avoiding inherited stream
wrapper state in spawned Python processes.
"""

from __future__ import annotations

import asyncio
import anyio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import mcp_types as types
from mcp.server import MCPServer
from mcp.shared.message import SessionMessage


async def run_stdio_server(server: MCPServer) -> None:
    incoming_send, incoming_receive = anyio.create_memory_object_stream[SessionMessage | Exception](10)
    outgoing_send, outgoing_receive = anyio.create_memory_object_stream[SessionMessage](0)

    async def deliver(chunk: bytes, buffered: bytes) -> bytes:
        buffered += chunk
        lines = buffered.split(b"\n")
        buffered = lines.pop()
        for line in lines:
            try:
                message = types.jsonrpc_message_adapter.validate_json(line)
                await incoming_send.send(SessionMessage(message))
            except Exception as exc:
                await incoming_send.send(exc)
        return buffered

    async def read_stdin() -> None:
        # Do not use AnyIO's thread-based stdin wrapper here.  In the
        # application runtime it can consume the first protocol frame and
        # then leave subsequent frames unread.  The asyncio pipe transport is
        # event-loop native and keeps the long-lived JSON-RPC session alive.
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        stdin_transport, _ = await loop.connect_read_pipe(
            lambda: protocol, sys.stdin.buffer
        )
        async with incoming_send:
            buffered = b""
            try:
                while chunk := await reader.read(65536):
                    buffered = await deliver(chunk, buffered)
            finally:
                stdin_transport.close()

    async def write_stdout() -> None:
        async with outgoing_receive:
            async for session_message in outgoing_receive:
                line = session_message.message.model_dump_json(
                    by_alias=True, exclude_unset=True
                ) + "\n"
                _write_protocol_line(line)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(read_stdin)
        tasks.start_soon(write_stdout)
        await anyio.sleep(0)
        try:
            await server._lowlevel_server.run(
                incoming_receive,
                outgoing_send,
                server._lowlevel_server.create_initialization_options(),
            )
        finally:
            tasks.cancel_scope.cancel()


def _write_protocol_line(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


@asynccontextmanager
async def stdio_client_transport(command: str, args: list[str], cwd: Path):
    """Connect the official MCP client to a child process over stdio."""
    process = await asyncio.create_subprocess_exec(
        command, *args,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    incoming_send, incoming_receive = anyio.create_memory_object_stream[SessionMessage | Exception](10)
    outgoing_send, outgoing_receive = anyio.create_memory_object_stream[SessionMessage](10)

    async def read_stdout() -> None:
        assert process.stdout
        async with incoming_send:
            while line := await process.stdout.readline():
                try:
                    await incoming_send.send(SessionMessage(types.jsonrpc_message_adapter.validate_json(line)))
                except Exception as exc:
                    await incoming_send.send(exc)

    async def write_stdin() -> None:
        assert process.stdin
        async with outgoing_receive:
            async for message in outgoing_receive:
                process.stdin.write((message.message.model_dump_json(by_alias=True, exclude_unset=True) + "\n").encode())
                await process.stdin.drain()

    async def read_stderr() -> None:
        assert process.stderr
        while line := await process.stderr.readline():
            sys.stderr.buffer.write(line)
            sys.stderr.flush()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(read_stdout)
        tasks.start_soon(write_stdin)
        tasks.start_soon(read_stderr)
        try:
            yield incoming_receive, outgoing_send
        finally:
            outgoing_send.close()
            if process.stdin:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()
            tasks.cancel_scope.cancel()
