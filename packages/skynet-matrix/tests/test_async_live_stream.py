"""Tests for :class:`skynet_matrix.async_live_stream.AsyncLiveStream` and
the streaming wrap inside ``CommandBot.handle_text_event``.

Offline — we mock ``nio.AsyncClient`` the same way ``test_commandbot.py``
does (``AsyncMock`` on ``room_send`` / ``room_typing``) and assert on
the recorded Matrix events to cover the new streaming path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from skynet_matrix import (
    AsyncLiveStream,
    CommandBot,
    EventType,
    current_live_stream,
    emit_if_live,
)


def _fake_client() -> MagicMock:
    client = MagicMock()
    client.room_send = AsyncMock(return_value=SimpleNamespace(event_id="$live:1"))
    client.room_typing = AsyncMock(return_value=None)
    client.room_put_state = AsyncMock(return_value=SimpleNamespace(event_id="$s:1"))
    client.room_get_state_event = AsyncMock(return_value=SimpleNamespace(content={"pinned": []}))
    client.room_get_event = AsyncMock()
    client.join = AsyncMock()
    client.joined_rooms = AsyncMock(return_value=SimpleNamespace(rooms=[]))
    client.whoami = AsyncMock(return_value=SimpleNamespace(user_id="@bot:test"))
    client.add_event_callback = MagicMock()
    client.sync_forever = AsyncMock()
    client.stop_sync_forever = MagicMock()
    client.close = AsyncMock()
    client.access_token = ""
    client.user_id = ""
    client.device_id = None
    return client


# -- AsyncLiveStream standalone ---------------------------------------------


@pytest.mark.asyncio
async def test_live_stream_starts_with_placeholder_and_typing():
    client = _fake_client()
    stream = AsyncLiveStream(
        nio_client=client,
        room_id="!room:t",
        bot_name="Skynet",
    )
    await stream.start()

    client.room_typing.assert_awaited_once()
    client.room_send.assert_awaited_once()
    content = client.room_send.await_args.kwargs["content"]
    assert content["msgtype"] == "m.text"
    # Placeholder body carries the bot name + progress bar
    assert "Skynet" in content["body"]
    assert "▰" in content["body"]
    assert stream.event_id == "$live:1"


@pytest.mark.asyncio
async def test_live_stream_tool_calls_force_edits():
    client = _fake_client()
    async with AsyncLiveStream(client, "!room:t", bot_name="Skynet", typing=False) as stream:
        await stream.emit(
            EventType.TOOL_CALL,
            'query="vibe decay"',
            tool_name="search_memory",
        )
        await stream.emit(
            EventType.TOOL_RESULT,
            "found 3 notes",
            tool_name="search_memory",
            duration_s=0.42,
        )

    # Initial send + tool_call edit + tool_result edit + complete edit = 4+
    assert client.room_send.await_count >= 3
    # One of the edits embeds the completed tool row
    bodies = [c.kwargs["content"]["body"] for c in client.room_send.await_args_list]
    assert any("search_memory" in b for b in bodies)
    assert any("found 3 notes" in b for b in bodies)
    # Final edit carries a done-bar (all filled)
    assert any("▰▰▰▰▰▰" in b for b in bodies)


@pytest.mark.asyncio
async def test_live_stream_complete_with_final_text_appends_footer():
    client = _fake_client()
    stream = AsyncLiveStream(client, "!room:t", bot_name="Skynet", typing=False)
    await stream.start()
    await stream.emit(EventType.TOOL_CALL, "", tool_name="foo")
    await stream.emit(EventType.TOOL_RESULT, "", tool_name="foo", duration_s=0.1)
    await stream.complete(final_text="Ready.")

    final_call = client.room_send.await_args_list[-1]
    new_body = final_call.kwargs["content"]["m.new_content"]["body"]
    assert "Ready." in new_body
    assert "1 tools" in new_body


@pytest.mark.asyncio
async def test_live_stream_thread_relation_on_send_and_edit():
    client = _fake_client()
    async with AsyncLiveStream(
        client,
        "!room:t",
        thread_root="$root:t",
        bot_name="Skynet",
        typing=False,
    ) as stream:
        await stream.emit(EventType.TOOL_CALL, "", tool_name="foo")

    # First call: send. Must carry m.thread.
    first = client.room_send.await_args_list[0].kwargs["content"]
    assert first["m.relates_to"]["rel_type"] == "m.thread"
    assert first["m.relates_to"]["event_id"] == "$root:t"
    # Edit must carry m.replace at the top level AND m.thread under m.new_content.
    edit = client.room_send.await_args_list[1].kwargs["content"]
    assert edit["m.relates_to"]["rel_type"] == "m.replace"
    assert edit["m.new_content"]["m.relates_to"]["rel_type"] == "m.thread"


@pytest.mark.asyncio
async def test_live_stream_redis_xadd_invoked():
    client = _fake_client()
    redis = MagicMock()
    redis.execute_command = MagicMock(return_value=None)
    redis.expire = MagicMock()
    async with AsyncLiveStream(
        client,
        "!room:t",
        bot_name="Skynet",
        typing=False,
        redis_client=redis,
    ) as stream:
        await stream.emit(EventType.TOOL_CALL, "", tool_name="foo")

    # start() emits no XADD by itself; emit() + complete() do
    assert redis.execute_command.call_count >= 2
    redis.expire.assert_called()


@pytest.mark.asyncio
async def test_emit_if_live_is_noop_without_stream():
    # No ContextVar set → must not raise.
    await emit_if_live(EventType.THINKING, "hello")


@pytest.mark.asyncio
async def test_live_stream_token_streaming():
    client = _fake_client()
    async with AsyncLiveStream(client, "!room:t", bot_name="Skynet", typing=False) as stream:
        await stream.start_llm(url="https://api.mistral.ai/v1", model="mistral-medium-latest")
        # Feed ~5 tokens quickly — first emit + finish_llm are force-edits,
        # appends in between obey the 2.5s debounce.
        for tok in ["hello", " ", "world", "!", " done"]:
            await stream.append_token(tok)
        await stream.finish_llm(tokens=5)

    bodies = [c.kwargs["content"]["body"] for c in client.room_send.await_args_list]
    # Endpoint anchor entry is visible once opened
    assert any("POST api.mistral.ai" in b for b in bodies)
    assert any("mistral-medium-latest" in b for b in bodies)
    # Final body (after finish_llm) no longer carries the live tail block
    final = bodies[-1]
    assert "`hello world! done`" not in final
    # … but the endpoint entry is marked done (✓)
    assert "✓ → POST api.mistral.ai" in final


@pytest.mark.asyncio
async def test_token_if_live_is_noop_without_stream():
    from skynet_matrix import emit_llm_end_if_live, emit_llm_start_if_live, emit_token_if_live

    # Without a live stream set — all three helpers must no-op.
    await emit_llm_start_if_live(url="http://x", model="m")
    await emit_token_if_live("abc")
    await emit_llm_end_if_live(tokens=3)


# -- CommandBot streaming wrap ---------------------------------------------


def _make_bot_with_on_text(on_text, **overrides):
    client = overrides.pop("client", None) or _fake_client()
    bot = CommandBot(
        homeserver="http://hs.test",
        access_token="tok",
        user_id="@bot:test",
        rooms=["!room:test"],
        on_text=on_text,
        client=client,
        **overrides,
    )
    return bot, client


@pytest.mark.asyncio
async def test_on_text_runs_under_live_stream_by_default():
    seen: dict = {}

    async def on_text(event, body, thread_root=None):
        # Inside the handler, a live stream MUST be active.
        seen["stream"] = current_live_stream.get()
        await emit_if_live(EventType.THINKING, "pondering")
        return "final"

    bot, client = _make_bot_with_on_text(on_text)
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        body="hello world",  # no command prefix — takes free-text path
        event_id="$msg:1",
        server_timestamp=None,
        source={"content": {}},
    )
    await bot.handle_text_event(room, event)

    assert seen["stream"] is not None
    # Placeholder + at least one edit + final edit
    assert client.room_send.await_count >= 2
    # Final edit contains the handler reply
    bodies = [c.kwargs["content"].get("m.new_content", {}).get("body", "") for c in client.room_send.await_args_list]
    assert any("final" in b for b in bodies)


@pytest.mark.asyncio
async def test_live_stream_disabled_falls_back_to_send_result():
    async def on_text(event, body, thread_root=None):
        return "legacy"

    bot, client = _make_bot_with_on_text(on_text, live_stream=False)
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        body="hello",
        event_id="$msg:2",
        server_timestamp=None,
        source={"content": {}},
    )
    await bot.handle_text_event(room, event)

    # One message, no edits.
    assert client.room_send.await_count == 1
    assert client.room_send.await_args.kwargs["content"]["body"] == "legacy"


@pytest.mark.asyncio
async def test_silent_handler_still_closes_stream():
    async def on_text(event, body, thread_root=None):
        return None  # silent

    bot, client = _make_bot_with_on_text(on_text)
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        body="hi",
        event_id="$msg:3",
        server_timestamp=None,
        source={"content": {}},
    )
    await bot.handle_text_event(room, event)

    # Placeholder got sent and then edited to a "done" state; no crash.
    assert client.room_send.await_count >= 1


@pytest.mark.asyncio
async def test_heartbeat_re_edits_on_silence():
    """With no events, the heartbeat should still edit every N seconds."""
    import asyncio as _asyncio

    client = _fake_client()
    async with AsyncLiveStream(
        client,
        "!room:t",
        bot_name="Skynet",
        typing=False,
        heartbeat_seconds=0.05,  # tight for test
    ):
        # Let the heartbeat fire a few times while we do nothing.
        await _asyncio.sleep(0.25)

    # Initial send + at least one heartbeat edit + final complete edit
    assert client.room_send.await_count >= 3


@pytest.mark.asyncio
async def test_heartbeat_does_not_fire_on_busy_turn():
    """When events keep firing, heartbeat should stay idle (no extra sends)."""
    import asyncio as _asyncio

    client = _fake_client()
    async with AsyncLiveStream(
        client,
        "!room:t",
        bot_name="Skynet",
        typing=False,
        heartbeat_seconds=0.1,
        debounce_seconds=0.0,  # every emit edits
    ) as stream:
        for i in range(3):
            await stream.emit(EventType.TOOL_CALL, "", tool_name=f"t{i}")
            await _asyncio.sleep(0.05)  # shorter than heartbeat_seconds

    # 1 placeholder + 3 tool_call edits + final = 5. Heartbeat should NOT fire
    # because `_last_edit_at` gets refreshed inside the silence window.
    assert client.room_send.await_count <= 7


@pytest.mark.asyncio
async def test_complete_silent_turn_redacts_placeholder():
    """Regression guard: silent-turn cleanup must leave zero artefacts."""
    client = _fake_client()
    client.room_redact = AsyncMock(return_value=None)
    stream = AsyncLiveStream(client, "!room:t", bot_name="Skynet", typing=False)
    await stream.start()
    await stream.complete()  # silent — no final_text, no entries

    client.room_redact.assert_awaited_once()
    # Only the initial placeholder was sent; complete() redacted it instead
    # of editing to a "done" header.
    assert client.room_send.await_count == 1
