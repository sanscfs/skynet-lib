"""Tests for ``skynet_matrix.chronicle_mirror``.

Guarantees enforced here:

* Envelope field shape matches what the Chronicle ingester expects
  (``_envelope_from_stream_fields`` in
  ``skynet_chronicle_service.ingester``).
* ``direction`` + ``user`` fields preserve the identity distinction
  from ``project_chat_identity_fix`` — user-originated messages never
  collapse into the bot's ``user=skynet-<component>`` bucket and vice
  versa.
* A missing redis client makes mirroring a silent no-op.
* An exploding redis client never raises back to the caller.
* CommandBot wires both inbound and outbound messages through the
  mirror when ``CHRONICLE_REDIS_URL`` is configured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from skynet_matrix import CommandBot
from skynet_matrix.chronicle_mirror import (
    DEFAULT_STREAM,
    mirror_message,
)


def _fake_client() -> MagicMock:
    client = MagicMock()
    client.room_send = AsyncMock(return_value=SimpleNamespace(event_id="$reply:m"))
    client.room_put_state = AsyncMock(return_value=SimpleNamespace(event_id="$state:m"))
    client.room_get_state_event = AsyncMock(
        return_value=SimpleNamespace(content={"pinned": []}),
    )
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


def test_mirror_none_client_is_noop():
    """Bots without CHRONICLE_REDIS_URL must keep working silently."""
    # Must not raise and must do nothing observable.
    mirror_message(
        None,
        direction="in",
        room_id="!room:test",
        sender="@user:test",
        body="hello",
        event_id="$evt:1",
        ts=1234567890.0,
        user="sanscfs",
    )


def test_mirror_xadd_envelope_shape_matches_ingester_expectations():
    """Verify the wire fields are exactly what the ingester decodes."""
    r = MagicMock()
    r.xadd = MagicMock(return_value=b"1-0")

    mirror_message(
        r,
        direction="in",
        room_id="!abc:matrix.sanscfs.dev",
        sender="@sanscfs:matrix.sanscfs.dev",
        body="привіт",
        event_id="$evt:abc",
        ts=1700000000.25,
        user="sanscfs",
    )

    r.xadd.assert_called_once()
    args, kwargs = r.xadd.call_args
    assert args[0] == DEFAULT_STREAM  # matrix:events
    fields = args[1]

    # Top-level flat fields (browsable with XINFO without parsing JSON).
    assert fields["source"] == "matrix"
    assert fields["kind"] == "message"
    assert fields["subject"] == "!abc:matrix.sanscfs.dev"
    assert fields["direction"] == "in"
    assert fields["user"] == "sanscfs"

    # The preferred ingester shape: single "envelope" JSON string.
    envelope = json.loads(fields["envelope"])
    assert envelope["source"] == "matrix"
    assert envelope["kind"] == "message"
    assert envelope["subject"] == "!abc:matrix.sanscfs.dev"
    assert envelope["text"] == "привіт"  # raw text, no transformation
    assert envelope["labels"]["direction"] == "in"
    assert envelope["labels"]["user"] == "sanscfs"
    assert envelope["labels"]["sender"] == "@sanscfs:matrix.sanscfs.dev"
    assert envelope["labels"]["room"] == "!abc:matrix.sanscfs.dev"
    assert envelope["payload"]["event_id"] == "$evt:abc"
    assert envelope["payload"]["body"] == "привіт"

    # Bounded stream so Redis memory stays predictable.
    assert kwargs["maxlen"] == 100_000
    assert kwargs["approximate"] is True


def test_mirror_swallows_redis_errors():
    """A flaky Redis must never crash the bot."""
    r = MagicMock()
    r.xadd = MagicMock(side_effect=RuntimeError("redis down"))

    # Must not raise.
    mirror_message(
        r,
        direction="out",
        room_id="!room:test",
        sender="@bot:test",
        body="pong",
        event_id="$reply:1",
        ts=1700000001.0,
        user="skynet-movies",
    )


def test_mirror_preserves_user_distinction_for_bot_replies():
    """Bot replies land under user=skynet-*, never user=sanscfs."""
    r = MagicMock()
    r.xadd = MagicMock(return_value=b"1-1")

    mirror_message(
        r,
        direction="out",
        room_id="!room:test",
        sender="@skynet-movies:matrix.sanscfs.dev",
        body="Переглянуто: The Parallax View",
        event_id="$rec:7",
        ts=1700000002.5,
        user="skynet-movies",
    )

    fields = r.xadd.call_args.args[1]
    envelope = json.loads(fields["envelope"])
    assert envelope["labels"]["user"] == "skynet-movies"
    assert envelope["labels"]["direction"] == "out"
    assert fields["user"] == "skynet-movies"


# -- CommandBot integration --------------------------------------------------


def _make_bot_with_mirror(monkeypatch: pytest.MonkeyPatch) -> tuple[CommandBot, MagicMock, MagicMock]:
    """Build a CommandBot with CHRONICLE_REDIS_URL pointing at a mock."""
    monkeypatch.setenv("CHRONICLE_REDIS_URL", "redis://fake:6379/7")
    fake_redis = MagicMock()
    fake_redis.xadd = MagicMock(return_value=b"1-0")
    fake_redis.ping = MagicMock(return_value=True)

    # Patch get_mirror_client BEFORE CommandBot is instantiated so
    # __init__ picks up our fake instead of trying to resolve a real
    # redis URL.
    import skynet_matrix.bot as bot_module

    monkeypatch.setattr(bot_module, "get_mirror_client", lambda *a, **kw: fake_redis)

    client = _fake_client()
    bot = CommandBot(
        homeserver="http://hs.test",
        access_token="tok",
        user_id="@bot:test",
        rooms=["!room:test"],
        client=client,
    )
    return bot, client, fake_redis


@pytest.mark.asyncio
async def test_commandbot_mirrors_inbound_user_message(monkeypatch: pytest.MonkeyPatch):
    bot, client, fake_redis = _make_bot_with_mirror(monkeypatch)

    @bot.command(name="ping", description="say pong")
    async def ping(event, args):
        return "pong"

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@sanscfs:matrix.sanscfs.dev",
        body="!ping",
        event_id="$in:1",
        server_timestamp=1_700_000_000_000,
    )
    await bot.handle_text_event(room, event)

    # Two XADDs: one for the inbound !ping, one for the outbound pong.
    assert fake_redis.xadd.call_count == 2

    first_fields = fake_redis.xadd.call_args_list[0].args[1]
    first_env = json.loads(first_fields["envelope"])
    assert first_env["labels"]["direction"] == "in"
    assert first_env["labels"]["user"] == "sanscfs"
    assert first_env["text"] == "!ping"
    # server_timestamp=1_700_000_000_000 ms => 2023-11-14T22:13:20 UTC.
    assert first_env["ts"].startswith("2023-11-14T22:13:20")
    assert first_env["ts"].endswith("+00:00")

    second_fields = fake_redis.xadd.call_args_list[1].args[1]
    second_env = json.loads(second_fields["envelope"])
    assert second_env["labels"]["direction"] == "out"
    # Bot localpart, not the human user.
    assert second_env["labels"]["user"] == "bot"
    assert second_env["text"] == "pong"


@pytest.mark.asyncio
async def test_commandbot_does_not_mirror_own_messages(monkeypatch: pytest.MonkeyPatch):
    """Bot's own echoed messages (sender==user_id) must not inflate the stream."""
    bot, client, fake_redis = _make_bot_with_mirror(monkeypatch)

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@bot:test",  # same as user_id
        body="anything",
        event_id="$self:1",
        server_timestamp=1_700_000_000_000,
    )
    await bot.handle_text_event(room, event)

    fake_redis.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_commandbot_mirrors_inbound_non_command_text(monkeypatch: pytest.MonkeyPatch):
    """Plain chat (not a !command) still lands in Chronicle."""
    bot, client, fake_redis = _make_bot_with_mirror(monkeypatch)

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@sanscfs:matrix.sanscfs.dev",
        body="just chatting, not a command",
        event_id="$in:2",
        server_timestamp=1_700_000_003_000,
    )
    await bot.handle_text_event(room, event)

    # Exactly one XADD: the inbound message. No reply => no outbound mirror.
    assert fake_redis.xadd.call_count == 1
    env = json.loads(fake_redis.xadd.call_args.args[1]["envelope"])
    assert env["labels"]["direction"] == "in"
    assert env["text"] == "just chatting, not a command"


@pytest.mark.asyncio
async def test_commandbot_post_message_mirrors_out(monkeypatch: pytest.MonkeyPatch):
    """post_message (background / scheduled posts) also goes to Chronicle."""
    bot, client, fake_redis = _make_bot_with_mirror(monkeypatch)

    await bot.post_message("!room:test", "scheduled digest: nothing to report")

    fake_redis.xadd.assert_called_once()
    env = json.loads(fake_redis.xadd.call_args.args[1]["envelope"])
    assert env["labels"]["direction"] == "out"
    assert env["labels"]["user"] == "bot"
    assert env["text"] == "scheduled digest: nothing to report"


@pytest.mark.asyncio
async def test_commandbot_without_chronicle_url_is_noop(monkeypatch: pytest.MonkeyPatch):
    """No env var => no mirror client => existing bots unaffected."""
    monkeypatch.delenv("CHRONICLE_REDIS_URL", raising=False)
    client = _fake_client()
    bot = CommandBot(
        homeserver="http://hs.test",
        access_token="tok",
        user_id="@bot:test",
        rooms=["!room:test"],
        client=client,
    )

    @bot.command(name="ping", description="say pong")
    async def ping(event, args):
        return "pong"

    # Must not raise; no mirror side-effects observable.
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@sanscfs:matrix.sanscfs.dev",
        body="!ping",
        event_id="$in:1",
    )
    await bot.handle_text_event(room, event)

    assert bot._chronicle_redis is None
    # Reply still posted to Matrix.
    client.room_send.assert_awaited_once()


# -- Identity mapping -------------------------------------------------------


def test_mxid_to_user_strips_at_and_host():
    assert CommandBot._mxid_to_user("@sanscfs:matrix.sanscfs.dev") == "sanscfs"
    assert CommandBot._mxid_to_user("@skynet-movies:matrix.sanscfs.dev") == "skynet-movies"
    assert CommandBot._mxid_to_user("") == ""
    # Malformed input must not crash the observability hook.
    assert CommandBot._mxid_to_user("no-at-sign") == "no-at-sign"
