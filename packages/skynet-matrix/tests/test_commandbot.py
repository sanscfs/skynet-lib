"""Tests for ``skynet_matrix.bot.CommandBot`` and its helpers.

All tests are fully offline — the real ``nio.AsyncClient`` is swapped
for an ``AsyncMock``-shaped fake so we never open a socket.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from skynet_matrix import (
    BotConfig,
    Command,
    CommandBot,
    build_bot_commands_content,
    parse_command_line,
)
from skynet_matrix.state_events import STATE_EVENT_TYPE

# -- Helpers -----------------------------------------------------------------


def _fake_client() -> MagicMock:
    """Build a MagicMock shaped like ``nio.AsyncClient`` for injection."""
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


def _make_bot(**overrides) -> tuple[CommandBot, MagicMock]:
    client = overrides.pop("client", None) or _fake_client()
    bot = CommandBot(
        homeserver="http://hs.test",
        access_token="tok",
        user_id="@bot:test",
        rooms=["!room:test"],
        client=client,
        **overrides,
    )
    return bot, client


# -- 1. Command registration + lookup ---------------------------------------


def test_register_and_lookup_command():
    bot, _ = _make_bot()

    @bot.command(name="echo", description="echo back", args_hint="<text>")
    async def echo(event, args):
        return " ".join(args)

    cmd = bot.get_command("echo")
    assert cmd is not None
    assert cmd.name == "echo"
    assert cmd.description == "echo back"
    assert cmd.args_hint == "<text>"
    # built-in help is also there
    assert bot.get_command("help") is not None


# -- 2. Dispatch by message text --------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_by_message_text():
    bot, client = _make_bot()

    seen: dict = {}

    @bot.command(name="ping", description="say pong")
    async def ping(event, args):
        seen["args"] = args
        seen["sender"] = event.sender
        return "pong"

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        body="!ping hello world",
        event_id="$msg:1",
    )
    await bot.handle_text_event(room, event)

    assert seen["args"] == ["hello", "world"]
    assert seen["sender"] == "@user:test"

    # One reply sent to the right room.
    client.room_send.assert_awaited_once()
    call = client.room_send.await_args
    assert call.kwargs["room_id"] == "!room:test"
    assert call.kwargs["content"]["body"] == "pong"


# -- 3. Arg parsing (quoted args, multi-word) --------------------------------


def test_parse_quoted_args():
    assert parse_command_line('!note "hello world" tag1') == (
        "note",
        ["hello world", "tag1"],
    )


def test_parse_plain_args():
    assert parse_command_line("!list watched 5") == ("list", ["watched", "5"])


def test_parse_no_prefix_returns_none():
    assert parse_command_line("just a chat message") is None


def test_parse_only_prefix_returns_none():
    assert parse_command_line("!") is None


def test_parse_malformed_quote_falls_back():
    # Unterminated quote -> shlex ValueError -> whitespace split.
    assert parse_command_line('!note "unterminated') == ("note", ['"unterminated'])


def test_parse_custom_prefix():
    assert parse_command_line("/help foo", prefix="/") == ("help", ["foo"])


# -- 4. Unknown command behaviour -------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_silent_by_default():
    bot, client = _make_bot()
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!nonexistent")
    await bot.handle_text_event(room, event)
    client.room_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_command_replies_when_enabled():
    bot, client = _make_bot(reply_unknown_command=True)
    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!nonexistent")
    await bot.handle_text_event(room, event)
    client.room_send.assert_awaited_once()
    body = client.room_send.await_args.kwargs["content"]["body"]
    assert "nonexistent" in body


# -- 5. Ignoring messages from self -----------------------------------------


@pytest.mark.asyncio
async def test_ignores_own_messages():
    bot, client = _make_bot()

    @bot.command(name="ping", description="")
    async def ping(event, args):  # pragma: no cover -- must not run
        return "pong"

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@bot:test", body="!ping")
    await bot.handle_text_event(room, event)
    client.room_send.assert_not_awaited()


# -- 6. Handler exception is caught, never crashes loop ---------------------


@pytest.mark.asyncio
async def test_handler_exception_is_caught():
    bot, client = _make_bot()

    @bot.command(name="boom", description="")
    async def boom(event, args):
        raise RuntimeError("kaboom")

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!boom")
    # Must not raise.
    await bot.handle_text_event(room, event)

    # Error reply posted.
    client.room_send.assert_awaited_once()
    body = client.room_send.await_args.kwargs["content"]["body"]
    assert "boom" in body and "fail" in body.lower()


# -- 7. State event payload shape -------------------------------------------


def test_state_event_payload_shape():
    cmds = [
        Command(
            name="list-watched-movies",
            description="Show recent watched movies",
            handler=AsyncMock(),
            args_hint="[limit]",
            emoji="\U0001f3ac",
        )
    ]
    content = build_bot_commands_content(
        bot_name="Skynet Movies",
        prefix="!",
        commands=cmds,
    )
    assert content == {
        "bot_name": "Skynet Movies",
        "prefix": "!",
        "commands": [
            {
                "name": "list-watched-movies",
                "description": "Show recent watched movies",
                "args_hint": "[limit]",
                "emoji": "\U0001f3ac",
            }
        ],
    }


@pytest.mark.asyncio
async def test_publish_commands_state_writes_state_event():
    bot, client = _make_bot()

    @bot.command(name="echo", description="", args_hint="[text]")
    async def echo(event, args):
        return None

    await bot.publish_commands_state()

    client.room_put_state.assert_awaited()
    call = client.room_put_state.await_args
    assert call.kwargs["event_type"] == STATE_EVENT_TYPE
    assert call.kwargs["state_key"] == "@bot:test"
    assert call.kwargs["room_id"] == "!room:test"
    content = call.kwargs["content"]
    names = [c["name"] for c in content["commands"]]
    assert "echo" in names and "help" in names
    assert content["prefix"] == "!"


# -- 8. !help auto-generation ------------------------------------------------


@pytest.mark.asyncio
async def test_help_command_auto_generated():
    bot, client = _make_bot()

    @bot.command(name="list", description="List things", args_hint="[limit]")
    async def _list(event, args):
        return None

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!help")
    await bot.handle_text_event(room, event)

    client.room_send.assert_awaited_once()
    content = client.room_send.await_args.kwargs["content"]
    assert "!list" in content["body"]
    assert "List things" in content["body"]
    # HTML path also populated
    assert "formatted_body" in content
    assert "<table>" in content["formatted_body"]


# -- 9. Reaction -> command dispatch ----------------------------------------


@pytest.mark.asyncio
async def test_reaction_dispatches_command():
    bot, client = _make_bot()
    # Pretend we already posted the pinned menu.
    bot._menu_message_ids["!room:test"] = "$menu:1"

    seen = {}

    @bot.command(name="watched", description="Show watched", emoji="\U0001f3ac")
    async def watched(event, args):
        seen["hit"] = True
        return "list"

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        key="\U0001f3ac",
        reacts_to="$menu:1",
    )
    await bot.handle_reaction_event(room, event)
    assert seen.get("hit") is True
    client.room_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_reaction_on_other_message_ignored():
    bot, client = _make_bot()
    bot._menu_message_ids["!room:test"] = "$menu:1"

    @bot.command(name="watched", description="", emoji="\U0001f3ac")
    async def watched(event, args):  # pragma: no cover -- must not run
        return "nope"

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(
        sender="@user:test",
        key="\U0001f3ac",
        reacts_to="$not-the-menu",
    )
    await bot.handle_reaction_event(room, event)
    client.room_send.assert_not_awaited()


# -- 10. BotConfig sanity ---------------------------------------------------


def test_bot_config_defaults():
    bot, _ = _make_bot(command_prefix="/")
    assert isinstance(bot.config, BotConfig)
    assert bot.config.command_prefix == "/"
    # bot_name derives from user_id localpart
    assert bot.config.bot_name == "bot"


# -- 11. Dict-shaped handler return is honoured -----------------------------


@pytest.mark.asyncio
async def test_handler_dict_return_sets_html():
    bot, client = _make_bot()

    @bot.command(name="rich", description="")
    async def rich(event, args):
        return {"text": "plain", "html": "<b>bold</b>"}

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!rich")
    await bot.handle_text_event(room, event)

    content = client.room_send.await_args.kwargs["content"]
    assert content["body"] == "plain"
    assert content["format"] == "org.matrix.custom.html"
    assert content["formatted_body"] == "<b>bold</b>"


# -- 12. None return is silent ---------------------------------------------


@pytest.mark.asyncio
async def test_handler_none_is_silent():
    bot, client = _make_bot()

    @bot.command(name="silent", description="")
    async def silent(event, args):
        return None

    room = SimpleNamespace(room_id="!room:test")
    event = SimpleNamespace(sender="@user:test", body="!silent")
    await bot.handle_text_event(room, event)
    client.room_send.assert_not_awaited()


# -- 13. post_message public helper -----------------------------------------


@pytest.mark.asyncio
async def test_post_message_plain_text_builds_escaped_html():
    """Without an explicit ``html=``, the helper auto-formats the text
    to HTML so clients that use ``formatted_body`` render consistently."""
    bot, client = _make_bot()

    resp = await bot.post_message("!room:test", "line1\nline2 & <x>")
    assert resp is not None  # mock returns SimpleNamespace(event_id=...)

    client.room_send.assert_awaited_once()
    call = client.room_send.await_args
    assert call.kwargs["room_id"] == "!room:test"
    assert call.kwargs["message_type"] == "m.room.message"
    content = call.kwargs["content"]
    assert content["msgtype"] == "m.text"
    assert content["body"] == "line1\nline2 & <x>"
    assert content["format"] == "org.matrix.custom.html"
    # Newlines -> <br/>, HTML-special chars escaped.
    assert "<br/>" in content["formatted_body"]
    assert "&amp;" in content["formatted_body"]
    assert "&lt;x&gt;" in content["formatted_body"]


@pytest.mark.asyncio
async def test_post_message_explicit_html_passthrough():
    bot, client = _make_bot()

    await bot.post_message(
        "!room:test",
        "plain fallback",
        html="<b>rich</b>",
    )

    content = client.room_send.await_args.kwargs["content"]
    assert content["body"] == "plain fallback"
    assert content["format"] == "org.matrix.custom.html"
    assert content["formatted_body"] == "<b>rich</b>"


@pytest.mark.asyncio
async def test_post_message_thread_root_sets_relates_to():
    bot, client = _make_bot()

    await bot.post_message(
        "!room:test",
        "reply in thread",
        thread_root="$root:event",
    )

    content = client.room_send.await_args.kwargs["content"]
    assert content["m.relates_to"] == {
        "rel_type": "m.thread",
        "event_id": "$root:event",
    }


@pytest.mark.asyncio
async def test_post_message_swallows_send_errors():
    """Background callsites must not crash if the homeserver hiccups."""
    bot, client = _make_bot()
    client.room_send.side_effect = RuntimeError("server went away")

    # Must not raise.
    resp = await bot.post_message("!room:test", "anything")
    assert resp is None


# -- 14. react public helper ------------------------------------------------


@pytest.mark.asyncio
async def test_react_builds_correct_annotation_payload():
    bot, client = _make_bot()

    resp = await bot.react("!room:test", "$target:event", "\U0001f44d")
    assert resp is not None

    client.room_send.assert_awaited_once()
    call = client.room_send.await_args
    assert call.kwargs["room_id"] == "!room:test"
    assert call.kwargs["message_type"] == "m.reaction"
    content = call.kwargs["content"]
    assert content["m.relates_to"] == {
        "rel_type": "m.annotation",
        "event_id": "$target:event",
        "key": "\U0001f44d",
    }


@pytest.mark.asyncio
async def test_react_swallows_send_errors():
    bot, client = _make_bot()
    client.room_send.side_effect = RuntimeError("boom")

    resp = await bot.react("!room:test", "$e:1", "\u2705")
    assert resp is None
