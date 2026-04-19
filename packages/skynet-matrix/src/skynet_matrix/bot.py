"""``CommandBot`` — Matrix bot framework for slash-command services.

High-level API::

    from skynet_matrix import CommandBot

    bot = CommandBot(
        homeserver="http://conduwuit.matrix.svc:6167",
        access_token=os.environ["MATRIX_ACCESS_TOKEN"],
        user_id="@skynet-movies:matrix.sanscfs.dev",
        rooms=["!xaGPso...:matrix.sanscfs.dev"],
    )

    @bot.command(name="list-watched", description="Show watched movies",
                 args_hint="[limit]")
    async def list_watched(event, args):
        limit = int(args[0]) if args else 10
        ...
        return "Переглянуто: ..."

    await bot.run()

Runtime behaviour:

* Uses ``nio.AsyncClient`` for sync, reconnect, rate-limit handling.
  This is intentional — the framework is meant to replace every place
  we currently write our own sync loop.
* On start: validates auth via ``whoami``, joins every configured
  ``room_id`` that the bot isn't in yet, and publishes a
  ``dev.sanscfs.bot_commands`` state event into each room.
* On every ``!<command>`` message: parses, dispatches, sends the
  handler's reply as ``m.text`` with HTML formatting.  Handler
  exceptions are caught — the sync loop never dies from a bad command.
* On reactions to a registered pinned "command menu" message: the
  reacting user is treated as the command invoker, and the matching
  ``Command`` (by ``emoji=``) is dispatched.
* Dispatch latency is instrumented with an OTel span
  ``matrix.command.dispatch`` if ``opentelemetry`` is available.  It
  falls back silently when no tracer is wired up.

State event schema: see ``skynet_matrix.state_events``.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from skynet_matrix.commands import Command, HandlerCoro, parse_command_line
from skynet_matrix.state_events import (
    STATE_EVENT_TYPE,
    publish_bot_commands_state,
)

logger = logging.getLogger("skynet_matrix.bot")


# -- OTel (soft dep) ---------------------------------------------------------
try:  # pragma: no cover — trivial import shim
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("skynet_matrix.bot")
except Exception:  # pragma: no cover
    _TRACER = None


# -- Configuration -----------------------------------------------------------


@dataclass
class BotConfig:
    """Structured config for ``CommandBot``.

    Separate from the constructor kwargs so callers can build the
    config from Vault / env in one place and then pass it around.
    """

    homeserver: str
    access_token: str
    user_id: str
    rooms: list[str] = field(default_factory=list)
    command_prefix: str = "!"
    bot_name: Optional[str] = None  # defaults to user_id localpart
    next_batch_store_path: Optional[str] = None  # None -> in-memory
    reply_unknown_command: bool = False  # stay silent by default
    state_event_type: str = STATE_EVENT_TYPE
    menu_message_body: Optional[str] = None  # optional pinned menu header
    sync_timeout_ms: int = 30_000


# -- Menu owner marker used to find & edit our own pinned menu -------------
MENU_OWNER_FIELD = "dev.sanscfs.menu_owner"


# -- Bot ---------------------------------------------------------------------


class CommandBot:
    """Slash-command Matrix bot.

    Wraps ``nio.AsyncClient`` with a command registry, automatic state
    event advertising, reaction-based command dispatch, and a built-in
    ``!help`` command.
    """

    def __init__(
        self,
        *,
        homeserver: str,
        access_token: str,
        user_id: str,
        rooms: Optional[list[str]] = None,
        command_prefix: str = "!",
        bot_name: Optional[str] = None,
        next_batch_store_path: Optional[str] = None,
        reply_unknown_command: bool = False,
        state_event_type: str = STATE_EVENT_TYPE,
        menu_message_body: Optional[str] = None,
        sync_timeout_ms: int = 30_000,
        client: Optional[Any] = None,  # injected for tests
    ) -> None:
        self.config = BotConfig(
            homeserver=homeserver,
            access_token=access_token,
            user_id=user_id,
            rooms=list(rooms or []),
            command_prefix=command_prefix,
            bot_name=bot_name or user_id.split(":", 1)[0].lstrip("@"),
            next_batch_store_path=next_batch_store_path,
            reply_unknown_command=reply_unknown_command,
            state_event_type=state_event_type,
            menu_message_body=menu_message_body,
            sync_timeout_ms=sync_timeout_ms,
        )

        self._commands: dict[str, Command] = {}
        self._emoji_commands: dict[str, Command] = {}
        self._client = client  # may be None until _ensure_client
        self._owns_client = client is None
        self._stop_event: asyncio.Event = asyncio.Event()
        # room_id -> event_id of our pinned command menu, so we can
        # idempotently edit it instead of spamming new messages.
        self._menu_message_ids: dict[str, str] = {}

        self._register_builtin_commands()

    # -- Public API --------------------------------------------------------

    def command(
        self,
        name: Optional[str] = None,
        *,
        description: str = "",
        args_hint: Optional[str] = None,
        emoji: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Callable[[HandlerCoro], HandlerCoro]:
        """Decorator that registers ``func`` as a command handler.

        ``name`` defaults to the function's name (underscores stay; callers
        who want kebab-case must pass it explicitly).
        """

        def decorator(func: HandlerCoro) -> HandlerCoro:
            cmd_name = name or func.__name__
            desc = description
            if not desc and func.__doc__:
                desc = func.__doc__.strip().splitlines()[0]
            cmd = Command(
                name=cmd_name,
                description=desc,
                handler=func,
                args_hint=args_hint,
                emoji=emoji,
                metadata=dict(metadata or {}),
            )
            self.register(cmd)
            return func

        return decorator

    def register(self, cmd: Command) -> None:
        """Register a ``Command`` instance programmatically."""
        if cmd.name in self._commands:
            logger.warning("CommandBot: overriding already-registered %s", cmd.name)
        self._commands[cmd.name] = cmd
        if cmd.emoji:
            self._emoji_commands[cmd.emoji] = cmd

    def commands(self) -> list[Command]:
        """Return the list of registered commands in insertion order."""
        return list(self._commands.values())

    def get_command(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    async def run(self) -> None:
        """Log in, join rooms, publish state, sync forever."""
        await self._ensure_client()
        await self._whoami_check()
        await self._join_configured_rooms()
        await self.publish_commands_state()
        await self._post_or_update_menu()

        self._bind_event_handlers()

        logger.info(
            "CommandBot %s online: %d rooms, %d commands",
            self.config.user_id,
            len(self.config.rooms),
            len(self._commands),
        )

        try:
            await self._client.sync_forever(
                timeout=self.config.sync_timeout_ms,
                full_state=False,
            )
        finally:
            if self._owns_client:
                await self._client.close()

    async def stop(self) -> None:
        """Graceful shutdown — next sync tick will return."""
        self._stop_event.set()
        if self._client is not None and hasattr(self._client, "stop_sync_forever"):
            # nio 0.24 synchronous call; 0.25+ is coroutine.  Duck-type.
            result = self._client.stop_sync_forever()
            if asyncio.iscoroutine(result):
                await result

    async def publish_commands_state(self) -> None:
        """Write the bot_commands state event into every configured room."""
        assert self._client is not None, "call _ensure_client() first"
        for room_id in self.config.rooms:
            ok = await publish_bot_commands_state(
                self._client,
                room_id,
                bot_user_id=self.config.user_id,
                bot_name=self.config.bot_name or self.config.user_id,
                prefix=self.config.command_prefix,
                commands=self.commands(),
                event_type=self.config.state_event_type,
            )
            if ok:
                logger.info("advertised %d commands in %s", len(self._commands), room_id)

    async def post_message(
        self,
        room_id: str,
        text: str,
        html: Optional[str] = None,
        thread_root: Optional[str] = None,
    ) -> Any:
        """Post a plain or HTML message to a room outside the command path.

        Use this for module-initiated posts that don't originate from a
        ``!cmd`` dispatch -- scheduled jobs, webhook receivers, periodic
        notifications, etc. For posting a reply from inside a command
        handler, just ``return`` the text (or a ``{"text", "html"}``
        dict) from the handler; the dispatcher calls the same underlying
        code path.

        Parameters
        ----------
        room_id:
            Matrix room ID (``!...:homeserver``).
        text:
            Plain-text body. Always sent as ``body``.
        html:
            Optional HTML body. When provided, the event is sent with
            ``format=org.matrix.custom.html`` + ``formatted_body=html``.
            When omitted, a safe auto-formatted HTML body is generated
            from ``text`` (HTML-escaped, newlines -> ``<br/>``) so
            rendering is consistent across clients.
        thread_root:
            Optional event_id of the thread root. When set, the message
            is sent as a threaded reply (``m.relates_to`` with
            ``rel_type=m.thread``).

        Returns the ``room_send`` response from the underlying client,
        or ``None`` if the send failed (errors are logged, not raised,
        so a misbehaving callsite never takes the bot down).
        """
        assert self._client is not None, "call _ensure_client() first"
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": (html if html is not None else html_lib.escape(text).replace("\n", "<br/>")),
        }
        if thread_root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
            }
        try:
            return await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("post_message failed room=%s: %s", room_id, exc)
            return None

    async def react(self, room_id: str, event_id: str, emoji: str) -> Any:
        """Post an ``m.reaction`` to ``event_id`` in ``room_id``.

        Useful for background acknowledgments from scheduled jobs or
        webhook handlers that want to signal "I saw this" on a user's
        message without posting a full reply.

        Returns the ``room_send`` response, or ``None`` on failure
        (errors are logged, not raised).
        """
        assert self._client is not None, "call _ensure_client() first"
        content: dict[str, Any] = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": emoji,
            }
        }
        try:
            return await self._client.room_send(
                room_id=room_id,
                message_type="m.reaction",
                content=content,
            )
        except Exception as exc:
            logger.warning("react failed room=%s event=%s: %s", room_id, event_id, exc)
            return None

    # -- Dispatch (sync-loop entry points; also called directly by tests) --

    async def handle_text_event(self, room: Any, event: Any) -> None:
        """Route an ``m.room.message`` / ``RoomMessageText`` to a command."""
        if getattr(event, "sender", None) == self.config.user_id:
            return  # never dispatch our own messages

        body = getattr(event, "body", "") or ""
        parsed = parse_command_line(body, prefix=self.config.command_prefix)
        if parsed is None:
            return

        cmd_name, args = parsed
        room_id = getattr(room, "room_id", None) or getattr(event, "room_id", None)
        if not room_id:
            return

        await self._dispatch(cmd_name, args, room_id=room_id, event=event)

    async def handle_reaction_event(self, room: Any, event: Any) -> None:
        """Route an ``m.reaction`` on our pinned menu to the matching cmd."""
        if getattr(event, "sender", None) == self.config.user_id:
            return

        key = getattr(event, "key", None)
        reacts_to = getattr(event, "reacts_to", None)
        if not key or not reacts_to:
            return

        room_id = getattr(room, "room_id", None) or getattr(event, "room_id", None)
        if not room_id:
            return

        # Only react to reactions on our own pinned menu.
        if self._menu_message_ids.get(room_id) != reacts_to:
            return

        cmd = self._emoji_commands.get(key)
        if cmd is None:
            return

        await self._dispatch(cmd.name, [], room_id=room_id, event=event)

    # -- Internal ----------------------------------------------------------

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        # Import locally so tests can inject a mock without importing nio.
        from nio import AsyncClient

        self._client = AsyncClient(self.config.homeserver)
        self._client.access_token = self.config.access_token
        self._client.user_id = self.config.user_id
        # nio wants a device_id when an access_token is provided directly;
        # for a non-E2EE service bot any stable string is fine.
        if not getattr(self._client, "device_id", None):
            self._client.device_id = "skynet-commandbot"

    async def _whoami_check(self) -> None:
        """Verify that the access token resolves to ``user_id``."""
        assert self._client is not None
        if not hasattr(self._client, "whoami"):
            return
        try:
            resp = await self._client.whoami()
        except Exception as exc:
            logger.warning("whoami failed: %s", exc)
            return
        resp_user = getattr(resp, "user_id", None)
        if resp_user and resp_user != self.config.user_id:
            logger.warning(
                "whoami user_id=%s does not match configured %s",
                resp_user,
                self.config.user_id,
            )

    async def _join_configured_rooms(self) -> None:
        """Join every configured room we're not in yet."""
        assert self._client is not None
        joined: set[str] = set()
        if hasattr(self._client, "joined_rooms"):
            try:
                resp = await self._client.joined_rooms()
                joined = set(getattr(resp, "rooms", []) or [])
            except Exception as exc:
                logger.warning("joined_rooms failed: %s", exc)

        for room_id in self.config.rooms:
            if room_id in joined:
                continue
            try:
                await self._client.join(room_id)
                logger.info("joined %s", room_id)
            except Exception as exc:
                logger.warning("join %s failed: %s", room_id, exc)

    def _bind_event_handlers(self) -> None:
        """Attach our handlers to nio's event dispatcher."""
        assert self._client is not None
        from nio import ReactionEvent, RoomMessageText

        self._client.add_event_callback(self.handle_text_event, RoomMessageText)
        self._client.add_event_callback(self.handle_reaction_event, ReactionEvent)

    async def _dispatch(
        self,
        cmd_name: str,
        args: list[str],
        *,
        room_id: str,
        event: Any,
    ) -> None:
        cmd = self._commands.get(cmd_name)
        start = time.perf_counter()

        span_ctx = _TRACER.start_as_current_span("matrix.command.dispatch") if _TRACER else None
        if span_ctx is not None:
            span = span_ctx.__enter__()  # type: ignore[union-attr]
            span.set_attribute("matrix.command.name", cmd_name)
            span.set_attribute("matrix.command.args_count", len(args))
            span.set_attribute("matrix.room_id", room_id)

        try:
            if cmd is None:
                logger.info("unknown command %r in %s", cmd_name, room_id)
                if self.config.reply_unknown_command:
                    await self._reply(
                        room_id,
                        f"Unknown command `{cmd_name}`. Try `{self.config.command_prefix}help`.",
                    )
                return

            try:
                result = await cmd.handler(event, args)
            except Exception as exc:
                logger.exception("command %s raised: %s", cmd_name, exc)
                await self._reply(room_id, f"Command `{cmd_name}` failed: {exc}")
                return

            await self._send_result(room_id, result)

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "dispatched %s in %s (%.1fms, result=%s)",
                cmd_name,
                room_id,
                elapsed_ms,
                "dict" if isinstance(result, dict) else type(result).__name__,
            )
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(None, None, None)  # type: ignore[union-attr]

    async def _send_result(self, room_id: str, result: Any) -> None:
        if result is None:
            return
        if isinstance(result, str):
            await self._reply(room_id, result)
            return
        if isinstance(result, dict):
            text = result.get("text") or ""
            html = result.get("html")
            thread = bool(result.get("thread", False))
            await self._reply(room_id, text, html=html, thread=thread)
            return
        logger.warning("unknown handler return type: %r", type(result))

    async def _reply(
        self,
        room_id: str,
        text: str,
        *,
        html: Optional[str] = None,
        thread: bool = False,
    ) -> None:
        assert self._client is not None
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
        }
        if html is None:
            # Auto-format: escape HTML so plain text bodies render identically
            # in clients that use formatted_body.
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html_lib.escape(text).replace("\n", "<br/>")
        else:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html
        # ``thread`` currently left unwired to avoid depending on a specific
        # thread_root plumbing; callers who need threads should use the
        # low-level httpx AsyncMatrixClient directly.
        try:
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("room_send failed room=%s: %s", room_id, exc)

    # -- Built-in commands & menu -----------------------------------------

    def _register_builtin_commands(self) -> None:
        async def _help(event: Any, args: list[str]) -> dict:
            return self._render_help()

        self.register(
            Command(
                name="help",
                description="Show the list of commands",
                handler=_help,
            )
        )

    def _render_help(self) -> dict[str, Any]:
        """Render the help reply as text + HTML."""
        prefix = self.config.command_prefix
        lines = [f"Commands for {self.config.bot_name}:"]
        html_rows = ["<table><thead><tr><th>Command</th><th>Args</th><th>Description</th></tr></thead><tbody>"]
        for cmd in self.commands():
            args_hint = cmd.args_hint or ""
            lines.append(f"  {prefix}{cmd.name} {args_hint}  -- {cmd.description}")
            html_rows.append(
                f"<tr><td><code>{html_lib.escape(prefix + cmd.name)}</code></td>"
                f"<td><code>{html_lib.escape(args_hint)}</code></td>"
                f"<td>{html_lib.escape(cmd.description)}</td></tr>"
            )
        html_rows.append("</tbody></table>")
        return {
            "text": "\n".join(lines),
            "html": (f"<p><b>Commands for {html_lib.escape(self.config.bot_name or '')}</b></p>" + "".join(html_rows)),
        }

    def _render_menu(self) -> dict[str, Any]:
        """Render the pinned command menu (HTML + plain)."""
        header = self.config.menu_message_body or (f"{self.config.bot_name} — available commands")
        help_body = self._render_help()
        return {
            "text": f"{header}\n\n{help_body['text']}",
            "html": f"<h3>{html_lib.escape(header)}</h3>{help_body['html']}",
        }

    async def _post_or_update_menu(self) -> None:
        """Post (or edit) a pinned command menu in every configured room.

        Tagged with ``dev.sanscfs.menu_owner = user_id`` inside
        ``content`` so we can find our own menu without a lookup table.
        """
        assert self._client is not None
        menu = self._render_menu()
        for room_id in self.config.rooms:
            existing = await self._find_existing_menu(room_id)
            if existing:
                await self._edit_menu(room_id, existing, menu)
                self._menu_message_ids[room_id] = existing
            else:
                event_id = await self._send_menu(room_id, menu)
                if event_id:
                    self._menu_message_ids[room_id] = event_id
                    await self._pin_event(room_id, event_id)

    async def _find_existing_menu(self, room_id: str) -> Optional[str]:
        """Look up the event_id of a previous menu we posted in ``room_id``.

        Strategy: ask for the room's pinned events (state) and filter to
        those where the source event's content has
        ``dev.sanscfs.menu_owner == self.user_id``.  nio only gives us
        the event_ids via state, so we also fetch each candidate via
        ``room_get_event`` to verify ownership.
        """
        assert self._client is not None
        try:
            state = await self._client.room_get_state_event(
                room_id=room_id,
                event_type="m.room.pinned_events",
                state_key="",
            )
        except Exception:
            return None

        content = getattr(state, "content", None) or {}
        pinned: list[str] = list(content.get("pinned", []) or [])
        if not pinned:
            return None

        # Best-effort: just return the first pinned id if room_get_event
        # confirms menu_owner matches. Room-state lookup of an arbitrary
        # event is only supported on some servers; on failure we treat
        # the room as menu-less and fall back to posting a new one.
        for event_id in pinned:
            if not hasattr(self._client, "room_get_event"):
                return None
            try:
                resp = await self._client.room_get_event(room_id, event_id)
            except Exception:
                continue
            src_event = getattr(resp, "event", None)
            if src_event is None:
                continue
            src_content = getattr(src_event, "source", {}).get("content", {})
            if src_content.get(MENU_OWNER_FIELD) == self.config.user_id:
                return event_id
        return None

    async def _send_menu(self, room_id: str, menu: dict[str, Any]) -> Optional[str]:
        assert self._client is not None
        content = {
            "msgtype": "m.text",
            "body": menu["text"],
            "format": "org.matrix.custom.html",
            "formatted_body": menu["html"],
            MENU_OWNER_FIELD: self.config.user_id,
        }
        try:
            resp = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("menu send failed room=%s: %s", room_id, exc)
            return None
        return getattr(resp, "event_id", None)

    async def _edit_menu(
        self,
        room_id: str,
        event_id: str,
        menu: dict[str, Any],
    ) -> None:
        assert self._client is not None
        content = {
            "msgtype": "m.text",
            "body": f"* {menu['text']}",
            "format": "org.matrix.custom.html",
            "formatted_body": f"* {menu['html']}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": menu["text"],
                "format": "org.matrix.custom.html",
                "formatted_body": menu["html"],
                MENU_OWNER_FIELD: self.config.user_id,
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": event_id,
            },
        }
        try:
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("menu edit failed room=%s: %s", room_id, exc)

    async def _pin_event(self, room_id: str, event_id: str) -> None:
        """Add ``event_id`` to ``m.room.pinned_events``, preserving existing pins."""
        assert self._client is not None
        pinned: list[str] = []
        try:
            state = await self._client.room_get_state_event(
                room_id=room_id,
                event_type="m.room.pinned_events",
                state_key="",
            )
            pinned = list((getattr(state, "content", {}) or {}).get("pinned", []) or [])
        except Exception:
            pinned = []
        if event_id in pinned:
            return
        pinned.append(event_id)
        try:
            await self._client.room_put_state(
                room_id=room_id,
                event_type="m.room.pinned_events",
                content={"pinned": pinned},
                state_key="",
            )
        except Exception as exc:
            logger.warning("pin failed room=%s: %s", room_id, exc)
