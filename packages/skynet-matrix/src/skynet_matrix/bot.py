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
from typing import Any, Awaitable, Callable, Optional

from skynet_matrix.async_live_stream import AsyncLiveStream, current_live_stream
from skynet_matrix.chronicle_mirror import get_mirror_client, mirror_message
from skynet_matrix.commands import Command, HandlerCoro, parse_command_line
from skynet_matrix.state_events import (
    STATE_EVENT_TYPE,
    publish_bot_commands_state,
)
from skynet_matrix.stream_events import EventType

OnTextCallback = Callable[..., Awaitable[Optional[Any]]]
# (event, thread_root_event_id, body) → reply or None
OnThreadReplyCallback = Callable[[Any, str, str], Awaitable[Optional[Any]]]

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
    # Streaming UX: when ``True`` (default), free-text handlers
    # (``on_text`` / ``on_thread_reply``) run under an
    # :class:`AsyncLiveStream`. A placeholder message is posted as soon
    # as the user's turn starts and edited live with every tool call
    # while the handler runs; the final edit carries the clean reply.
    # Command handlers (``!cmd``) are unaffected — they stay one-shot
    # because their latency is already low.
    live_stream: bool = True
    live_stream_bot_name: Optional[str] = None  # default: bot_name
    live_stream_initial_text: str = "\u25b8 _thinking..._"
    # Cross-restart event dedup. When ``stream_redis_client`` is set,
    # every inbound ``event_id`` is claimed via a Redis SETNX; a claim
    # failure (key existed) means a previous handler turn \u2014 possibly
    # from a now-dead pod \u2014 already processed the event, so we skip.
    # ``dedup_ttl_seconds`` MUST exceed the Matrix /sync ``timeline_limit``
    # window for the busiest room (default 24h covers any sane case).
    dedup_enabled: bool = True
    dedup_ttl_seconds: int = 86_400


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
        on_text: Optional[OnTextCallback] = None,
        on_thread_reply: Optional[OnThreadReplyCallback] = None,
        live_stream: bool = True,
        live_stream_bot_name: Optional[str] = None,
        live_stream_initial_text: str = "\u25b8 _thinking..._",
        stream_redis_client: Any = None,
        dedup_enabled: bool = True,
        dedup_ttl_seconds: int = 86_400,
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
            live_stream=live_stream,
            live_stream_bot_name=live_stream_bot_name,
            live_stream_initial_text=live_stream_initial_text,
            dedup_enabled=dedup_enabled,
            dedup_ttl_seconds=dedup_ttl_seconds,
        )
        # Optional Redis client for per-turn stream logging (XADD into a
        # Redis Stream). When ``None`` the live stream degrades to
        # Matrix-edit-only (the primary UX anyway); Redis is extra
        # observability for the rag-atlas / watcher consumers.
        self._stream_redis = stream_redis_client

        self._commands: dict[str, Command] = {}
        self._emoji_commands: dict[str, Command] = {}
        self._client = client  # may be None until _ensure_client
        self._owns_client = client is None
        self._stop_event: asyncio.Event = asyncio.Event()
        # room_id -> event_id of our pinned command menu, so we can
        # idempotently edit it instead of spamming new messages.
        self._menu_message_ids: dict[str, str] = {}

        # Chronicle mirror: XADDs every in/out Matrix message onto a
        # Redis stream consumed by the Chronicle ingester. Opt-in via
        # ``CHRONICLE_REDIS_URL`` — unset => ``None`` and all mirror
        # calls become silent no-ops. See ``chronicle_mirror`` module
        # for the zero-LLM guarantee.
        self._chronicle_redis = get_mirror_client()

        # Free-text handler. Invoked on any non-command inbound message so
        # services (movies/music/...) can wire an LLM that routes the text
        # to one of their registered tools. Silent-by-default: the
        # callback returns ``None`` to skip, or a ``str`` / ``dict`` in
        # the same shape a command handler returns.
        self._on_text: Optional[OnTextCallback] = on_text
        self._on_thread_reply: Optional[OnThreadReplyCallback] = on_thread_reply

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

    def set_on_thread_reply(self, handler: Optional[OnThreadReplyCallback]) -> None:
        self._on_thread_reply = handler

    def set_on_text(self, handler: Optional[OnTextCallback]) -> None:
        """Install or replace the free-text handler after construction.

        Useful when the handler needs ``CommandBot``'s final command list
        (built from registered modules) to compose its LLM tool schema.
        """
        self._on_text = handler

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
            resp = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("post_message failed room=%s: %s", room_id, exc)
            return None
        self._mirror_out(
            room_id=room_id,
            body=text,
            event_id=getattr(resp, "event_id", "") or "",
        )
        return resp

    async def post_recommendation(
        self,
        room_id: str,
        text: str,
        rec_id: str,
        *,
        html: Optional[str] = None,
        thread_root: Optional[str] = None,
        domain: Optional[str] = None,
        title: Optional[str] = None,
        redis_client: Any = None,
        ttl_seconds: int = 90 * 24 * 3600,
    ) -> Optional[str]:
        """Post a recommendation and record the ``rec_id`` -> event_id mapping.

        Behaves like :meth:`post_message` but additionally persists a small
        trail record in Redis so subsequent user reactions on the posted
        message can be correlated back to the recommendation that
        originated them.

        Redis keys written (best-effort, never raises):

        * ``skynet:rec_trail:{rec_id}`` — a hash with fields
          ``room_id``, ``event_id``, ``title``, ``suggested_at``,
          ``domain``. Set with a 90-day TTL.
        * ``skynet:vibe:rec_id:{event_id}`` — a plain string pointing
          back at ``rec_id``. Matches the key already consumed by
          ``skynet_agent.modules.vibe_capture.absorber._resolve_rec_id``
          so reactions on the event flow a ``linked_rec_id`` into the
          vibe engine.

        ``redis_client`` may be a sync or async redis client. If it is
        ``None``, the trail is silently skipped (the message still
        posts). This lets tests and degraded environments drop the
        tracking without losing the recommendation itself.

        Parameters
        ----------
        room_id, text, html, thread_root:
            Same as :meth:`post_message`.
        rec_id:
            Recommendation identifier the caller assigns. Required — an
            empty / missing rec_id skips the trail writes (use
            :meth:`post_message` instead in that case).
        domain, title:
            Stored on the trail hash for later inspection.
        redis_client:
            Optional redis client (sync or async). Passed in by the
            caller — CommandBot does not open its own connection.
        ttl_seconds:
            TTL on the trail hash. Defaults to 90 days.

        Returns the posted message's ``event_id`` (or ``None`` on send
        failure). The trail is only written when the send succeeds; we
        never record a rec_id for a message that was never delivered.
        """
        import time

        resp = await self.post_message(
            room_id,
            text,
            html=html,
            thread_root=thread_root,
        )
        if resp is None:
            return None
        event_id = getattr(resp, "event_id", None)
        if not event_id or not rec_id:
            return event_id

        if redis_client is None:
            return event_id

        trail = {
            "room_id": room_id,
            "event_id": event_id,
            "title": title or "",
            "domain": domain or "",
            "suggested_at": str(int(time.time())),
        }
        trail_key = f"skynet:rec_trail:{rec_id}"
        reverse_key = f"skynet:vibe:rec_id:{event_id}"
        try:
            _maybe = redis_client.hset(trail_key, mapping=trail)
            if asyncio.iscoroutine(_maybe):
                await _maybe
            _maybe = redis_client.expire(trail_key, int(ttl_seconds))
            if asyncio.iscoroutine(_maybe):
                await _maybe
            _maybe = redis_client.set(reverse_key, rec_id, ex=int(ttl_seconds))
            if asyncio.iscoroutine(_maybe):
                await _maybe
        except Exception as exc:
            logger.warning(
                "rec_trail write failed rec_id=%s event=%s: %s",
                rec_id,
                event_id,
                exc,
            )
        return event_id

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

    @staticmethod
    def _mxid_to_user(mxid: str) -> str:
        """Turn ``@sanscfs:matrix.sanscfs.dev`` -> ``sanscfs``.

        Used to fill the ``user`` label in Chronicle envelopes so the
        query layer can filter by identity without parsing full MXIDs.
        See ``project_chat_identity_fix``: user-originated rows belong
        under ``user=sanscfs`` and bot-originated rows under
        ``user=skynet-<component>``; everything downstream keys off that
        distinction.
        """
        if not mxid:
            return ""
        # "@local:host" -> "local"; fall back to the raw string if the
        # shape is unfamiliar (don't raise, this is observability glue).
        local = mxid.split(":", 1)[0] if ":" in mxid else mxid
        return local.lstrip("@")

    def _mirror_in(self, event: Any, room_id: str) -> None:
        """Chronicle-mirror an inbound ``m.room.message`` immediately.

        Called before ``_dispatch`` so the raw text lands in Chronicle
        even if a buggy command handler crashes, and critically BEFORE
        any LLM / analyzer touches the body. Never raises.
        """
        if self._chronicle_redis is None:
            return
        body = getattr(event, "body", "") or ""
        if not body:
            return
        sender = getattr(event, "sender", "") or ""
        event_id = getattr(event, "event_id", "") or ""
        ts_ms = getattr(event, "server_timestamp", None)
        ts = (ts_ms / 1000.0) if isinstance(ts_ms, (int, float)) else time.time()
        mirror_message(
            self._chronicle_redis,
            direction="in",
            room_id=room_id,
            sender=sender,
            body=body,
            event_id=event_id,
            ts=ts,
            user=self._mxid_to_user(sender),
        )

    def _mirror_out(
        self,
        *,
        room_id: str,
        body: str,
        event_id: str = "",
    ) -> None:
        """Chronicle-mirror an outbound reply right after ``room_send`` succeeded.

        ``event_id`` may be empty when nio's response didn't carry one;
        Chronicle accepts that — the ``direction=out`` + ``user`` +
        ``ts`` + ``body`` tuple is still enough to reconstruct the
        transcript. Never raises.
        """
        if self._chronicle_redis is None or not body:
            return
        mirror_message(
            self._chronicle_redis,
            direction="out",
            room_id=room_id,
            sender=self.config.user_id,
            body=body,
            event_id=event_id or "",
            ts=time.time(),
            user=self.config.bot_name or self._mxid_to_user(self.config.user_id),
        )

    async def handle_text_event(self, room: Any, event: Any) -> None:
        """Route an ``m.room.message`` / ``RoomMessageText`` to a command."""
        if getattr(event, "sender", None) == self.config.user_id:
            return  # never dispatch our own messages

        room_id = getattr(room, "room_id", None) or getattr(event, "room_id", None)

        # Cluster-wide event dedup. ``Recreate`` deploy strategy + slow
        # pod startup leave a 30s+ window where Matrix queues events;
        # the new pod's first /sync then replays the backlog and we
        # process each event N times across consecutive restarts. With
        # the silent-turn redact path (d0171fa), each replay turns into
        # a "Message deleted" stub in Element. Redis SETNX on the event
        # id collapses those replays into one real processing per turn,
        # transparent to all downstream callbacks.
        if not await self._claim_event(event):
            logger.debug("dedup: skipping replayed event %s", getattr(event, "event_id", "?"))
            return

        # Chronicle-mirror EVERY inbound text, not just !commands.
        # This is the "raw transcript" layer — we want the full record
        # before any LLM / analyzer sees the message body.
        if room_id:
            self._mirror_in(event, room_id)

        body = getattr(event, "body", "") or ""

        # Detect Matrix thread replies (m.relates_to rel_type=m.thread).
        thread_root_id: Optional[str] = None
        source = getattr(event, "source", None)
        if isinstance(source, dict):
            relates = source.get("content", {}).get("m.relates_to", {})
            if isinstance(relates, dict) and relates.get("rel_type") == "m.thread":
                thread_root_id = relates.get("event_id") or None

        parsed = parse_command_line(body, prefix=self.config.command_prefix)
        if parsed is None:
            if thread_root_id and self._on_thread_reply is not None and room_id and body.strip():
                if await self._run_free_text(
                    event,
                    room_id=room_id,
                    thread_root=thread_root_id,
                    body=body,
                    is_thread_reply=True,
                ):
                    return
            if self._on_text is not None and room_id and body.strip():
                await self._run_free_text(
                    event,
                    room_id=room_id,
                    thread_root=thread_root_id,
                    body=body,
                    is_thread_reply=False,
                )
            return

        cmd_name, args = parsed
        if not room_id:
            return

        await self._dispatch(cmd_name, args, room_id=room_id, event=event)

    async def _claim_event(self, event: Any) -> bool:
        """Try to mark an event as processed; return False if it was already.

        Implementation: ``SET key NX EX dedup_ttl`` against the configured
        Redis client. The key is keyed by ``bot_name`` + Matrix event_id
        so a single homeserver event is processed exactly once *across
        the entire pod lifetime + restarts*. When dedup is disabled
        (no ``stream_redis_client`` passed to the bot, or no
        ``event_id`` on the event), this falls open to the legacy
        "process every event" behaviour.
        """
        redis = self._stream_redis
        if redis is None or not self.config.dedup_enabled:
            return True
        event_id = getattr(event, "event_id", "") or ""
        if not event_id:
            return True
        key = f"skynet:bot:{self.config.bot_name}:seen:{event_id}"
        try:
            res = redis.set(key, "1", nx=True, ex=self.config.dedup_ttl_seconds)
            if asyncio.iscoroutine(res):
                res = await res
        except Exception as exc:  # noqa: BLE001
            logger.debug("dedup SETNX failed (open-fail): %s", exc)
            return True
        # redis-py: returns True when set, None when key existed.
        # Some async clients return ``b"OK"`` / ``"OK"`` / ``True``.
        return bool(res)

    async def _run_free_text(
        self,
        event: Any,
        *,
        room_id: str,
        thread_root: Optional[str],
        body: str,
        is_thread_reply: bool,
    ) -> bool:
        """Invoke the free-text handler (optionally under a live stream).

        Returns ``True`` when a reply was produced (used by the caller to
        short-circuit the fallback from ``on_thread_reply`` to ``on_text``).
        """
        handler = self._on_thread_reply if is_thread_reply else self._on_text
        if handler is None:
            return False

        async def _invoke() -> Any:
            if is_thread_reply:
                # thread-reply signature: (event, thread_root, body)
                assert thread_root is not None
                return await handler(event, thread_root, body)  # type: ignore[misc]
            return await handler(event, body, thread_root=thread_root)  # type: ignore[misc]

        # Non-streaming path keeps the pre-streaming behaviour intact so
        # anyone opting out via ``live_stream=False`` sees the same one-shot
        # send path that predates streaming.
        if not self.config.live_stream or self._client is None:
            try:
                result = await _invoke()
            except Exception as exc:
                logger.exception("free-text handler raised: %s", exc)
                return False
            if result is None:
                return False
            await self._send_result(room_id, result, thread_root=thread_root)
            return True

        bot_name = self.config.live_stream_bot_name or self.config.bot_name or "Skynet"
        stream = AsyncLiveStream(
            nio_client=self._client,
            room_id=room_id,
            thread_root=thread_root,
            bot_name=bot_name,
            redis_client=self._stream_redis,
            initial_text=self.config.live_stream_initial_text,
        )

        async with stream:
            tok = current_live_stream.set(stream)
            try:
                try:
                    result = await _invoke()
                except Exception as exc:
                    logger.exception("free-text handler raised: %s", exc)
                    await stream.emit(
                        EventType.ERROR,
                        f"{type(exc).__name__}: {exc}"[:300],
                        force_edit=True,
                    )
                    return False
            finally:
                current_live_stream.reset(tok)

            if result is None:
                # Stay silent — collapse the live placeholder into a short
                # "done" header so the user sees the turn closed without
                # an extra message in the room.
                await stream.complete()
                return False

            final_text, final_html = self._result_to_text_html(result)
            await stream.complete(final_text=final_text, html_body=final_html)

            if final_text:
                self._mirror_out(
                    room_id=room_id,
                    body=final_text,
                    event_id=stream.event_id or "",
                )
            return True

    @staticmethod
    def _result_to_text_html(result: Any) -> tuple[Optional[str], Optional[str]]:
        """Normalise the variety of shapes a free-text handler may return."""
        if result is None:
            return None, None
        if isinstance(result, str):
            return result, None
        if isinstance(result, dict):
            text = result.get("text")
            html = result.get("html")
            if isinstance(text, str) and text:
                return text, html if isinstance(html, str) else None
            return None, None
        return str(result), None

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

    async def _send_result(self, room_id: str, result: Any, *, thread_root: Optional[str] = None) -> Optional[str]:
        """Send result to room. Returns sent event_id or None."""
        if result is None:
            return None
        if isinstance(result, str):
            return await self._reply(room_id, result, thread_root=thread_root)
        if isinstance(result, dict):
            text = result.get("text") or ""
            html = result.get("html")
            tr = result.get("thread_root") or thread_root
            return await self._reply(room_id, text, html=html, thread_root=tr)
        logger.warning("unknown handler return type: %r", type(result))
        return None

    async def _reply(
        self,
        room_id: str,
        text: str,
        *,
        html: Optional[str] = None,
        thread_root: Optional[str] = None,
    ) -> Optional[str]:
        """Send a message and return the sent event_id (or None on failure)."""
        assert self._client is not None
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
        }
        if html is None:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html_lib.escape(text).replace("\n", "<br/>")
        else:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html
        if thread_root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
            }
        try:
            resp = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            logger.warning("room_send failed room=%s: %s", room_id, exc)
            return None
        event_id = getattr(resp, "event_id", "") or ""
        self._mirror_out(room_id=room_id, body=text, event_id=event_id)
        return event_id or None

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
        event_id = getattr(resp, "event_id", None)
        self._mirror_out(
            room_id=room_id,
            body=menu.get("text", "") or "",
            event_id=event_id or "",
        )
        return event_id

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
