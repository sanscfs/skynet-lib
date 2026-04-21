"""Free-text handler for ``CommandBot`` — LLM routes text to a tool.

Each recommender service (movies, music, ...) plugs one of these into
``CommandBot.on_text`` so the bot can respond to plain messages like
"подивився X, норм" by picking a registered tool (``mark_watched``,
``store_preference``, ...) instead of silently ignoring the message.

Design:

* **Tool schemas are provided by the caller**, not imported from
  ``skynet-mcp``, so this module stays a leaf dependency. A tool is a
  dict ``{"name": str, "description": str, "inputSchema": dict}``.
* **Dispatch is a callback**: ``tool_dispatch(name, args) -> Any``.
  The caller maps from the tool name to the actual handler (usually a
  ``ToolRegistry.get(name).handler`` call).
* **LLM is reached through ``skynet_providers.async_chat_completion``**
  with ``response_format={"type": "json_object"}`` — OpenRouter and
  Ollama both honour that, so the reply parses deterministically.
* **Silence is default**: if the LLM replies ``{"silent": true}`` or
  fails / returns invalid JSON, ``handle`` returns ``None`` and the
  bot sends nothing. Matches the project's "silence-is-default" rule.

The handler returns one of:

* ``None`` — do not reply (normal silent case, or any error).
* ``str`` — text to send to the room.
* ``dict`` — ``{"text": ..., "html": ...}`` shape ``CommandBot``
  already accepts from a command handler.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("skynet_matrix.chat_agent")

# A tool schema in the shape ``{"name", "description", "inputSchema"}``.
ToolSchema = dict[str, Any]
# ``(name, args) -> result`` (result is whatever the tool returns; str /
# dict / None). Coroutine-only; sync handlers should be adapted upstream.
ToolDispatch = Callable[[str, dict[str, Any]], Awaitable[Any]]
# Async adapter around ``/chat/completions``. Receives ``system`` and
# ``user`` strings and returns the raw assistant content.
LLMCaller = Callable[[str, str], Awaitable[str]]
# History-aware variant: also receives the pre-built messages list so the
# caller can pass a multi-turn context window to the LLM.
HistoryLLMCaller = Callable[[str, str, "list[dict[str, Any]]"], Awaitable[str]]


_DEFAULT_SYSTEM = """You help the user manage {scope}. \
User writes in Ukrainian, bot is ``{bot_name}``.

Return ONLY valid JSON — no markdown, no prose:
- Tool call:  {{"tool": "<name>", "args": {{...}}, "reply": "<1-2 sentence Ukrainian ack>"}}
- Reply only: {{"reply": "<short Ukrainian reply>"}}
- Silent:     {{"silent": true}}

Silence is default. Use tools only when intent clearly matches description. No emojis unless the user used one.

Available tools:
{tool_schemas}
"""


@dataclass
class ChatAgent:
    """LLM-driven adapter that turns free-text messages into tool calls.

    Instantiate once per service in ``main.py`` lifespan, pass
    ``agent.handle`` as ``CommandBot(on_text=...)`` or
    ``bot.set_on_text(agent.handle)``.
    """

    tools: list[ToolSchema]
    dispatch: ToolDispatch
    llm_call: LLMCaller
    bot_name: str = "assistant"
    scope: str = "your data"
    system_prompt: str = _DEFAULT_SYSTEM
    # Messages shorter than this skip the LLM entirely (cheap guard
    # against single-emoji reactions / "ok" noise). Set to 0 to disable.
    min_body_chars: int = 3
    # Ignore messages from self-identified bots (sender contains this).
    self_mxid: Optional[str] = None
    # Optional sender filter (e.g. ``{"@sanscfs:matrix.sanscfs.dev"}``)
    # — when non-empty, only those senders can trigger ``handle``.
    allowed_senders: set[str] = field(default_factory=set)
    # Guard against historical-message bursts: on first ``/sync`` the
    # Matrix client replays room backlog, which otherwise fires an LLM
    # call per stale message. Events older than this (seconds since the
    # ``ChatAgent`` was constructed) are silently dropped. Set to 0 to
    # disable the guard entirely.
    skip_older_than_seconds: int = 120
    # When set, a tool dispatch exception is surfaced to the user via
    # ``template.format(tool=..., exc=...)`` instead of silently dropping
    # the LLM-parsed intent. Default ``None`` preserves silence-is-default;
    # services that want the bot to acknowledge a failed action opt in.
    tool_error_template: Optional[str] = None
    # --- Optional conversation history -----------------------------------
    # When set, ChatAgent maintains per-room history in Redis and passes
    # it to the LLM so context carries across messages.
    # Provide all three or none; mixing is silently degraded.
    #
    # history_loader(room_id, thread_root) -> list of {"role","content"}
    # history_appender(room_id, role, content, thread_root) -> None
    # history_llm_call(system, user, messages) -> str
    #   (same as llm_call but receives pre-built multi-turn messages list)
    history_loader: Optional[Callable[[str, Optional[str]], list[dict[str, Any]]]] = None
    history_appender: Optional[Callable[[str, str, str, Optional[str]], None]] = None
    history_llm_call: Optional[HistoryLLMCaller] = None
    _started_at: float = field(default_factory=time.time, init=False, repr=False)

    async def handle(self, event: Any, body: str) -> Optional[Any]:
        """``CommandBot.on_text`` entry point. Never raises."""
        body = (body or "").strip()
        if len(body) < self.min_body_chars:
            return None
        sender = getattr(event, "sender", "") or ""
        if self.self_mxid and sender == self.self_mxid:
            return None
        if self.allowed_senders and sender not in self.allowed_senders:
            return None
        if not self.tools:
            logger.debug("chat_agent: no tools registered, staying silent")
            return None
        # Skip historical messages replayed by the first ``/sync`` after
        # boot. ``server_timestamp`` is milliseconds since epoch per nio.
        # We compare against both the bot start time (drops ALL backlog
        # events) and an absolute "stale message" cutoff (drops anything
        # surprisingly old even after bot has been up a while — could
        # happen if Matrix replays on reconnect).
        if self.skip_older_than_seconds > 0:
            ts_ms = getattr(event, "server_timestamp", None)
            if isinstance(ts_ms, (int, float)):
                ts_s = ts_ms / 1000.0
                now = time.time()
                if ts_s < self._started_at or (now - ts_s) > self.skip_older_than_seconds:
                    logger.debug(
                        "chat_agent: skipping stale event (age=%.1fs, started=%.1fs ago)",
                        now - ts_s,
                        now - self._started_at,
                    )
                    return None

        room_id: str = getattr(event, "room_id", "") or ""

        system = self._render_system()
        history: list[dict[str, Any]] = []
        if self.history_loader and room_id:
            history = self.history_loader(room_id, None)

        try:
            if history and self.history_llm_call:
                messages = [
                    {"role": "system", "content": system},
                    *history,
                    {"role": "user", "content": body},
                ]
                raw = await self.history_llm_call(system, body, messages)
            else:
                raw = await self.llm_call(system, body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent LLM call failed: %s", exc)
            return None

        decision = _parse_decision(raw)
        if decision is None:
            return None
        if decision.get("silent") is True:
            return None

        tool_name = decision.get("tool")
        if tool_name:
            response = await self._call_tool(tool_name, decision)
        else:
            reply = decision.get("reply")
            response = reply.strip() if isinstance(reply, str) and reply.strip() else None

        if self.history_appender and room_id and response is not None:
            self.history_appender(room_id, "user", body, None)
            resp_text = _render_result(response)
            if resp_text:
                self.history_appender(room_id, "assistant", resp_text, None)

        return response

    # -- Internals -------------------------------------------------------

    def _render_system(self) -> str:
        return self.system_prompt.format(
            bot_name=self.bot_name,
            scope=self.scope,
            tool_schemas=json.dumps(self.tools, ensure_ascii=False, indent=2),
        )

    async def _call_tool(self, tool_name: str, decision: dict[str, Any]) -> Optional[Any]:
        args = decision.get("args") or {}
        if not isinstance(args, dict):
            logger.warning("chat_agent: tool %r args not a dict: %r", tool_name, args)
            args = {}
        valid_names = {t.get("name") for t in self.tools}
        if tool_name not in valid_names:
            logger.warning("chat_agent: unknown tool %r", tool_name)
            reply = decision.get("reply")
            return reply if isinstance(reply, str) else None
        try:
            result = await self.dispatch(tool_name, args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent: tool %s failed: %s", tool_name, exc)
            if self.tool_error_template:
                try:
                    return self.tool_error_template.format(tool=tool_name, exc=type(exc).__name__)
                except Exception:  # noqa: BLE001 -- malformed template
                    logger.debug("chat_agent: tool_error_template format failed")
            return None

        # {"_self_reply": True}: tool posted its own Matrix message; suppress ack.
        if isinstance(result, dict) and result.get("_self_reply"):
            return None

        ack = decision.get("reply")
        ack = ack.strip() if isinstance(ack, str) and ack.strip() else ""
        result_text = _render_result(result)
        if ack and result_text and ack != result_text:
            return f"{ack}\n\n{result_text}"
        return ack or result_text or None


def _parse_decision(raw: str) -> Optional[dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    # Strip markdown fences just in case.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json\n"):
            raw = raw[5:]
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("chat_agent: non-JSON LLM reply: %r", raw[:200])
        return None
    if not isinstance(decision, dict):
        logger.debug("chat_agent: LLM reply not a JSON object: %r", raw[:200])
        return None
    return decision


def _render_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("result", "text", "message"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    return str(result)


__all__ = ["ChatAgent", "ToolSchema", "ToolDispatch", "LLMCaller", "HistoryLLMCaller"]
