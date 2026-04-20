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


_DEFAULT_SYSTEM = """You help the user manage {scope}. \
The user writes in Ukrainian in a chat with a bot called ``{bot_name}``.

Decide ONE of these actions for each user message and return ONLY valid JSON (no markdown fences, no prose):

1. Call a tool:
   {{"tool": "<exact tool name>", "args": {{...}}, "reply": "<short Ukrainian acknowledgement shown to the user>"}}
2. Just reply (no tool fits but the user clearly expects a response):
   {{"reply": "<short Ukrainian reply>"}}
3. Stay silent (message is small-talk, meta, not addressed to the bot, or outside its scope):
   {{"silent": true}}

Guidelines:
- Prefer silence when in doubt. Better to miss a signal than to spam replies.
- Use tools only when the user's intent matches a tool's description; never invent tools.
- Tool arguments MUST satisfy the tool's input schema. Omit optional fields when unsure.
- Keep ``reply`` short — one or two sentences max. No emojis unless the user used one.

Available tools (JSON):
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

        system = self._render_system()
        try:
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
            return await self._call_tool(tool_name, decision)

        reply = decision.get("reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()
        return None

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


__all__ = ["ChatAgent", "ToolSchema", "ToolDispatch", "LLMCaller"]
