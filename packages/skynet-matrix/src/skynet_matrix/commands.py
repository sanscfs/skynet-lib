"""Command dataclass and parser for ``CommandBot``.

A ``Command`` is a single registered slash-command entry: name,
description, hint for arguments, optional emoji (for reaction-based
invocation), and the handler coroutine.

Argument parsing is deliberately simple: it splits on whitespace but
respects double-quoted substrings, so ``!note "hello world" tag1`` is
parsed as ``["hello world", "tag1"]``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

# A command handler receives the raw nio event (duck-typed as ``Any`` so
# we don't impose nio as a type-time dependency on callers) and the
# parsed argument list.  It may return:
#   * ``str``   — sent as plain text + HTML-formatted reply
#   * ``dict``  — ``{"text": str, "html": str, "thread": bool}`` shape
#   * ``None``  — silent (handler already did any side effects)
HandlerReturn = Optional[str] | dict
HandlerCoro = Callable[[Any, list[str]], Awaitable[HandlerReturn]]


@dataclass
class Command:
    """A single registered bot command."""

    name: str
    description: str
    handler: HandlerCoro
    args_hint: Optional[str] = None
    emoji: Optional[str] = None
    # Extra metadata consumers can put into the state event payload
    # without us having to add fields for every future knob.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state_entry(self) -> dict[str, Any]:
        """Render this command as an entry for the state event payload."""
        entry: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.args_hint:
            entry["args_hint"] = self.args_hint
        if self.emoji:
            entry["emoji"] = self.emoji
        if self.metadata:
            entry["metadata"] = self.metadata
        return entry


def parse_command_line(
    body: str,
    prefix: str = "!",
) -> Optional[tuple[str, list[str]]]:
    """Parse a raw message body into ``(command_name, args)``.

    Returns ``None`` if the body does not start with ``prefix``, is empty
    after the prefix, or can't be tokenized.

    The parser uses ``shlex.split`` with ``posix=True`` so quoted
    arguments (``"hello world"``) are preserved as a single token.  On
    malformed quoting we fall back to a plain whitespace split so users
    never get an "error parsing your command" reply for something like
    a stray quote.
    """
    if not body:
        return None
    body = body.lstrip()
    if not body.startswith(prefix):
        return None
    stripped = body[len(prefix):].strip()
    if not stripped:
        return None

    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        tokens = stripped.split()

    if not tokens:
        return None

    name, *args = tokens
    return name, args
