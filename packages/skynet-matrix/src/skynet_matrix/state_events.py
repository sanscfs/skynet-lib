"""Helpers for publishing the ``dev.sanscfs.bot_commands`` state event.

The state event advertises the list of slash-commands supported by a
bot so clients (and other bots / dashboards) can render a command menu
without having to hardcode it.

Schema (intentionally shaped so we can rename the ``type`` to the
official ``m.room.bot_commands`` once MSC4391 lands — every field name
matches):

.. code-block:: json

    {
      "type": "dev.sanscfs.bot_commands",
      "state_key": "@skynet-movies:matrix.sanscfs.dev",
      "content": {
        "bot_name": "Skynet Movies",
        "prefix": "!",
        "commands": [
          {"name": "list-watched-movies", "description": "...",
           "args_hint": "[limit]", "emoji": "\ud83c\udfac"}
        ]
      }
    }

The ``state_key`` is the bot's Matrix user_id; clients MAY merge events
across multiple state_keys to discover all bots in a room.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from skynet_matrix.commands import Command

if TYPE_CHECKING:  # pragma: no cover
    from nio import AsyncClient

logger = logging.getLogger("skynet_matrix.bot")


STATE_EVENT_TYPE = "dev.sanscfs.bot_commands"


def build_bot_commands_content(
    *,
    bot_name: str,
    prefix: str,
    commands: Iterable[Command],
) -> dict[str, Any]:
    """Render the ``content`` payload for the state event."""
    return {
        "bot_name": bot_name,
        "prefix": prefix,
        "commands": [c.to_state_entry() for c in commands],
    }


async def publish_bot_commands_state(
    client: "AsyncClient",
    room_id: str,
    *,
    bot_user_id: str,
    bot_name: str,
    prefix: str,
    commands: Iterable[Command],
    event_type: str = STATE_EVENT_TYPE,
) -> bool:
    """Write the ``dev.sanscfs.bot_commands`` state event into ``room_id``.

    Returns ``True`` on success.  Errors (missing PL, 403, network) are
    logged and swallowed — a bot can happily keep running with no
    advertised commands.
    """
    content = build_bot_commands_content(
        bot_name=bot_name,
        prefix=prefix,
        commands=commands,
    )

    try:
        resp = await client.room_put_state(
            room_id=room_id,
            event_type=event_type,
            content=content,
            state_key=bot_user_id,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "publish_bot_commands_state: room=%s error=%s",
            room_id,
            exc,
        )
        return False

    # nio returns either a RoomPutStateResponse or an error response.
    # We don't import the error classes here to keep optional typing
    # lightweight; duck-typed check is enough.
    if hasattr(resp, "event_id"):
        return True

    logger.warning(
        "publish_bot_commands_state: room=%s unexpected response=%r",
        room_id,
        resp,
    )
    return False
