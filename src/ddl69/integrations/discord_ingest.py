from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional


@dataclass
class DiscordMessage:
    id: str
    channel_id: str
    channel_name: str | None
    author_id: str | None
    author_name: str | None
    content: str | None
    created_at: str | None
    jump_url: str | None
    attachments: list[dict[str, Any]]
    embeds: list[dict[str, Any]]


def parse_channel_ids(value: str | Iterable[str]) -> list[int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    ids = []
    for part in parts:
        if not part:
            continue
        ids.append(int(part))
    return ids


def _serialize_embed(embed: Any) -> dict[str, Any]:
    try:
        return embed.to_dict()
    except Exception:
        return {"type": str(getattr(embed, "type", "unknown"))}


def _serialize_attachment(att: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(att, "id", "")),
        "filename": getattr(att, "filename", None),
        "content_type": getattr(att, "content_type", None),
        "size": getattr(att, "size", None),
        "url": getattr(att, "url", None),
    }


async def _fetch_messages(
    token: str,
    channel_ids: list[int],
    limit: int,
    after: Optional[datetime],
    before: Optional[datetime],
) -> list[dict[str, Any]]:
    try:
        import discord
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("discord.py not installed; pip install discord.py") from exc

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    results: list[dict[str, Any]] = []

    @client.event
    async def on_ready() -> None:
        for cid in channel_ids:
            channel = client.get_channel(cid)
            if channel is None:
                try:
                    channel = await client.fetch_channel(cid)
                except Exception:
                    continue

            if not hasattr(channel, "history"):
                continue

            messages: list[dict[str, Any]] = []
            async for msg in channel.history(
                limit=limit,
                after=after,
                before=before,
                oldest_first=True,
            ):
                author = getattr(msg, "author", None)
                messages.append(
                    DiscordMessage(
                        id=str(msg.id),
                        channel_id=str(cid),
                        channel_name=getattr(channel, "name", None),
                        author_id=str(getattr(author, "id", "")) if author else None,
                        author_name=str(author) if author else None,
                        content=getattr(msg, "content", None),
                        created_at=getattr(msg, "created_at", None).isoformat()
                        if getattr(msg, "created_at", None)
                        else None,
                        jump_url=getattr(msg, "jump_url", None),
                        attachments=[_serialize_attachment(a) for a in getattr(msg, "attachments", [])],
                        embeds=[_serialize_embed(e) for e in getattr(msg, "embeds", [])],
                    ).__dict__
                )

            results.append(
                {
                    "channel_id": str(cid),
                    "channel_name": getattr(channel, "name", None),
                    "messages": messages,
                }
            )

        await client.close()

    await client.start(token)
    return results


def pull_messages(
    token: str,
    channel_ids: list[int],
    limit: int = 200,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    return asyncio.run(_fetch_messages(token, channel_ids, limit, after, before))
