"""Demo: end-to-end match() pipeline with the packaged v2 prototypes.

Uses the offline hash embedder (same one the test suite uses) so the
demo can run without Ollama / Qdrant. Output is illustrative, NOT
semantic quality -- the hash embedder is non-semantic by design. For a
real semantic demo, wire up ``skynet_embedding.async_embed`` against
the Ollama endpoint on the Mac.

Run::

    uv run --package skynet-vibe python packages/skynet-vibe/scripts/demo_match.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math

from skynet_vibe import PrototypeRegistry, VibeEngine, VibeStore


def _hash_embed(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        b = digest[i % len(digest)]
        values.append((b / 255.0) - 0.5)
    norm = sum(x * x for x in values) ** 0.5
    return [x / norm for x in values] if norm else values


class _NullQdrant:
    async def upsert(self, *a, **kw):
        return {"status": "ok"}

    async def search(self, *a, **kw):
        return []

    async def get_point(self, *a, **kw):
        return None

    async def set_payload(self, *a, **kw):
        return {"status": "ok"}

    async def count(self, *a, **kw):
        return 0

    async def scroll(self, *a, **kw):
        return [], None


async def main() -> None:
    async def embedder(text: str) -> list[float]:
        return _hash_embed(text)

    store = VibeStore(_NullQdrant(), collection="demo")
    prototypes = PrototypeRegistry(embedder)
    engine = VibeEngine(
        store=store,
        prototypes=prototypes,
        embedder=embedder,
        llm_client=lambda _p: "unused",
    )

    # Load the packaged v2 bank + auto-calibrate τ.
    prototypes.start_warmup()
    assert await prototypes.wait_ready(timeout=60.0)
    print(f"prototypes: {len(prototypes.names())}  tau={prototypes.tau:.4f}  "
          f"H_max={math.log2(len(prototypes.names())):.3f} bits")
    print()

    events = [
        # A phrase plucked from a training prototype -> should accept.
        "крок за кроком",
        # Off-axis text that matches no prototype well -> should reject.
        "aslkdjf qpwoei zxcvbm lajsdhf",
    ]
    for text in events:
        result = await engine.match(text)
        top3 = sorted(result.softmax_probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"INPUT: {text!r}")
        print(f"  winner={result.winner!r}  confidence={result.confidence:.3f}")
        print(f"  entropy_bits={result.entropy_bits:.3f}  accepted={result.accepted}")
        print(f"  softmax top3={json.dumps(top3, ensure_ascii=False)}")
        print()

    # Pathological example: a uniform embedder (every text -> same vector).
    # Cosines all equal -> entropy = H_max -> accepted must be False.
    async def uniform_embedder(text: str) -> list[float]:
        return [1.0] * 16

    store2 = VibeStore(_NullQdrant(), collection="demo2")
    prototypes2 = PrototypeRegistry(uniform_embedder)
    engine2 = VibeEngine(
        store=store2,
        prototypes=prototypes2,
        embedder=uniform_embedder,
        llm_client=lambda _p: "unused",
    )
    prototypes2.start_warmup()
    await prototypes2.wait_ready(timeout=60.0)
    bad = await engine2.match("уніформне повідомлення — всі косинуси рівні")
    print("PATHOLOGICAL (uniform-cosine) INPUT:")
    print(f"  winner={bad.winner!r}  confidence={bad.confidence:.3f}")
    print(f"  entropy_bits={bad.entropy_bits:.3f}  accepted={bad.accepted}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
