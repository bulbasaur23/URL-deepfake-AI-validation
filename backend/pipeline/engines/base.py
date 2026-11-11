"""Engine contract (design §7.2) — the single seam every detector implements.

Adding a detector later means one class satisfying ``Engine`` plus one
``registry._ENGINE_SOURCES`` entry — no pipeline edits. Engines are pure
consumers of ``Item``: they never mutate the item; the Task 12 engine runner
owns ownership of results (appends ``EngineResult`` to ``Item.engine_results``).

``EngineFault`` is the contract for "this engine is broken on this item":
raising it inside ``analyze`` lets the runner degrade just that item and
still ship the scan (§7.3: engine faults never fail the scan).
"""

from __future__ import annotations

from typing import Protocol

from pipeline.types import EngineResult, Item


class EngineFault(Exception):
    """Raised inside ``analyze`` when the engine is broken on this item.

    The Task 12 engine runner catches it alongside ``TimeoutError`` /
    ``OSError`` / ``torch.OutOfMemoryError`` and marks the item degraded —
    never the whole scan.
    """


class Engine(Protocol):
    """Structural type for a detector engine (spec §7.2 shape, verbatim).

    Implementations may declare ``id``/``version`` as class attributes;
    ``analyze`` is async so the runner can ``asyncio.wait_for`` it against
    ``settings.engine_timeout_s`` (GATE 3 memory-cap call, §11.14).
    """

    id: str
    version: str

    def supports(self, item: Item) -> bool: ...

    async def analyze(self, item: Item) -> EngineResult: ...
