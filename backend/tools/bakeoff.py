"""Offline bake-off harness (Task 11) — throwaway spike tool, NOT a product feature.

Evaluates every registered candidate engine on the ``data/eval/`` fixture set and
prints one markdown results table for ``docs/bakeoff.md``.

Contracts consumed (unchanged seams):
- ``pipeline.engines.registry.get_by_id(engine_id, settings)`` — the two-arg
  dispatch-pinned signature (T8 review fix); construction loads real weights.
- ``Engine.supports/analyze`` + ``EngineResult.latency_ms`` (score/confidence/
  rationale are unused here — bake-off bands need the score only).
- ``app.settings.settings`` — bands (``band_unclear_min``/``band_synthetic_min``)
  and the per-item ``engine_timeout_s`` mirror of the Task-12 runner.

Accounting (ruling 3, pinned):
- verdict bands from settings: authentic < band_unclear_min, synthetic >=
  band_synthetic_min, unclear in between;
- accuracy = (human items < band_unclear_min + AI items >= band_synthetic_min) / N,
  where per-item EngineFault/timeout failures count as incorrect;
- FP-rate = human items >= band_synthetic_min / human N;
- unclear-rate = items in [band_unclear_min, band_synthetic_min) / N;
- mean latency_ms = mean of ``EngineResult.latency_ms`` over successful items;
  failed items carry no latency.

Resilience (ruling 8): a candidate that cannot be constructed becomes a single
"blocked" row with the failure reason (never a fabricated number); a per-item
fault degrades only that item (counted incorrect, disclosed); one candidate's
crash never aborts the run.

Usage (from ``backend/``)::

    .venv/bin/python -m tools.bakeoff                      # all candidates
    .venv/bin/python -m tools.bakeoff --candidates image-ai-smaug \
        --env-label overlay                                # pass 2 (T10 recipe)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.settings import settings
from pipeline.engines import registry
from pipeline.engines.base import Engine, EngineFault
from pipeline.types import EngineResult, Item

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "eval"

# All registered candidates (registry._ENGINE_SOURCES keys, hardwired for this
# spike so the tool stays independent of registry internals).
CANDIDATES = ("text-ai-roberta", "image-ai-vit", "image-ai-smaug")

# Truth side of the fixture tree (dirs under --data-dir).
TREE: dict[str, dict[str, tuple[str, str]]] = {
    "text": {"human": ("text/human", "human"), "ai": ("text/ai", "ai")},
    "image": {"human": ("images/real", "human"), "ai": ("images/ai", "ai")},
}

TEXT_SUFFIX = ".txt"
IMAGE_SUFFIX = ".jpg"


@dataclass
class ItemScore:
    """One fixture's outcome for one candidate."""

    label: str
    truth: str  # "human" (not AI-generated) | "ai"
    score: float | None
    latency_ms: int | None
    band: str | None  # authentic | unclear | synthetic (None when failed)
    error: str | None  # failure reason (per-item degrade, counts incorrect)


@dataclass
class CandidateReport:
    """One candidate's full run (or its blocked row)."""

    engine_id: str
    modality: str
    env_label: str
    blocked: str | None  # construct/load failure reason -> whole-candidate row
    items: list[ItemScore]

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def failed(self) -> list[ItemScore]:
        return [item for item in self.items if item.error is not None]


def modality_of(engine_id: str) -> str:
    """Candidate id prefix -> modality (must match the registry's modalities)."""
    prefix = engine_id.split("-", 1)[0]
    if prefix not in TREE:
        raise ValueError(f"cannot infer a modality from engine id {engine_id!r}")
    return prefix


def band_for(score: float) -> str:
    """Verdict band for one score, straight from the app's settings."""
    if score < settings.band_unclear_min:
        return "authentic"
    if score >= settings.band_synthetic_min:
        return "synthetic"
    return "unclear"


def load_items(modality: str, data_dir: Path) -> list[tuple[str, str, Item]]:
    """Build harness Items from pipeline.types for one modality's eval tree.

    Image items carry ABSOLUTE paths (the engines just ``Image.open(path)``, so
    any resolvable form works; absolute is cwd-independent).
    """
    suffix = TEXT_SUFFIX if modality == "text" else IMAGE_SUFFIX
    triples: list[tuple[str, str, Item]] = []
    for subtree, truth in TREE[modality].values():
        for path in sorted((data_dir / subtree).glob(f"*{suffix}")):
            item = (
                Item(
                    kind=modality,
                    label=path.stem,
                    source_ref=str(path),
                    text=path.read_text("utf-8"),
                )
                if modality == "text"
                else Item(
                    kind=modality,
                    label=path.stem,
                    source_ref=str(path),
                    image_path=str(path.resolve()),
                )
            )
            triples.append((path.stem, truth, item))
    return triples


def missing_truth_classes(modality: str, data_dir: Path) -> list[str]:
    """Truth classes whose fixture directory yielded no fixtures (degenerate eval tree).

    A wrong ``--data-dir`` or a deleted subtree silently scores nothing and
    would otherwise emit a fake ``measured | 0.00% (0/0)`` row; the caller
    turns any empty class into a blocked row instead.
    """
    suffix = TEXT_SUFFIX if modality == "text" else IMAGE_SUFFIX
    return [
        truth
        for subtree, truth in TREE[modality].values()
        if not any((data_dir / subtree).glob(f"*{suffix}"))
    ]


async def run_candidate(engine_id: str, env_label: str, data_dir: Path) -> CandidateReport:
    """Construct the candidate and score its modality's eval set (per-item degrade)."""
    modality = modality_of(engine_id)
    triples = load_items(modality, data_dir)
    missing = missing_truth_classes(modality, data_dir)
    if missing:
        # Degenerate eval tree: fail the candidate as a blocked row BEFORE
        # constructing the engine (no weight load on a wrong --data-dir).
        return CandidateReport(
            engine_id,
            modality,
            env_label,
            f"no fixtures found for class(es) {', '.join(missing)} under {data_dir}",
            [],
        )
    try:
        engine: Engine = registry.get_by_id(engine_id, settings)
    except Exception as exc:  # noqa: BLE001 — whole-candidate blocked row, never a fake number
        return CandidateReport(engine_id, modality, env_label, blocked_reason(exc), [])

    scored: list[ItemScore] = []
    for label, truth, item in triples:
        try:
            result: EngineResult = await asyncio.wait_for(
                engine.analyze(item), settings.engine_timeout_s
            )
        except EngineFault as exc:
            scored.append(ItemScore(label, truth, None, None, None, f"EngineFault: {exc}"))
        except TimeoutError:
            scored.append(
                ItemScore(
                    label, truth, None, None, None, f"timeout after {settings.engine_timeout_s}s"
                )
            )
        except Exception as exc:  # noqa: BLE001 — contract violation on this item; run continues
            scored.append(
                ItemScore(label, truth, None, None, None, f"unexpected {type(exc).__name__}: {exc}")
            )
        else:
            scored.append(
                ItemScore(
                    label, truth, result.score, result.latency_ms, band_for(result.score), None
                )
            )
    return CandidateReport(engine_id, modality, env_label, None, scored)


def blocked_reason(exc: Exception) -> str:
    """Compress a construct/load failure into a one-line row reason."""
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {first}"[:300]


def summarize(report: CandidateReport) -> dict[str, Any]:
    """The ruling-3 metrics for one candidate report."""
    successful = [item for item in report.items if item.error is None]
    human = [item for item in successful if item.truth == "human"]
    correct = sum(
        1
        for item in successful
        if (item.truth == "human" and item.score < settings.band_unclear_min)
        or (item.truth == "ai" and item.score >= settings.band_synthetic_min)
    )
    n_human_all = sum(1 for item in report.items if item.truth == "human")
    latencies = [item.latency_ms for item in successful if item.latency_ms is not None]
    mean_latency: float | None = sum(latencies) / len(latencies) if latencies else None
    # FP convention (ruling 3): failed items have no score (never FPs) but stay in the denominator.
    fp = sum(1 for item in human if item.score >= settings.band_synthetic_min)
    unclear = sum(1 for item in successful if item.band == "unclear")
    return {
        "accuracy": correct / report.n if report.n else 0.0,
        "correct": correct,
        "n": report.n,
        "fp_rate": fp / n_human_all if n_human_all else 0.0,
        "fp": fp,
        "n_human_all": n_human_all,
        "unclear_rate": unclear / report.n if report.n else 0.0,
        "mean_latency_ms": mean_latency,
        "failures": len(report.failed),
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def table_row(report: CandidateReport) -> str:
    if report.blocked is not None:
        return (
            f"| `{report.engine_id}` | {report.modality} | {report.env_label} | blocked | "
            f"— | — | — | — | {report.blocked} |"
        )
    stats = summarize(report)
    notes = (
        f"{stats['failures']} per-item failures (counted incorrect)" if stats["failures"] else ""
    )
    # Zero successful latencies (all items failed) means NO mean latency exists:
    # render "—" rather than a fabricated "0 ms" measured row.
    mean_cell = (
        f"{stats['mean_latency_ms']:.0f} ms" if stats["mean_latency_ms"] is not None else "—"
    )
    return (
        f"| `{report.engine_id}` | {report.modality} | {report.env_label} | measured | "
        f"{fmt_pct(stats['accuracy'])} ({stats['correct']}/{stats['n']}) | "
        f"{fmt_pct(stats['fp_rate'])} ({stats['fp']}/{stats['n_human_all']}) | "
        f"{fmt_pct(stats['unclear_rate'])} | {mean_cell} | {notes} |"
    )


def print_details(reports: list[CandidateReport]) -> None:
    """Per-item evidence lines (the numbers behind the table)."""
    for report in reports:
        print(f"\n### per-item: {report.engine_id} [{report.env_label}]")
        if report.blocked is not None:
            print(f"  BLOCKED: {report.blocked}")
            continue
        for item in report.items:
            if item.error is not None:
                print(f"  {item.label} truth={item.truth} FAILED: {item.error}")
            else:
                verdict = (
                    "correct"
                    if (
                        (item.truth == "human" and item.band == "authentic")
                        or (item.truth == "ai" and item.band == "synthetic")
                    )
                    else "miss"
                )
                print(
                    f"  {item.label} truth={item.truth} score={item.score:.2f} "
                    f"band={item.band} {verdict} latency={item.latency_ms}ms"
                )


def environment_footer() -> list[str]:
    """Runtime facts for the docs (GPU + versions; best-effort, never fatal)."""
    lines = []
    try:
        import torch

        lines.append(
            f"torch {torch.__version__}, torch.cuda.is_available()={torch.cuda.is_available()}"
        )
        if torch.cuda.is_available():
            lines.append(f"cuda device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        lines.append(f"torch probe failed: {exc}")
    try:
        import transformers

        lines.append(f"transformers {transformers.__version__}")
    except Exception:  # noqa: BLE001
        lines.append("transformers: import failed")
    try:
        import sentence_transformers

        lines.append(f"sentence-transformers {sentence_transformers.__version__}")
    except Exception:  # noqa: BLE001
        lines.append("sentence-transformers: import failed")
    return lines


async def main_async(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Task 11 offline bake-off (throwaway spike)")
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATES),
        help=f"comma-separated engine ids (default: all of {', '.join(CANDIDATES)})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"eval fixture root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--env-label",
        default="native",
        help="environment label for the table (native | overlay)",
    )
    args = parser.parse_args(argv)
    engine_ids = [part.strip() for part in args.candidates.split(",") if part.strip()]

    reports: list[CandidateReport] = []
    for engine_id in engine_ids:
        try:
            report = await run_candidate(engine_id, args.env_label, args.data_dir)
        except Exception as exc:  # noqa: BLE001 — a broken candidate must never kill the run
            report = CandidateReport(
                engine_id,
                modality_of(engine_id),
                args.env_label,
                f"harness-level failure: {blocked_reason(exc)}",
                [],
            )
        reports.append(report)

    print_details(reports)

    print(
        "\n| engine | modality | environment | status | accuracy | FP-rate (human) | unclear-rate | mean latency | notes |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for report in reports:
        print(table_row(report))

    print("\n[environment]")
    for line in environment_footer():
        print(f"  {line}")
    return 0


def main() -> int:
    return asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
