"""Content extractor (GATE 2 input side): fetched HTML → structured items.

``extract(html, *, url, settings)`` turns bytes/str HTML into an
``ExtractionResult``: page ``title``/``domain`` metadata, ``text_items`` (one
``Item`` per top-level ``<p>``/``<li>`` of the main content, kind=text,
status always "ok"), ``image_refs`` (``ImageRef`` candidates for Task 7's
image pipeline), ``warnings`` (data, never error codes) and
``analyzable_count``. No fetching happens here — Task 7 fetches image bytes
through the Task 3 fetcher; this module only reads HTML.

Main-content resolution, in the prescribed order:

1. explicit ``<article>`` or ``<main>`` — the first match in document order;
2. otherwise the BeautifulSoup "readability" fallback on the boilerplate-
   stripped tree: the subtree with the highest paragraph-density score;
3. neither possible (no paragraph anywhere) → the stripped page root scopes
   image collection without inventing text items.

The paragraph-density score of a subtree sums, over its descendant p/li
blocks, the normalized block text length in characters — a paragraph-mass
metric that preserves containment: a container always outscores or ties its
children. Ties resolve deterministically to the first-scoring element, which
is the nearest (smallest enclosing) ancestor of the earliest-discovered
unit; among disjoint tied siblings that is the first in document order. A
pure text/total ratio would let a container lose to its own child and
fragment itemization.

Boilerplate — the exact brief list ``<nav> <header> <footer> <aside>
<script> <style> <template> <noscript>`` plus HTML comments — is removed
from the whole document BEFORE scoring and selection, so the fallback never
sees it, and content-internal boilerplate (e.g. an ``<aside>`` inside the
article) is stripped too.

Caps are enforced as data, not exceptions:

- text: UTF-8 byte total over the joined item texts (``"\n\n"`` separators
  included) against ``Settings.text_cap_bytes``; items are admitted wholly in
  document order ("skip truncation" — no mid-item cutting); the first item
  that no longer fits ends collection, everything discovered after is
  dropped, and exactly one ``"text_cap_truncated"`` warning (the Task 12
  token) is emitted.
- images: ``Settings.max_images`` per page; refs are kept in document order,
  the rest dropped, one ``"image_cap_exceeded"`` warning. ``data:`` URIs and
  candidates without an http(s) URL are filtered before the cap, so they can
  neither occupy a slot nor trip it.

Deferred extraction problems surface as extraction warnings only — the
extractor never invents item statuses (extraction only assigns "ok"); a
block shorter than ``Settings.block_min_chars`` is simply skipped, silently.

Relative image URLs are absolutized against the scanned ``url`` (Task 7
needs absoluteness). The SSRF story is unchanged: unfriendly candidates like
protocol-relative hosts resolve into ordinary URLs that Task 7 still fetches
through the fully guarded ``fetch()``, so a hostile hop is rejected there,
never here. When ``url`` is not given, unusable relative candidates are
dropped rather than carried.

Labels are deterministic and simple, never embedding long text: kept text
blocks count independently as ``"Paragraph N"`` / ``"List item N"`` (emitted
blocks only — skipped short blocks do not burn a number); image labels use
the element's ``alt`` when present and ≤ 60 chars (the DB's ScanItem.label
comment carries alt text "when present"), otherwise ``"Image N"``.

``source_ref`` locates a payload in the source document:
``{scope}:{tag}[i]`` with ``scope`` the winning element's name
(article/main/body/div/…) and ``[i]`` the 0-based document-order ordinal of
that tag inside the scope — stable even when neighbors are filtered.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag

from app.settings import Settings
from app.settings import settings as _app_settings
from pipeline.types import ImageRef, Item

# The exact boilerplate list from the brief + controller ruling 1, removed
# before any scoring or itemization. HTML comments are handled separately.
_BOILERPLATE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "template", "noscript"]
_UNIT_TAGS = ["p", "li"]  # the itemizable block vocabulary of the brief
# Elements the density fallback may not pick: page/document roots (the page
# itself is not a content subtree — "[document]" is bs4's BeautifulSoup
# object, which subclasses Tag) and unit blocks (a plain <p> under body is a
# unit, not a scope).
_NON_SCOPE_TAGS = {"html", "body", "p", "li", "[document]"}
_ALT_LABEL_MAX_CHARS = 60  # longer alts get the "Image N" fallback label
_SEPARATOR_BYTES = 2  # the "\n\n" between joined item texts under the cap

TEXT_CAP_WARNING = "text_cap_truncated"  # matches the Task 12 warning token
IMAGE_CAP_WARNING = "image_cap_exceeded"


@dataclass
class ExtractionResult:
    """Everything extraction produced for one page (the Task 12 seam)."""

    title: str | None
    domain: str | None
    text_items: list[Item] = field(default_factory=list)
    image_refs: list[ImageRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analyzable_count: int = 0


def _normalize(text: str) -> str:
    """Strip ends and collapse every whitespace run to one space (ruling 5)."""
    return " ".join(text.split())


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    """Remove the boilerplate tags + HTML comments from the whole document.

    Ruling 1: this happens BEFORE any density scoring, so the same rules
    apply regardless of how the main content is later resolved — including
    boilerplate nested inside the winning subtree.
    """
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()


def _top_level_units(scope: Tag | BeautifulSoup) -> list[Tag]:
    """Document-order p/li units of ``scope``, skipping nested duplicates.

    A unit is top-level when no other p/li element inside the scope subtree
    is its strict ancestor — a ``<p>`` inside an ``<li>`` belongs to the list
    item's text and must not become a second item.
    """
    inside = {id(element) for element in scope.find_all(True)}
    inside.add(id(scope))
    units: list[Tag] = []
    for unit in scope.find_all(_UNIT_TAGS):
        nearest = unit.find_parent(_UNIT_TAGS)
        # Skip when a unit ancestor exists INSIDE the scope (its text is
        # already carried by that ancestor's own item); a matching ancestor
        # above the scope does not dedupe (it never becomes an item either).
        if nearest is None or id(nearest) not in inside:
            units.append(unit)
    return units


def _is_scopable(element: Any) -> bool:
    """True for a Tag the density fallback may pick as the content scope."""
    return isinstance(element, Tag) and element.name not in _NON_SCOPE_TAGS


def _select_scope(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    """Explicit article/main first; else highest paragraph-density subtree.

    Scoring attributes each top-level unit's normalized length to every unit
    ancestor that is a legal scope candidate, nearest ancestor first. With
    strict ``>`` comparison the first-registered (nearest) ancestor wins
    equal-score ties; among disjoint tied siblings that is the first in
    document order (units register in document order). With no paragraphs at
    all the page root is returned so images still get a scope.
    """
    explicit = soup.find(["article", "main"])
    if explicit is not None:
        return explicit

    root: Tag | BeautifulSoup = soup.body if soup.body is not None else soup
    units = _top_level_units(root)
    if not units:
        return root
    scores: dict[int, tuple[Tag, int]] = {}
    for unit in units:
        length = len(_normalize_unit_text(unit))
        if not length:
            continue
        for ancestor in unit.parents:  # nearest ancestor first
            if not _is_scopable(ancestor):
                continue
            entry = scores.get(id(ancestor))
            scores[id(ancestor)] = (ancestor, (entry[1] if entry else 0) + length)
    if not scores:
        return root
    best: tuple[Tag, int] | None = None
    for entry in scores.values():
        if best is None or entry[1] > best[1]:  # strict > keeps the earliest
            best = entry
    return best[0] if best is not None else root


def _normalize_unit_text(unit: Tag) -> str:
    """One normalized text blob for a p/li unit (ruling 5)."""
    return _normalize(unit.get_text(" "))


def _itemize_text(scope: Tag | BeautifulSoup, settings: Settings) -> tuple[list[Item], list[str]]:
    """Turn the scope's top-level units into Items, enforcing the two rules.

    Blocks shorter than ``block_min_chars`` (after normalization, in
    characters) are skipped silently. The 60 kB cap counts UTF-8 bytes of the
    joined texts (2-byte ``"\n\n"`` separators included); on the first item
    that no longer fits, collection stops, everything discovered after is
    dropped, and exactly one ``text_cap_truncated`` warning is emitted.
    """
    items: list[Item] = []
    warnings: list[str] = []
    scope_name = _scope_name(scope)
    seen: dict[str, int] = {"p": 0, "li": 0}  # per-tag doc-order ordinals
    emitted: dict[str, int] = {"p": 0, "li": 0}  # kept blocks only → labels
    total_bytes = 0
    for unit in _top_level_units(scope):
        tag = unit.name
        seen[tag] += 1
        text = _normalize_unit_text(unit)
        if len(text) < settings.block_min_chars:
            continue  # blocks shorter than the minimum are skipped, silently
        extra = len(text.encode("utf-8"))
        separator = _SEPARATOR_BYTES if items else 0
        if total_bytes + separator + extra > settings.text_cap_bytes:
            warnings.append(TEXT_CAP_WARNING)  # exactly one, then stop
            break  # keep items discovered first; drop the rest
        total_bytes += separator + extra
        emitted[tag] += 1
        word = "List item" if tag == "li" else "Paragraph"
        items.append(
            Item(
                kind="text",
                label=f"{word} {emitted[tag]}",
                source_ref=f"{scope_name}:{tag}[{seen[tag] - 1}]",
                text=text,
            )
        )
    return items, warnings


def _parse_srcset(value: str) -> list[tuple[str, int | None]]:
    """Parse a srcset attribute into (url, width-or-None) entries in order.

    The HTML comma split is naive about ``data:`` URLs (their payload may
    contain commas), so any chunk starting with ``data:`` is skipped, and a
    chunk whose first token is no plausible URL (no scheme, no ``/`` and no
    ``.`` — a base64 comma-residue such as ``BBBB 10w``) is dropped as
    debris. Widths come from ``w`` descriptors only; ``x`` descriptors keep
    ``None`` like plain srcs. Redundant with real parsers on ordinary
    attributes; the simplification trades exotic encoded payloads for zero
    garbage candidates.
    """
    entries: list[tuple[str, int | None]] = []
    for chunk in value.split(","):
        tokens = chunk.strip().split()
        if not tokens:
            continue
        url = tokens[0]
        if url.lower().startswith("data:"):
            continue
        if not any(mark in url for mark in ("/", ".")) and "://" not in url:
            continue  # base64 comma-residue, not a URL
        width: int | None = None
        if len(tokens) > 1 and tokens[1].endswith("w"):
            digits = tokens[1][:-1]
            if digits.isascii() and digits.isdecimal():
                width = int(digits)
        entries.append((url, width))
    return entries


def _absolute_candidate(candidate: str, base_url: str | None) -> str | None:
    """Absolutize one image candidate against the scanned URL; drop unusable.

    ``#fragments`` (they point at the page itself) and ``data:`` URIs are
    dropped up front; without a base URL only already-absolute http(s)
    candidates survive, because Task 7 can fetch nothing from a bare
    relative URL. The http(s) scheme check is the extractor's only URL
    verdict — bigger host/IP dangers are the fetcher's business (Task 7
    goes through the guarded channel).
    """
    candidate = candidate.strip()
    if not candidate or candidate.startswith("#") or candidate.lower().startswith("data:"):
        return None
    absolute = candidate
    if base_url:
        try:
            absolute = urllib.parse.urljoin(base_url, candidate)
        except ValueError:
            return None
    parts = urllib.parse.urlsplit(absolute)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return absolute


def _image_label(alt: str, fallback_index: int) -> str:
    """Alt text when useful (ruling 4: simple, no long text), else Image N."""
    alt = _normalize(alt)
    if alt and len(alt) <= _ALT_LABEL_MAX_CHARS:
        return alt
    return f"Image {fallback_index}"


def _collect_images(
    scope: Tag | BeautifulSoup, base_url: str | None, settings: Settings
) -> tuple[list[ImageRef], list[str]]:
    """Document-order ImageRefs within the scope, capped at max_images.

    Each surviving ``<img>`` carries ``src`` first (width None — Task 7's
    "last entry when no widths" then resolves to the last srcset entry, or
    to ``src`` when srcset is absent outright), followed by its srcset
    entries in attribute order. ``data:`` URIs are excluded BEFORE the cap:
    they can neither occupy a slot nor trip ``image_cap_exceeded``. Every
    img element burns a scope ordinal, so source_refs stay stable even when
    an element yields no candidates.
    """
    refs: list[ImageRef] = []
    warnings: list[str] = []
    scope_name = _scope_name(scope)
    ordinal = 0
    for img in scope.find_all("img"):
        index = ordinal
        ordinal += 1
        candidates: list[tuple[str, int | None]] = []
        src = img.get("src")
        if isinstance(src, str):
            resolved = _absolute_candidate(src, base_url)
            if resolved is not None:
                candidates.append((resolved, None))
        for url, width in _parse_srcset(img.get("srcset") or ""):
            resolved = _absolute_candidate(url, base_url)
            if resolved is not None:
                candidates.append((resolved, width))
        if not candidates:
            continue
        if len(refs) >= settings.max_images:
            warnings.append(IMAGE_CAP_WARNING)  # exactly one, then stop
            break  # keep images discovered first; drop the rest
        alt = img.get("alt")
        alt_text = alt if isinstance(alt, str) else ""
        refs.append(
            ImageRef(
                label=_image_label(alt_text, len(refs) + 1),
                url_candidates=candidates,
                source_ref=f"{scope_name}:img[{index}]",
            )
        )
    return refs, warnings


def _page_title(soup: BeautifulSoup) -> str | None:
    """``<title>`` — else the og:title meta — else None; normalized."""
    title = soup.find("title")
    if title is not None:
        text = _normalize(title.get_text())
        if text:
            return text
    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta is not None:
        content = meta.get("content")
        if isinstance(content, str):
            text = _normalize(content)
            if text:
                return text
    return None


def _domain(url: str | None) -> str | None:
    """Hostname of the scanned URL (lowercase, port-less), else None."""
    if not url:
        return None
    try:
        hostname = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return None
    return hostname.lower() if hostname else None


def _scope_name(scope: Tag | BeautifulSoup) -> str:
    """The scope's naming half of every source_ref; fragments report "body"."""
    name = getattr(scope, "name", None)
    return "body" if not name or name == "[document]" else name


def extract(
    html: bytes | str,
    *,
    url: str | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Extract main-content text items and image refs from fetched HTML.

    ``html`` is raw payload as ``FetchResult.content`` (bytes: the page's own
    charset declaration drives decoding, so latin-1 fixtures decode) or str.
    ``url`` should be the fetch's ``final_url``: it absolutizes relative
    image candidates and names ``domain``. Callers without ``settings`` run
    on the boot Settings singleton; tests pass explicit ``Settings()``.
    """
    settings = settings if settings is not None else _app_settings
    soup = BeautifulSoup(html if html else "", "html.parser")
    _strip_boilerplate(soup)

    title = _page_title(soup)
    domain = _domain(url)
    scope = _select_scope(soup)
    text_items, text_warnings = _itemize_text(scope, settings)
    image_refs, image_warnings = _collect_images(scope, url, settings)

    return ExtractionResult(
        title=title,
        domain=domain,
        text_items=text_items,
        image_refs=image_refs,
        warnings=text_warnings + image_warnings,
        analyzable_count=len(text_items) + len(image_refs),
    )
