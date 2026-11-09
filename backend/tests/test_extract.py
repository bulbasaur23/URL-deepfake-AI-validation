"""Task 5: content extractor tests — HTML fixture strings, fully offline.

No fetching happens in this module: ``extract()`` is exercised directly with
BeautifulSoup-friendly fixture strings/bytes. The brief pins five tests here;
the supporting tests pin the controller rulings the brief left open (density
fallback + tie-break, image refs/srcset/data-URI exclusion, caps, title/domain
metadata) and Task 7's seams (``ImageRef`` lives in ``pipeline/types.py``).

Settings are built through ``Settings()`` defaults (no ``backend/.env``
required); the extractor-relevant fields are pinned explicitly so a stray
``.env`` cannot skew any assertion.
"""

from app.settings import Settings
from pipeline.extract import ExtractionResult, extract
from pipeline.types import ImageRef

_URL = "https://news.example.org/story/slug"

_LONG_PARAGRAPH = (
    "This opening paragraph describes the scan pipeline and carries enough "
    "ordinary filler text to comfortably pass the eighty character minimum "
    "in the content extractor tests"
)


def _mk_settings(**overrides: object) -> Settings:
    """Settings pinned to the extractor's spec defaults so backend/.env cannot skew."""
    values: dict[str, object] = {
        "text_cap_bytes": 61_440,
        "block_min_chars": 80,
        "max_images": 20,
    }
    values.update(overrides)
    return Settings(**values)


# --- brief-pinned test 1: boilerplate never reaches the items -------------------

_NAV_TEXT = (
    "Navigation boilerplate marker NAVMARKER link home about service pricing "
    "that browsers repeat on every single page of a site"
)
_HEADER_TEXT = (
    "Header boilerplate marker HEADERMARKER with a repeated masthead sentence "
    "so the assertion can prove it was excluded from the item list"
)
_FOOTER_TEXT = (
    "Footer boilerplate marker FOOTERMARKER copyright terms privacy sitemap "
    "repeated word for word on every page of the whole site"
)
_ASIDE_TEXT = (
    "Aside boilerplate marker ASIDEMARKER related posts widget with several "
    "links and enough filler text to pass the length gate"
)
_NOSCRIPT_TEXT = (
    "Noscript boilerplate marker NOSCRIPTMARKER please enable javascript to "
    "view this content because the site is fully dynamic"
)
_TEMPLATE_TEXT = (
    "Template boilerplate marker TEMPLATEMARKER cached card content for "
    "repeated layout usage on the listing pages of the site"
)

_BOILER_HTML = f"""
<html><head>
    <title>Boilerplate Fixture Title</title>
    <style>.nav {{ color: red; }} /* STYLEMARKER */</style>
    <script>var tracker = "SCRIPTMARKER";</script>
</head>
<body>
    <nav><p>{_NAV_TEXT}</p></nav>
    <header><p>{_HEADER_TEXT}</p></header>
    <aside><p>{_ASIDE_TEXT}</p></aside>
    <template><p>{_TEMPLATE_TEXT}</p></template>
    <noscript><p>{_NOSCRIPT_TEXT}</p></noscript>
    <article>
        <!-- structural comment COMMENTMARKER between blocks -->
        <p>Article paragraph one carries the actual analysis content with
        enough words to be safely itemizable by the extractor</p>
        <p>Article paragraph two completes the evidence text and must survive
        all boilerplate stripping in the fixture<!-- inline secret COMMENTMARKER --></p>
        <script>var inline = "SCRIPTMARKER";</script>
    </article>
    <footer><p>{_FOOTER_TEXT}</p></footer>
</body></html>
"""

_MARKERS = (
    "NAVMARKER",
    "HEADERMARKER",
    "FOOTERMARKER",
    "ASIDEMARKER",
    "SCRIPTMARKER",
    "STYLEMARKER",
    "COMMENTMARKER",
    "NOSCRIPTMARKER",
    "TEMPLATEMARKER",
)


def test_boilerplate_excluded() -> None:
    """nav/header/footer/aside/script/style/comments/noscript/template never leak."""
    result = extract(_BOILER_HTML, url=_URL, settings=_mk_settings())
    texts = [item.text or "" for item in result.text_items]
    joined = " ".join(texts)
    for marker in _MARKERS:
        assert marker not in joined, marker
    labels = [item.label for item in result.text_items]
    assert labels == ["Paragraph 1", "Paragraph 2"]
    assert result.title == "Boilerplate Fixture Title"
    assert result.domain == "news.example.org"
    assert result.warnings == []


# --- brief-pinned test 2: itemization, labels, normalization, doc order ---------

_P1 = (
    "Alpha paragraph one about generated headlines and detector engines with a "
    "couple more filler words so that it clears the eighty character minimum "
    "threshold in the extractor test"
)
_LI1 = (
    "first list item mentions rewriting and polishing and detection with more "
    "filler text to reach the minimum length for itemization"
)
_LI2 = (
    "second list item mentions another engine id and rounds the fixture out "
    "with a few more filler tokens for the length gate"
)
_P2 = (
    "second paragraph closes the piece with a short concluding sentence plus "
    "enough filler words to pass the same eighty character gate"
)
_PROBE = (
    "probe paragraph embedding awkward spacing that must collapse into single "
    "spaces by normalization before itemization happens"
)

_ITEMIZE_HTML = f"""
<html><head><title>Itemization Fixture</title></head><body>
<article>
    <p>{_P1}</p>
    <ul>
        <li>{_LI1}</li>
        <li>{_LI2}</li>
    </ul>
    <p>{_P2}</p>
    <p> {_PROBE.replace(" ", "     ")} </p>
</article>
</body></html>
"""


def test_paragraph_itemization_and_labels() -> None:
    """One item per top-level p/li, document order, simple labels, normalized text."""
    result = extract(_ITEMIZE_HTML, url=_URL, settings=_mk_settings())
    assert isinstance(result, ExtractionResult)  # the pinned Task 12 seam shape
    items = result.text_items
    assert [i.text for i in items] == [_P1, _LI1, _LI2, _P2, _PROBE]  # doc order
    assert [i.label for i in items] == [
        "Paragraph 1",
        "List item 1",
        "List item 2",
        "Paragraph 2",
        "Paragraph 3",
    ]
    assert [i.source_ref for i in items] == [
        "article:p[0]",
        "article:li[0]",
        "article:li[1]",
        "article:p[1]",
        "article:p[2]",
    ]
    for item in items:  # the closed Item contract for extractor output
        assert item.kind == "text"
        assert item.status == "ok"
        assert item.image_path is None and item.mime is None
        assert item.language is None and item.verdict is None
        assert item.confidence is None and item.engine_results == []
        assert item.text == (item.text or "").strip()
        assert "  " not in (item.text or "")
    assert result.warnings == []
    assert result.analyzable_count == len(items) + len(result.image_refs)


# --- brief-pinned test 3: short blocks skipped ----------------------------------

_SHORT_HTML = f"""
<html><body><article>
    <p>{"A" * 79}</p>
    <p>{"B" * 80}</p>
    <p>{"C" * 200}</p>
</article></body></html>
"""


def test_short_blocks_skipped() -> None:
    """Blocks under block_min_chars=80 chars are skipped; exactly 80 chars stays."""
    result = extract(_SHORT_HTML, url=_URL, settings=_mk_settings())
    texts = [i.text for i in result.text_items]
    assert texts == ["B" * 80, "C" * 200]  # 79 skipped; the <80 rule keeps 80
    assert [i.label for i in result.text_items] == ["Paragraph 1", "Paragraph 2"]
    assert result.warnings == []


# --- brief-pinned test 4: text cap keeps first-discovered, drops the rest --------

_CAP_A, _CAP_B, _CAP_C = "X" * 100, "Y" * 200, "Z" * 100
_CAP_HTML = (
    f"<html><body><article><p>{_CAP_A}</p><p>{_CAP_B}</p><p>{_CAP_C}</p></article></body></html>"
)
_HUGE_HTML = f"<html><body><article><p>{'W' * 70_000}</p><p>{'V' * 120}</p></article></body></html>"


def test_text_cap_truncates_with_warning() -> None:
    """60 kB cap: utf-8 bytes over joined items; keep first-discovered, one warning.

    With text_cap_bytes=250 the X block (100 B) fits, Y (200 B) overflows and is
    dropped together with everything discovered after it — the Z block is NOT
    resurrected later (ruling: keep items discovered first, drop the rest).
    Also checks the default 61,440 cap: one 70,000-byte block → zero items.
    """
    capped = extract(_CAP_HTML, url=_URL, settings=_mk_settings(text_cap_bytes=250))
    assert [i.text for i in capped.text_items] == [_CAP_A]  # B overflow → stop
    assert capped.warnings == ["text_cap_truncated"]  # exactly one warning
    joined = "\n\n".join(i.text or "" for i in capped.text_items)
    assert len(joined.encode("utf-8")) <= 250

    huge = extract(_HUGE_HTML, url=_URL, settings=_mk_settings())
    assert huge.text_items == []  # first block alone exceeds the default cap
    assert huge.warnings == ["text_cap_truncated"]


# --- brief-pinned test 5: no content → no items, no crash -----------------------

_NO_CONTENT_URL = "https://site.test/empty"


def test_no_content_yields_no_items_not_crash() -> None:
    """Empty/JS-only pages produce empty items + refs and never raise."""
    for html in (
        "<html><body><script>var t = 1;</script></body></html>",
        "",
        "<a href='/'>nav link</a>",
    ):
        result = extract(html, url=_NO_CONTENT_URL, settings=_mk_settings())
        assert result.text_items == []
        assert result.image_refs == []
        assert result.warnings == []
        assert result.analyzable_count == 0
        assert result.title is None
    still = extract("<html><body></body></html>", url=_NO_CONTENT_URL, settings=_mk_settings())
    assert still.domain == "site.test"
    assert still.text_items == [] and still.image_refs == []


# --- supporting: density fallback + deterministic tie-break ----------------------

_DENSITY_HTML = (
    "<html><body>"
    '<div class="thin"><p>' + "T" * 90 + "</p></div>"
    '<div class="rich"><p>' + "R" * 150 + "</p><p>" + "R" * 150 + "</p></div>"
    "</body></html>"
)
_TIE_HTML = (
    "<html><body>"
    '<div class="a"><p>' + "E" * 100 + "</p></div>"
    '<div class="b"><p>' + "E" * 100 + "</p></div>"
    "</body></html>"
)


def test_fallback_density_selects_richest_subtree() -> None:
    """No article/main → the subtree with the most paragraph text wins."""
    result = extract(_DENSITY_HTML, url=_URL, settings=_mk_settings())
    texts = [i.text for i in result.text_items]
    assert texts == ["R" * 150, "R" * 150]  # the rich div; the thin div excluded
    assert "T" * 90 not in texts
    assert [i.label for i in result.text_items] == ["Paragraph 1", "Paragraph 2"]
    assert [i.source_ref for i in result.text_items] == ["div:p[0]", "div:p[1]"]


def test_fallback_tie_breaks_to_document_order() -> None:
    """Equal scores resolve deterministically to the first subtree in doc order."""
    result = extract(_TIE_HTML, url=_URL, settings=_mk_settings())
    texts = [i.text for i in result.text_items]
    assert texts == ["E" * 100]  # only the first div is itemized
    assert result.warnings == []


# --- supporting: image refs (srcset, labels, data-URIs), cap, doc order ----------

_IMG_PAGE = "https://site.test/articles/story"
_ALT_TOO_LONG = "A" * 61  # one over the alt-label budget → "Image N" fallback

_IMG_HTML = f"""
<html><head><title>Images Fixture</title></head><body>
<article>
    <p>{_LONG_PARAGRAPH}</p>
    <img src="/hero.png" alt="A hero image">
    <img srcset=" {_IMG_PAGE.rsplit("/", 1)[0]}/sq.png 200w, {_IMG_PAGE.rsplit("/", 1)[0]}/lg.png 800w"
        src="{_IMG_PAGE.rsplit("/", 1)[0]}/default.png" alt="{_ALT_TOO_LONG}">
    <img src="data:image/png;base64,AAAA">
    <img srcset="data:image/png;base64,BBBB 10w, {_IMG_PAGE.rsplit("/", 1)[0]}/real.png 300w"
        src="spacer.png" alt="Data in srcset">
</article>
</body></html>
"""


def test_images_itemized_with_srcset_and_data_excluded() -> None:
    """document-order ImageRefs; data: URIs excluded; src-first candidate order."""
    result = extract(_IMG_HTML, url=_IMG_PAGE, settings=_mk_settings())
    refs = result.image_refs
    assert len(refs) == 3  # the data: <img> never appears
    assert all(isinstance(ref, ImageRef) for ref in refs)
    assert [ref.source_ref for ref in refs] == [
        "article:img[0]",
        "article:img[1]",
        "article:img[3]",
    ]
    assert refs[0].label == "A hero image"
    assert refs[0].url_candidates == [("https://site.test/hero.png", None)]
    assert refs[1].label == "Image 2"  # long alt → deterministic simple label
    assert refs[1].url_candidates == [
        ("https://site.test/articles/default.png", None),  # src first
        ("https://site.test/articles/sq.png", 200),
        ("https://site.test/articles/lg.png", 800),
    ]
    assert refs[2].label == "Data in srcset"
    assert refs[2].url_candidates == [
        ("https://site.test/articles/spacer.png", None),
        ("https://site.test/articles/real.png", 300),  # data: srcset entry dropped
    ]
    # text side untouched: 1 paragraph + 3 images = analyzable
    assert result.analyzable_count == len(result.text_items) + len(refs) == 4
    # downstream label sanity even for the ImageRef dataclass used by Task 7
    assert refs[0].source_ref != refs[1].source_ref


def test_image_cap_keeps_first_document_order() -> None:
    """max_images cap: keep images discovered first, drop the rest, one warning."""
    html = (
        "<html><body><article>"
        '<img src="/one.png" alt="first">'
        '<img src="/two.png" alt="second">'
        '<img src="data:image/png;base64,CCCC">'
        '<img src="/three.png" alt="third">'
        "</article></body></html>"
    )
    capped = extract(html, url=_URL, settings=_mk_settings(max_images=2))
    assert [ref.url_candidates[0][0] for ref in capped.image_refs] == [
        "https://news.example.org/one.png",
        "https://news.example.org/two.png",
    ]
    assert capped.warnings == ["image_cap_exceeded"]


# --- supporting: title fallbacks + domain ---------------------------------------

_META_OG_HTML = (
    "<html><head>"
    '<meta property="og:title" content="OG Title Here">'
    "</head><body><article><p>" + _LONG_PARAGRAPH + "</p></article></body></html>"
)


def test_title_og_fallback_and_domain() -> None:
    """title → og:title → None; domain is the scanned hostname, port stripped."""
    titled = extract(
        "<html><head><title>  Multi \n Space\tTitle </title></head><body>"
        "<article><p>" + _LONG_PARAGRAPH + "</p></article></body></html>",
        url="https://Example.COM:8443/page",
        settings=_mk_settings(),
    )
    assert titled.title == "Multi Space Title"
    assert titled.domain == "example.com"

    og = extract(_META_OG_HTML, url=_URL, settings=_mk_settings())
    assert og.title == "OG Title Here"

    bare = extract(_DENSITY_HTML, settings=_mk_settings())  # no url
    assert bare.title is None and bare.domain is None


# --- supporting: bytes input decodes via meta charset ---------------------------


def test_bytes_input_decodes_via_meta_charset() -> None:
    """FetchResult.content is bytes: extractor decodes via the page's meta charset."""
    body = ("café " * 18) + "and then the rest of this latin-1 encoded paragraph continues"
    html = (
        "<html><head><meta charset='latin-1'></head><body><article>"
        f"<p>{body}</p>"
        "</article></body></html>"
    ).encode("latin-1")
    result = extract(html, url=_URL, settings=_mk_settings())
    texts = [i.text for i in result.text_items]
    assert len(texts) == 1
    assert "café" in texts[0]


# --- supporting: nested units are not duplicated --------------------------------


def test_nested_units_not_duplicated() -> None:
    """A <p> inside an <li> is part of the list item, never a second item."""
    li_text = (
        "Outer list item paragraph text that alone would already pass the "
        "eighty character gate for itemization without any assistance"
    )
    p_text = (
        "Inner paragraph likewise long enough to pass the eighty character "
        "threshold without any trouble at all in this nesting fixture"
    )
    html = (
        f"<html><body><article><ul><li>{li_text}<p>{p_text}</p></li></ul></article></body></html>"
    )
    result = extract(html, url=_URL, settings=_mk_settings())
    items = result.text_items
    assert len(items) == 1  # the li only
    assert items[0].label == "List item 1"
    assert li_text in (items[0].text or "")
    assert p_text in (items[0].text or "")  # nested text carried inside
