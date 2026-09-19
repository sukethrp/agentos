"""Landing HTML must paint text with JavaScript disabled.

Invariant: Vercel "Ready" is not "the page works". After the v0.4.0 docs cut,
mocha served docs/index.html — a meta-refresh and location.replace back to
itself — so production was blank while the dashboard file still rendered.
This check reads the file vercel.json says it deploys, and refuses a stub
that redirects mocha to mocha.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard" / "index.html"
DOCS = ROOT / "docs" / "index.html"
VERCEL = ROOT / "vercel.json"
HERO_WORDS = ("Record", "Replay", "Diff", "Bisect")


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._in_h1 = 0
        self.body_chunks: list[str] = []
        self.h1_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag == "h1":
            self._in_h1 += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag == "h1" and self._in_h1:
            self._in_h1 -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        self.body_chunks.append(text)
        if self._in_h1:
            self.h1_chunks.append(text)


def _dashboard_html() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def _visible(html: str) -> VisibleTextParser:
    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()
    return parser


def test_vercel_json_deploys_dashboard() -> None:
    cfg = json.loads(VERCEL.read_text(encoding="utf-8"))
    assert cfg.get("outputDirectory") == "dashboard", (
        "vercel.json outputDirectory must be dashboard; mocha went blank "
        f"when the deployed index was docs/index.html instead. got={cfg.get('outputDirectory')!r}"
    )
    build = cfg.get("buildCommand")
    assert isinstance(build, str) and build.strip(), (
        "buildCommand must be a non-empty string. JSON null / empty skips "
        "the build and Vercel serves the project root, which on this repo "
        "is how docs/index.html became https://agentos-mocha.vercel.app/."
    )


def test_dashboard_h1_has_text_without_js() -> None:
    parser = _visible(_dashboard_html())
    h1 = " ".join(parser.h1_chunks).strip()
    assert h1, (
        "dashboard/index.html <h1 id='heroTagline'> is empty. The hero is "
        "filled only by JS, so a script error or a GSAP word-mask at "
        "yPercent 110 paints a blank heading."
    )
    missing = [word for word in HERO_WORDS if word not in h1]
    assert not missing, f"hero h1 missing {missing}: {h1!r}"


def test_dashboard_has_visible_body_copy() -> None:
    parser = _visible(_dashboard_html())
    body = " ".join(parser.body_chunks)
    assert len(body) > 800, (
        f"dashboard/index.html has too little visible text without JS "
        f"({len(body)} chars). A landing page that is only a shell until "
        "scripts run will ship blank when those scripts fail."
    )
    assert "pip install agentos-platform" in body
    assert "Bisect the commit that broke the run" in body


def test_dashboard_does_not_prehide_content() -> None:
    html = _dashboard_html()
    assert "gsap.set(revealSel, { opacity: 0" not in html, (
        "GSAP pre-hides .card/.install/.stat at opacity 0 before "
        "ScrollTrigger. If the batch never fires, the page stays empty."
    )
    assert "opacity: 0, y: 40" not in html, (
        "Scroll reveals must not start by setting opacity 0. Content has to "
        "be visible if the tween never completes."
    )
    for selector in ("html", "body", "main", r"\.hero"):
        assert re.search(
            rf"\b{selector}\s*\{{[^}}]*\bopacity\s*:\s*0\b", html
        ) is None, f"{selector} sets opacity:0 without a non-JS fallback"
        assert re.search(
            rf"\b{selector}\s*\{{[^}}]*\bvisibility\s*:\s*hidden\b", html
        ) is None, f"{selector} sets visibility:hidden without a non-JS fallback"


def test_dashboard_motion_init_is_try_caught() -> None:
    html = _dashboard_html()
    assert "Motion init failed; leaving content visible" in html, (
        "Motion init must be wrapped in try/catch that logs and leaves "
        "the already-rendered copy visible."
    )


def test_docs_stub_does_not_redirect_mocha_to_itself() -> None:
    html = DOCS.read_text(encoding="utf-8")
    assert "http-equiv=\"refresh\"" not in html.lower(), (
        "docs/index.html has a meta refresh to agentos-mocha.vercel.app. "
        "GitHub Pages source is /docs, and mocha was serving this file, so "
        "the refresh targeted the page it was already on. That is a blank loop."
    )
    assert "github.io" in html, (
        "GitHub Pages still needs a host-guarded redirect to mocha."
    )
    assert "agentos-mocha.vercel.app" in html
    assert re.search(
        r"github\.io[\s\S]{0,400}location\.replace\(\"https://agentos-mocha\.vercel\.app/\"\)",
        html,
    ), (
        "location.replace to mocha must sit behind a github.io hostname check. "
        "An unconditional replace blanks mocha when Vercel serves this stub."
    )


def main() -> int:
    tests = [
        test_vercel_json_deploys_dashboard,
        test_dashboard_h1_has_text_without_js,
        test_dashboard_has_visible_body_copy,
        test_dashboard_does_not_prehide_content,
        test_dashboard_motion_init_is_try_caught,
        test_docs_stub_does_not_redirect_mocha_to_itself,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"ok   {test.__name__}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
