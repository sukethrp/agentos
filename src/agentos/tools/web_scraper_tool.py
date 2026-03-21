"""Web Scraper Tool — extract text content from any URL.

Uses httpx for HTTP requests and basic HTML tag stripping (no
BeautifulSoup dependency required, but uses it when available for
better extraction).
"""

from __future__ import annotations

import re

from agentos.core.tool import Tool


def web_scraper_tool(
    *,
    max_chars: int = 3000,
    timeout: float = 15.0,
) -> Tool:
    """Create a tool that scrapes text content from a URL."""

    def scrape_url(url: str) -> str:
        """Fetch a web page and return its text content."""
        import httpx

        resp = httpx.get(url, follow_redirects=True, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        try:
            from html.parser import HTMLParser

            class _TextExtractor(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self._parts: list[str] = []
                    self._skip = False

                def handle_starttag(self, tag: str, attrs: list) -> None:
                    if tag in ("script", "style", "noscript"):
                        self._skip = True

                def handle_endtag(self, tag: str) -> None:
                    if tag in ("script", "style", "noscript"):
                        self._skip = False

                def handle_data(self, data: str) -> None:
                    if not self._skip:
                        text = data.strip()
                        if text:
                            self._parts.append(text)

            parser = _TextExtractor()
            parser.feed(html)
            text = " ".join(parser._parts)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html)

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    return Tool(
        fn=scrape_url,
        name="web_scraper",
        description=(
            "Scrape text content from a URL. Returns the visible text "
            "of the page (HTML tags stripped). Provide the full URL."
        ),
        timeout_seconds=timeout,
    )
