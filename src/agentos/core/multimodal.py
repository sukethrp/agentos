"""Multi-modal utilities — image encoding, PDF text extraction, and vision API helpers.

Uses only Python built-ins + the openai SDK.  No external dependencies
like PyPDF2 or Pillow are required.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def encode_image(file_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
        )
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_media_type(file_path: str) -> str:
    """Return the MIME type for an image file (e.g. 'image/png')."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "image/png"


def is_url(path_or_url: str) -> bool:
    """Quick check whether a string is an HTTP(S) URL."""
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


# ---------------------------------------------------------------------------
# PDF text extraction (pure-Python, no external libs)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract readable text from a PDF using only Python built-ins.

    This handles the most common PDF encodings (uncompressed streams and
    FlateDecode / zlib-compressed streams).  It won't handle encrypted PDFs,
    image-only PDFs, or exotic encodings — for those, recommend a dedicated
    library.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    data = path.read_bytes()

    # Collect all stream…endstream blocks
    text_parts: list[str] = []
    stream_re = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
    flate_re = re.compile(rb"/FlateDecode")

    for match in stream_re.finditer(data):
        raw = match.group(1)
        # Try FlateDecode decompression first
        decoded: bytes | None = None
        try:
            decoded = zlib.decompress(raw)
        except zlib.error:
            # Not compressed — use raw bytes
            decoded = raw

        if decoded:
            # Extract text from content stream operators: Tj, TJ, '
            text = _extract_text_operators(decoded)
            if text.strip():
                text_parts.append(text.strip())

    result = "\n".join(text_parts)
    if not result.strip():
        # Fallback: grab any printable ASCII runs
        result = _extract_printable_text(data)

    return result if result.strip() else "(No extractable text found in this PDF)"


def _extract_text_operators(stream: bytes) -> str:
    """Parse PDF text operators (Tj, TJ, ') from a decoded content stream."""
    text = stream.decode("latin-1", errors="replace")
    parts: list[str] = []

    # Tj — show a single string:  (Hello World) Tj
    for m in re.finditer(r"\(([^)]*)\)\s*Tj", text):
        parts.append(_unescape_pdf(m.group(1)))

    # TJ — show an array of strings:  [(Hello ) -100 (World)] TJ
    for m in re.finditer(r"\[([^\]]*)\]\s*TJ", text):
        array_text = m.group(1)
        for s in re.finditer(r"\(([^)]*)\)", array_text):
            parts.append(_unescape_pdf(s.group(1)))

    # ' operator — move to next line and show string: (text) '
    for m in re.finditer(r"\(([^)]*)\)\s*'", text):
        parts.append(_unescape_pdf(m.group(1)))

    return " ".join(parts)


def _unescape_pdf(s: str) -> str:
    """Handle basic PDF string escapes like \\n, \\(, \\)."""
    return (
        s.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\\\", "\\")
    )


def _extract_printable_text(data: bytes) -> str:
    """Last-resort fallback: grab long runs of printable ASCII from the PDF."""
    text = data.decode("latin-1", errors="replace")
    # Filter to runs of >=20 printable characters
    runs = re.findall(r"[ -~]{20,}", text)
    # Remove PDF-internal lines that look like operators
    filtered = [
        r
        for r in runs
        if not r.strip().startswith("/")
        and "endobj" not in r
        and "endstream" not in r
        and "xref" not in r
    ]
    return "\n".join(filtered[:100])  # Cap to avoid massive output


# ---------------------------------------------------------------------------
# Document readers (text, markdown)
# ---------------------------------------------------------------------------

SUPPORTED_DOC_TYPES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}


def read_document(file_path: str, max_chars: int = 50_000) -> str:
    """Read a text-based document and return its content (truncated if large)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    if suffix not in SUPPORTED_DOC_TYPES:
        raise ValueError(
            f"Unsupported document type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_DOC_TYPES | {'.pdf'}))}"
        )

    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... [truncated at {max_chars:,} characters]"
    return content


# ---------------------------------------------------------------------------
# OpenAI Vision API helper
# ---------------------------------------------------------------------------

def analyze_image(
    image_path_or_url: str,
    prompt: str = "Describe this image in detail.",
    model: str = "gpt-4o",
    max_tokens: int = 1024,
) -> str:
    """Send an image to the OpenAI Vision API and return the analysis.

    *image_path_or_url* can be:
      - A local file path  (encoded as base64 and sent as a data URI)
      - An HTTPS URL        (sent directly)
    """
    client = OpenAI()

    if is_url(image_path_or_url):
        image_content: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": image_path_or_url},
        }
    else:
        b64 = encode_image(image_path_or_url)
        mime = image_media_type(image_path_or_url)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or "(No response from vision model)"
