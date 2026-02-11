"""Document chunking — split text, markdown, and PDF files into overlapping chunks."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A piece of a document."""

    text: str
    metadata: dict = field(default_factory=dict)
    index: int = 0


class DocumentChunker:
    """Split documents into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ──

    def chunk_file(self, path: str) -> list[Chunk]:
        """Load and chunk a file. Supports .txt, .md, and .pdf."""
        p = Path(path)
        ext = p.suffix.lower()
        filename = p.name

        if ext == ".pdf":
            text = self._read_pdf(p)
        elif ext in (".md", ".markdown"):
            text = p.read_text(encoding="utf-8")
        elif ext in (".txt", ".text", ".rst", ".csv", ".log"):
            text = p.read_text(encoding="utf-8")
        else:
            text = p.read_text(encoding="utf-8")

        base_meta = {"source": str(p), "filename": filename, "filetype": ext}
        return self.chunk_text(text, base_meta)

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split raw text into overlapping chunks."""
        metadata = metadata or {}
        text = self._clean(text)

        if not text:
            return []

        # Try semantic splitting first (by paragraphs), fall back to character splitting
        paragraphs = self._split_paragraphs(text)
        chunks = self._merge_paragraphs(paragraphs)

        result = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                result.append(Chunk(
                    text=chunk_text.strip(),
                    metadata={**metadata, "chunk_index": i},
                    index=i,
                ))
        return result

    # ── Internal ──

    def _clean(self, text: str) -> str:
        """Normalize whitespace."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on double newlines, preserving paragraph structure."""
        parts = re.split(r"\n\n+", text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Merge paragraphs into chunks that respect size limits with overlap."""
        if not paragraphs:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk_size, split it by characters
            if para_len > self.chunk_size:
                # Flush current buffer
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                # Split the long paragraph
                chunks.extend(self._split_by_chars(para))
                continue

            # If adding this paragraph exceeds the limit, flush
            if current_len + para_len + 2 > self.chunk_size and current:
                chunks.append("\n\n".join(current))

                # Keep overlap — take trailing paragraphs that fit in overlap window
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) + 2 > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 2

                current = overlap_parts
                current_len = overlap_len

            current.append(para)
            current_len += para_len + 2

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_by_chars(self, text: str) -> list[str]:
        """Hard-split by character count with overlap. Used for oversized paragraphs."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF file. Requires PyPDF2 or pymupdf."""
        # Try PyPDF2 first
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            pass

        # Try pymupdf (fitz)
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        raise ImportError(
            "PDF support requires PyPDF2 or pymupdf. "
            "Install with: pip install PyPDF2  or  pip install pymupdf"
        )
