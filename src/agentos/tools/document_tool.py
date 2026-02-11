"""Document Tool — let agents read text files, markdown, and PDFs.

Usage:
    from agentos.tools.document_tool import document_reader_tool, document_qa_tool
    agent = Agent(tools=[document_reader_tool(), document_qa_tool()])
"""

from __future__ import annotations

from agentos.core.tool import Tool
from agentos.core.multimodal import read_document


def document_reader_tool(max_chars: int = 50_000) -> Tool:
    """Create a tool that reads a document and returns its text content.

    Supported formats: .txt, .md, .markdown, .rst, .csv, .json, .log, .pdf
    """

    def read_doc(file_path: str) -> str:
        """Read a document file and return its text content.

        Args:
            file_path: Path to the document to read.
        """
        try:
            content = read_document(file_path, max_chars=max_chars)
            lines = content.count("\n") + 1
            chars = len(content)
            header = f"[Document: {file_path} | {lines} lines | {chars:,} chars]\n\n"
            return header + content
        except FileNotFoundError as e:
            return f"Document not found: {e}"
        except ValueError as e:
            return f"Unsupported format: {e}"
        except Exception as e:
            return f"Error reading document: {e}"

    return Tool(
        fn=read_doc,
        name="read_document",
        description=(
            "Read a text document (txt, md, pdf, csv, json, log) and return its content. "
            "Use this when the user wants to examine, summarize, or ask questions about "
            "a file. Provide the file path."
        ),
    )


def document_qa_tool(max_chars: int = 30_000) -> Tool:
    """Create a tool that reads a document and prepares it for question-answering.

    This tool returns the document content formatted with metadata so the LLM
    can directly answer questions about it.
    """

    def read_and_prepare(file_path: str, question: str = "Summarize the key points.") -> str:
        """Read a document and prepare context for answering a question about it.

        Args:
            file_path: Path to the document to analyze.
            question: The question to answer about the document.
        """
        try:
            content = read_document(file_path, max_chars=max_chars)
            return (
                f"=== Document: {file_path} ===\n"
                f"Question: {question}\n\n"
                f"--- Document Content ---\n"
                f"{content}\n"
                f"--- End Document ---\n\n"
                f"Based on the document above, answer the question: {question}"
            )
        except FileNotFoundError as e:
            return f"Document not found: {e}"
        except ValueError as e:
            return f"Unsupported format: {e}"
        except Exception as e:
            return f"Error reading document: {e}"

    return Tool(
        fn=read_and_prepare,
        name="analyze_document",
        description=(
            "Read a document and answer a question about it. Supports txt, md, pdf, csv, "
            "json, and log files. Use this when the user wants to understand, summarize, "
            "extract key points, or ask specific questions about a document."
        ),
    )
