"""
Text extraction from resume and JD files.
Supports .pdf, .docx, and .txt
"""

import os
import pdfplumber
from docx import Document


def extract_from_pdf(path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_from_docx(path: str) -> str:
    """Extract all text from a .docx file using python-docx."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_from_txt(path: str) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(path: str) -> str:
    """
    Detect file type and extract text.
    Raises ValueError for unsupported formats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = extract_from_pdf(path)
    elif ext == ".docx":
        text = extract_from_docx(path)
    elif ext == ".txt":
        text = extract_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return text.strip()


if __name__ == "__main__":
    # Quick manual test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.extract <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    text = extract_text(path)
    print(f"--- Extracted {len(text)} characters ---")
    print(text[:1000])