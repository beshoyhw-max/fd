"""
PDF Extraction Module for Invoice Fraud Detection System.

Handles:
- Document classification (text-based vs scanned PDF)
- Direct text extraction from text-based PDFs (fast path)
- Page-to-image rendering for scanned PDFs (standard path)
- PDF metadata extraction for forensics
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts content from PDF documents.

    Classifies PDFs as text-based or scanned:
    - Text-based: Direct text extraction via PyMuPDF (fast path).
    - Scanned: Page rendering to images for vision/OCR processing (standard path).
    """

    # Minimum character count to consider a page as having a text layer
    TEXT_THRESHOLD = 50

    def __init__(self):
        pass

    def classify_and_extract(
        self, pdf_path: str
    ) -> Dict:
        """
        Classify a PDF and extract content accordingly.

        Returns:
            {
                "document_type": "text_pdf" | "scanned_pdf",
                "page_count": int,
                "text": str | None,           # Full text (text PDFs only)
                "images": [PIL.Image, ...],    # Page images (scanned PDFs only)
                "metadata": {...},             # PDF metadata dict
                "pdf_path": str,
            }
        """
        pdf_path = str(pdf_path)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {pdf_path} — {e}")
            raise ValueError(f"Cannot open PDF file: {e}")

        try:
            page_count = len(doc)
            metadata = self._extract_metadata(doc)
            has_text = self._has_text_layer(doc)

            if has_text:
                # Fast path — direct text extraction
                text = self._extract_text(doc)
                logger.info(f"Text PDF detected: {pdf_path} ({page_count} pages, {len(text)} chars)")
                return {
                    "document_type": "text_pdf",
                    "page_count": page_count,
                    "text": text,
                    "images": [],
                    "metadata": metadata,
                    "pdf_path": pdf_path,
                }
            else:
                # Standard path — render pages to images
                images = self._render_pages(doc)
                logger.info(f"Scanned PDF detected: {pdf_path} ({page_count} pages)")
                return {
                    "document_type": "scanned_pdf",
                    "page_count": page_count,
                    "text": None,
                    "images": images,
                    "metadata": metadata,
                    "pdf_path": pdf_path,
                }
        finally:
            doc.close()

    def _has_text_layer(self, doc: fitz.Document) -> bool:
        """
        Check if the PDF has a selectable text layer.

        Returns True if enough text is found in at least one page.
        """
        for page in doc:
            text = page.get_text("text").strip()
            if len(text) >= self.TEXT_THRESHOLD:
                return True
        return False

    def _extract_text(self, doc: fitz.Document) -> str:
        """Extract all text from a text-based PDF."""
        pages_text = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(pages_text)

    def _render_pages(
        self, doc: fitz.Document, dpi: int = 300
    ) -> List[Image.Image]:
        """
        Render each page of a PDF to a PIL Image.

        Args:
            doc: Open PyMuPDF document.
            dpi: Rendering resolution (default 300 for quality OCR).

        Returns:
            List of PIL Images, one per page.
        """
        images = []
        zoom = dpi / 72  # 72 is the default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img.convert("RGB"))

        return images

    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """
        Extract PDF metadata for forensic analysis.

        Returns:
            Dictionary with creator, producer, dates, etc.
        """
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),       # Application that created the PDF
            "producer": meta.get("producer", ""),     # PDF library used
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
            "page_count": len(doc),
            "format": meta.get("format", ""),
            "encryption": meta.get("encryption", None),
        }

    def extract_metadata_only(self, pdf_path: str) -> Dict:
        """Extract only PDF metadata (lightweight, no rendering)."""
        try:
            doc = fitz.open(pdf_path)
            metadata = self._extract_metadata(doc)
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata: {pdf_path} — {e}")
            return {}
