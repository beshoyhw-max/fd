"""
Processing Pipeline — End-to-End Invoice Processing Orchestrator.

Orchestrates: Ingest → Extract → Detect → Store → Queue

Supports:
- Configurable concurrency via asyncio.Semaphore
- Document type routing (text PDF fast path vs scanned standard path)
- Progress callbacks for Streamlit progress bars
- Error handling and retry logic
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from src.config import Config
from src.models import (
    DocumentType,
    ExtractedFields,
    Invoice,
    ProcessingResult,
    ReviewStatus,
)
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.image_extractor import ImageExtractor
from src.extraction.vision_extractor import VisionExtractor, ExtractionError
from src.extraction.ocr_fallback import OCRFallback, OCRError
from src.fraud.engine import FraudEngine
from src.storage.file_store import FileStore

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end invoice processing pipeline.

    Handles: Ingestion → Extraction → Fraud Detection → Storage → Review Queue.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()
        self._config.ensure_data_dirs()

        self._pdf_extractor = PDFExtractor()
        self._image_preprocessor = ImageExtractor()
        self._vision_extractor = VisionExtractor(self._config)
        self._ocr_fallback = OCRFallback(self._config)
        self._fraud_engine = FraudEngine(self._config)
        self._store = FileStore(self._config)

        self._semaphore = asyncio.Semaphore(self._config.concurrency)

    @property
    def store(self) -> FileStore:
        """Access the file store."""
        return self._store

    # ── Single Invoice Processing ─────────────────────────────

    async def process_file(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[ProcessingResult]:
        """
        Process a single invoice file end-to-end.

        The LLM determines if the document contains one or multiple invoices.

        Args:
            file_path: Path to the invoice file (PDF or image).
            progress_callback: Optional callback(stage_name, progress_pct).

        Returns:
            List of ProcessingResult (one per invoice detected). Use results[0] for
            backward compatibility with code expecting a single result.
        """
        async with self._semaphore:
            return await self._process_single(file_path, progress_callback)

    async def _process_single(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[ProcessingResult]:
        """Internal single-file processing implementation."""
        start_time = time.time()
        errors: List[str] = []

        # Generate invoice ID
        invoice_id = self._generate_id()
        source_file = os.path.basename(file_path)

        def _progress(stage: str, pct: float):
            if progress_callback:
                progress_callback(stage, pct)

        _progress("Initializing", 0.0)

        # ── Step 1: Copy source file to data/invoices/ ────────
        try:
            dest_path = self._config.invoices_dir / source_file
            if not dest_path.exists():
                shutil.copy2(file_path, str(dest_path))
        except Exception as e:
            logger.warning(f"Failed to copy source file: {e}")

        # ── Step 2: Ingest — classify and extract content ─────
        _progress("Ingesting document", 0.1)

        ext = Path(file_path).suffix.lower()
        page_images = []  # Initialize before try block for scope safety
        invoice = Invoice(
            invoice_id=invoice_id,
            source_file=source_file,
        )

        try:
            if ext == ".pdf":
                ingestion = self._pdf_extractor.classify_and_extract(file_path)
                doc_type = ingestion["document_type"]
                invoice.document_type = DocumentType(doc_type)
                invoice.page_count = ingestion["page_count"]
                invoice.pdf_path = file_path

                if doc_type == "text_pdf":
                    invoice.raw_text = ingestion["text"]
                    page_images = []
                else:
                    page_images = ingestion["images"]
                    # Save image paths for forensic reference
                    saved_paths = self._save_page_images(invoice_id, page_images)
                    invoice.image_paths = saved_paths
            else:
                # Direct image file
                invoice.document_type = DocumentType.IMAGE
                img = self._image_preprocessor.load_and_preprocess(file_path)
                page_images = [img]
                saved_paths = self._save_page_images(invoice_id, page_images)
                invoice.image_paths = saved_paths

        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            errors.append(f"Ingestion error: {e}")
            return self._build_error_result(
                invoice_id, source_file, errors, start_time
            )

        # ── Step 3: Extract structured fields ─────────────────
        _progress("Extracting fields", 0.3)

        extracted_invoices: List[ExtractedFields] = []

        try:
            if invoice.document_type == DocumentType.TEXT_PDF and invoice.raw_text:
                # Fast path: send text to LLM for structured parsing
                extracted_invoices = await self._vision_extractor.extract_from_text(invoice.raw_text)
            elif page_images:
                # Standard path: send images to Vision LLM
                try:
                    extracted_invoices = await self._vision_extractor.extract_from_images(page_images)
                except ExtractionError:
                    # Fallback to OCR (returns single ExtractedFields, wrap in list)
                    logger.info("Vision LLM failed, falling back to OCR")
                    _progress("OCR fallback", 0.4)
                    try:
                        ocr_fields = await self._ocr_fallback.extract_with_ocr(page_images)
                        extracted_invoices = [ocr_fields] if ocr_fields else []
                    except OCRError as e:
                        logger.error(f"OCR fallback also failed: {e}")
                        errors.append(f"Extraction error: Vision LLM and OCR both failed")
                        extracted_invoices = []
            else:
                errors.append("No content available for field extraction")

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            errors.append(f"Extraction error: {e}")
            extracted_invoices = []

        # If no invoices were detected, create one with empty fields
        if not extracted_invoices:
            extracted_invoices = [ExtractedFields()]

        # ── Step 4 & 5: Process each detected invoice ─────────
        _progress("Running fraud checks", 0.5)

        results: List[ProcessingResult] = []
        elapsed = time.time() - start_time

        # Get historical data for cross-invoice checks
        try:
            historical = self._store.get_historical_invoices(
                self._config.duplicate_lookback_days
            )
        except Exception:
            historical = []

        # Get page image for forensic checks
        first_page_image = page_images[0] if page_images else None

        # Get OCR data for font consistency
        ocr_data = None
        if first_page_image and invoice.document_type != DocumentType.TEXT_PDF:
            try:
                ocr_data = self._ocr_fallback.get_per_char_confidence(first_page_image)
            except Exception:
                pass

        for idx, fields in enumerate(extracted_invoices):
            invoice_for_check = Invoice(
                invoice_id=f"{invoice_id}_{idx + 1}" if len(extracted_invoices) > 1 else invoice_id,
                source_file=source_file,
                document_type=invoice.document_type,
                page_count=invoice.page_count,
                extracted_fields=fields,
                raw_text=invoice.raw_text,
                image_paths=invoice.image_paths,
                pdf_path=invoice.pdf_path,
            )

            try:
                fraud_result = self._fraud_engine.run_all_checks(
                    invoice=invoice_for_check,
                    historical_invoices=historical,
                    page_image=first_page_image,
                    ocr_data=ocr_data,
                )
            except Exception as e:
                logger.error(f"Fraud detection failed for invoice {idx + 1}: {e}")
                from src.models import FraudResult
                fraud_result = FraudResult()

            # Determine review status
            if fraud_result.any_triggered:
                review_status = ReviewStatus.PENDING
            else:
                review_status = ReviewStatus.AUTO_APPROVED

            result = ProcessingResult(
                invoice_id=invoice_for_check.invoice_id,
                source_file=source_file,
                document_type=invoice.document_type,
                extracted_fields=fields,
                fraud_result=fraud_result,
                review_status=review_status,
                processing_time_seconds=round(time.time() - start_time, 2),
                errors=errors.copy() if idx == 0 else [],
            )

            try:
                self._store.save_result(result)
            except Exception as e:
                logger.error(f"Failed to save result: {e}")

            results.append(result)

            logger.info(
                f"Processed {source_file} [invoice {idx + 1}/{len(extracted_invoices)}] — "
                f"vendor={fields.vendor_name}, total={fields.grand_total}, "
                f"score={fraud_result.aggregate_score}, status={review_status.value}"
            )

        _progress("Complete", 1.0)

        # Return list of results (or single result for backward compatibility)
        return results

    # ── Batch Processing ──────────────────────────────────────

    async def process_batch(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[ProcessingResult]:
        """
        Process multiple invoice files concurrently.

        Each file may contain one or multiple invoices - the LLM decides.
        All extracted invoices are returned as a flat list.

        Args:
            file_paths: List of file paths to process.
            progress_callback: Optional callback(filename, current, total).

        Returns:
            List of ProcessingResult objects (one per invoice detected).
        """
        total = len(file_paths)

        async def process_with_tracking(i: int, path: str):
            if progress_callback:
                progress_callback(os.path.basename(path), i + 1, total)
            return await self.process_file(path)

        tasks = [
            process_with_tracking(i, path)
            for i, path in enumerate(file_paths)
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        final_results: List[ProcessingResult] = []
        for i, r in enumerate(batch_results):
            if isinstance(r, Exception):
                logger.error(f"Batch item {i} failed: {r}")
                final_results.append(self._build_error_result(
                    self._generate_id(),
                    os.path.basename(file_paths[i]),
                    [str(r)],
                    time.time(),
                ))
            elif isinstance(r, list):
                # Each file may return multiple invoices
                final_results.extend(r)
            else:
                # Single result (backward compatibility)
                final_results.append(r)

        return final_results

    # ── Helpers ────────────────────────────────────────────────

    def _generate_id(self) -> str:
        """Generate a unique invoice ID."""
        ts = time.strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"INV-{ts}-{short_uuid}"

    def _save_page_images(
        self, invoice_id: str, images: List[Image.Image]
    ) -> List[str]:
        """Save page images to the invoices directory."""
        paths = []
        for i, img in enumerate(images):
            filename = f"{invoice_id}_page{i+1}.png"
            path = str(self._config.invoices_dir / filename)
            try:
                img.save(path)
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to save page image: {e}")
        return paths

    def _build_error_result(
        self,
        invoice_id: str,
        source_file: str,
        errors: List[str],
        start_time: float,
    ) -> ProcessingResult:
        """Build a ProcessingResult for a failed processing attempt."""
        from src.models import FraudResult
        return ProcessingResult(
            invoice_id=invoice_id,
            source_file=source_file,
            processing_time_seconds=round(time.time() - start_time, 2),
            errors=errors,
            fraud_result=FraudResult(),
            review_status=ReviewStatus.PENDING,
        )
