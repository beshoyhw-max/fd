"""
Vision-Based Field Extraction for Invoice Fraud Detection System.

Sends invoice page images to a Vision LLM and extracts 10 structured
fields as JSON. Falls back to OCR if the LLM fails.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from PIL import Image

from src.config import Config
from src.extraction.llm_client import LLMClient, LLMError
from src.models import BankDetails, ExtractedFields, LineItem

logger = logging.getLogger(__name__)

# System prompt for the Vision LLM
SYSTEM_PROMPT = """You are an expert financial document extraction system. You analyze invoices, 
receipts, and purchase folios to extract structured information with high accuracy. You support documents 
in any language including English, Arabic, Chinese, French, German, Spanish, and others.

IMPORTANT: Return ONLY valid JSON. Do not include explanations or markdown formatting.
If a field cannot be found (which is common on receipts), use null. For line items, extract as many as visible."""

# Extraction prompt template
EXTRACTION_PROMPT = """This document has MULTIPLE PAGES. Each page may contain a DIFFERENT invoice.

CRITICAL INSTRUCTIONS:
1. Examine EVERY page of this document carefully
2. If there are 3 pages, there may be 3 different invoices (one per page or one per multiple pages)
3. Look for: different vendor names, different invoice numbers, different totals, different dates
4. Even if pages look similar, they may be separate invoices

If you find MULTIPLE invoices (even if spread across multiple pages), return ALL of them as an array.
If you find ONLY ONE invoice, return it as a single-element array.

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "vendor_name": "Company name of the invoice issuer",
      "vendor_address": "Full address of the vendor",
      "invoice_number": "Invoice ID/number",
      "invoice_date": "Invoice date in YYYY-MM-DD format",
      "due_date": "Payment due date in YYYY-MM-DD format",
      "currency": "3-letter currency code (USD, EUR, SAR, etc.)",
      "line_items": [
        {{
          "description": "Item or service description",
          "quantity": 0.0,
          "unit_price": 0.0,
          "line_total": 0.0
        }}
      ],
      "tax_rate": 0.0,
      "tax_amount": 0.0,
      "grand_total": 0.0,
      "po_number": "Purchase order number if present",
      "customer_name": "Name of the customer / bill-to entity",
      "bank_details": {{
        "iban": "IBAN if present",
        "swift": "SWIFT/BIC if present",
        "account_number": "Account number if present",
        "bank_name": "Bank name if present"
      }}
    }}
  ]
}}

Return ONLY the JSON object. No markdown, no explanations."""

# Text-based extraction prompt (when we already have OCR/text)
TEXT_EXTRACTION_PROMPT = """Analyze this invoice text and determine if it contains ONE or MULTIPLE invoices.
Multiple invoices might appear as:
- Multiple vendor names
- Multiple invoice numbers
- Multiple grand totals
- Distinct invoice blocks separated by headers or dividers

If it contains MULTIPLE invoices, return them as an array.
If it contains only ONE invoice, return it as a single-element array.

INVOICE TEXT:
{text}

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "vendor_name": "Company name of the invoice issuer",
      "vendor_address": "Full address of the vendor",
      "invoice_number": "Invoice ID/number",
      "invoice_date": "Invoice date in YYYY-MM-DD format",
      "due_date": "Payment due date in YYYY-MM-DD format",
      "currency": "3-letter currency code (USD, EUR, SAR, etc.)",
      "line_items": [
        {{
          "description": "Item or service description",
          "quantity": 0.0,
          "unit_price": 0.0,
          "line_total": 0.0
        }}
      ],
      "tax_rate": 0.0,
      "tax_amount": 0.0,
      "grand_total": 0.0,
      "po_number": "Purchase order number if present",
      "customer_name": "Name of the customer / bill-to entity",
      "bank_details": {{
        "iban": "IBAN if present",
        "swift": "SWIFT/BIC if present",
        "account_number": "Account number if present",
        "bank_name": "Bank name if present"
      }}
    }}
  ]
}}

Return ONLY the JSON object. No markdown, no explanations."""


class VisionExtractor:
    """
    Extracts structured invoice fields using a Vision LLM.

    For scanned PDFs / images: sends page images to the LLM.
    For text PDFs: sends extracted text to the LLM for structured parsing.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()
        self._llm = LLMClient(self._config)

    async def extract_from_images(
        self, images: List[Image.Image]
    ) -> List[ExtractedFields]:
        """
        Extract fields from invoice page images using Vision LLM.

        The LLM determines if the document contains one or multiple invoices.

        Args:
            images: List of page images (PIL Image).

        Returns:
            List of ExtractedFields (one per invoice detected).

        Raises:
            ExtractionError if LLM fails and no fallback is available.
        """
        try:
            # Process each page SEPARATELY to detect invoices on each page
            from src.ingestion.image_extractor import ImageExtractor
            preprocessor = ImageExtractor()

            all_invoices = []

            for page_idx, img in enumerate(images):
                # Resize this single page
                resized = preprocessor.resize_for_llm(img)

                # Prompt specifically for this page
                page_prompt = f"""This is PAGE {page_idx + 1} of a multi-page document.

Extract the financial data from THIS PAGE ONLY. 
CRITICAL: This page might be a formal INVOICE or a simple RECEIPT. Treat both as valid!

Look for:
- Vendor name and address
- Document number (Invoice Number, Receipt Number, or Transaction ID)
- Date (Invoice date or Purchase date)
- Line items with quantities, unit prices, and totals
- Tax amount and rate
- Grand total
- If present: PO number, customer name, or bank details (often missing on receipts, use null if so)

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "vendor_name": "...",
      "vendor_address": "...",
      "invoice_number": "...",
      "invoice_date": "YYYY-MM-DD",
      "due_date": "YYYY-MM-DD or null",
      "currency": "...",
      "line_items": [...],
      "tax_rate": 0.0,
      "tax_amount": 0.0,
      "grand_total": 0.0,
      "po_number": "...",
      "customer_name": "...",
      "bank_details": {{...}}
    }}
  ]
}}

ONLY return {{"invoices": []}} if the page is 100% blank or contains absolutely no financial data.
Return ONLY valid JSON."""

                response = await self._llm.chat_with_images(
                    prompt=page_prompt,
                    images=[resized],
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.05,
                )

                # DEBUG: Write each page response
                debug_file = f"data/debug_page_{page_idx + 1}.txt"
                import os
                os.makedirs("data", exist_ok=True)
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"Page {page_idx + 1} response written to: {debug_file}")

                # Parse this page's results
                page_invoices = self._parse_response(response)
                all_invoices.extend(page_invoices)

            return all_invoices

        except LLMError as e:
            logger.error(f"Vision extraction failed: {e}")
            raise ExtractionError(f"Vision LLM failed: {e}")
        except Exception as e:
            logger.error(f"Vision extraction unexpected error: {e}")
            raise ExtractionError(f"Vision extraction error: {e}")

    async def extract_from_text(self, text: str) -> List[ExtractedFields]:
        """
        Extract fields from pre-extracted text using LLM.

        The LLM determines if the text contains one or multiple invoices.

        Args:
            text: Raw text extracted from the PDF.

        Returns:
            List of ExtractedFields (one per invoice detected).
        """
        try:
            # Truncate very long text to avoid token limits
            truncated = text[:8000] if len(text) > 8000 else text

            prompt = TEXT_EXTRACTION_PROMPT.format(text=truncated)

            response = await self._llm.chat_text(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.05,
            )

            return self._parse_response(response)

        except LLMError as e:
            logger.error(f"Text extraction failed: {e}")
            raise ExtractionError(f"Text LLM failed: {e}")
        except Exception as e:
            logger.error(f"Text extraction unexpected error: {e}")
            raise ExtractionError(f"Text extraction error: {e}")

    def _parse_response(self, response: str) -> List[ExtractedFields]:
        """
        Parse LLM JSON response into list of ExtractedFields.

        Handles the new format where LLM returns {"invoices": [...]} and
        common response formatting issues (markdown fences, etc.)
        """
        # Strip markdown code fences if present
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Try to find JSON object in the response
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error(f"Cannot parse LLM response as JSON: {cleaned[:500]}")
                    raise ExtractionError(f"Invalid JSON from LLM: {e}")
            else:
                logger.error(f"No JSON found in LLM response: {cleaned[:500]}")
                raise ExtractionError(f"No JSON in LLM response: {e}")

        # Handle both old format (direct fields) and new format (invoices array)
        invoices_data = data.get("invoices")
        if invoices_data is None:
            # Old format: single invoice without wrapper - wrap it
            invoices_data = [data]

        if not invoices_data:
            logger.warning("No invoices found in LLM response, returning empty list")
            return []

        return [self._dict_to_fields(invoice_data) for invoice_data in invoices_data]

    def _dict_to_fields(self, data: Dict) -> ExtractedFields:
        """Convert a raw dict to ExtractedFields model."""
        # Parse line items
        line_items = []
        for item in data.get("line_items", []) or []:
            if isinstance(item, dict):
                line_items.append(LineItem(
                    description=str(item.get("description", "")),
                    quantity=self._safe_float(item.get("quantity")),
                    unit_price=self._safe_float(item.get("unit_price")),
                    line_total=self._safe_float(item.get("line_total")),
                ))

        # Parse bank details
        bank_data = data.get("bank_details")
        bank_details = None
        if bank_data and isinstance(bank_data, dict):
            bank_details = BankDetails(
                iban=bank_data.get("iban"),
                swift=bank_data.get("swift"),
                account_number=bank_data.get("account_number"),
                bank_name=bank_data.get("bank_name"),
            )

        return ExtractedFields(
            vendor_name=data.get("vendor_name"),
            vendor_address=data.get("vendor_address"),
            invoice_number=data.get("invoice_number"),
            invoice_date=data.get("invoice_date"),
            due_date=data.get("due_date"),
            currency=data.get("currency"),
            line_items=line_items,
            tax_rate=self._safe_float(data.get("tax_rate")),
            tax_amount=self._safe_float(data.get("tax_amount")),
            grand_total=self._safe_float(data.get("grand_total")),
            po_number=data.get("po_number"),
            customer_name=data.get("customer_name"),
            bank_details=bank_details,
        )

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """Safely convert a value to float."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


class ExtractionError(Exception):
    """Raised when field extraction fails."""
    pass
