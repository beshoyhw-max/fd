"""
Vision-Based Field Extraction for Invoice Fraud Detection System.

Sends invoice/receipt page images or text chunks to a Vision LLM and extracts 
structured fields as JSON. Handles continuation pages and multiple invoices per file.
Optimized for maximum accuracy and exhaustive extraction without token/cost constraints.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.config import Config
from src.extraction.llm_client import LLMClient, LLMError
from src.models import BankDetails, ExtractedFields, LineItem

logger = logging.getLogger(__name__)

# Maximum Quality System Prompt - Explicitly forbids laziness and summarization
SYSTEM_PROMPT = """You are an elite, state-of-the-art financial document extraction system. 
You analyze invoices, receipts, and purchase folios to extract structured information with perfect accuracy. 
You support documents in any language including English, Arabic, Chinese, French, German, Spanish, and others.

CRITICAL DIRECTIVES:
1. EXHAUSTIVE EXTRACTION: Do NOT skip, abbreviate, or summarize line items. Extract EVERY SINGLE line item, even if there are hundreds.
2. EXACT TRANSCRIPTION: Transcribe names, addresses, and numbers exactly as they appear. Do not correct typos in the source document.
3. FLEXIBILITY: If a field cannot be found (which is common on receipts or continuation pages), use null. Do not guess.
4. FORMATTING: Return ONLY valid, parseable JSON. Do not include explanations, preambles, or markdown formatting."""

# Extraction prompt template for legacy/fallback pipelines
EXTRACTION_PROMPT = """This document has MULTIPLE PAGES. Each page may contain a DIFFERENT invoice/receipt, 
or some pages may be continuation sheets of a previous page's invoice.

CRITICAL INSTRUCTIONS:
1. Examine EVERY page of this document meticulously.
2. Look for: different vendor names, different document numbers, different totals, different dates.
3. Identify if a page is a continuation of the previous page (i.e., contains more line items but has the same or no header).
4. Extract ALL line items. Do not truncate the list.

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "is_continuation": false, // Set to true ONLY if this page is a continuation page of the previous page's invoice
      "vendor_name": "Company name of the issuer (null if continuation page)",
      "vendor_address": "Full address of the vendor",
      "invoice_number": "Invoice ID/Receipt number/Transaction ID (null if continuation page)",
      "invoice_date": "Document date in YYYY-MM-DD format",
      "due_date": "Payment due date in YYYY-MM-DD format (or null if paid receipt)",
      "currency": "3-letter currency code (USD, EUR, SAR, MAD, etc.)",
      "line_items": [
        {{
          "description": "Exact item or service description",
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

# Text-based extraction prompt updated for exhaustive document handling
TEXT_EXTRACTION_PROMPT = """Analyze this text block completely and extract ALL financial data.
The text may represent an Invoice, a Receipt, or a Continuation Page. 

CRITICAL: Extract every single line item found in the text. Do not stop early.

INVOICE/RECEIPT TEXT:
{text}

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "is_continuation": false, // Set to true ONLY if this page is a continuation page of the previous page's invoice
      "vendor_name": "Company name of the issuer (null if continuation page)",
      "vendor_address": "Full address of the vendor",
      "invoice_number": "Invoice ID/Receipt number/Transaction ID (null if continuation page)",
      "invoice_date": "Document date in YYYY-MM-DD format",
      "due_date": "Payment due date in YYYY-MM-DD format or null",
      "currency": "3-letter currency code",
      "line_items": [
        {{
          "description": "Exact item or service description",
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
        Processes each page separately to catch individual invoices or receipts,
        then dynamically stitches together continuous multi-page sheets.
        """
        try:
            from src.ingestion.image_extractor import ImageExtractor
            preprocessor = ImageExtractor()

            all_extracted_tuples: List[Tuple[ExtractedFields, bool]] = []

            for page_idx, img in enumerate(images):
                # We retain resizing only to ensure it fits within the Vision model's hard maximum resolution
                resized = preprocessor.resize_for_llm(img)

                # Broadened page prompt to capture structural groupings & continuations exhaustively
                page_prompt = f"""This is PAGE {page_idx + 1} of a multi-page document.

Extract the financial data from THIS PAGE ONLY. 
CRITICAL CHECK:
- Is this page a CONTINUATION of a table of line items from the previous page? (i.e. has no main header, no logo, or same invoice number). If so, set "is_continuation": true.
- Is this page a standalone INVOICE or a simple RECEIPT? If so, set "is_continuation": false.

Look for and extract EVERYTHING:
- Vendor name and address
- Document number (Invoice Number, Receipt Number, or Transaction ID)
- Date (Invoice date or Purchase date)
- Line items with quantities, unit prices, and totals. (EXTRACT EVERY SINGLE ROW)
- Tax amount and rate
- Grand total
- If present: PO number, customer name, or bank details

Return as JSON with this structure:

{{
  "invoices": [
    {{
      "is_continuation": false, // Set to true ONLY if this page is a continuation sheet of the previous page
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

                # Dropped temperature to 0.0 for maximum deterministic accuracy
                response = await self._llm.chat_with_images(
                    prompt=page_prompt,
                    images=[resized],
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0,
                )

                # DEBUG: Write each page response
                debug_file = f"data/debug_page_{page_idx + 1}.txt"
                import os
                os.makedirs("data", exist_ok=True)
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"Page {page_idx + 1} response written to: {debug_file}")

                page_invoices_raw = self._parse_response_raw(response)
                for d in page_invoices_raw:
                    fields = self._dict_to_fields(d)
                    is_cont = bool(d.get("is_continuation", False))
                    all_extracted_tuples.append((fields, is_cont))

            # Safely merge continuous invoices before returning
            return self._merge_continuations(all_extracted_tuples)

        except LLMError as e:
            logger.error(f"Vision extraction failed: {e}")
            raise ExtractionError(f"Vision LLM failed: {e}")
        except Exception as e:
            logger.error(f"Vision extraction unexpected error: {e}")
            raise ExtractionError(f"Vision extraction error: {e}")

    async def extract_from_text(self, text: str) -> List[ExtractedFields]:
        """
        Extract fields from pre-extracted text using LLM.
        Processes text page-by-page without length truncation for maximum accuracy.
        """
        try:
            # Split the combined text block back into individual pages using extraction headers
            pages = re.split(r'--- Page \d+ ---\n?', text)
            all_extracted_tuples = []

            for page_text in pages:
                page_text = page_text.strip()
                if not page_text:
                    continue

                # REMOVED TRUNCATION: Pass the entire page text to guarantee no line items are dropped
                prompt = TEXT_EXTRACTION_PROMPT.format(text=page_text)

                try:
                    # Dropped temperature to 0.0 for maximum analytical precision
                    response = await self._llm.chat_text(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        temperature=0.0,
                    )
                    raw_dicts = self._parse_response_raw(response)
                    for d in raw_dicts:
                        fields = self._dict_to_fields(d)
                        is_cont = bool(d.get("is_continuation", False))
                        all_extracted_tuples.append((fields, is_cont))
                except Exception as e:
                    logger.error(f"Text extraction failed for page slice: {e}")
                    continue

            # Safely merge continuous text layers
            return self._merge_continuations(all_extracted_tuples)

        except Exception as e:
            logger.error(f"Text extraction unexpected pipeline error: {e}")
            raise ExtractionError(f"Text extraction error: {e}")

    def _parse_response_raw(self, response: str) -> List[Dict]:
        """Parse raw LLM response cleanly to dictionaries before object wrapping."""
        cleaned = response.strip()
        
        # FIX: Replaced literal triple backticks with `{3}` to prevent SyntaxErrors
        # when copying and pasting into markdown environments
        cleaned = re.sub(r"^`{3}(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?`{3}\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
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

        invoices_data = data.get("invoices")
        if invoices_data is None:
            invoices_data = [data]

        if not invoices_data:
            return []

        return invoices_data

    def _parse_response(self, response: str) -> List[ExtractedFields]:
        """Legacy parser preservation mapping."""
        dicts = self._parse_response_raw(response)
        return [self._dict_to_fields(d) for d in dicts]

    def _merge_continuations(self, tuples: List[Tuple[ExtractedFields, bool]]) -> List[ExtractedFields]:
        """
        Sequentially groups and merges continuous invoices based on LLM flags
        and reliable structural heuristics.
        """
        if not tuples:
            return []

        merged: List[ExtractedFields] = []

        for fields, is_cont in tuples:
            # Safely check heuristic fallbacks if LLM omitted the is_continuation flag
            if not is_cont and merged:
                prev = merged[-1]
                # Heuristic 1: Match exact identical non-empty invoice numbers
                same_invoice_num = (
                    fields.invoice_number and 
                    prev.invoice_number and 
                    str(fields.invoice_number).strip().lower() == str(prev.invoice_number).strip().lower()
                )
                # Heuristic 2: Continuation page has no header info but contains line items
                is_anonymous_continuation = (
                    not fields.vendor_name and 
                    not fields.invoice_number and 
                    fields.line_items
                )
                if same_invoice_num or is_anonymous_continuation:
                    is_cont = True

            # Perform the merge operation
            if is_cont and merged:
                parent = merged[-1]

                # Stitch line items
                if fields.line_items:
                    parent.line_items.extend(fields.line_items)

                # Capture updated grand totals and taxes from the continuation page
                if fields.grand_total is not None and fields.grand_total > 0:
                    parent.grand_total = fields.grand_total
                if fields.tax_amount is not None and fields.tax_amount > 0:
                    parent.tax_amount = fields.tax_amount
                if fields.tax_rate is not None and fields.tax_rate > 0:
                    parent.tax_rate = fields.tax_rate

                # Carry over key missing objects from the bottom of the transaction page
                if not parent.due_date and fields.due_date:
                    parent.due_date = fields.due_date
                if not parent.bank_details and fields.bank_details:
                    parent.bank_details = fields.bank_details
            else:
                merged.append(fields)

        return merged

    def _dict_to_fields(self, data: Dict) -> ExtractedFields:
        """Convert a raw dict data block to an ExtractedFields data model."""
        line_items = []
        for item in data.get("line_items", []) or []:
            if isinstance(item, dict):
                line_items.append(LineItem(
                    description=str(item.get("description", "")),
                    quantity=self._safe_float(item.get("quantity")),
                    unit_price=self._safe_float(item.get("unit_price")),
                    line_total=self._safe_float(item.get("line_total")),
                ))

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
        """Safely convert a field value into a float descriptor."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


class ExtractionError(Exception):
    """Raised when field extraction fails."""
    pass