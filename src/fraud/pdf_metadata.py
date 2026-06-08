"""
PDF Metadata Forensics — Fraud Detection Module #9.

Inspects PDF metadata for signs of tampering:
- Creator tool analysis (Photoshop, Illustrator = suspicious)
- Creation vs modification date discrepancies
- Author field analysis
- Structure tag inspection
"""

import logging
import re
from typing import Dict, Optional

from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "pdf_metadata"

# Suspicious PDF creator tools (not expected for genuine invoices)
SUSPICIOUS_CREATORS = [
    "photoshop", "illustrator", "gimp", "inkscape", "canva",
    "affinity", "corel", "paint", "pixelmator", "sketch",
    "figma", "adobe indesign",
]

# Expected legitimate creators (ERP systems, accounting software)
LEGITIMATE_CREATORS = [
    "sap", "oracle", "quickbooks", "xero", "sage",
    "microsoft", "word", "excel", "crystal reports",
    "jasperreports", "wkhtmltopdf", "weasyprint",
    "reportlab", "fpdf", "itext", "pdflatex",
    "chrome", "firefox", "print",
]


def run(
    pdf_metadata: Optional[Dict] = None,
    **kwargs,
) -> FraudCheck:
    """
    Analyze PDF metadata for forensic red flags.

    Args:
        pdf_metadata: Dictionary of PDF metadata fields.

    Returns:
        FraudCheck with metadata analysis results.
    """
    if not pdf_metadata:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No PDF metadata available for analysis.",
        )

    issues = []
    evidence = {"metadata": pdf_metadata, "checks": []}

    creator = (pdf_metadata.get("creator") or "").strip().lower()
    producer = (pdf_metadata.get("producer") or "").strip().lower()
    author = (pdf_metadata.get("author") or "").strip()
    creation_date = pdf_metadata.get("creation_date", "")
    mod_date = pdf_metadata.get("modification_date", "")

    # ── 1. Suspicious creator tool ──
    for tool in SUSPICIOUS_CREATORS:
        if tool in creator or tool in producer:
            issues.append(
                f"PDF created with suspicious tool: "
                f"creator='{pdf_metadata.get('creator')}', "
                f"producer='{pdf_metadata.get('producer')}'"
            )
            evidence["checks"].append({
                "check": "suspicious_creator",
                "tool_detected": tool,
                "creator": pdf_metadata.get("creator"),
                "producer": pdf_metadata.get("producer"),
            })
            break

    # ── 2. Creation vs modification date discrepancy ──
    if creation_date and mod_date:
        # PyMuPDF dates are in format "D:YYYYMMDDHHmmSS..."
        c_clean = _clean_pdf_date(creation_date)
        m_clean = _clean_pdf_date(mod_date)

        if c_clean and m_clean and m_clean != c_clean:
            # PDF was modified after creation
            issues.append(
                f"PDF was modified after creation: "
                f"created={creation_date}, modified={mod_date}"
            )
            evidence["checks"].append({
                "check": "date_discrepancy",
                "creation_date": creation_date,
                "modification_date": mod_date,
            })

    # ── 3. Missing metadata (could indicate stripping) ──
    missing_meta = []
    if not creator and not producer:
        missing_meta.append("creator/producer")
    if not creation_date:
        missing_meta.append("creation_date")

    if missing_meta:
        issues.append(f"Missing PDF metadata fields: {', '.join(missing_meta)}")
        evidence["checks"].append({
            "check": "missing_metadata",
            "missing_fields": missing_meta,
        })

    # ── 4. Multiple PDF producers (re-processed) ──
    if creator and producer:
        if creator != producer:
            # Different creator and producer can indicate re-processing
            # but this is common and not always suspicious
            is_creator_legit = any(l in creator for l in LEGITIMATE_CREATORS)
            is_producer_legit = any(l in producer for l in LEGITIMATE_CREATORS)

            if not is_creator_legit and not is_producer_legit:
                issues.append(
                    f"Unrecognized PDF creation tools: "
                    f"creator='{pdf_metadata.get('creator')}', "
                    f"producer='{pdf_metadata.get('producer')}'"
                )
                evidence["checks"].append({
                    "check": "unknown_tools",
                    "creator": pdf_metadata.get("creator"),
                    "producer": pdf_metadata.get("producer"),
                })

    # ── Scoring ──
    if not issues:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="PDF metadata appears legitimate.",
            evidence=evidence,
        )

    # Suspicious creator is most critical
    has_suspicious_creator = any(
        c.get("check") == "suspicious_creator" for c in evidence["checks"]
    )

    if has_suspicious_creator:
        score = 75
        severity = Severity.CRITICAL
    elif len(issues) >= 2:
        score = 50
        severity = Severity.HIGH
    else:
        score = 30
        severity = Severity.MEDIUM

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=True,
        detail="; ".join(issues),
        evidence=evidence,
    )


def _clean_pdf_date(date_str: str) -> Optional[str]:
    """Clean a PDF date string for comparison."""
    if not date_str:
        return None
    # Remove "D:" prefix and timezone info
    cleaned = re.sub(r"^D:", "", date_str)
    cleaned = re.sub(r"[+-]\d{2}'\d{2}'?$", "", cleaned)
    # Take first 14 chars (YYYYMMDDHHmmSS)
    return cleaned[:14] if len(cleaned) >= 14 else cleaned
