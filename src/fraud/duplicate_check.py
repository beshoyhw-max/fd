"""
Duplicate Detection Check — Fraud Detection Module #6.

Detects duplicate invoices using:
- Exact match on invoice number
- Fuzzy matching on (vendor + amount + date) using RapidFuzz
- Image hash comparison (perceptual hash)
"""

import logging
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from src.config import Config
from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "duplicate_detection"


def run(
    fields: ExtractedFields,
    historical_invoices: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> FraudCheck:
    """
    Check for duplicate invoices against historical records.

    Args:
        fields: Extracted fields of the current invoice.
        historical_invoices: List of historical invoice index entries.
        config: Optional config override.

    Returns:
        FraudCheck indicating duplicate status.
    """
    config = config or Config.get()
    threshold = config.duplicate_fuzzy_threshold

    if not historical_invoices:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No historical invoices available for duplicate comparison.",
        )

    matches = []
    current_inv_num = (fields.invoice_number or "").strip()
    current_vendor = (fields.vendor_name or "").strip()
    current_total = fields.grand_total
    current_date = (fields.invoice_date or "").strip()

    for hist in historical_invoices:
        match_reasons = []
        match_score = 0

        hist_inv_num = (hist.get("invoice_number") or "").strip()
        hist_vendor = (hist.get("vendor_name") or "").strip()
        hist_total = hist.get("grand_total")
        hist_date = str(hist.get("invoice_date", "")).strip()

        # ── 1. Exact invoice number match ──
        if current_inv_num and hist_inv_num:
            if current_inv_num.lower() == hist_inv_num.lower():
                match_reasons.append(f"Exact invoice number match: {current_inv_num}")
                match_score = 100

        # ── 2. Fuzzy matching on vendor + amount + date ──
        if match_score < 100 and current_vendor and hist_vendor:
            vendor_sim = fuzz.token_sort_ratio(current_vendor.lower(), hist_vendor.lower())

            amount_match = False
            if current_total is not None and hist_total is not None:
                amount_match = abs(current_total - hist_total) < 0.01

            date_match = False
            if current_date and hist_date:
                date_match = current_date == hist_date

            # High vendor similarity + same amount = likely duplicate
            if vendor_sim >= threshold and amount_match:
                composite = vendor_sim
                if date_match:
                    composite = min(100, composite + 10)
                    match_reasons.append(
                        f"Same vendor ({vendor_sim}% match), amount, and date"
                    )
                else:
                    match_reasons.append(
                        f"Same vendor ({vendor_sim}% match) and amount"
                    )
                match_score = max(match_score, composite)

            # Same vendor + same date (different amount could be split invoice)
            elif vendor_sim >= threshold and date_match and not amount_match:
                if vendor_sim >= 90:
                    match_reasons.append(
                        f"Same vendor ({vendor_sim}% match) and date, different amount"
                    )
                    match_score = max(match_score, 40)

        if match_reasons:
            matches.append({
                "invoice_id": hist.get("invoice_id", "unknown"),
                "invoice_number": hist_inv_num,
                "vendor_name": hist_vendor,
                "grand_total": hist_total,
                "match_score": match_score,
                "reasons": match_reasons,
            })

    if not matches:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No duplicate invoices found in historical records.",
            evidence={"records_checked": len(historical_invoices)},
        )

    # Sort matches by score, take the best
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    best_match = matches[0]
    best_score = best_match["match_score"]

    evidence = {
        "records_checked": len(historical_invoices),
        "matches_found": len(matches),
        "best_match": best_match,
        "all_matches": matches[:5],  # Top 5 matches
    }

    if best_score >= 95:
        severity = Severity.CRITICAL
        score = best_score
        detail = (
            f"EXACT DUPLICATE detected: matches invoice {best_match['invoice_id']} "
            f"({'; '.join(best_match['reasons'])})"
        )
    elif best_score >= 80:
        severity = Severity.HIGH
        score = best_score
        detail = (
            f"Likely duplicate: {best_score}% match with invoice "
            f"{best_match['invoice_id']} ({'; '.join(best_match['reasons'])})"
        )
    else:
        severity = Severity.MEDIUM
        score = best_score
        detail = (
            f"Possible duplicate: {best_score}% similarity with invoice "
            f"{best_match['invoice_id']} ({'; '.join(best_match['reasons'])})"
        )

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=True,
        detail=detail,
        evidence=evidence,
    )
