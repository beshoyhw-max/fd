"""
Vendor Pattern Analysis — Fraud Detection Module #8.

Analyzes vendor behavior patterns across historical invoices:
- New vendor with unusually large first invoice
- Sudden frequency spikes from existing vendors
- Outlier analysis on vendor volume and amounts
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.config import Config
from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "vendor_pattern"

# Thresholds
NEW_VENDOR_HIGH_AMOUNT = 5000.0   # First invoice > this = suspicious
FREQUENCY_SPIKE_MULTIPLIER = 3.0  # 3x normal frequency = spike
AMOUNT_SPIKE_MULTIPLIER = 3.0     # 3x average amount = spike


def run(
    fields: ExtractedFields,
    historical_invoices: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> FraudCheck:
    """
    Analyze vendor behavior patterns for anomalies.

    Args:
        fields: Current invoice extracted fields.
        historical_invoices: Historical invoice index entries.
        config: Optional config override.

    Returns:
        FraudCheck with vendor pattern analysis.
    """
    config = config or Config.get()
    current_vendor = (fields.vendor_name or "").strip().lower()
    current_total = fields.grand_total

    evidence = {
        "vendor_name": fields.vendor_name,
        "is_new_vendor": False,
        "checks": [],
    }

    if not current_vendor:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No vendor name available for pattern analysis.",
        )

    issues = []

    # Build vendor history
    vendor_history: Dict[str, List[Dict]] = defaultdict(list)
    if historical_invoices:
        for hist in historical_invoices:
            v = (hist.get("vendor_name") or "").strip().lower()
            if v:
                vendor_history[v].append(hist)

    # ── 1. New vendor check ──
    is_new = current_vendor not in vendor_history
    evidence["is_new_vendor"] = is_new

    if is_new and current_total is not None and current_total > NEW_VENDOR_HIGH_AMOUNT:
        issues.append(
            f"New vendor '{fields.vendor_name}' with high first invoice: "
            f"{current_total} (threshold: {NEW_VENDOR_HIGH_AMOUNT})"
        )
        evidence["checks"].append({
            "check": "new_vendor_high_amount",
            "amount": current_total,
            "threshold": NEW_VENDOR_HIGH_AMOUNT,
        })

    # ── 2. Frequency spike (existing vendor) ──
    if not is_new and current_vendor in vendor_history:
        hist = vendor_history[current_vendor]
        invoice_count = len(hist)

        # Calculate monthly rate (rough estimate)
        if invoice_count >= 3:
            # Sort by date
            dates = sorted(h.get("processed_at", "") for h in hist if h.get("processed_at"))
            if len(dates) >= 2:
                # Count recent vs historical
                mid = len(dates) // 2
                recent_count = len(dates) - mid
                old_count = mid
                if old_count > 0:
                    ratio = recent_count / old_count
                    if ratio >= FREQUENCY_SPIKE_MULTIPLIER:
                        issues.append(
                            f"Vendor '{fields.vendor_name}' shows {ratio:.1f}x "
                            f"frequency increase in recent invoices "
                            f"({recent_count} recent vs {old_count} historical)"
                        )
                        evidence["checks"].append({
                            "check": "frequency_spike",
                            "ratio": round(ratio, 2),
                            "recent_count": recent_count,
                            "historical_count": old_count,
                        })

        # ── 3. Amount spike ──
        if current_total is not None:
            hist_amounts = [
                h.get("grand_total") for h in hist
                if h.get("grand_total") is not None
            ]
            if hist_amounts:
                avg_amount = sum(hist_amounts) / len(hist_amounts)
                if avg_amount > 0 and current_total > avg_amount * AMOUNT_SPIKE_MULTIPLIER:
                    ratio = current_total / avg_amount
                    issues.append(
                        f"Current invoice amount ({current_total}) is {ratio:.1f}x "
                        f"the vendor's average ({avg_amount:.2f})"
                    )
                    evidence["checks"].append({
                        "check": "amount_spike",
                        "current_amount": current_total,
                        "average_amount": round(avg_amount, 2),
                        "ratio": round(ratio, 2),
                    })

    # ── Scoring ──
    if not issues:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="Vendor behavior patterns are normal.",
            evidence=evidence,
        )

    score = min(100, len(issues) * 35)

    has_new_vendor_issue = any(
        c.get("check") == "new_vendor_high_amount" for c in evidence["checks"]
    )
    if has_new_vendor_issue:
        severity = Severity.HIGH
        score = max(score, 55)
    elif len(issues) >= 2:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=True,
        detail="; ".join(issues),
        evidence=evidence,
    )
