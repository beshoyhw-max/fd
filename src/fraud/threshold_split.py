"""
Threshold Splitting Check — Fraud Detection Module #7.

Detects clusters of invoices from the same vendor just below approval limits.
This is a common fraud pattern where large amounts are split into multiple
smaller invoices to bypass executive authorization requirements.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import Config
from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "threshold_splitting"

# How close to the limit counts as "just below" (percentage)
PROXIMITY_THRESHOLD = 0.15  # Within 15% below the limit


def run(
    fields: ExtractedFields,
    historical_invoices: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> FraudCheck:
    """
    Detect potential threshold splitting patterns.

    Looks for:
    1. Current invoice amount just below the approval limit
    2. Multiple recent invoices from same vendor near the limit
    """
    config = config or Config.get()
    limit = config.approval_limit

    current_total = fields.grand_total
    current_vendor = (fields.vendor_name or "").strip().lower()

    evidence = {
        "approval_limit": limit,
        "current_total": current_total,
        "near_threshold": False,
        "vendor_cluster": None,
    }

    issues = []

    # ── 1. Is the current invoice just below the limit? ──
    if current_total is not None:
        lower_bound = limit * (1 - PROXIMITY_THRESHOLD)

        if lower_bound <= current_total < limit:
            pct_of_limit = round((current_total / limit) * 100, 1)
            issues.append(
                f"Amount {current_total} is {pct_of_limit}% of approval limit "
                f"({limit}) — just below threshold"
            )
            evidence["near_threshold"] = True
            evidence["pct_of_limit"] = pct_of_limit

    # ── 2. Cluster analysis: same vendor, multiple invoices near limit ──
    if historical_invoices and current_vendor:
        vendor_near_limit = []

        for hist in historical_invoices:
            hist_vendor = (hist.get("vendor_name") or "").strip().lower()
            hist_total = hist.get("grand_total")

            if hist_vendor and hist_total is not None:
                # Check if same vendor (fuzzy)
                if hist_vendor == current_vendor or (
                    len(current_vendor) > 3
                    and (current_vendor in hist_vendor or hist_vendor in current_vendor)
                ):
                    lower = limit * (1 - PROXIMITY_THRESHOLD)
                    if lower <= hist_total < limit:
                        vendor_near_limit.append({
                            "invoice_id": hist.get("invoice_id"),
                            "amount": hist_total,
                            "date": hist.get("processed_at", ""),
                        })

        # Include current invoice if near threshold
        if evidence.get("near_threshold"):
            total_near = len(vendor_near_limit) + 1  # +1 for current
        else:
            total_near = len(vendor_near_limit)

        if total_near >= 2:
            issues.append(
                f"Vendor '{fields.vendor_name}' has {total_near} invoices "
                f"just below the approval limit of {limit}"
            )
            evidence["vendor_cluster"] = {
                "vendor": fields.vendor_name,
                "count_near_limit": total_near,
                "historical_matches": vendor_near_limit[:10],
            }

    # ── Scoring ──
    if not issues:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No threshold splitting patterns detected.",
            evidence=evidence,
        )

    # Score based on pattern strength
    score = 0
    if evidence.get("near_threshold"):
        score += 30
    if evidence.get("vendor_cluster"):
        cluster_count = evidence["vendor_cluster"]["count_near_limit"]
        score += min(70, cluster_count * 20)

    score = min(100, score)

    if score >= 60:
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
