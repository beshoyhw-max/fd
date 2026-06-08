"""
Missing Fields Check — Fraud Detection Module #2.

Checks which of the 10 required fields are missing or empty.
Incomplete documents may be designed to evade extraction pipelines.
"""

import logging
from typing import Optional

from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "missing_fields"

# Required fields and their importance weights
REQUIRED_FIELDS = {
    "vendor_name": 3,       # Critical
    "invoice_number": 3,    # Critical
    "invoice_date": 2,      # Important
    "grand_total": 3,       # Critical
    "currency": 1,          # Useful
    "line_items": 2,        # Important
    "customer_name": 1,     # Useful
    "due_date": 1,          # Useful
    "tax_amount": 1,        # Useful
    "po_number": 1,         # Optional but useful
}

MAX_WEIGHT = sum(REQUIRED_FIELDS.values())


def run(fields: ExtractedFields, **kwargs) -> FraudCheck:
    """
    Check for missing or empty required fields.

    Returns a FraudCheck scored by the number and importance of missing fields.
    """
    missing = []
    missing_weight = 0
    evidence = {"missing_fields": [], "present_fields": [], "completeness_pct": 0}

    for field_name, weight in REQUIRED_FIELDS.items():
        value = getattr(fields, field_name, None)

        # Check if the field is effectively empty
        is_empty = False
        if value is None:
            is_empty = True
        elif isinstance(value, str) and not value.strip():
            is_empty = True
        elif isinstance(value, list) and len(value) == 0:
            is_empty = True

        if is_empty:
            missing.append(field_name)
            missing_weight += weight
            evidence["missing_fields"].append({
                "field": field_name,
                "weight": weight,
            })
        else:
            evidence["present_fields"].append(field_name)

    present_weight = MAX_WEIGHT - missing_weight
    completeness = round((present_weight / MAX_WEIGHT) * 100, 1)
    evidence["completeness_pct"] = completeness

    if not missing:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=f"All {len(REQUIRED_FIELDS)} required fields are present.",
        )

    # Score: proportional to weight of missing fields
    score = min(100, int((missing_weight / MAX_WEIGHT) * 100))

    # Determine severity based on what's missing
    critical_missing = [f for f in missing if REQUIRED_FIELDS.get(f, 0) >= 3]
    if critical_missing:
        severity = Severity.HIGH
    elif len(missing) >= 5:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM

    detail = (
        f"Missing {len(missing)}/{len(REQUIRED_FIELDS)} fields "
        f"({completeness}% complete): {', '.join(missing)}"
    )

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=True,
        detail=detail,
        evidence=evidence,
    )
