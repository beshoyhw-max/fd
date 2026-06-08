"""
Math Verification Check — Fraud Detection Module #1.

Verifies mathematical consistency of invoice line items:
- qty × unit_price = line_total (per line)
- sum(line_totals) + tax = grand_total
- Tax rate reasonableness
"""

import logging
from typing import Optional

from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "math_verification"


def run(fields: ExtractedFields, **kwargs) -> FraudCheck:
    """
    Verify mathematical integrity of the invoice.

    Returns a FraudCheck with score 0 (clean) to 100 (definite mismatch).
    """
    issues = []
    evidence = {
        "line_mismatches": [],
        "total_mismatch": None,
        "tax_issues": [],
    }

    # ── 1. Line item verification: qty × unit_price = line_total ──
    computed_line_sum = 0.0
    line_items = fields.line_items or []

    for i, item in enumerate(line_items):
        if item.quantity is not None and item.unit_price is not None and item.line_total is not None:
            expected = round(item.quantity * item.unit_price, 2)
            actual = round(item.line_total, 2)

            if abs(expected - actual) > 0.01:  # Allow 1 cent tolerance
                diff = round(actual - expected, 2)
                issues.append(
                    f"Line {i+1}: {item.quantity} × {item.unit_price} = {expected}, "
                    f"but stated as {actual} (diff: {diff})"
                )
                evidence["line_mismatches"].append({
                    "line": i + 1,
                    "expected": expected,
                    "actual": actual,
                    "difference": diff,
                })

            computed_line_sum += item.line_total
        elif item.line_total is not None:
            computed_line_sum += item.line_total

    # ── 2. Grand total verification: sum(line_totals) + tax = grand_total ──
    if fields.grand_total is not None and len(line_items) > 0:
        tax = fields.tax_amount or 0.0
        expected_total = round(computed_line_sum + tax, 2)
        actual_total = round(fields.grand_total, 2)

        if abs(expected_total - actual_total) > 0.01:
            diff = round(actual_total - expected_total, 2)
            issues.append(
                f"Grand total mismatch: sum({computed_line_sum}) + tax({tax}) = {expected_total}, "
                f"but stated as {actual_total} (diff: {diff})"
            )
            evidence["total_mismatch"] = {
                "line_sum": computed_line_sum,
                "tax": tax,
                "expected_total": expected_total,
                "actual_total": actual_total,
                "difference": diff,
            }

    # ── 3. Tax rate reasonableness ──
    if fields.tax_rate is not None:
        if fields.tax_rate < 0:
            issues.append(f"Negative tax rate: {fields.tax_rate}")
            evidence["tax_issues"].append("negative_rate")
        elif fields.tax_rate > 0.5:  # More than 50% tax is suspicious
            issues.append(f"Unusually high tax rate: {fields.tax_rate * 100:.1f}%")
            evidence["tax_issues"].append("excessive_rate")

    # Tax amount vs computed tax
    if (fields.tax_rate is not None and fields.tax_amount is not None
            and computed_line_sum > 0):
        expected_tax = round(computed_line_sum * fields.tax_rate, 2)
        actual_tax = round(fields.tax_amount, 2)
        if abs(expected_tax - actual_tax) > 0.01:
            issues.append(
                f"Tax amount mismatch: {computed_line_sum} × {fields.tax_rate} = {expected_tax}, "
                f"but stated as {actual_tax}"
            )
            evidence["tax_issues"].append({
                "expected_tax": expected_tax,
                "actual_tax": actual_tax,
            })

    # ── Scoring ──
    if not issues:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="All math checks passed — line items and totals are consistent.",
        )

    # Score based on severity: line mismatches are most critical
    score = 0
    if evidence["line_mismatches"]:
        score += min(50, len(evidence["line_mismatches"]) * 25)
    if evidence["total_mismatch"]:
        score += 35
    if evidence["tax_issues"]:
        score += 15

    score = min(score, 100)

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=Severity.CRITICAL,
        triggered=True,
        detail="; ".join(issues),
        evidence=evidence,
    )
