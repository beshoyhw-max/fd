"""
Benford's Law Check — Fraud Detection Module #4.

Analyzes the leading-digit frequency distribution of monetary amounts.
Human-fabricated numbers fail to follow Benford's Law (the natural logarithmic
distribution of leading digits in real-world datasets).

Uses the benford_py library for battle-tested chi-squared, KS, and MAD tests
with proper edge-case handling.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.models import ExtractedFields, FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "benford_law"

# ── benford_py availability ──────────────────────────────────────

_BENFORD_PY_AVAILABLE = False
try:
    import benford as bf
    _BENFORD_PY_AVAILABLE = True
except ImportError:
    logger.warning(
        "benford_py not installed — falling back to manual Benford analysis. "
        "Install with: pip install benford-py"
    )

# Benford's Law expected probabilities for leading digits 1-9 (manual fallback)
BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
    6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046,
}


def _collect_amounts(fields: ExtractedFields) -> List[float]:
    """Collect all monetary amounts from the invoice."""
    amounts = []

    # Line item amounts
    for item in fields.line_items or []:
        if item.unit_price is not None and item.unit_price > 0:
            amounts.append(item.unit_price)
        if item.line_total is not None and item.line_total > 0:
            amounts.append(item.line_total)
        if item.quantity is not None and item.quantity > 0:
            amounts.append(item.quantity)

    # Grand total
    if fields.grand_total is not None and fields.grand_total > 0:
        amounts.append(fields.grand_total)

    # Tax amount
    if fields.tax_amount is not None and fields.tax_amount > 0:
        amounts.append(fields.tax_amount)

    return amounts


def run(
    fields: ExtractedFields,
    historical_amounts: Optional[List[float]] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> FraudCheck:
    """
    Run Benford's Law analysis on invoice amounts.

    Uses benford_py (if available) for chi-squared, KS, and MAD tests.
    Falls back to manual scipy-based analysis if benford_py is not installed.

    Args:
        fields: Extracted invoice fields.
        historical_amounts: Optional list of historical amounts from the same
            vendor or all invoices (provides a larger sample).
        config: Optional config override.

    Returns:
        FraudCheck with Benford's Law test results.
    """
    config = config or Config.get()
    min_samples = config.benford_min_samples

    # Collect amounts from current invoice + any historical data
    amounts = _collect_amounts(fields)
    if historical_amounts:
        amounts.extend(historical_amounts)

    # Filter positive amounts only
    amounts = [a for a in amounts if a > 0]

    # Insufficient data
    if len(amounts) < min_samples:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=(
                f"Insufficient data for Benford's Law analysis: "
                f"{len(amounts)} amounts (minimum {min_samples} required)."
            ),
            evidence={"sample_size": len(amounts), "minimum_required": min_samples},
        )

    # Route to the appropriate implementation
    if _BENFORD_PY_AVAILABLE:
        return _run_with_benford_py(amounts, config)
    else:
        return _run_manual_fallback(amounts, config)


# ── benford_py Implementation (Primary) ──────────────────────────


def _run_with_benford_py(amounts: List[float], config: Config) -> FraudCheck:
    """
    Run Benford's Law analysis using the benford_py library.

    Advantages over manual implementation:
    - Battle-tested edge case handling
    - Multiple test statistics (Chi², KS, MAD) for more robust detection
    - MAD is recommended over chi-squared for Benford analysis because
      chi-squared is overly sensitive to sample size
    """
    series = pd.Series(amounts)

    try:
        # Run first-digit test (suppressing the auto-plot)
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        f1d = bf.first_digits(series, digs=1, decimals=2, show_plot=False)

        # Extract test statistics from the returned DataFrame
        # f1d is a DataFrame with columns including 'Found', 'Expected', etc.
        chi2 = float(f1d.chi_square) if hasattr(f1d, 'chi_square') else None
        ks_stat = float(f1d.KS) if hasattr(f1d, 'KS') else None
        mad = float(f1d.MAD) if hasattr(f1d, 'MAD') else None

        # Build digit distribution from the results DataFrame
        digit_distribution = {}
        if hasattr(f1d, 'index'):
            for digit in f1d.index:
                found_pct = float(f1d.loc[digit, 'Found']) * 100 if 'Found' in f1d.columns else 0
                expected_pct = float(f1d.loc[digit, 'Expected']) * 100 if 'Expected' in f1d.columns else 0
                digit_distribution[str(int(digit))] = {
                    "observed_pct": round(found_pct, 1),
                    "expected_pct": round(expected_pct, 1),
                    "deviation": round(abs(found_pct - expected_pct), 1),
                }

        evidence = {
            "sample_size": len(amounts),
            "chi_squared": round(chi2, 4) if chi2 is not None else None,
            "ks_statistic": round(ks_stat, 4) if ks_stat is not None else None,
            "mad": round(mad, 6) if mad is not None else None,
            "digit_distribution": digit_distribution,
            "engine": "benford_py",
        }

        # Scoring based on MAD (Mean Absolute Deviation)
        # MAD thresholds from the literature (Nigrini, 2012):
        #   Close conformity:    0.000 – 0.006
        #   Acceptable:          0.006 – 0.012
        #   Marginally acceptable: 0.012 – 0.015
        #   Nonconformity:       > 0.015
        if mad is not None:
            return _score_from_mad(mad, chi2, evidence)
        elif chi2 is not None:
            # Fallback to chi-squared if MAD not available
            return _score_from_chi2_manual(chi2, len(amounts), evidence)
        else:
            # Could not compute any statistic
            return FraudCheck(
                name=CHECK_NAME,
                score=0,
                severity=Severity.LOW,
                triggered=False,
                detail="Benford's Law analysis could not compute test statistics.",
                evidence=evidence,
            )

    except Exception as e:
        logger.warning(f"benford_py analysis failed: {e}, falling back to manual")
        return _run_manual_fallback(amounts, config)


def _score_from_mad(
    mad: float,
    chi2: Optional[float],
    evidence: dict,
) -> FraudCheck:
    """
    Score based on MAD (recommended for Benford's Law).

    MAD thresholds (Nigrini, 2012):
    - Close conformity:      MAD ≤ 0.006
    - Acceptable conformity: MAD ≤ 0.012
    - Marginally acceptable: MAD ≤ 0.015
    - Nonconformity:         MAD > 0.015
    """
    if mad > 0.015:
        # Nonconformity — suspicious
        score = min(100, int(60 + mad * 2000))
        severity = Severity.HIGH
        triggered = True
        detail = (
            f"SIGNIFICANT Benford's Law violation: MAD={mad:.4f} "
            f"(nonconformity threshold: 0.015). "
            f"Leading digit distribution is inconsistent with natural numbers."
        )
        if chi2 is not None:
            detail += f" Chi²={chi2:.2f}."

    elif mad > 0.012:
        # Marginally acceptable — flag but lower score
        score = int(30 + mad * 1500)
        severity = Severity.MEDIUM
        triggered = True
        detail = (
            f"Moderate Benford's Law deviation: MAD={mad:.4f} "
            f"(marginally acceptable: 0.012–0.015). "
            f"Leading digit distribution shows some irregularity."
        )
        if chi2 is not None:
            detail += f" Chi²={chi2:.2f}."

    elif mad > 0.006:
        # Acceptable conformity — no trigger
        score = 0
        severity = Severity.LOW
        triggered = False
        detail = (
            f"Benford's Law check passed: MAD={mad:.4f} "
            f"(acceptable conformity: ≤0.012). "
            f"Leading digit distribution is consistent with natural numbers."
        )

    else:
        # Close conformity — clean
        score = 0
        severity = Severity.LOW
        triggered = False
        detail = (
            f"Benford's Law check passed: MAD={mad:.4f} "
            f"(close conformity: ≤0.006). "
            f"Leading digit distribution closely matches expected Benford distribution."
        )

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )


# ── Manual Fallback (if benford_py not installed) ────────────────


def _run_manual_fallback(amounts: List[float], config: Config) -> FraudCheck:
    """
    Manual Benford's Law analysis using scipy.

    This is the original implementation kept as a fallback in case
    benford_py is not installed.
    """
    from collections import Counter
    from scipy import stats

    # Extract leading digits
    digits = []
    for amount in amounts:
        d = _extract_leading_digit(amount)
        if d is not None:
            digits.append(d)

    if len(digits) < config.benford_min_samples:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=(
                f"Insufficient data for Benford's Law analysis: "
                f"{len(digits)} digits (minimum {config.benford_min_samples} required)."
            ),
            evidence={"sample_size": len(digits), "minimum_required": config.benford_min_samples},
        )

    # Count digit frequencies
    digit_counts = Counter(digits)
    total = len(digits)

    # Build observed and expected arrays for chi-squared test
    observed = []
    expected = []
    observed_pct = {}

    for d in range(1, 10):
        obs = digit_counts.get(d, 0)
        exp = BENFORD_EXPECTED[d] * total
        observed.append(obs)
        expected.append(exp)
        observed_pct[d] = round(obs / total * 100, 1)

    # Chi-squared test
    chi2, p_value = stats.chisquare(observed, f_exp=expected)

    evidence = {
        "sample_size": total,
        "chi_squared": round(chi2, 4),
        "p_value": round(p_value, 6),
        "digit_distribution": {
            str(d): {
                "observed_pct": observed_pct[d],
                "expected_pct": round(BENFORD_EXPECTED[d] * 100, 1),
            }
            for d in range(1, 10)
        },
        "engine": "scipy_fallback",
    }

    return _score_from_chi2_manual(chi2, total, evidence, p_value)


def _score_from_chi2_manual(
    chi2: float,
    sample_size: int,
    evidence: dict,
    p_value: Optional[float] = None,
) -> FraudCheck:
    """Score from chi-squared statistic (fallback scoring)."""
    from scipy import stats as sp_stats

    if p_value is None:
        # Compute p-value from chi2 with 8 degrees of freedom (digits 1-9)
        p_value = float(sp_stats.chi2.sf(chi2, df=8))

    evidence["p_value"] = round(p_value, 6)

    if p_value < 0.01:
        score = min(100, int(80 + (1 - p_value) * 20))
        triggered = True
        detail = (
            f"SIGNIFICANT Benford's Law violation: χ²={chi2:.2f}, p={p_value:.4f}. "
            f"Leading digit distribution is inconsistent with natural numbers "
            f"(sample size: {sample_size})."
        )
        severity = Severity.HIGH
    elif p_value < 0.05:
        score = int(40 + (0.05 - p_value) * 800)
        triggered = True
        detail = (
            f"Moderate Benford's Law deviation: χ²={chi2:.2f}, p={p_value:.4f}. "
            f"Leading digit distribution shows some irregularity "
            f"(sample size: {sample_size})."
        )
        severity = Severity.MEDIUM
    else:
        score = 0
        triggered = False
        detail = (
            f"Benford's Law check passed: χ²={chi2:.2f}, p={p_value:.4f}. "
            f"Leading digit distribution is consistent with natural numbers "
            f"(sample size: {sample_size})."
        )
        severity = Severity.LOW

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )


def _extract_leading_digit(value: float) -> Optional[int]:
    """Extract the leading significant digit from a number."""
    if value <= 0:
        return None
    # Get the leading digit by stripping to first non-zero digit
    s = f"{value:.10f}".lstrip("0").lstrip(".").lstrip("0")
    if s and s[0].isdigit() and s[0] != "0":
        return int(s[0])
    return None
