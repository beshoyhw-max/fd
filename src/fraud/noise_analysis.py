"""
Noise Pattern Analysis — Fraud Detection Module #13.

Detects image manipulation through noise consistency analysis:
- Wavelet-based noise estimation (MAD on HH subband)
- PRNU (Photo Response Non-Uniformity) sensor noise fingerprinting
- Per-region noise variance analysis
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "noise_analysis"


def run(
    image: Optional[Image.Image] = None,
    **kwargs,
) -> FraudCheck:
    """
    Analyze noise patterns for signs of image manipulation.

    Args:
        image: PIL Image of the invoice page.

    Returns:
        FraudCheck with noise analysis results.
    """
    if image is None:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No image available for noise analysis (text PDF — skipped).",
        )

    try:
        wavelet_score, wavelet_evidence = _wavelet_noise_analysis(image)
        region_score, region_evidence = _region_noise_analysis(image)
        prnu_score, prnu_evidence = _prnu_analysis(image)
    except Exception as e:
        logger.error(f"Noise analysis failed: {e}")
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=f"Noise analysis failed: {e}",
        )

    # Weighted combination
    combined = int(wavelet_score * 0.35 + region_score * 0.40 + prnu_score * 0.25)

    evidence = {
        "wavelet_analysis": {"score": wavelet_score, **wavelet_evidence},
        "region_analysis": {"score": region_score, **region_evidence},
        "prnu_analysis": {"score": prnu_score, **prnu_evidence},
        "combined_score": combined,
    }

    if combined >= 60:
        severity = Severity.HIGH
        triggered = True
        detail = (
            f"Significant noise inconsistencies detected: "
            f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score} "
            f"(combined: {combined}/100). Possible image compositing."
        )
    elif combined >= 35:
        severity = Severity.MEDIUM
        triggered = True
        detail = (
            f"Moderate noise irregularities: "
            f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score} "
            f"(combined: {combined}/100)."
        )
    else:
        severity = Severity.LOW
        triggered = False
        detail = (
            f"Noise patterns appear consistent: "
            f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score} "
            f"(combined: {combined}/100)."
        )

    return FraudCheck(
        name=CHECK_NAME,
        score=combined,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )


def _wavelet_noise_analysis(image: Image.Image) -> Tuple[int, Dict]:
    """
    Wavelet-based noise estimation using Median Absolute Deviation
    on the HH (diagonal detail) subband.

    Real scans have consistent noise levels; composited images don't.
    """
    import pywt

    gray = np.array(image.convert("L"), dtype=np.float64)
    gray = cv2.resize(gray, (512, 512))

    # 2-level wavelet decomposition
    coeffs = pywt.wavedec2(gray, "db4", level=2)

    # Analyze HH subband (diagonal details — contains mainly noise)
    noise_levels = []
    for level in range(1, len(coeffs)):
        lh, hl, hh = coeffs[level]
        # Robust noise estimation via MAD
        mad = float(np.median(np.abs(hh - np.median(hh))))
        sigma = mad / 0.6745  # MAD-based noise std estimation
        noise_levels.append({
            "level": level,
            "mad": round(mad, 4),
            "sigma": round(sigma, 4),
        })

    if len(noise_levels) < 2:
        return 0, {"insufficient_levels": True}

    # Compare noise levels between decomposition levels
    sigmas = [n["sigma"] for n in noise_levels]
    sigma_ratio = sigmas[0] / sigmas[1] if sigmas[1] > 0 else 0

    # Expected ratio for natural images is typically 1.8-2.5
    # AI/composited images deviate significantly
    if sigma_ratio < 1.2 or sigma_ratio > 4.0:
        score = min(100, int(50 + abs(sigma_ratio - 2.0) * 20))
    elif sigma_ratio < 1.5 or sigma_ratio > 3.0:
        score = int(20 + abs(sigma_ratio - 2.0) * 15)
    else:
        score = 0

    evidence = {
        "noise_levels": noise_levels,
        "sigma_ratio": round(sigma_ratio, 3),
        "expected_range": "1.8-2.5",
    }

    return score, evidence


def _region_noise_analysis(image: Image.Image) -> Tuple[int, Dict]:
    """
    Per-region noise variance analysis.

    Splits the image into blocks and estimates local noise variance.
    Significant variance differences between blocks indicate manipulation.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    gray = cv2.resize(gray, (512, 512))

    block_size = 64
    noise_variances = []

    for y in range(0, gray.shape[0] - block_size, block_size):
        for x in range(0, gray.shape[1] - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]

            # Skip uniform blocks (background)
            block_std = np.std(block)
            if block_std < 2.0:
                continue

            # Estimate noise using Laplacian variance
            laplacian = cv2.Laplacian(block.astype(np.float64), cv2.CV_64F)
            noise_var = float(np.var(laplacian))
            noise_variances.append({
                "x": x, "y": y,
                "noise_variance": round(noise_var, 2),
            })

    if len(noise_variances) < 8:
        return 0, {"block_count": len(noise_variances), "insufficient_data": True}

    variances = np.array([n["noise_variance"] for n in noise_variances])
    mean_var = float(np.mean(variances))
    std_var = float(np.std(variances))
    cv_var = std_var / mean_var if mean_var > 0 else 0  # Coefficient of variation

    # Count outlier blocks (noise variance > 2 std from mean)
    outliers = int(np.sum(np.abs(variances - mean_var) > 2 * std_var))
    outlier_pct = outliers / len(noise_variances) * 100

    # High CV or many outliers = suspicious
    if cv_var > 1.0 or outlier_pct > 15:
        score = min(100, int(50 + cv_var * 20 + outlier_pct))
    elif cv_var > 0.6 or outlier_pct > 8:
        score = int(20 + cv_var * 25 + outlier_pct * 2)
    else:
        score = 0

    evidence = {
        "block_count": len(noise_variances),
        "mean_noise_variance": round(mean_var, 2),
        "std_noise_variance": round(std_var, 2),
        "coefficient_of_variation": round(cv_var, 3),
        "outlier_blocks": outliers,
        "outlier_pct": round(outlier_pct, 1),
    }

    return score, evidence


def _prnu_analysis(image: Image.Image) -> Tuple[int, Dict]:
    """
    PRNU (Photo Response Non-Uniformity) analysis.

    Real scans from the same scanner share a sensor noise fingerprint.
    AI-generated or composited images lack a consistent PRNU pattern.

    We estimate the noise residual and check for spatial consistency.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    gray = cv2.resize(gray, (512, 512))

    # Estimate the "clean" image using Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Noise residual = original - denoised
    residual = gray - denoised

    # Analyze spatial consistency of the noise residual
    # Split into quadrants
    h, w = residual.shape
    quadrants = [
        residual[:h//2, :w//2],   # Top-left
        residual[:h//2, w//2:],   # Top-right
        residual[h//2:, :w//2],   # Bottom-left
        residual[h//2:, w//2:],   # Bottom-right
    ]

    quad_stats = []
    for i, q in enumerate(quadrants):
        quad_stats.append({
            "quadrant": i,
            "mean": round(float(np.mean(q)), 4),
            "std": round(float(np.std(q)), 4),
            "energy": round(float(np.sum(q**2) / q.size), 4),
        })

    # Compare quadrant statistics
    stds = [q["std"] for q in quad_stats]
    energies = [q["energy"] for q in quad_stats]

    std_variation = float(np.std(stds) / np.mean(stds)) if np.mean(stds) > 0 else 0
    energy_variation = float(np.std(energies) / np.mean(energies)) if np.mean(energies) > 0 else 0

    # High variation between quadrants = inconsistent noise = suspicious
    combined_variation = (std_variation + energy_variation) / 2

    if combined_variation > 0.4:
        score = min(100, int(50 + combined_variation * 100))
    elif combined_variation > 0.2:
        score = int(20 + combined_variation * 60)
    else:
        score = 0

    evidence = {
        "quadrant_stats": quad_stats,
        "std_variation": round(std_variation, 4),
        "energy_variation": round(energy_variation, 4),
        "combined_variation": round(combined_variation, 4),
    }

    return score, evidence
