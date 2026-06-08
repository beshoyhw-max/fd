"""
AI-Generated Image Detection — Fraud Detection Module #11.

Detects AI-generated (GAN, Stable Diffusion, LLM) documents using:
- DCT spectral signature analysis
- JPEG ghost curves
- GLCM (Gray-Level Co-occurrence Matrix) texture descriptors
"""

import io
import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "ai_generated"


def run(
    image: Optional[Image.Image] = None,
    **kwargs,
) -> FraudCheck:
    """
    Detect AI-generated document images.

    Args:
        image: PIL Image of the invoice page.

    Returns:
        FraudCheck with AI-generation detection results.
    """
    if image is None:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No image available for AI-generated detection (text PDF — skipped).",
        )

    try:
        dct_score, dct_evidence = _analyze_dct_spectrum(image)
        ghost_score, ghost_evidence = _analyze_jpeg_ghosts(image)
        glcm_score, glcm_evidence = _analyze_glcm_texture(image)
    except Exception as e:
        logger.error(f"AI-generated detection failed: {e}")
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=f"AI-generated detection failed: {e}",
        )

    # Weighted combination
    combined_score = int(dct_score * 0.35 + ghost_score * 0.30 + glcm_score * 0.35)

    evidence = {
        "dct_analysis": {"score": dct_score, **dct_evidence},
        "jpeg_ghost": {"score": ghost_score, **ghost_evidence},
        "glcm_texture": {"score": glcm_score, **glcm_evidence},
        "combined_score": combined_score,
    }

    if combined_score >= 65:
        severity = Severity.CRITICAL
        triggered = True
        detail = (
            f"HIGH probability of AI-generated document: "
            f"DCT={dct_score}, JPEG-ghost={ghost_score}, GLCM={glcm_score} "
            f"(combined: {combined_score}/100)."
        )
    elif combined_score >= 40:
        severity = Severity.HIGH
        triggered = True
        detail = (
            f"Moderate AI-generation indicators: "
            f"DCT={dct_score}, JPEG-ghost={ghost_score}, GLCM={glcm_score} "
            f"(combined: {combined_score}/100)."
        )
    else:
        severity = Severity.LOW
        triggered = False
        detail = (
            f"No significant AI-generation artifacts detected: "
            f"DCT={dct_score}, JPEG-ghost={ghost_score}, GLCM={glcm_score} "
            f"(combined: {combined_score}/100)."
        )

    return FraudCheck(
        name=CHECK_NAME,
        score=combined_score,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )


def _analyze_dct_spectrum(image: Image.Image) -> Tuple[int, Dict]:
    """
    Analyze DCT (Discrete Cosine Transform) spectral signatures.

    AI-generated images often have distinctive high-frequency patterns
    in the DCT domain that differ from real photographs/scans.
    """
    img = np.array(image.convert("L"), dtype=np.float32)

    # Resize to standard size for consistent analysis
    img = cv2.resize(img, (512, 512))

    # Compute 2D DCT
    dct = cv2.dct(img)

    # Analyze frequency distribution
    h, w = dct.shape
    # Split into low, mid, high frequency regions
    low_freq = np.abs(dct[:h//4, :w//4])
    mid_freq = np.abs(dct[h//4:h//2, w//4:w//2])
    high_freq = np.abs(dct[h//2:, w//2:])

    # AI images tend to have unusual energy in mid/high frequencies
    low_energy = float(np.mean(low_freq))
    mid_energy = float(np.mean(mid_freq))
    high_energy = float(np.mean(high_freq))

    # Ratio of high-to-low frequency energy
    if low_energy > 0:
        freq_ratio = (mid_energy + high_energy) / low_energy
    else:
        freq_ratio = 0

    # AI images typically show freq_ratio > 0.15
    if freq_ratio > 0.25:
        score = min(100, int(60 + freq_ratio * 100))
    elif freq_ratio > 0.15:
        score = int(30 + freq_ratio * 100)
    else:
        score = 0

    evidence = {
        "low_energy": round(low_energy, 4),
        "mid_energy": round(mid_energy, 4),
        "high_energy": round(high_energy, 4),
        "freq_ratio": round(freq_ratio, 4),
    }

    return score, evidence


def _analyze_jpeg_ghosts(image: Image.Image) -> Tuple[int, Dict]:
    """
    JPEG ghost analysis.

    Re-compress at multiple quality levels and measure error curves.
    AI-generated images produce different error curves than real scans.
    """
    original = np.array(image.convert("RGB"), dtype=np.float32)
    rgb_image = image.convert("RGB")  # Ensure JPEG-compatible format
    errors = {}

    for quality in [60, 70, 80, 90, 95]:
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer).convert("RGB"), dtype=np.float32)

        error = float(np.mean(np.abs(original - recompressed)))
        errors[quality] = round(error, 4)

    # Analyze error curve shape
    # Real images show a characteristic step at their original quality
    # AI images show a smooth, monotonic decrease
    error_values = list(errors.values())

    # Check for monotonic decrease (suspicious — suggests AI)
    diffs = [error_values[i] - error_values[i+1] for i in range(len(error_values)-1)]
    is_monotonic = all(d >= -0.5 for d in diffs)

    # Check variance of differences (AI images have very consistent differences)
    diff_variance = float(np.var(diffs)) if diffs else 0

    if is_monotonic and diff_variance < 1.0:
        score = min(100, int(50 + (1.0 - diff_variance) * 50))
    elif diff_variance < 2.0:
        score = int(20 + (2.0 - diff_variance) * 15)
    else:
        score = 0

    evidence = {
        "quality_errors": errors,
        "is_monotonic": is_monotonic,
        "diff_variance": round(diff_variance, 4),
    }

    return score, evidence


def _analyze_glcm_texture(image: Image.Image) -> Tuple[int, Dict]:
    """
    GLCM (Gray-Level Co-occurrence Matrix) texture analysis.

    AI-generated images have unnaturally smooth or repetitive
    texture distributions compared to real scanned documents.
    """
    from skimage.feature import graycomatrix, graycoprops

    gray = np.array(image.convert("L"))

    # Resize for consistent analysis
    gray = cv2.resize(gray, (256, 256))

    # Reduce to fewer gray levels for GLCM computation
    gray = (gray // 4).astype(np.uint8)  # 64 levels

    # Compute GLCM at multiple angles
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

    # Extract texture properties
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))

    # AI images tend to have higher homogeneity and lower contrast
    # Real scans have more natural texture variation
    suspicion = 0
    if homogeneity > 0.7:
        suspicion += 30
    if contrast < 5.0:
        suspicion += 20
    if energy > 0.1:
        suspicion += 25
    if correlation > 0.95:
        suspicion += 25

    score = min(100, suspicion)

    evidence = {
        "contrast": round(contrast, 4),
        "homogeneity": round(homogeneity, 4),
        "energy": round(energy, 4),
        "correlation": round(correlation, 4),
    }

    return score, evidence
