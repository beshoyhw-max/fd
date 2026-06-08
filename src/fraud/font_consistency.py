"""
Font Consistency Analysis — Fraud Detection Module #12.

Detects document tampering by analyzing font rendering consistency:
- Stroke width uniformity across text regions
- Anti-aliasing profile comparison
- OCR confidence variance (edited text has different patterns)
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "font_consistency"


def run(
    image: Optional[Image.Image] = None,
    ocr_data: Optional[List[Dict]] = None,
    **kwargs,
) -> FraudCheck:
    """
    Analyze font rendering consistency across the document.

    Args:
        image: PIL Image of the invoice page.
        ocr_data: Per-word OCR data (from Surya OCR or Tesseract) if available.

    Returns:
        FraudCheck with font consistency analysis.
    """
    if image is None:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No image available for font analysis (text PDF — skipped).",
        )

    try:
        stroke_score, stroke_evidence = _analyze_stroke_consistency(image)
        aa_score, aa_evidence = _analyze_antialiasing(image)

        # OCR confidence analysis (if data available)
        ocr_score = 0
        ocr_evidence = {}
        if ocr_data and len(ocr_data) > 5:
            ocr_score, ocr_evidence = _analyze_ocr_confidence(ocr_data)
    except Exception as e:
        logger.error(f"Font consistency analysis failed: {e}")
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=f"Font consistency analysis failed: {e}",
        )

    # Weighted combination
    weights = [0.40, 0.35, 0.25] if ocr_data else [0.55, 0.45, 0.0]
    combined = int(stroke_score * weights[0] + aa_score * weights[1] + ocr_score * weights[2])

    evidence = {
        "stroke_analysis": {"score": stroke_score, **stroke_evidence},
        "antialiasing_analysis": {"score": aa_score, **aa_evidence},
        "ocr_confidence": {"score": ocr_score, **ocr_evidence},
        "combined_score": combined,
    }

    if combined >= 60:
        severity = Severity.CRITICAL
        triggered = True
        detail = (
            f"Font inconsistencies detected: stroke={stroke_score}, "
            f"antialiasing={aa_score}, OCR-conf={ocr_score} "
            f"(combined: {combined}/100). Possible text region editing."
        )
    elif combined >= 35:
        severity = Severity.HIGH
        triggered = True
        detail = (
            f"Moderate font inconsistencies: stroke={stroke_score}, "
            f"antialiasing={aa_score}, OCR-conf={ocr_score} "
            f"(combined: {combined}/100)."
        )
    else:
        severity = Severity.LOW
        triggered = False
        detail = (
            f"Font rendering appears consistent: stroke={stroke_score}, "
            f"antialiasing={aa_score}, OCR-conf={ocr_score} "
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


def _analyze_stroke_consistency(image: Image.Image) -> Tuple[int, Dict]:
    """
    Analyze stroke width consistency across text regions.

    Segments the image into blocks and compares stroke widths.
    Tampered regions show different stroke characteristics.
    """
    gray = np.array(image.convert("L"))
    gray = cv2.resize(gray, (512, 512))

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Divide into blocks and measure stroke width per block
    block_size = 64
    stroke_widths = []

    for y in range(0, gray.shape[0] - block_size, block_size):
        for x in range(0, gray.shape[1] - block_size, block_size):
            block = binary[y:y+block_size, x:x+block_size]

            # Only analyze blocks with text (>5% ink)
            ink_ratio = np.sum(block > 0) / block.size
            if ink_ratio < 0.05 or ink_ratio > 0.8:
                continue

            # Estimate stroke width using distance transform
            dist = cv2.distanceTransform(block, cv2.DIST_L2, 5)
            if np.max(dist) > 0:
                # Average stroke width is ~2x the mean distance in ink regions
                mean_stroke = float(np.mean(dist[block > 0])) * 2
                stroke_widths.append({
                    "x": x, "y": y,
                    "stroke_width": round(mean_stroke, 2),
                    "ink_ratio": round(ink_ratio, 3),
                })

    if len(stroke_widths) < 4:
        return 0, {"block_count": len(stroke_widths), "insufficient_data": True}

    widths = [s["stroke_width"] for s in stroke_widths]
    mean_width = float(np.mean(widths))
    std_width = float(np.std(widths))
    cv_width = std_width / mean_width if mean_width > 0 else 0  # Coefficient of variation

    # High coefficient of variation suggests mixed fonts / editing
    if cv_width > 0.5:
        score = min(100, int(50 + cv_width * 50))
    elif cv_width > 0.3:
        score = int(20 + cv_width * 60)
    else:
        score = 0

    evidence = {
        "block_count": len(stroke_widths),
        "mean_stroke_width": round(mean_width, 3),
        "std_stroke_width": round(std_width, 3),
        "coefficient_of_variation": round(cv_width, 3),
    }

    return score, evidence


def _analyze_antialiasing(image: Image.Image) -> Tuple[int, Dict]:
    """
    Analyze anti-aliasing consistency across text regions.

    Different rendering engines produce different anti-aliasing patterns.
    Pasted text from a different source will have different AA profiles.
    """
    gray = np.array(image.convert("L"))
    gray = cv2.resize(gray, (512, 512))

    # Edge detection to find text boundaries
    edges = cv2.Canny(gray, 50, 150)

    # Analyze edge gradient profiles in blocks
    block_size = 64
    gradient_profiles = []

    for y in range(0, gray.shape[0] - block_size, block_size):
        for x in range(0, gray.shape[1] - block_size, block_size):
            edge_block = edges[y:y+block_size, x:x+block_size]
            gray_block = gray[y:y+block_size, x:x+block_size]

            edge_density = np.sum(edge_block > 0) / edge_block.size
            if edge_density < 0.02 or edge_density > 0.5:
                continue

            # Compute gradient magnitude
            grad_x = cv2.Sobel(gray_block, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_block, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # AA profile: mean gradient at edges
            edge_mask = edge_block > 0
            if np.any(edge_mask):
                mean_gradient = float(np.mean(magnitude[edge_mask]))
                gradient_profiles.append(mean_gradient)

    if len(gradient_profiles) < 4:
        return 0, {"profile_count": len(gradient_profiles), "insufficient_data": True}

    profiles = np.array(gradient_profiles)
    mean_profile = float(np.mean(profiles))
    std_profile = float(np.std(profiles))
    cv_profile = std_profile / mean_profile if mean_profile > 0 else 0

    # High variance in AA profiles suggests mixed sources
    if cv_profile > 0.6:
        score = min(100, int(50 + cv_profile * 40))
    elif cv_profile > 0.4:
        score = int(20 + cv_profile * 40)
    else:
        score = 0

    evidence = {
        "profile_count": len(gradient_profiles),
        "mean_gradient": round(mean_profile, 3),
        "std_gradient": round(std_profile, 3),
        "coefficient_of_variation": round(cv_profile, 3),
    }

    return score, evidence


def _analyze_ocr_confidence(ocr_data: List[Dict]) -> Tuple[int, Dict]:
    """
    Analyze OCR confidence variance across text regions.

    Edited/AI-generated text often has different confidence patterns
    than original typed/printed text.
    """
    if not ocr_data or len(ocr_data) < 5:
        return 0, {"word_count": len(ocr_data) if ocr_data else 0}

    confidences = [d["conf"] for d in ocr_data if d.get("conf", 0) > 0]

    if len(confidences) < 5:
        return 0, {"word_count": len(confidences)}

    conf_array = np.array(confidences, dtype=np.float32)
    mean_conf = float(np.mean(conf_array))
    std_conf = float(np.std(conf_array))

    # Count low-confidence regions (potential edited areas)
    low_conf = sum(1 for c in confidences if c < 50)
    low_conf_pct = low_conf / len(confidences) * 100

    # High std in confidence or many low-confidence words = suspicious
    score = 0
    if std_conf > 30 and low_conf_pct > 20:
        score = min(100, int(40 + std_conf + low_conf_pct))
    elif std_conf > 25:
        score = int(std_conf)
    elif low_conf_pct > 30:
        score = int(low_conf_pct)

    evidence = {
        "word_count": len(confidences),
        "mean_confidence": round(mean_conf, 1),
        "std_confidence": round(std_conf, 1),
        "low_confidence_pct": round(low_conf_pct, 1),
    }

    return score, evidence
