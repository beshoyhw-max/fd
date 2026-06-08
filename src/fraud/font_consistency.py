"""
Font Consistency Analysis — Fraud Detection Module #12.

Detects document tampering by analyzing font rendering consistency:
- Stroke width uniformity across text regions
- Anti-aliasing profile comparison
- OCR confidence variance (edited text has different patterns)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.config import Config
from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "font_consistency"


def run(
    image: Optional[Image.Image] = None,
    ocr_data: Optional[List[Dict]] = None,
    invoice_id: Optional[str] = None,
    image_path: Optional[str] = None,
    **kwargs,
) -> FraudCheck:
    """
    Analyze font rendering consistency across the document.

    Args:
        image: PIL Image of the invoice page.
        ocr_data: Per-word OCR data (from Surya OCR or Tesseract) if available.
        invoice_id: Optional invoice ID for exporting detailed results.
        image_path: Optional path to original image file (for high-quality export).

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

    result = FraudCheck(
        name=CHECK_NAME,
        score=combined,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )

    # Export detailed results if invoice_id provided
    if invoice_id:
        _export_font_consistency(
            invoice_id=invoice_id,
            image=image,
            image_path=image_path,
            stroke_evidence=stroke_evidence,
            aa_evidence=aa_evidence,
            ocr_evidence=ocr_evidence,
            stroke_score=stroke_score,
            aa_score=aa_score,
            ocr_score=ocr_score,
            combined=combined,
            severity=severity,
            triggered=triggered,
        )

    return result


def _export_font_consistency(
    invoice_id: str,
    image: Image.Image,
    image_path: Optional[str] = None,
    stroke_evidence: Dict = None,
    aa_evidence: Dict = None,
    ocr_evidence: Dict = None,
    stroke_score: int = 0,
    aa_score: int = 0,
    ocr_score: int = 0,
    combined: int = 0,
    severity: Severity = Severity.LOW,
    triggered: bool = False,
):
    """Export detailed font consistency analysis to JSON file and draw boxes on image."""
    try:
        config = Config.get()
        output_dir = config.font_consistency_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load original image from file path for best quality
        if image_path and Path(image_path).exists():
            original_img = Image.open(image_path)
        else:
            original_img = image.copy()
        output_dir = config.font_consistency_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get outlier blocks from both analyses
        stroke_outliers = stroke_evidence.get("outlier_blocks", [])
        aa_outliers = aa_evidence.get("outlier_blocks", [])

        # Draw red boxes on the image
        img_array = np.array(original_img)
        original_height, original_width = img_array.shape[:2]

        # Scale factor from 512 (analysis size) to original image
        scale_x = original_width / 512
        scale_y = original_height / 512
        block_size = int(64 * scale_x)

        # Draw boxes for stroke outliers (red)
        for block in stroke_outliers:
            x = int(block["x"] * scale_x)
            y = int(block["y"] * scale_y)
            cv2.rectangle(img_array, (x, y), (x + block_size, y + block_size), (0, 0, 255), 3)

        # Draw boxes for anti-aliasing outliers (blue - to distinguish)
        for block in aa_outliers:
            x = int(block["x"] * scale_x)
            y = int(block["y"] * scale_y)
            cv2.rectangle(img_array, (x, y), (x + block_size, y + block_size), (255, 0, 0), 3)

        # Save the image with boxes
        result_image = Image.fromarray(img_array)
        image_path = output_dir / f"{invoice_id}.png"
        result_image.save(image_path)

        export_data = {
            "invoice_id": invoice_id,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "combined_score": combined,
            "severity": severity.value,
            "triggered": triggered,
            "stroke_analysis": {
                "score": stroke_score,
                "blocks": stroke_evidence.get("block_count", 0),
                "mean_stroke_width": stroke_evidence.get("mean_stroke_width"),
                "std_stroke_width": stroke_evidence.get("std_stroke_width"),
                "coefficient_of_variation": stroke_evidence.get("coefficient_of_variation"),
                "outlier_blocks": stroke_outliers,
            },
            "antialiasing_analysis": {
                "score": aa_score,
                "profiles": aa_evidence.get("profile_count", 0),
                "mean_gradient": aa_evidence.get("mean_gradient"),
                "std_gradient": aa_evidence.get("std_gradient"),
                "coefficient_of_variation": aa_evidence.get("coefficient_of_variation"),
                "outlier_blocks": aa_outliers,
            },
            "ocr_confidence": {
                "score": ocr_score,
                "word_count": ocr_evidence.get("word_count", 0),
                "mean_confidence": ocr_evidence.get("mean_confidence"),
                "std_confidence": ocr_evidence.get("std_confidence"),
                "low_confidence_pct": ocr_evidence.get("low_confidence_pct"),
            },
            "box_colors": {
                "stroke_outliers": "red",
                "antialiasing_outliers": "blue",
            },
        }

        output_path = output_dir / f"{invoice_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported font consistency analysis to {output_path} and {image_path}")
    except Exception as e:
        logger.error(f"Failed to export font consistency analysis: {e}")


def _analyze_stroke_consistency(image: Image.Image) -> Tuple[int, Dict]:
    """
    Analyze stroke width consistency across text regions.

    Segments the image into blocks and compares stroke widths.
    Tampered regions show different stroke characteristics.
    """
    gray = np.array(image.convert("L"))
    original_shape = gray.shape
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

    # Find outlier blocks (those with stroke width > 1.5 std from mean)
    outlier_blocks = []
    threshold = mean_width + 1.5 * std_width if std_width > 0 else mean_width * 2
    for sw in stroke_widths:
        if sw["stroke_width"] > threshold:
            outlier_blocks.append({
                "x": sw["x"],
                "y": sw["y"],
                "stroke_width": sw["stroke_width"],
            })

    evidence = {
        "block_count": len(stroke_widths),
        "mean_stroke_width": round(mean_width, 3),
        "std_stroke_width": round(std_width, 3),
        "coefficient_of_variation": round(cv_width, 3),
        "outlier_blocks": outlier_blocks,
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
    block_positions = []

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
                block_positions.append((x, y))

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

    # Find outlier blocks (those with gradient > 1.5 std from mean)
    outlier_blocks = []
    threshold_high = mean_profile + 1.5 * std_profile if std_profile > 0 else mean_profile * 2
    threshold_low = mean_profile - 1.5 * std_profile if std_profile > 0 else 0
    for i, gp in enumerate(gradient_profiles):
        if gp > threshold_high or (std_profile > 0 and gp < threshold_low):
            x, y = block_positions[i]
            outlier_blocks.append({
                "x": x,
                "y": y,
                "gradient": round(gp, 3),
            })

    evidence = {
        "profile_count": len(gradient_profiles),
        "mean_gradient": round(mean_profile, 3),
        "std_gradient": round(std_profile, 3),
        "coefficient_of_variation": round(cv_profile, 3),
        "outlier_blocks": outlier_blocks,
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
