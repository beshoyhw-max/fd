"""
Error Level Analysis (ELA) — Fraud Detection Module #10.

Detects image tampering by re-saving the image at a known JPEG quality
and comparing error levels. Edited/pasted regions show different
compression artifacts than the original content.

Produces an ELA heatmap image saved to data/ela_output/.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.config import Config
from src.models import FraudCheck, Severity

logger = logging.getLogger(__name__)

CHECK_NAME = "ela_analysis"

# ELA parameters
QUALITY = 95           # JPEG re-compression quality
SCALE_FACTOR = 15      # Multiply error levels for visibility
SUSPICIOUS_THRESHOLD = 25   # Mean error above this is suspicious
HIGH_THRESHOLD = 40    # Mean error above this is highly suspicious
REGION_THRESHOLD = 60  # Per-region threshold for localized tampering


def run(
    image: Optional[Image.Image] = None,
    invoice_id: str = "",
    config: Optional[Config] = None,
    **kwargs,
) -> FraudCheck:
    """
    Run Error Level Analysis on an invoice image.

    Args:
        image: PIL Image of the invoice page.
        invoice_id: ID for saving the ELA output image.
        config: Optional config override.

    Returns:
        FraudCheck with ELA analysis results.
    """
    config = config or Config.get()

    if image is None:
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail="No image available for ELA analysis (text PDF — skipped).",
        )

    try:
        ela_image, stats = _compute_ela(image)
    except Exception as e:
        logger.error(f"ELA computation failed: {e}")
        return FraudCheck(
            name=CHECK_NAME,
            score=0,
            severity=Severity.LOW,
            triggered=False,
            detail=f"ELA analysis failed: {e}",
        )

    # Save ELA heatmap
    ela_path = None
    if invoice_id:
        try:
            output_dir = config.ela_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            ela_path = str(output_dir / f"{invoice_id}_ela.png")
            ela_image.save(ela_path)
            logger.info(f"ELA heatmap saved: {ela_path}")
        except Exception as e:
            logger.warning(f"Failed to save ELA image: {e}")

    # Analyze results
    mean_error = stats["mean_error"]
    max_error = stats["max_error"]
    high_error_pct = stats["high_error_pct"]

    evidence = {
        "mean_error": round(mean_error, 2),
        "max_error": round(max_error, 2),
        "std_error": round(stats["std_error"], 2),
        "high_error_pct": round(high_error_pct, 2),
        "high_error_regions": stats.get("high_error_regions", 0),
        "ela_image_path": ela_path,
    }

    # Score based on error levels
    if mean_error >= HIGH_THRESHOLD or high_error_pct > 15:
        score = min(100, int(60 + mean_error))
        severity = Severity.CRITICAL
        triggered = True
        detail = (
            f"HIGH tampering indicators: mean error={mean_error:.1f}, "
            f"max={max_error:.1f}, {high_error_pct:.1f}% of regions show "
            f"elevated compression artifacts."
        )
    elif mean_error >= SUSPICIOUS_THRESHOLD or high_error_pct > 5:
        score = min(80, int(30 + mean_error))
        severity = Severity.HIGH
        triggered = True
        detail = (
            f"Suspicious compression artifacts detected: mean error={mean_error:.1f}, "
            f"{high_error_pct:.1f}% of regions show elevated error levels."
        )
    else:
        score = 0
        severity = Severity.LOW
        triggered = False
        detail = (
            f"ELA check passed: mean error={mean_error:.1f}, "
            f"compression artifacts are consistent across the image."
        )

    return FraudCheck(
        name=CHECK_NAME,
        score=score,
        severity=severity,
        triggered=triggered,
        detail=detail,
        evidence=evidence,
    )


def _compute_ela(
    original: Image.Image,
) -> Tuple[Image.Image, dict]:
    """
    Compute Error Level Analysis.

    1. Re-compress the image at QUALITY level
    2. Compute the absolute difference
    3. Scale for visibility
    4. Analyze error distribution

    Returns:
        (ela_image, stats_dict)
    """
    # Ensure RGB
    original = original.convert("RGB")

    # Re-save at known quality
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=QUALITY)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # Compute difference
    orig_array = np.array(original, dtype=np.float32)
    resaved_array = np.array(resaved, dtype=np.float32)

    diff = np.abs(orig_array - resaved_array)

    # Scale for visibility
    ela_array = np.clip(diff * SCALE_FACTOR, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(ela_array)

    # Compute statistics
    gray_diff = np.mean(diff, axis=2)  # Average across channels
    mean_error = float(np.mean(gray_diff))
    max_error = float(np.max(gray_diff))
    std_error = float(np.std(gray_diff))

    # Percentage of pixels with high error
    high_error_mask = gray_diff > REGION_THRESHOLD
    high_error_pct = float(np.sum(high_error_mask) / gray_diff.size * 100)

    # Count distinct high-error regions using connected components
    high_error_regions = 0
    if high_error_pct > 0:
        binary = (gray_diff > REGION_THRESHOLD).astype(np.uint8) * 255
        num_labels, _ = cv2.connectedComponents(binary)
        high_error_regions = max(0, num_labels - 1)  # Subtract background

    stats = {
        "mean_error": mean_error,
        "max_error": max_error,
        "std_error": std_error,
        "high_error_pct": high_error_pct,
        "high_error_regions": high_error_regions,
    }

    return ela_image, stats
