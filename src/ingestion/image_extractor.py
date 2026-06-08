"""
Image Preprocessing Module for Invoice Fraud Detection System.

Handles:
- Loading images from various formats (JPEG, PNG, TIFF, BMP)
- Deskewing (rotation correction)
- Denoising
- Contrast enhancement
- Resolution normalization
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Preprocesses invoice images for optimal Vision LLM / OCR extraction.

    Pipeline: Load → Deskew → Denoise → Enhance contrast → Normalize
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def load_and_preprocess(self, image_path: str) -> Image.Image:
        """
        Load an image file and apply full preprocessing pipeline.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed PIL Image.
        """
        img = Image.open(image_path).convert("RGB")
        logger.info(f"Loaded image: {image_path} ({img.size[0]}×{img.size[1]})")
        return self.preprocess(img)

    def preprocess(self, img: Image.Image) -> Image.Image:
        """
        Apply full preprocessing pipeline to a PIL Image.

        Steps: Deskew → Denoise → Enhance contrast.
        """
        # Convert to OpenCV format
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # 1. Deskew
        cv_img = self._deskew(cv_img)

        # 2. Denoise
        cv_img = self._denoise(cv_img)

        # 3. Enhance contrast
        cv_img = self._enhance_contrast(cv_img)

        # Convert back to PIL
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Correct skew/rotation in scanned documents.

        Uses Hough Line Transform to detect dominant angle.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10,
            )

            if lines is None or len(lines) < 5:
                return img

            # Calculate dominant angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 15:
                    angles.append(angle)

            if not angles:
                return img

            median_angle = np.median(angles)

            # Only correct if skew is significant but not extreme
            if abs(median_angle) < 0.5 or abs(median_angle) > 10:
                return img

            # Rotate
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            logger.debug(f"Deskewed by {median_angle:.2f}°")
            return rotated

        except Exception as e:
            logger.warning(f"Deskew failed, using original: {e}")
            return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Remove scanner noise while preserving text edges.

        Uses Non-Local Means Denoising (conservative settings to preserve detail).
        """
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                img,
                h=6,        # Luminance filter strength (lower = less aggressive)
                hColor=6,   # Color filter strength
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised
        except Exception as e:
            logger.warning(f"Denoise failed, using original: {e}")
            return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Applied to the L channel in LAB color space to avoid color shifts.
        """
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)

            enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced
        except Exception as e:
            logger.warning(f"Contrast enhancement failed, using original: {e}")
            return img

    def to_grayscale(self, img: Image.Image) -> Image.Image:
        """Convert PIL image to grayscale (useful for OCR)."""
        return img.convert("L")

    def resize_for_llm(
        self, img: Image.Image, max_dimension: int = 2048
    ) -> Image.Image:
        """
        Resize image to fit within max_dimension while preserving aspect ratio.

        Vision LLMs often have input size limits.
        """
        w, h = img.size
        if max(w, h) <= max_dimension:
            return img

        if w > h:
            new_w = max_dimension
            new_h = int(h * max_dimension / w)
        else:
            new_h = max_dimension
            new_w = int(w * max_dimension / h)

        return img.resize((new_w, new_h), Image.LANCZOS)
