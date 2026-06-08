"""
OCR Fallback Module for Invoice Fraud Detection System.

Uses Surya OCR (primary) with Tesseract fallback for when the Vision LLM fails.
Surya provides dramatically better quality for multilingual (Arabic+English)
documents via its 650M-parameter model supporting 90+ languages.

Falls back to Tesseract if Surya is not installed.
"""

import logging
from typing import Dict, List, Optional

from PIL import Image

from src.config import Config
from src.extraction.vision_extractor import VisionExtractor, ExtractionError
from src.models import ExtractedFields

logger = logging.getLogger(__name__)


# ── OCR Backend Detection ────────────────────────────────────────


def _detect_surya() -> bool:
    """Check if Surya OCR is available."""
    try:
        from surya.recognition import RecognitionPredictor  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_tesseract() -> bool:
    """Check if Tesseract OCR is available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _detect_easyocr() -> bool:
    """Check if EasyOCR is available."""
    try:
        import easyocr
        return True
    except ImportError:
        return False


def _detect_paddleocr() -> bool:
    """Check if PaddleOCR is available."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


SURYA_AVAILABLE = _detect_surya()
TESSERACT_AVAILABLE = _detect_tesseract()
EASYOCR_AVAILABLE = _detect_easyocr()
PADDLEOCR_AVAILABLE = _detect_paddleocr()

if SURYA_AVAILABLE:
    logger.info("Surya OCR detected — using as primary OCR engine")
elif PADDLEOCR_AVAILABLE:
    logger.info("Surya OCR not found, PaddleOCR detected — using as fallback")
elif EASYOCR_AVAILABLE:
    logger.info("Surya/PaddleOCR not found, EasyOCR detected — using as fallback")
elif TESSERACT_AVAILABLE:
    logger.info("Surya/PaddleOCR/EasyOCR not found, Tesseract OCR detected — using as fallback")
else:
    logger.warning("No OCR engine available — OCR fallback disabled")


# ── Language Code Mapping ────────────────────────────────────────

# Map Tesseract language codes → Surya/ISO language codes
_LANG_MAP: Dict[str, str] = {
    "eng": "en",
    "ara": "ar",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "chi_sim": "zh",
    "chi_tra": "zh",
    "jpn": "ja",
    "kor": "ko",
    "tur": "tr",
    "hin": "hi",
    "urd": "ur",
    "fas": "fa",
    "nld": "nl",
}


def _map_languages(tesseract_langs: List[str]) -> List[str]:
    """Convert Tesseract language codes to Surya-compatible codes."""
    return [_LANG_MAP.get(lang, lang) for lang in tesseract_langs]


# ── Surya OCR Engine ─────────────────────────────────────────────


class _SuryaEngine:
    """
    Surya OCR engine wrapper.

    Lazily initializes the heavy model objects on first use to avoid
    loading the 650M model at import time.
    """

    def __init__(self):
        self._manager = None
        self._recognition = None

    def _ensure_loaded(self):
        """Lazy-load Surya models on first OCR call."""
        if self._manager is not None:
            return

        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        logger.info("Initializing Surya OCR models (first-time load)...")
        self._manager = SuryaInferenceManager()
        self._recognition = RecognitionPredictor(self._manager)
        logger.info("Surya OCR models loaded successfully")

    def ocr_image(self, img: Image.Image, languages: List[str]) -> str:
        """
        Run Surya OCR on a single image.

        Args:
            img: PIL Image to OCR.
            languages: List of Surya language codes (e.g., ["en", "ar"]).

        Returns:
            Extracted text string.
        """
        self._ensure_loaded()

        try:
            predictions = self._recognition([img])

            # Extract text from predictions
            lines = []
            for page in predictions:
                for line in page.text_lines:
                    text = line.text.strip()
                    if text:
                        lines.append(text)

            result = "\n".join(lines)
            logger.info(f"Surya OCR extracted {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Surya OCR failed: {e}")
            raise OCRError(f"Surya OCR failed: {e}")

    def get_per_word_data(
        self, img: Image.Image, languages: List[str]
    ) -> List[dict]:
        """
        Get per-word OCR data with bounding boxes (for font consistency check).

        Returns list of {char, conf, left, top, width, height} dicts.
        """
        self._ensure_loaded()

        try:
            predictions = self._recognition([img])

            results = []
            for page in predictions:
                for line in page.text_lines:
                    text = line.text.strip()
                    if not text:
                        continue

                    bbox = line.bbox  # [x1, y1, x2, y2]
                    confidence = getattr(line, "confidence", 0.9)
                    conf_pct = int(confidence * 100)

                    results.append({
                        "char": text,
                        "conf": conf_pct,
                        "left": int(bbox[0]),
                        "top": int(bbox[1]),
                        "width": int(bbox[2] - bbox[0]),
                        "height": int(bbox[3] - bbox[1]),
                    })

            return results

        except Exception as e:
            logger.warning(f"Surya per-word OCR failed: {e}")
            return []


# Singleton — lazy init means no cost until first use
_surya_engine = _SuryaEngine() if SURYA_AVAILABLE else None


# ── OCRFallback (Public API) ─────────────────────────────────────


class OCRFallback:
    """
    Multi-engine OCR fallback for when Vision LLM is unavailable.

    Priority: Surya OCR → EasyOCR → Tesseract OCR → Error

    Pipeline: Image → OCR → Raw text → LLM text parsing → ExtractedFields
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()
        self._vision_extractor = VisionExtractor(self._config)
        self._easyocr_reader = None
        self._paddleocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                import easyocr
                # Initialize EasyOCR reader - lazy load on first use
                logger.info("Initializing EasyOCR reader (lazy load)...")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

    def _get_surya_languages(self) -> List[str]:
        """Get Surya-compatible language codes from config."""
        return _map_languages(self._config.ocr_languages)

    # ── Single Image OCR ──────────────────────────────────────

    def ocr_image(self, img: Image.Image) -> str:
        """
        Run OCR on a single image using the best available engine.

        Args:
            img: PIL Image to OCR.

        Returns:
            Extracted text string.
        """
        # Try Surya first (much better quality)
        if SURYA_AVAILABLE and _surya_engine is not None:
            try:
                return _surya_engine.ocr_image(
                    img, self._get_surya_languages()
                )
            except OCRError:
                logger.warning("Surya OCR failed, falling back to PaddleOCR")

        # Try PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                return self._paddleocr_ocr_image(img)
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}, falling back to EasyOCR")

        # Try EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                return self._easyocr_ocr_image(img)
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}, falling back to Tesseract")

        # Fall back to Tesseract
        if TESSERACT_AVAILABLE:
            return self._tesseract_ocr_image(img)

        raise OCRError(
            "No OCR engine available. Install surya-ocr (recommended) "
            "or pytesseract+Tesseract."
        )

    def _easyocr_ocr_image(self, img: Image.Image) -> str:
        """Run EasyOCR on a single image."""
        import easyocr
        import numpy as np

        if self._easyocr_reader is None:
            # Lazy load EasyOCR reader with common languages
            self._easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

        # Convert PIL Image to numpy array
        img_array = np.array(img)
        results = self._easyocr_reader.readtext(img_array)
        lines = []
        for bbox, text, confidence in results:
            if text.strip():
                lines.append(text.strip())
        result = "\n".join(lines)
        logger.info(f"EasyOCR extracted {len(result)} characters")
        return result

    def _paddleocr_ocr_image(self, img: Image.Image) -> str:
        """Run PaddleOCR on a single image."""
        from paddleocr import PaddleOCR

        if self._paddleocr_reader is None:
            # Lazy load PaddleOCR reader
            self._paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='ch')

        results = self._paddleocr_reader.ocr(img, cls=True)
        lines = []
        if results and results[0]:
            for line in results[0]:
                text = line[1][0]
                if text.strip():
                    lines.append(text.strip())
        result = "\n".join(lines)
        logger.info(f"PaddleOCR extracted {len(result)} characters")
        return result

    def _tesseract_ocr_image(self, img: Image.Image) -> str:
        """Run Tesseract OCR on a single image (legacy fallback)."""
        import pytesseract

        languages = "+".join(self._config.ocr_languages)

        try:
            text = pytesseract.image_to_string(
                img,
                lang=languages,
                config=f"--psm {self._config.ocr_psm}",
            )
            logger.info(f"Tesseract OCR extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise OCRError(f"Tesseract OCR failed: {e}")

    # ── Multi-Page OCR ────────────────────────────────────────

    def ocr_images(self, images: List[Image.Image]) -> str:
        """
        Run OCR on multiple page images and concatenate results.

        Args:
            images: List of PIL Images (one per page).

        Returns:
            Combined text from all pages.
        """
        pages = []
        for i, img in enumerate(images):
            text = self.ocr_image(img)
            if text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(pages)

    # ── Full Extraction Pipeline ──────────────────────────────

    async def extract_with_ocr(
        self, images: List[Image.Image]
    ) -> ExtractedFields:
        """
        Full OCR fallback pipeline:
        1. Run OCR on images to get raw text
        2. Send text to LLM for structured parsing

        Args:
            images: List of page images.

        Returns:
            ExtractedFields parsed from OCR text.
        """
        # Step 1: OCR
        raw_text = self.ocr_images(images)

        if not raw_text.strip():
            raise OCRError("OCR produced no text from the images")

        logger.info(f"OCR produced {len(raw_text)} chars, sending to LLM for parsing")

        # Step 2: Send OCR text to LLM for structured extraction
        try:
            return await self._vision_extractor.extract_from_text(raw_text)
        except ExtractionError as e:
            raise OCRError(f"LLM parsing of OCR text failed: {e}")

    # ── Per-Character/Word Data (for font_consistency) ────────

    def get_per_char_confidence(self, img: Image.Image) -> List[dict]:
        """
        Get per-character/word OCR confidence data (used by font_consistency check).

        Returns list of {char, conf, left, top, width, height} dicts.
        """
        # Try Surya first
        if SURYA_AVAILABLE and _surya_engine is not None:
            try:
                result = _surya_engine.get_per_word_data(
                    img, self._get_surya_languages()
                )
                if result:
                    return result
            except Exception:
                logger.warning("Surya per-word data failed, trying PaddleOCR")

        # Try PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                return self._paddleocr_per_char(img)
            except Exception as e:
                logger.warning(f"PaddleOCR per-word data failed: {e}, trying EasyOCR")

        # Try EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                return self._easyocr_per_char(img)
            except Exception as e:
                logger.warning(f"EasyOCR per-word data failed: {e}, trying Tesseract")

        # Fall back to Tesseract
        if TESSERACT_AVAILABLE:
            return self._tesseract_per_char(img)

        return []

    def _paddleocr_per_char(self, img: Image.Image) -> List[dict]:
        """Get per-word OCR data from PaddleOCR."""
        from paddleocr import PaddleOCR

        if self._paddleocr_reader is None:
            self._paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='ch')

        results = self._paddleocr_reader.ocr(img, cls=True)
        word_data = []

        if results and results[0]:
            for line in results[0]:
                bbox = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = line[1][0]
                confidence = line[1][1]

                if not text.strip():
                    continue

                x1 = min(p[0] for p in bbox)
                y1 = min(p[1] for p in bbox)
                x2 = max(p[0] for p in bbox)
                y2 = max(p[1] for p in bbox)
                width = x2 - x1
                height = y2 - y1
                conf_pct = int(confidence * 100) if confidence else 0

                word_data.append({
                    "char": text.strip(),
                    "conf": conf_pct,
                    "left": int(x1),
                    "top": int(y1),
                    "width": int(width),
                    "height": int(height),
                })

        logger.info(f"PaddleOCR extracted {len(word_data)} words")
        return word_data

    def _easyocr_per_char(self, img: Image.Image) -> List[dict]:
        """Get per-word OCR data from EasyOCR."""
        import easyocr
        import numpy as np

        if self._easyocr_reader is None:
            # Lazy load EasyOCR reader with common languages
            self._easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

        # Convert PIL Image to numpy array
        img_array = np.array(img)
        results = self._easyocr_reader.readtext(img_array)
        word_data = []
        for bbox, text, confidence in results:
            if not text.strip():
                continue
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1 = min(p[0] for p in bbox)
            y1 = min(p[1] for p in bbox)
            x2 = max(p[0] for p in bbox)
            y2 = max(p[1] for p in bbox)
            width = x2 - x1
            height = y2 - y1
            conf_pct = int(confidence * 100) if confidence else 0

            word_data.append({
                "char": text.strip(),
                "conf": conf_pct,
                "left": int(x1),
                "top": int(y1),
                "width": int(width),
                "height": int(height),
            })

        logger.info(f"EasyOCR extracted {len(word_data)} words")
        return word_data

    def _tesseract_per_char(self, img: Image.Image) -> List[dict]:
        """Get per-character OCR data from Tesseract (legacy fallback)."""
        try:
            import pytesseract

            languages = "+".join(self._config.ocr_languages)

            data = pytesseract.image_to_data(
                img,
                lang=languages,
                config=f"--psm {self._config.ocr_psm}",
                output_type=pytesseract.Output.DICT,
            )

            results = []
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if text and conf >= 0:
                    results.append({
                        "char": text,
                        "conf": conf,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    })
            return results

        except Exception as e:
            logger.warning(f"Tesseract per-character OCR failed: {e}")
            return []


class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass
