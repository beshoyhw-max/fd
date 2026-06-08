"""
Configuration Management for Invoice Fraud Detection System.

Loads/saves config.yaml, provides typed access to all settings,
supports runtime updates from the Streamlit settings page.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Centralized configuration manager backed by config.yaml."""

    _instance: Optional["Config"] = None

    def __init__(self, config_path: Optional[Path] = None):
        self._path = config_path or CONFIG_PATH
        self._data: Dict = {}
        self.load()

    @classmethod
    def get(cls, config_path: Optional[Path] = None) -> "Config":
        """Singleton access to the configuration."""
        if cls._instance is None or (config_path and cls._instance._path != config_path):
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None

    # ── Loading / Saving ──────────────────────────────────────

    def load(self):
        """Load configuration from YAML file."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

    def save(self):
        """Persist current configuration to YAML file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)

    def update(self, updates: Dict):
        """Merge updates into configuration and save."""
        self._deep_merge(self._data, updates)
        self.save()

    # ── LLM Settings ──────────────────────────────────────────

    @property
    def llm_base_url(self) -> str:
        return self._data.get("llm", {}).get("base_url", "http://localhost:11434/v1")

    @property
    def llm_model(self) -> str:
        return self._data.get("llm", {}).get("model", "llava")

    @property
    def llm_temperature(self) -> float:
        return self._data.get("llm", {}).get("temperature", 0.1)

    @property
    def llm_max_tokens(self) -> int:
        return self._data.get("llm", {}).get("max_tokens", 4096)

    @property
    def llm_timeout(self) -> int:
        return self._data.get("llm", {}).get("timeout_seconds", 120)

    @property
    def llm_retry_attempts(self) -> int:
        return self._data.get("llm", {}).get("retry_attempts", 3)

    @property
    def llm_retry_delay(self) -> float:
        return self._data.get("llm", {}).get("retry_delay_seconds", 2)

    @property
    def llm_api_key(self) -> str:
        return self._data.get("llm", {}).get("api_key", "")

    # ── OCR Settings ──────────────────────────────────────────

    @property
    def ocr_enabled(self) -> bool:
        return self._data.get("ocr", {}).get("enabled", True)

    @property
    def ocr_languages(self) -> List[str]:
        return self._data.get("ocr", {}).get("languages", ["eng", "ara"])

    @property
    def ocr_psm(self) -> int:
        return self._data.get("ocr", {}).get("psm", 6)

    # ── Pipeline Settings ─────────────────────────────────────

    @property
    def concurrency(self) -> int:
        return self._data.get("pipeline", {}).get("concurrency", 3)

    @property
    def supported_extensions(self) -> List[str]:
        return self._data.get("pipeline", {}).get(
            "supported_extensions",
            [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
        )

    @property
    def max_file_size_mb(self) -> int:
        return self._data.get("pipeline", {}).get("max_file_size_mb", 50)

    # ── Watcher Settings ──────────────────────────────────────

    @property
    def watcher_enabled(self) -> bool:
        return self._data.get("watcher", {}).get("enabled", False)

    @property
    def watcher_folder(self) -> str:
        return self._data.get("watcher", {}).get("folder_path", "data/invoices/watch")

    @property
    def watcher_poll_interval(self) -> int:
        return self._data.get("watcher", {}).get("poll_interval_seconds", 5)

    # ── Risk Thresholds ───────────────────────────────────────

    @property
    def risk_threshold_low(self) -> int:
        return self._data.get("risk_thresholds", {}).get("low", 30)

    @property
    def risk_threshold_medium(self) -> int:
        return self._data.get("risk_thresholds", {}).get("medium", 60)

    # ── Fraud Detection ───────────────────────────────────────

    @property
    def approval_limit(self) -> float:
        return self._data.get("approval_limit", 10000.0)

    @property
    def fraud_weights(self) -> Dict[str, int]:
        return self._data.get("fraud_weights", {
            "math_verification": 15,
            "duplicate_detection": 12,
            "benford_law": 5,
            "round_number_bias": 4,
            "date_anomalies": 6,
            "threshold_splitting": 10,
            "ela_analysis": 12,
            "pdf_metadata": 8,
            "missing_fields": 5,
            "vendor_pattern": 8,
            "ai_generated": 12,
            "font_consistency": 8,
            "noise_analysis": 7,
        })

    @property
    def fraud_severities(self) -> Dict[str, str]:
        return self._data.get("fraud_severities", {
            "math_verification": "CRITICAL",
            "duplicate_detection": "CRITICAL",
            "benford_law": "MEDIUM",
            "round_number_bias": "MEDIUM",
            "date_anomalies": "MEDIUM",
            "threshold_splitting": "HIGH",
            "ela_analysis": "CRITICAL",
            "pdf_metadata": "HIGH",
            "missing_fields": "MEDIUM",
            "vendor_pattern": "HIGH",
            "ai_generated": "CRITICAL",
            "font_consistency": "CRITICAL",
            "noise_analysis": "HIGH",
        })

    @property
    def benford_min_samples(self) -> int:
        return self._data.get("benford_min_samples", 10)

    @property
    def duplicate_fuzzy_threshold(self) -> int:
        return self._data.get("duplicate", {}).get("fuzzy_threshold", 85)

    @property
    def duplicate_lookback_days(self) -> int:
        return self._data.get("duplicate", {}).get("lookback_days", 365)

    @property
    def date_max_past_days(self) -> int:
        return self._data.get("date_check", {}).get("max_past_days", 365)

    @property
    def date_weekend_flag(self) -> bool:
        return self._data.get("date_check", {}).get("weekend_flag", True)

    # ── Data Directories ──────────────────────────────────────

    @property
    def invoices_dir(self) -> Path:
        rel = self._data.get("data", {}).get("invoices_dir", "data/invoices")
        return PROJECT_ROOT / rel

    @property
    def results_dir(self) -> Path:
        rel = self._data.get("data", {}).get("results_dir", "data/results")
        return PROJECT_ROOT / rel

    @property
    def exports_dir(self) -> Path:
        rel = self._data.get("data", {}).get("exports_dir", "data/exports")
        return PROJECT_ROOT / rel

    @property
    def ela_output_dir(self) -> Path:
        rel = self._data.get("data", {}).get("ela_output_dir", "data/ela_output")
        return PROJECT_ROOT / rel

    def ensure_data_dirs(self):
        """Create all data directories if they don't exist."""
        for d in [self.invoices_dir, self.results_dir, self.exports_dir, self.ela_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Raw access ────────────────────────────────────────────

    @property
    def raw(self) -> Dict:
        """Direct access to the raw config dictionary."""
        return self._data

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _deep_merge(base: Dict, override: Dict):
        """Recursively merge override into base."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = v
