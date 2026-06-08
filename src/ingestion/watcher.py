"""
Folder Watcher — Monitors a directory for new invoice files.

Uses watchdog to detect new files dropped into the watched folder
and automatically triggers the processing pipeline.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from src.config import Config

logger = logging.getLogger(__name__)


class InvoiceFileHandler(FileSystemEventHandler):
    """Handles new file events in the watched folder."""

    def __init__(
        self,
        supported_extensions: list,
        callback: Callable[[str], None],
        debounce_seconds: float = 2.0,
    ):
        super().__init__()
        self._extensions = [ext.lower() for ext in supported_extensions]
        self._callback = callback
        self._debounce = debounce_seconds
        self._seen: dict = {}  # path → last_seen_time

    def on_created(self, event):
        if event.is_directory:
            return

        if not isinstance(event, FileCreatedEvent):
            return

        file_path = event.src_path
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self._extensions:
            logger.debug(f"Ignoring non-invoice file: {file_path}")
            return

        # Debounce: ignore duplicate events for the same file
        now = time.time()
        if file_path in self._seen:
            if now - self._seen[file_path] < self._debounce:
                return
        self._seen[file_path] = now

        # Wait briefly for file to finish writing
        self._wait_for_file(file_path)

        logger.info(f"New invoice detected: {file_path}")
        try:
            self._callback(file_path)
        except Exception as e:
            logger.error(f"Callback failed for {file_path}: {e}")

    def _wait_for_file(self, path: str, timeout: float = 10.0):
        """Wait until the file size stabilizes (finished writing)."""
        last_size = -1
        stable_count = 0
        start = time.time()

        while time.time() - start < timeout:
            try:
                size = os.path.getsize(path)
                if size == last_size and size > 0:
                    stable_count += 1
                    if stable_count >= 3:
                        return
                else:
                    stable_count = 0
                last_size = size
            except OSError:
                pass
            time.sleep(0.3)


class FolderWatcher:
    """
    Watches a folder for new invoice files and triggers processing.

    Usage:
        watcher = FolderWatcher(callback=process_fn)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        config: Optional[Config] = None,
    ):
        self._config = config or Config.get()
        self._callback = callback
        self._observer: Optional[Observer] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def watch_path(self) -> str:
        return str(Path(self._config.watcher_folder).resolve())

    def start(self, folder_path: Optional[str] = None):
        """Start watching the configured folder."""
        if self._running:
            logger.warning("Watcher is already running")
            return

        path = folder_path or self._config.watcher_folder
        path = str(Path(path).resolve())

        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        handler = InvoiceFileHandler(
            supported_extensions=self._config.supported_extensions,
            callback=self._callback,
        )

        self._observer = Observer()
        self._observer.schedule(handler, path, recursive=False)
        self._observer.start()
        self._running = True

        logger.info(f"Folder watcher started: {path}")

    def stop(self):
        """Stop the folder watcher."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info("Folder watcher stopped")

    def __del__(self):
        self.stop()
