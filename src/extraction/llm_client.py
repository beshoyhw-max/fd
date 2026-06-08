"""
OpenAI-Compatible LLM Client for Invoice Fraud Detection System.

Generic async HTTP client that supports:
- Text chat completions
- Vision (image) inputs
- Retry with exponential backoff
- Configurable base URL, model, temperature
"""

import asyncio
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx
from PIL import Image

from src.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async client for OpenAI-compatible LLM APIs.

    Supports both text-only and vision (image) messages.
    Works with Ollama, LM Studio, vLLM, or any OpenAI-compatible endpoint.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config.get()
        self._base_url = self._config.llm_base_url.rstrip("/")
        self._api_key = self._config.llm_api_key
        self._model = self._config.llm_model
        self._temperature = self._config.llm_temperature
        self._max_tokens = self._config.llm_max_tokens
        self._timeout = self._config.llm_timeout
        self._retry_attempts = self._config.llm_retry_attempts
        self._retry_delay = self._config.llm_retry_delay

    def _image_to_base64(self, img: Image.Image, format: str = "JPEG") -> str:
        """Convert a PIL Image to a base64-encoded data URI."""
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=95)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts (role + content).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.

        Raises:
            LLMError: If all retry attempts fail.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": False,
        }

        url = f"{self._base_url}"
        print(f"DEBUG LLM URL: {url}")
        print(f"DEBUG LLM API_KEY: {self._api_key}")
        print(f"DEBUG LLM MODEL: {self._model}")
        logger.info(f"LLM request URL: {url}")
        logger.info(f"LLM request payload: model={self._model}, max_tokens={max_tokens or self._max_tokens}")
        last_error = None

        for attempt in range(1, self._retry_attempts + 1):
            try:
                headers = {}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
                logger.info(f"LLM attempt {attempt}: posting to {url}")
                async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    # Extract the assistant message content
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMError("No choices in LLM response")

                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        raise LLMError("Empty content in LLM response")

                    logger.debug(f"LLM response (attempt {attempt}): {len(content)} chars")
                    return content

            except httpx.HTTPStatusError as e:
                last_error = e
                print(f"DEBUG HTTP ERROR: {e.response.status_code}")
                print(f"DEBUG HTTP RESPONSE: {e.response.text[:500]}")
                logger.warning(
                    f"LLM HTTP error (attempt {attempt}/{self._retry_attempts}): "
                    f"{e.response.status_code} — {e.response.text[:200]}"
                )
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"LLM connection error (attempt {attempt}/{self._retry_attempts}): {e}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM unexpected error (attempt {attempt}/{self._retry_attempts}): {e}"
                )

            if attempt < self._retry_attempts:
                delay = self._retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise LLMError(f"All {self._retry_attempts} LLM attempts failed. Last error: {last_error}")

    async def chat_with_images(
        self,
        prompt: str,
        images: List[Image.Image],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a vision chat completion with one or more images.

        Args:
            prompt: User text prompt.
            images: List of PIL Images to include.
            system_prompt: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.
        """
        # Build content array with text + images
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]

        for img in images:
            b64_uri = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": b64_uri},
            })

        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": content})

        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    async def chat_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a text-only chat completion.

        Args:
            prompt: User text prompt.
            system_prompt: Optional system message.
            temperature: Override default temperature.

        Returns:
            The assistant's response text.
        """
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self.chat(messages, temperature=temperature)

    async def health_check(self) -> bool:
        """Check if the LLM server is reachable."""
        try:
            url = f"{self._base_url}/models"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False


class LLMError(Exception):
    """Raised when LLM communication fails after all retries."""
    pass
