#!/usr/bin/env -S uv run

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, api_url: str, default_model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.default_model = default_model

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API request."""
        pass

    @abstractmethod
    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        """Encode request body for API."""
        pass

    @abstractmethod
    def decode_response(self, response_data: Dict[str, Any]) -> str:
        """Decode API response to get text."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (also used for OpenRouter)."""

    def get_headers(self) -> Dict[str, str]:
        return {"authorization": f"bearer {self.api_key}"}

    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        return {"model": model, "messages": messages, **opts}

    def decode_response(self, response_data: Dict[str, Any]) -> str:
        return response_data["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def get_headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        # Anthropic requires max_tokens
        if "max_tokens" not in opts:
            opts["max_tokens"] = 1024
        return {"model": model, "messages": messages, **opts}

    def decode_response(self, response_data: Dict[str, Any]) -> str:
        return response_data["content"][0]["text"]


class LLMClient:
    """
    Client for LLM API operations with OpenAI, OpenRouter, and Anthropic support.
    Follows the existing client pattern with dependency injection.
    """

    def __init__(
        self,
        http_client: httpx.Client,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_api_base: str = "https://api.openai.com/v1/chat/completions",
        openrouter_api_base: str = "https://openrouter.ai/api/v1/chat/completions",
        anthropic_api_url: str = "https://api.anthropic.com/v1/messages",
    ):
        """
        Initialize the LLM client with configuration.

        Args:
            http_client: HTTP client for making API requests
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            openrouter_api_key: OpenRouter API key
            openai_api_base: OpenAI API endpoint
            openrouter_api_base: OpenRouter API endpoint
            anthropic_api_url: Anthropic API endpoint
        """
        self.http_client = http_client

        # Initialize providers
        self.providers: List[LLMProvider] = []

        if openai_api_key:
            self.providers.append(OpenAIProvider(openai_api_key, openai_api_base, "gpt-4o"))

        if anthropic_api_key:
            self.providers.append(AnthropicProvider(anthropic_api_key, anthropic_api_url, "claude-3-5-sonnet-20241022"))

        if openrouter_api_key:
            # OpenRouter uses OpenAI-compatible API
            self.providers.append(OpenAIProvider(openrouter_api_key, openrouter_api_base, "openrouter/auto"))

    def is_available(self) -> bool:
        """
        Check if LLM client has at least one API key available.

        Returns:
            True if at least one API key is available, False otherwise
        """
        return len(self.providers) > 0

    def generate_text(self, prompt: str, **opts) -> str:
        """
        Generate text using available LLM API.

        Args:
            prompt: Text prompt for generation
            **opts: Additional options (model, max_tokens, temperature, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If no API keys are available
            Exception: For API call failures
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_text_with_messages(messages, **opts)

    def generate_text_with_messages(self, messages: List[Dict[str, str]], **opts) -> str:
        """
        Generate text using available LLM API with multi-message conversation support.

        Args:
            messages: List of messages in conversation format [{"role": "user/assistant/system", "content": "..."}]
            **opts: Additional options (model, max_tokens, temperature, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If no API keys are available
            Exception: For API call failures
        """
        if not self.is_available():
            raise RuntimeError("No LLM API keys available")

        # Try providers in order until one succeeds
        last_error = None
        logger.debug(f"Attempting LLM generation with {len(self.providers)} available provider(s)")

        for i, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            logger.debug(f"Trying provider {i + 1}/{len(self.providers)}: {provider_name} at {provider.api_url}")
            try:
                # Prepare request
                model = opts.pop("model", provider.default_model)
                body = provider.encode_request(model, messages, opts)
                headers = provider.get_headers()

                # Set timeout
                timeout = opts.get("timeout", 60)

                logger.debug(f"Making LLM API request to {provider.api_url} with model {model}")
                logger.debug(f"=== LLM REQUEST ({len(messages)} messages) ===")
                for i, msg in enumerate(messages):
                    logger.debug(f"Message {i + 1} [{msg['role']}]:")
                    logger.debug(msg["content"])
                    logger.debug("---")

                # Make API request
                response = self.http_client.post(provider.api_url, headers=headers, json=body, timeout=timeout)
                response.raise_for_status()

                # Decode response
                response_json = response.json()
                result = provider.decode_response(response_json)
                logger.debug(f"=== LLM RESPONSE ({len(result)} chars) ===")
                logger.debug(result)

                return result

            except httpx.HTTPStatusError as e:
                logger.debug(f"Provider {provider_name} failed with HTTP error: {e.response.status_code}")
                logger.error(f"LLM API HTTP error: {e.response.status_code} - {e.response.text}")
                last_error = Exception(f"LLM API request failed: {e.response.status_code}")
                continue
            except httpx.TimeoutException as e:
                logger.debug(f"Provider {provider_name} failed with timeout")
                logger.error(f"LLM API timeout: {e}")
                last_error = Exception("LLM API request timed out")
                continue
            except KeyError as e:
                logger.debug(f"Provider {provider_name} failed with response parsing error")
                logger.error(f"LLM API response parsing error: {e}")
                last_error = Exception(f"Invalid LLM API response format: {e}")
                continue
            except Exception as e:
                logger.debug(f"Provider {provider_name} failed with error: {e}")
                logger.error(f"LLM API error: {e}")
                last_error = Exception(f"LLM API request failed: {str(e)}")
                continue

        # If we get here, all providers failed
        if last_error:
            raise last_error
        else:
            raise Exception("All LLM providers failed")
