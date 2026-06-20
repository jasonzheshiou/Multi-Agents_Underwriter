"""LLM client module wrapping an OpenAI-compatible API."""

import logging
import os
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "[LLM unavailable - using deterministic fallback]"


class LLMClient:
    """Client for interacting with an OpenAI-compatible LLM API.

    Loads configuration from a YAML file and provides a generate method
    that gracefully degrades to a deterministic fallback when the LLM
    service is unavailable.

    Attributes:
        base_url: The base URL of the OpenAI-compatible API.
        model: The model name to use for completions.
        context_window: Maximum context window size in tokens.
        wait_time: Wait time between retries in seconds.
        temperature: Default sampling temperature.
        max_tokens: Default maximum number of tokens to generate.

    """

    def __init__(self, config_path: str = "./config.yaml") -> None:
        """Initialize the LLMClient with configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.
                Defaults to './config.yaml'.

        """
        config = self._load_config(config_path)
        llm_config = config.get("llm") or {}

        self.base_url: str = llm_config.get("baseURL", "")
        self.model: str = llm_config.get("model", "")
        self.context_window: int = llm_config.get("context_window", 200000)
        self.wait_time: float = llm_config.get("wait_time", 3600)
        self.temperature: float = llm_config.get("temperature", 0.1)
        self.max_tokens: int = llm_config.get("max_tokens", 2000)

        logger.info(
            "LLMClient initialized: model=%s, baseURL=%s",
            self.model,
            self.base_url,
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load and parse a YAML configuration file.

        Args:
            config_path: Path to the YAML file to load.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file is not valid YAML.

        """
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Attempts to call the OpenAI-compatible API. On any failure
        (timeout, connection error, invalid response, or missing
        configuration), returns a deterministic fallback message.

        Args:
            prompt: The text prompt to send to the LLM.
            max_tokens: Maximum tokens to generate.
                Defaults to the configured max_tokens.
            temperature: Sampling temperature. Defaults to configured value.

        Returns:
            The LLM-generated response text, or a fallback message
            if the LLM service is unavailable.

        """
        if not self.base_url or not self.model:
            logger.warning("LLM not configured (base_url=%s, model=%s)",
                           self.base_url, self.model)
            return FALLBACK_MESSAGE

        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            # Use the OpenAI Python client with a custom base_url
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv("LLM_API_KEY", "not-needed"),
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if response.choices:
                msg = response.choices[0].message
                result = msg.content

                # Log token usage details (including reasoning tokens for thinking models)
                if response.usage:
                    usage = response.usage
                    reasoning_tokens = 0
                    if usage.completion_tokens_details and hasattr(
                        usage.completion_tokens_details, "reasoning_tokens"
                    ):
                        reasoning_tokens = (
                            usage.completion_tokens_details.reasoning_tokens or 0
                        )
                    logger.info(
                        "LLM tokens: prompt=%d completion=%d reasoning=%d total=%d",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        reasoning_tokens,
                        usage.total_tokens,
                    )

                # Handle thinking models: content may be empty but reasoning_content present
                if result is None or not str(result).strip():
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning and str(reasoning).strip():
                        logger.warning(
                            "LLM content empty but reasoning_content present "
                            "(%d chars). Thinking model exhausted token budget "
                            "before producing output. Increase max_tokens.",
                            len(str(reasoning)),
                        )
                    else:
                        logger.warning("LLM returned null or empty content")
                    return FALLBACK_MESSAGE

                logger.info("LLM generate succeeded: %d chars", len(result))
                return result

            logger.warning("LLM returned no choices")
            return FALLBACK_MESSAGE

        except Exception as exc:
            logger.error("LLM generate failed: %s", exc)
            return FALLBACK_MESSAGE

    def is_available(self) -> bool:
        """Check whether the LLM service appears to be configured.

        Returns:
            True if base_url and model are configured, False otherwise.
        """
        return bool(self.base_url and self.model)

    def chat(self, prompt: str) -> str:
        """Convenience wrapper around generate() for agent rebuttals.

        Args:
            prompt: The text prompt to send to the LLM.

        Returns:
            The LLM-generated response text, or a fallback message.
        """
        return self.generate(prompt)

    def chat_completion(self, messages: list) -> dict:
        """Convenience wrapper for chat-completion style API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Dict with 'choices' key containing the LLM response,
            or a fallback dict if the LLM is unavailable.
        """
        if not self.base_url or not self.model:
            return {"choices": [{"message": {"content": FALLBACK_MESSAGE}}]}

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv("LLM_API_KEY", "not-needed"),
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if response.choices:
                msg = response.choices[0].message
                content = msg.content

                # Log reasoning tokens for thinking models
                if response.usage and response.usage.completion_tokens_details:
                    details = response.usage.completion_tokens_details
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                    if reasoning_tokens > 0:
                        logger.info(
                            "LLM chat_completion: reasoning_tokens=%d / completion_tokens=%d",
                            reasoning_tokens,
                            response.usage.completion_tokens,
                        )

                if content is None or not str(content).strip():
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning and str(reasoning).strip():
                        logger.warning(
                            "LLM chat_completion: content empty, reasoning_content "
                            "present (%d chars) — increase max_tokens for thinking model",
                            len(str(reasoning)),
                        )
                    else:
                        logger.warning("LLM chat_completion returned null or empty content")
                    return {"choices": [{"message": {"content": FALLBACK_MESSAGE}}]}
                return {"choices": [{"message": {"content": content}}]}

            return {"choices": [{"message": {"content": FALLBACK_MESSAGE}}]}

        except Exception as exc:
            logger.error("LLM chat_completion failed: %s", exc)
            return {"choices": [{"message": {"content": FALLBACK_MESSAGE}}]}
