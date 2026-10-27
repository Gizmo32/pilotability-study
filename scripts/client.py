"""
LLM Client for Ollama Integration

This module provides a client for interacting with Ollama API.
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import requests


@dataclass
class LLMResponse:
    """
    Response from LLM generation

    Attributes:
        text: Generated text
        model: Model name used
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        duration_ms: Duration of generation in milliseconds
    """
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)"""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMClient:
    """
    Client for interacting with Ollama LLM API

    Attributes:
        model: Model name to use (default: mistral:7b-instruct-q4_K_M)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        base_url: Ollama API base URL
        timeout: Request timeout in seconds
    """
    model: str = "mistral:7b-instruct-q4_K_M"
    temperature: float = 0.7
    max_tokens: int = 500
    base_url: str = "http://localhost:11434"
    timeout: int = 30

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from prompt using Ollama

        Args:
            prompt: User prompt
            system: System prompt (optional)
            **kwargs: Additional parameters to pass to Ollama

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If response is invalid
        """
        start_time = time.time()

        # Build request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        # Add system prompt if provided
        if system:
            payload["system"] = system

        # Add any additional kwargs to options
        for key, value in kwargs.items():
            payload["options"][key] = value

        # Make API call
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to generate text from Ollama: {e}"
            )

        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")

        # Validate response structure
        if "response" not in data:
            raise ValueError("Missing 'response' field in Ollama response")

        # Extract tokens and duration
        duration_ms = (time.time() - start_time) * 1000

        # Ollama returns total_duration in nanoseconds
        if "total_duration" in data:
            duration_ms = data["total_duration"] / 1_000_000

        # Extract token counts (Ollama provides these in response)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=data["response"],
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms
        )

    def is_available(self) -> bool:
        """
        Check if Ollama API is accessible

        Returns:
            True if Ollama is running and accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5  # Short timeout for availability check
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        List available models in Ollama

        Returns:
            List of model names

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to list models from Ollama: {e}"
            )

        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")

        models = data.get("models", [])
        return [model["name"] for model in models]

    def generate_with_retry(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text with automatic retry on failure

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_retries: Maximum number of retry attempts
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text

        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system, **kwargs)
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                continue

        # All retries failed
        raise requests.exceptions.RequestException(
            f"Failed after {max_retries} retries: {last_error}"
        )
