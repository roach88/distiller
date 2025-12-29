#!/usr/bin/env python3
"""
LLM provider abstraction for quality scoring.

Supports:
- OpenRouter (cloud API with 100+ models)
- Ollama (local models)
"""

import os
import sys
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import requests


class LLMProvider:
    """Base class for LLM providers."""

    def score_conversation(self, prompt: str, timeout: int = 30) -> Optional[str]:
        """Send prompt to LLM and return response."""
        raise NotImplementedError


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider using OpenAI-compatible interface."""

    def __init__(self, model: str, config: dict):
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model = model
        self.config = config

        # Get API key from environment
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at: https://openrouter.ai/keys"
            )

        # Initialize OpenAI client with OpenRouter base URL
        base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Optional headers for OpenRouter
        self.extra_headers = {}
        if config.get('site_url'):
            self.extra_headers['HTTP-Referer'] = config['site_url']
        if config.get('app_name'):
            self.extra_headers['X-Title'] = config['app_name']

    def score_conversation(self, prompt: str, timeout: int = 30) -> Optional[str]:
        """Call OpenRouter API for LLM judge evaluation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only response bot. You MUST respond with valid JSON only, no explanations or markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=timeout,
                extra_headers=self.extra_headers,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenRouter API error: {e}", file=sys.stderr)
            return None


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, model: str, config: dict):
        self.model = model
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:11434')

    def score_conversation(self, prompt: str, timeout: int = 120) -> Optional[str]:
        """Call Ollama API for LLM judge evaluation."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a JSON-only response bot. You MUST respond with valid JSON only, no explanations or markdown."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "format": "json",  # Request JSON format
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')

        except requests.exceptions.RequestException as e:
            print(f"Ollama API error: {e}", file=sys.stderr)
            return None


def load_scoring_config(config_path: Optional[Path] = None) -> dict:
    """Load scoring configuration from YAML file."""
    if config_path is None:
        # Default to config/scoring.yaml in distiller root
        distiller_root = Path(__file__).parent.parent
        config_path = distiller_root / "config" / "scoring.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Scoring config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_llm_provider(config_path: Optional[Path] = None) -> Optional[LLMProvider]:
    """
    Create LLM provider based on configuration.

    Returns None if provider is set to 'none' (heuristic-only mode).
    """
    config = load_scoring_config(config_path)

    provider_type = config.get('provider', 'none')

    if provider_type == 'none':
        return None

    model = config.get('model')
    if not model:
        raise ValueError("Model not specified in scoring config")

    if provider_type == 'openrouter':
        openrouter_config = config.get('openrouter', {})
        return OpenRouterProvider(model, openrouter_config)

    elif provider_type == 'ollama':
        ollama_config = config.get('ollama', {})
        return OllamaProvider(model, ollama_config)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
