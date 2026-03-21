from __future__ import annotations
from agentos.providers.anthropic_provider import AnthropicProvider
from agentos.providers.ollama_provider import OllamaProvider
from agentos.providers.mock import MockProvider

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}
