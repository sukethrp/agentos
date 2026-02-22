from __future__ import annotations
from agentos.providers.anthropic_provider import AnthropicProvider
from agentos.providers.ollama_provider import OllamaProvider

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}
