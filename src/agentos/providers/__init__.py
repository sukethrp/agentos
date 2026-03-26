from __future__ import annotations
from agentos.providers.mock import MockProvider

PROVIDERS: dict[str, type] = {"mock": MockProvider}

# Providers have optional third-party dependencies. Import them lazily and
# only register them when their requirements are available.
try:
    from agentos.providers.anthropic_provider import AnthropicProvider

    PROVIDERS["anthropic"] = AnthropicProvider
except ModuleNotFoundError:
    # e.g. missing `anthropic` package
    pass

try:
    from agentos.providers.ollama_provider import OllamaProvider

    PROVIDERS["ollama"] = OllamaProvider
except ModuleNotFoundError:
    # e.g. missing `ollama`-related deps
    pass
