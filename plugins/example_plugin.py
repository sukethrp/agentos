"""Example AgentOS Plugin — adds a "translate" tool.

This is a simple function-style plugin.  It defines module-level metadata
and a ``register(ctx)`` function that registers one tool.

Usage:
    pm = PluginManager()
    pm.discover_plugins("plugins/")
    pm.load_plugin("example_plugin")
    tools = pm.get_tools_list()  # includes "translate"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.core.tool import Tool

# ── Plugin metadata (optional but recommended) ──
PLUGIN_NAME = "example-translate"
PLUGIN_VERSION = "0.1.0"
PLUGIN_DESCRIPTION = "Adds a translate tool that converts text between languages."
PLUGIN_AUTHOR = "AgentOS Team"


# ── The tool ──

def _translate(text: str, target_language: str = "Spanish") -> str:
    """Translate text to a target language.

    This is a lightweight mock — for production you would call
    a real translation API.  It demonstrates the plugin pattern.
    """
    # Simple phrasebook for demo purposes
    phrasebook = {
        "hello": {"spanish": "hola", "french": "bonjour", "german": "hallo",
                  "japanese": "こんにちは", "italian": "ciao"},
        "goodbye": {"spanish": "adiós", "french": "au revoir", "german": "auf wiedersehen",
                    "japanese": "さようなら", "italian": "arrivederci"},
        "thank you": {"spanish": "gracias", "french": "merci", "german": "danke",
                      "japanese": "ありがとう", "italian": "grazie"},
        "yes": {"spanish": "sí", "french": "oui", "german": "ja",
                "japanese": "はい", "italian": "sì"},
        "no": {"spanish": "no", "french": "non", "german": "nein",
               "japanese": "いいえ", "italian": "no"},
        "please": {"spanish": "por favor", "french": "s'il vous plaît", "german": "bitte",
                   "japanese": "お願いします", "italian": "per favore"},
        "how are you": {"spanish": "¿cómo estás?", "french": "comment allez-vous?",
                        "german": "wie geht es Ihnen?", "japanese": "お元気ですか?",
                        "italian": "come stai?"},
    }

    lang = target_language.lower().strip()
    key = text.lower().strip()

    if key in phrasebook and lang in phrasebook[key]:
        translated = phrasebook[key][lang]
        return f'"{text}" in {target_language} → "{translated}"'

    # Fallback: tell the agent this phrase isn't in our phrasebook
    available_langs = ", ".join(["Spanish", "French", "German", "Japanese", "Italian"])
    available_phrases = ", ".join(sorted(phrasebook.keys()))
    return (
        f"[mock translator] Cannot translate \"{text}\" to {target_language}. "
        f"Known phrases: {available_phrases}. "
        f"Available languages: {available_langs}."
    )


translate_tool = Tool(
    fn=_translate,
    name="translate",
    description=(
        "Translate a word or phrase to another language. "
        "Parameters: text (str), target_language (str, default 'Spanish'). "
        "Supports: Spanish, French, German, Japanese, Italian. "
        "Known phrases: hello, goodbye, thank you, yes, no, please, how are you."
    ),
)


# ── register() — called by PluginManager ──

def register(ctx):
    """Register the translate tool with AgentOS."""
    ctx.add_tool("translate", translate_tool)
