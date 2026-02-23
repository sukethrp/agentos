"""Vision Tool — let agents analyze images via the OpenAI Vision API.

Usage:
    from agentos.tools.vision_tool import vision_tool
    agent = Agent(tools=[vision_tool()])
"""

from __future__ import annotations

from agentos.core.tool import Tool
from agentos.core.multimodal import analyze_image


def vision_tool(model: str = "gpt-4o") -> Tool:
    """Create a tool that accepts an image path or URL and returns a description / analysis.

    The agent supplies:
        - *image*: a local file path **or** an HTTPS URL to an image
        - *question*: what to analyze (defaults to "Describe this image in detail.")
    """

    def analyze_image_tool(
        image: str, question: str = "Describe this image in detail."
    ) -> str:
        """Analyze an image and answer a question about it using AI vision.

        Args:
            image: A file path or URL to the image to analyze.
            question: What to look for or ask about the image.
        """
        try:
            result = analyze_image(
                image_path_or_url=image,
                prompt=question,
                model=model,
            )
            return result
        except FileNotFoundError as e:
            return f"Image not found: {e}"
        except ValueError as e:
            return f"Invalid image: {e}"
        except Exception as e:
            return f"Vision analysis error: {e}"

    return Tool(
        fn=analyze_image_tool,
        name="analyze_image",
        description=(
            "Analyze an image using AI vision. Provide a file path or URL to an image "
            "and optionally a question about it. Returns a detailed description or answer. "
            "Use this when the user shares an image or asks about visual content."
        ),
    )
