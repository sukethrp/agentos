"""Slack Notification Tool — post messages to a Slack channel via webhook.

Configure via environment variable:
    SLACK_WEBHOOK_URL — an Incoming Webhook URL from Slack
"""

from __future__ import annotations

import os

from agentos.core.tool import Tool


def slack_notification_tool(
    *,
    webhook_url: str | None = None,
) -> Tool:
    """Create a tool that posts messages to Slack via an Incoming Webhook.

    Args:
        webhook_url: Slack webhook URL.  Falls back to ``SLACK_WEBHOOK_URL``.
    """

    def send_slack_message(message: str, channel: str = "") -> str:
        """Post a message to Slack. Optionally override the channel."""
        import httpx

        url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        if not url:
            return "Error: SLACK_WEBHOOK_URL not configured."

        payload: dict = {"text": message}
        if channel:
            payload["channel"] = channel

        try:
            resp = httpx.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                return f"Slack message sent: '{message[:80]}'"
            return f"Slack API error ({resp.status_code}): {resp.text[:200]}"
        except httpx.HTTPError as e:
            return f"HTTP Error: {e}"

    return Tool(
        fn=send_slack_message,
        name="slack_notify",
        description=(
            "Send a notification message to a Slack channel via webhook. "
            "Provide the message text and optionally a channel override."
        ),
        timeout_seconds=15.0,
    )
