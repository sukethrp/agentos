"""Email Sender Tool — send emails via SMTP.

Configure via environment variables:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

from agentos.core.tool import Tool


def email_sender_tool(
    *,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    from_address: str | None = None,
) -> Tool:
    """Create a tool that sends emails via SMTP.

    Parameters fall back to ``SMTP_*`` environment variables if not provided.
    """

    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email. Provide recipient address, subject line, and body text."""
        host = smtp_host or os.getenv("SMTP_HOST", "")
        port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        user = smtp_user or os.getenv("SMTP_USER", "")
        password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        sender = from_address or os.getenv("SMTP_FROM", user)

        if not host:
            return "Error: SMTP_HOST not configured. Set env var or pass smtp_host."

        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to

        try:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                if port != 25:
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.sendmail(sender, [to], msg.as_string())
            return f"Email sent to {to} with subject '{subject}'."
        except smtplib.SMTPException as e:
            return f"SMTP Error: {e}"

    return Tool(
        fn=send_email,
        name="send_email",
        description=(
            "Send an email via SMTP. Provide the recipient address (to), "
            "subject line, and body text. SMTP credentials are read from "
            "environment variables."
        ),
        timeout_seconds=20.0,
    )
