# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | ✅ Current release |
| < 0.3   | ❌ Not supported   |

## Reporting a Vulnerability

If you discover a security vulnerability in AgentOS, please report
it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email: [your-email] or use GitHub's private vulnerability
   reporting feature on this repository.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline
for a fix.

## Security Best Practices for AgentOS Users

- **Never commit API keys.** Use `.env` files and `.env.example`
  as a template.
- **Use budget guards** to prevent runaway agent costs.
- **Enable permission guards** to block dangerous tool access.
- **Review audit trails** regularly in production deployments.
- **Use the kill switch** when testing new agent configurations.
- **Keep dependencies updated** — run `pip install --upgrade
  agentos-platform` regularly.

## Known Security Considerations

- The example `calculator` tool previously used `eval()`. This has
  been replaced with `simpleeval` for safe expression evaluation.
- JSON file storage is not suitable for multi-tenant production
  deployments. Use a proper database for sensitive data.
- The embeddable widget should be served over HTTPS in production.