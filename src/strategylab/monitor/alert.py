"""Email alert sender."""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from strategylab.config import load_settings

logger = logging.getLogger(__name__)


def send_alert(
    subject: str,
    body_text: str,
    body_html: str | None = None,
    recipients: list[str] | None = None,
) -> bool:
    """Send an email alert.

    Args:
        subject: Email subject line.
        body_text: Plain text body.
        body_html: Optional HTML body.
        recipients: Override recipient list (defaults to config).

    Returns:
        True if sent successfully.
    """
    settings = load_settings()
    email_config = settings.get("email", {})

    sender = os.environ.get(email_config.get("sender_env_var", "ALERT_EMAIL_SENDER"), "")
    password = os.environ.get(email_config.get("password_env_var", "ALERT_EMAIL_PASSWORD"), "")

    if not sender or not password:
        logger.error("Email credentials not configured. Set %s and %s env vars.",
                      email_config.get("sender_env_var"), email_config.get("password_env_var"))
        return False

    if recipients is None:
        recipients = email_config.get("recipients", [])

    if not recipients:
        logger.error("No email recipients configured")
        return False

    smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
    smtp_port = email_config.get("smtp_port", 587)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Alert sent to %s: %s", recipients, subject)
        return True
    except Exception as e:
        logger.error("Failed to send alert: %s", e)
        return False


def send_test_email(to: str) -> bool:
    """Send a test email to verify configuration."""
    return send_alert(
        subject="[StrategyLab] Test Alert",
        body_text="This is a test email from StrategyLab. Your alert configuration is working.",
        body_html="<h2>StrategyLab Test Alert</h2><p>Your alert configuration is working.</p>",
        recipients=[to],
    )
