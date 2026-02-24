"""Tests for the alert system."""

from unittest.mock import MagicMock, patch

import pytest

from strategylab.monitor.alert import send_alert


class TestSendAlert:
    @patch("strategylab.monitor.alert.load_settings")
    @patch("strategylab.monitor.alert.os.environ.get")
    def test_missing_credentials(self, mock_env_get, mock_settings):
        mock_settings.return_value = {
            "email": {
                "sender_env_var": "ALERT_EMAIL_SENDER",
                "password_env_var": "ALERT_EMAIL_PASSWORD",
            }
        }
        mock_env_get.return_value = ""  # empty credentials

        result = send_alert("Test", "Body")
        assert result is False

    @patch("strategylab.monitor.alert.smtplib.SMTP")
    @patch("strategylab.monitor.alert.load_settings")
    @patch("strategylab.monitor.alert.os.environ.get")
    def test_successful_send(self, mock_env_get, mock_settings, mock_smtp):
        mock_settings.return_value = {
            "email": {
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "sender_env_var": "SENDER",
                "password_env_var": "PASS",
                "recipients": ["test@example.com"],
            }
        }
        mock_env_get.side_effect = lambda key, default="": {
            "SENDER": "from@test.com",
            "PASS": "password123",
        }.get(key, default)

        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        result = send_alert(
            "Test Subject",
            "Test body",
            body_html="<p>Test</p>",
        )
        assert result is True
        mock_server.sendmail.assert_called_once()

    @patch("strategylab.monitor.alert.load_settings")
    @patch("strategylab.monitor.alert.os.environ.get")
    def test_no_recipients(self, mock_env_get, mock_settings):
        mock_settings.return_value = {
            "email": {
                "sender_env_var": "SENDER",
                "password_env_var": "PASS",
                "recipients": [],
            }
        }
        mock_env_get.side_effect = lambda key, default="": {
            "SENDER": "from@test.com",
            "PASS": "password123",
        }.get(key, default)

        result = send_alert("Test", "Body")
        assert result is False
