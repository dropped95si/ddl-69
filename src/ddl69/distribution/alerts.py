"""Signal distribution: Discord alerts, Email summaries, Telegram streaming"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional
import json

logger = logging.getLogger(__name__)


class DiscordSignalAlerts:
    """Send trading signals to Discord webhook"""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Args:
          webhook_url: Discord webhook URL (or env var DISCORD_WEBHOOK_URL)
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")

    def send_signal(
        self,
        symbol: str,
        signal: str,  # BUY, SELL, HOLD
        probability: float,
        confidence: float,
        price: float,
        sharpe: Optional[float] = None,
    ) -> bool:
        """Send signal to Discord"""
        try:
            import requests

            # Color based on signal
            color = {"BUY": 0x00FF00, "SELL": 0xFF0000, "HOLD": 0xFFFF00}.get(signal, 0xFFFFFF)

            embed = {
                "title": f"{signal} {symbol}",
                "color": color,
                "fields": [
                    {"name": "Price", "value": f"${price:.2f}", "inline": True},
                    {"name": "Probability", "value": f"{probability:.1%}", "inline": True},
                    {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

            if sharpe is not None:
                embed["fields"].append({"name": "Sharpe", "value": f"{sharpe:.2f}", "inline": True})

            payload = {"embeds": [embed]}

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )

            if response.status_code != 204:
                logger.warning(f"Discord post failed: {response.status_code}")
                return False

            logger.info(f"Discord alert sent: {signal} {symbol} @ ${price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def send_portfolio_summary(self, portfolio_status: dict) -> bool:
        """Send portfolio-wide summary to Discord"""
        try:
            import requests

            buy_count = portfolio_status.get("buy_count", 0)
            sell_count = portfolio_status.get("sell_count", 0)
            hold_count = portfolio_status.get("hold_count", 0)
            sharpe = portfolio_status.get("portfolio_sharpe", 0)

            embed = {
                "title": "Portfolio Summary",
                "color": 0x0099FF,
                "fields": [
                    {"name": "🟢 BUY Signals", "value": str(buy_count), "inline": True},
                    {"name": "🔴 SELL Signals", "value": str(sell_count), "inline": True},
                    {"name": "🟡 HOLD", "value": str(hold_count), "inline": True},
                    {"name": "Portfolio Sharpe", "value": f"{sharpe:.2f}", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

            payload = {"embeds": [embed]}

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )

            return response.status_code == 204

        except Exception as e:
            logger.error(f"Failed to send portfolio summary: {e}")
            return False


class EmailSignalAlerts:
    """Send daily signal summaries via email"""

    def __init__(
        self,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
    ):
        """
        Args:
          sender_email: Email address (or env var EMAIL_SENDER)
          sender_password: Email password (or env var EMAIL_PASSWORD)
        """
        self.sender_email = sender_email or os.getenv("EMAIL_SENDER")
        self.sender_password = sender_password or os.getenv("EMAIL_PASSWORD")
        self.sender_name = os.getenv("EMAIL_SENDER_NAME", "DDL-69 Trading Signals")

    def send_daily_summary(
        self,
        recipient_email: str,
        symbols: list[str],
        signals: dict,
        metrics: dict,
    ) -> bool:
        """Send daily summary email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            if not self.sender_email or not self.sender_password:
                logger.warning("Email not configured")
                return False

            # Build HTML content
            html = f"""
            <html>
              <body style="font-family: Arial, sans-serif; background-color: #1a1a1a; color: #fff;">
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                  <h1>Trading Signals Summary - {datetime.now().strftime('%Y-%m-%d')}</h1>

                  <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                    <tr style="background-color: #333;">
                      <th style="padding: 10px; border: 1px solid #555;">Symbol</th>
                      <th style="padding: 10px; border: 1px solid #555;">Signal</th>
                      <th style="padding: 10px; border: 1px solid #555;">Probability</th>
                      <th style="padding: 10px; border: 1px solid #555;">Price</th>
                    </tr>
            """

            for symbol in symbols:
                sig = signals.get(symbol, {})
                signal = sig.get("signal", "N/A")
                prob = sig.get("probability", 0)
                price = sig.get("price", 0)

                signal_color = {"BUY": "#00FF00", "SELL": "#FF0000", "HOLD": "#FFFF00"}.get(
                    signal, "#FFFFFF"
                )

                html += f"""
                    <tr style="background-color: #222;">
                      <td style="padding: 10px; border: 1px solid #555;">{symbol}</td>
                      <td style="padding: 10px; border: 1px solid #555; color: {signal_color}; font-weight: bold;">{signal}</td>
                      <td style="padding: 10px; border: 1px solid #555;">{prob:.1%}</td>
                      <td style="padding: 10px; border: 1px solid #555;">${price:.2f}</td>
                    </tr>
                """

            # Summary metrics
            html += f"""
                  </table>

                  <div style="margin-top: 20px; padding: 10px; background-color: #333; border-radius: 5px;">
                    <h3>Portfolio Metrics</h3>
                    <p><strong>Sharpe Ratio:</strong> {metrics.get('sharpe', 0):.2f}</p>
                    <p><strong>Avg Accuracy:</strong> {metrics.get('accuracy', 0):.1%}</p>
                    <p><strong>Total Signals:</strong> {len(symbols)}</p>
                  </div>

                  <footer style="margin-top: 30px; text-align: center; color: #999; font-size: 12px;">
                    <p>This is an automated signal from DDL-69 Probability Engine</p>
                  </footer>
                </div>
              </body>
            </html>
            """

            # Send email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Trading Signals - {datetime.now().strftime('%Y-%m-%d')}"
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = recipient_email

            msg.attach(MIMEText(html, "html"))

            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=5)
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class TelegramSignalStream:
    """Send real-time signals to Telegram"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Args:
          bot_token: Telegram bot token (or env var TELEGRAM_BOT_TOKEN)
          chat_id: Telegram chat ID (or env var TELEGRAM_CHAT_ID)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured")

    def send_signal(
        self,
        symbol: str,
        signal: str,
        probability: float,
        price: float,
    ) -> bool:
        """Send signal to Telegram"""
        try:
            import requests

            # Emoji based on signal
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")

            message = (
                f"{emoji} <b>{signal}</b> {symbol}\n"
                f"Price: ${price:.2f}\n"
                f"Probability: {probability:.1%}\n"
                f"<i>{datetime.now().strftime('%H:%M:%S UTC')}</i>"
            )

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(url, json=payload, timeout=5)

            if response.status_code != 200:
                logger.warning(f"Telegram post failed: {response.status_code}")
                return False

            logger.info(f"Telegram alert sent: {signal} {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def send_portfolio_update(self, portfolio_status: dict) -> bool:
        """Send portfolio state to Telegram"""
        try:
            import requests

            buy = portfolio_status.get("buy_count", 0)
            sell = portfolio_status.get("sell_count", 0)
            hold = portfolio_status.get("hold_count", 0)
            sharpe = portfolio_status.get("portfolio_sharpe", 0)

            message = (
                f"📊 <b>Portfolio Update</b>\n\n"
                f"🟢 {buy} BUY\n"
                f"🔴 {sell} SELL\n"
                f"🟡 {hold} HOLD\n"
                f"📈 Sharpe: {sharpe:.2f}\n\n"
                f"<i>{datetime.now().strftime('%H:%M:%S %Z')}</i>"
            )

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send portfolio update: {e}")
            return False


class SignalDistributor:
    """Unified distributor for all alert channels"""

    def __init__(self):
        self.discord = DiscordSignalAlerts()
        self.email = EmailSignalAlerts()
        self.telegram = TelegramSignalStream()

    def distribute_signal(
        self,
        symbol: str,
        signal: str,
        probability: float,
        confidence: float,
        price: float,
        sharpe: Optional[float] = None,
    ) -> dict:
        """Send signal to all configured channels

        Returns:
          {discord: bool, email: bool, telegram: bool}
        """
        results = {
            "discord": self.discord.send_signal(symbol, signal, probability, confidence, price, sharpe),
            "telegram": self.telegram.send_signal(symbol, signal, probability, price),
        }

        logger.info(f"Signal distributed to {sum(results.values())} channels")
        return results

    def distribute_portfolio_update(self, portfolio_status: dict) -> dict:
        """Send portfolio summary to all channels"""
        results = {
            "discord": self.discord.send_portfolio_summary(portfolio_status),
            "telegram": self.telegram.send_portfolio_update(portfolio_status),
        }

        return results
