from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
import smtplib

from daily_news_agent.models import BriefingResult


class MailDeliveryError(RuntimeError):
    """Raised when an email could not be delivered."""


@dataclass(frozen=True)
class EmailMessagePayload:
    subject: str
    plain_text: str
    sender: str
    recipients: list[str]
    html_text: str | None = None


class MailClient:
    def send(self, payload: EmailMessagePayload) -> str | None:
        raise NotImplementedError


def build_email_message(payload: EmailMessagePayload) -> EmailMessage:
    if not payload.sender.strip():
        raise ValueError("발신자 이메일 주소가 비어 있습니다.")
    recipients = [recipient.strip() for recipient in payload.recipients if recipient.strip()]
    if not recipients:
        raise ValueError("수신자 이메일 주소가 비어 있습니다.")

    message = EmailMessage()
    message["Subject"] = payload.subject.strip() or "Daily News Briefing"
    message["From"] = payload.sender.strip()
    message["To"] = ", ".join(recipients)
    message.set_content(payload.plain_text or "")
    if payload.html_text:
        message.add_alternative(payload.html_text, subtype="html")
    return message


def build_briefing_email_payload(
    briefing_result: BriefingResult,
    sender: str,
    recipient: str,
) -> EmailMessagePayload:
    article_lines = []
    for article in briefing_result.selected_articles:
        article_lines.append(
            "\n".join(
                [
                    f"- 제목: {article.title}",
                    f"  출처: {article.source}",
                    f"  발행일: {article.published_at or '정보 없음'}",
                    f"  링크: {article.link}",
                ]
            )
        )

    sections = [briefing_result.briefing_markdown.strip()]
    if article_lines:
        sections.append("선정 기사 링크\n" + "\n\n".join(article_lines))
    if briefing_result.errors:
        sections.append("수집 중 오류\n" + "\n".join(f"- {error}" for error in briefing_result.errors))

    return EmailMessagePayload(
        subject=f"Daily News Agent 브리핑 - {briefing_result.interest}",
        plain_text="\n\n".join(section for section in sections if section).strip(),
        sender=sender,
        recipients=[recipient],
    )


class SMTPMailClient(MailClient):
    def __init__(
        self,
        host: str,
        port: int,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout_seconds: int = 10,
    ) -> None:
        self.host = host.strip()
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.timeout_seconds = timeout_seconds

    def send(self, payload: EmailMessagePayload) -> None:
        if not self.host:
            raise ValueError("SMTP 호스트가 비어 있습니다.")

        message = build_email_message(payload)
        smtp_factory = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP

        try:
            with smtp_factory(self.host, self.port, timeout=self.timeout_seconds) as client:
                if self.use_tls and not self.use_ssl:
                    client.starttls()
                if self.username:
                    client.login(self.username, self.password)
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:
            raise MailDeliveryError(f"메일 전송에 실패했습니다: {exc}") from exc
