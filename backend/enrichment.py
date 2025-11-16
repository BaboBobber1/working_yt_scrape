"""Background enrichment manager for channel processing."""
from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from . import database
from .youtube import EnrichmentError, enrich_channel_email_only, enrich_channel_with_options


NO_EMAIL_RETRY_WINDOW = dt.timedelta(days=30)


@dataclass(frozen=True)
class EnrichmentOptions:
    emails_from_channel: bool = True
    emails_from_videos: bool = True
    language_basic: bool = True
    language_precise: bool = True
    update_metadata: bool = True
    update_activity: bool = True
    telegram_enrichment: bool = True

    @classmethod
    def full(cls) -> "EnrichmentOptions":
        return cls()

    @classmethod
    def email_only(cls) -> "EnrichmentOptions":
        return cls(
            emails_from_channel=True,
            emails_from_videos=False,
            language_basic=False,
            language_precise=False,
            update_metadata=False,
            update_activity=False,
            telegram_enrichment=True,
        )

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "EnrichmentOptions":
        if not isinstance(payload, dict):
            return cls.full()
        return cls(
            emails_from_channel=bool(payload.get("emails_from_channel", True)),
            emails_from_videos=bool(payload.get("emails_from_videos", True)),
            language_basic=bool(payload.get("language_basic", True)),
            language_precise=bool(payload.get("language_precise", True)),
            update_metadata=bool(payload.get("update_metadata", True)),
            update_activity=bool(payload.get("update_activity", True)),
            telegram_enrichment=bool(payload.get("telegram_enrichment", True)),
        )

    @classmethod
    def with_modes(
        cls,
        *,
        base: Optional["EnrichmentOptions"] = None,
        emails_mode: Optional[str] = None,
        language_mode: Optional[str] = None,
    ) -> "EnrichmentOptions":
        options = base or cls.full()

        normalized_email_mode = (emails_mode or "").strip().lower()
        if normalized_email_mode:
            if normalized_email_mode == "off":
                options = replace(options, emails_from_channel=False, emails_from_videos=False)
            elif normalized_email_mode == "channel_only":
                options = replace(options, emails_from_channel=True, emails_from_videos=False)
            elif normalized_email_mode == "channel_and_videos":
                options = replace(options, emails_from_channel=True, emails_from_videos=True)
            elif normalized_email_mode == "videos_only":
                options = replace(options, emails_from_channel=False, emails_from_videos=True)

        normalized_language_mode = (language_mode or "").strip().lower()
        if normalized_language_mode:
            if normalized_language_mode == "off":
                options = replace(options, language_basic=False, language_precise=False)
            elif normalized_language_mode == "fast":
                options = replace(options, language_basic=True, language_precise=False)
            elif normalized_language_mode == "precise":
                options = replace(options, language_basic=True, language_precise=True)

        return options

    def mode_label(self) -> str:
        if self == self.full():
            return "full"
        if self == self.email_only():
            return "email_only"
        return "custom"

    def is_email_only(self) -> bool:
        return self == self.email_only()

    def asdict(self) -> Dict[str, Any]:
        return {
            "emails_from_channel": self.emails_from_channel,
            "emails_from_videos": self.emails_from_videos,
            "language_basic": self.language_basic,
            "language_precise": self.language_precise,
            "update_metadata": self.update_metadata,
            "update_activity": self.update_activity,
            "telegram_enrichment": self.telegram_enrichment,
        }


def _parse_iso_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@dataclass
class EnrichmentJob:
    """Represents a single enrichment batch run."""

    job_id: str
    channels: List[Dict]
    mode: str = "full"
    options: EnrichmentOptions = field(default_factory=EnrichmentOptions.full)
    started_at: float = field(default_factory=time.monotonic)
    completed: int = 0
    errors: int = 0
    requested: int = 0
    skipped: int = 0
    queue: "queue.Queue[Optional[Dict]]" = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    done_event: threading.Event = field(default_factory=threading.Event)

    def push_update(self, payload: Dict) -> None:
        self.queue.put(payload)

    def mark_done(self) -> None:
        if self.done_event.is_set():
            return
        self.done_event.set()
        self.queue.put(None)

    @property
    def total(self) -> int:
        return len(self.channels)

    def update_counts(self, *, completed: bool) -> None:
        with self.lock:
            if completed:
                self.completed += 1
            else:
                self.errors += 1
            summary = self.summary()
        self.push_update({"type": "progress", **summary})

    def summary(self) -> Dict:
        elapsed = time.monotonic() - self.started_at
        pending = max(0, self.total - self.completed - self.errors)
        return {
            "jobId": self.job_id,
            "total": self.total,
            "completed": self.completed,
            "errors": self.errors,
            "pending": pending,
            "durationSeconds": round(elapsed, 2),
            "mode": self.mode,
            "requested": self.requested,
            "skipped": self.skipped,
            "options": self.options.asdict(),
        }


class EnrichmentManager:
    """Coordinates enrichment jobs and exposes streaming progress."""

    def __init__(self, *, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, EnrichmentJob] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        limit: Optional[int],
        mode: str = "full",
        *,
        options: Optional[EnrichmentOptions] = None,
        scope: str = "all_active",
        category: database.ChannelCategory = database.ChannelCategory.ACTIVE,
        filters: Optional[database.ChannelFilters] = None,
        force_run: bool = False,
        never_reenrich: bool = False,
    ) -> EnrichmentJob:
        if options is None and mode not in {"full", "email_only"}:
            mode = "full"

        effective_options = options or (
            EnrichmentOptions.email_only() if mode == "email_only" else EnrichmentOptions.full()
        )
        mode = effective_options.mode_label()
        email_only_mode = effective_options.is_email_only()

        channels = database.get_enrichment_candidates(
            email_only_mode,
            scope=scope,
            category=category,
            filters=filters,
            limit=limit,
        )
        filtered, skipped = self._filter_channels(
            channels,
            options=effective_options,
            force_run=force_run,
            never_reenrich=never_reenrich,
        )
        job_id = str(uuid.uuid4())
        job = EnrichmentJob(
            job_id=job_id,
            channels=filtered,
            mode=mode,
            options=effective_options,
            requested=len(channels),
            skipped=len(skipped),
        )
        with self._lock:
            self._jobs[job_id] = job

        if not filtered:
            job.mark_done()
            return job

        for channel in filtered:
            self._executor.submit(self._process_channel, job, channel)

        # Emit initial summary to kick off UI progress display.
        job.push_update({"type": "progress", **job.summary()})
        return job

    def _filter_channels(
        self,
        channels: List[Dict],
        *,
        options: EnrichmentOptions,
        force_run: bool,
        never_reenrich: bool,
    ) -> Tuple[List[Dict], List[Dict]]:
        if force_run:
            return list(channels), []

        filtered: List[Dict] = []
        skipped: List[Dict] = []
        now = dt.datetime.utcnow()
        for channel in channels:
            should_skip = False
            last_enriched_at = _parse_iso_datetime(channel.get("last_enriched_at"))
            if never_reenrich and last_enriched_at:
                should_skip = True
            else:
                last_result = str(channel.get("last_enriched_result") or "").lower()
                if last_result == "no_emails" and last_enriched_at:
                    if now - last_enriched_at < NO_EMAIL_RETRY_WINDOW and not options.telegram_enrichment:
                        should_skip = True
            if should_skip:
                skipped.append(channel)
            else:
                filtered.append(channel)
        return filtered, skipped

    def stream(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)

        def event_stream():
            try:
                while True:
                    try:
                        item = job.queue.get(timeout=10)
                    except queue.Empty:
                        # Periodic heartbeat to keep connection alive.
                        yield "data: {}\n\n"
                        continue
                    if item is None:
                        summary = job.summary()
                        summary["done"] = True
                        yield f"data: {json.dumps({'type': 'progress', **summary})}\n\n"
                        break
                    yield f"data: {json.dumps(item)}\n\n"
            finally:
                job.mark_done()
                with self._lock:
                    self._jobs.pop(job_id, None)

        return event_stream()

    def get_job_summaries(self) -> Dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        summaries = []
        pending_total = 0
        for job in jobs:
            summary = job.summary()
            pending_total += int(summary.get("pending", 0) or 0)
            summaries.append(summary)
        return {
            "activeJobs": len(summaries),
            "pendingChannels": pending_total,
            "jobs": summaries,
        }

    def _process_channel(self, job: EnrichmentJob, channel: Dict) -> None:
        if job.options.is_email_only():
            self._process_channel_email_only(job, channel)
        else:
            self._process_channel_full(job, channel)

    def _process_channel_full(self, job: EnrichmentJob, channel: Dict) -> None:
        channel_id = channel["channel_id"]
        now = dt.datetime.utcnow().isoformat()
        database.update_channel_enrichment(
            channel_id,
            last_attempted=now,
        )
        database.set_channel_status(channel_id, "processing", reason=None, timestamp=now)
        job.push_update(
            {
                "type": "channel",
                "channelId": channel_id,
                "status": "processing",
                "statusReason": None,
                "lastStatusChange": now,
                "mode": job.mode,
            }
        )

        try:
            enriched = enrich_channel_with_options(channel, job.options)
        except EnrichmentError as exc:
            error_time = dt.datetime.utcnow().isoformat()
            reason = str(exc)
            database.update_channel_enrichment(
                channel_id,
                needs_enrichment=True,
                last_error=reason,
                status="error",
                status_reason=reason,
                last_status_change=error_time,
                last_enriched_at=error_time,
                last_enriched_result="error",
            )
            job.update_counts(completed=False)
            job.push_update(
                {
                    "type": "channel",
                    "channelId": channel_id,
                    "status": "error",
                    "statusReason": reason,
                    "lastStatusChange": error_time,
                    "mode": job.mode,
                }
            )
            if job.completed + job.errors >= job.total:
                job.mark_done()
            return
        except Exception as exc:  # Catch-all safety net
            error_time = dt.datetime.utcnow().isoformat()
            reason = f"Unexpected error: {exc}"[:500]
            database.update_channel_enrichment(
                channel_id,
                needs_enrichment=True,
                last_error=reason,
                status="error",
                status_reason=reason,
                last_status_change=error_time,
                last_enriched_at=error_time,
                last_enriched_result="error",
            )
            job.update_counts(completed=False)
            job.push_update(
                {
                    "type": "channel",
                    "channelId": channel_id,
                    "status": "error",
                    "statusReason": reason,
                    "lastStatusChange": error_time,
                    "mode": job.mode,
                }
            )
            if job.completed + job.errors >= job.total:
                job.mark_done()
            return

        success_time = dt.datetime.utcnow().isoformat()
        enriched_emails = enriched.get("emails") or []
        if enriched_emails:
            database.record_channel_emails(channel_id, enriched_emails, success_time)
        emails = ", ".join(enriched_emails) if enriched_emails else None
        email_gate_present = (
            enriched.get("email_gate_present")
            if job.options.emails_from_channel or job.options.emails_from_videos
            else None
        )
        telegram_account = enriched.get("telegram_account")
        telegram_value = (
            telegram_account if telegram_account is not None else "" if job.options.telegram_enrichment else None
        )
        result_value = None
        if job.options.emails_from_channel or job.options.emails_from_videos:
            result_value = "emails_found" if enriched_emails else "no_emails"
        if result_value is None:
            result_value = "completed"

        language_value = None
        language_confidence = None
        if job.options.language_basic or job.options.language_precise:
            language_value = enriched.get("language")
            language_confidence = enriched.get("language_confidence")
        database.update_channel_enrichment(
            channel_id,
            name=(
                enriched.get("name")
                if job.options.update_metadata
                else None
            ),
            subscribers=enriched.get("subscribers") if job.options.update_metadata else None,
            language=language_value,
            language_confidence=language_confidence,
            emails=emails,
            telegram_account=telegram_value,
            email_gate_present=email_gate_present,
            last_updated=enriched.get("last_updated") if job.options.update_activity else None,
            last_attempted=success_time,
            last_enriched_at=success_time,
            last_enriched_result=result_value,
            needs_enrichment=False,
            last_error=None,
            status="completed",
            status_reason=None,
            last_status_change=success_time,
        )

        job.update_counts(completed=True)
        job.push_update(
            {
                "type": "channel",
                "channelId": channel_id,
                "status": "completed",
                "statusReason": None,
                "lastStatusChange": success_time,
                "subscribers": enriched.get("subscribers") if job.options.update_metadata else None,
                "language": language_value,
                "languageConfidence": language_confidence,
                "emails": enriched_emails,
                "telegramAccount": telegram_value,
                "lastUpdated": enriched.get("last_updated") if job.options.update_activity else None,
                "emailGatePresent": email_gate_present,
                "mode": job.mode,
            }
        )

        if job.completed + job.errors >= job.total:
            job.mark_done()

    def _process_channel_email_only(self, job: EnrichmentJob, channel: Dict) -> None:
        channel_id = channel["channel_id"]
        start_time = dt.datetime.utcnow().isoformat()

        parsed_emails = database.parse_email_candidates(channel.get("emails"))
        stored_emails = database.get_channel_email_set(channel_id)
        display_emails: List[str] = list(parsed_emails)
        if not display_emails and stored_emails:
            display_emails = sorted(stored_emails)
        telegram_requested = job.options.telegram_enrichment
        should_skip = not telegram_requested and bool(stored_emails)
        if not should_skip and display_emails:
            should_skip = not telegram_requested and database.has_all_known_emails(display_emails)
        if should_skip:
            if display_emails:
                database.record_channel_emails(channel_id, display_emails, start_time)
            elif stored_emails:
                database.record_channel_emails(channel_id, stored_emails, start_time)
            emails_value = ", ".join(display_emails) if display_emails else channel.get("emails")
            if emails_value:
                database.update_channel_enrichment(
                    channel_id,
                    emails=emails_value,
                    email_gate_present=False,
                    last_enriched_at=start_time if display_emails or stored_emails else None,
                    last_enriched_result="emails_found" if display_emails or stored_emails else None,
                    telegram_account="" if telegram_requested else None,
                )
            job.update_counts(completed=True)
            job.push_update(
                {
                    "type": "channel",
                    "channelId": channel_id,
                    "status": "completed",
                    "statusReason": "emails unchanged",
                    "lastStatusChange": start_time,
                    "emails": display_emails,
                    "lastUpdated": channel.get("last_updated") or start_time,
                    "emailGatePresent": False,
                    "telegramAccount": "" if telegram_requested else None,
                    "mode": job.mode,
                }
            )
            if job.completed + job.errors >= job.total:
                job.mark_done()
            return

        job.push_update(
            {
                "type": "channel",
                "channelId": channel_id,
                "status": "processing",
                "statusReason": None,
                "lastStatusChange": start_time,
                "mode": job.mode,
            }
        )

        try:
            enriched = enrich_channel_email_only(
                channel, telegram_enabled=job.options.telegram_enrichment
            )
        except EnrichmentError as exc:
            error_time = dt.datetime.utcnow().isoformat()
            reason = str(exc)
            job.update_counts(completed=False)
            job.push_update(
                {
                    "type": "channel",
                    "channelId": channel_id,
                    "status": "error",
                    "statusReason": reason,
                    "lastStatusChange": error_time,
                    "mode": job.mode,
                }
            )
            database.update_channel_enrichment(
                channel_id,
                last_enriched_at=error_time,
                last_enriched_result="error",
            )
            if job.completed + job.errors >= job.total:
                job.mark_done()
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            error_time = dt.datetime.utcnow().isoformat()
            reason = f"Unexpected error: {exc}"[:500]
            job.update_counts(completed=False)
            job.push_update(
                {
                    "type": "channel",
                    "channelId": channel_id,
                    "status": "error",
                    "statusReason": reason,
                    "lastStatusChange": error_time,
                    "mode": job.mode,
                }
            )
            database.update_channel_enrichment(
                channel_id,
                last_enriched_at=error_time,
                last_enriched_result="error",
            )
            if job.completed + job.errors >= job.total:
                job.mark_done()
            return

        success_time = dt.datetime.utcnow().isoformat()
        emails = enriched.get("emails") or []
        if emails:
            database.record_channel_emails(channel_id, emails, success_time)
        emails_value = ", ".join(emails) if emails else None
        last_updated = enriched.get("last_updated") or success_time
        email_gate_present = enriched.get("email_gate_present")
        telegram_account = enriched.get("telegram_account")
        telegram_value = (
            telegram_account if telegram_account is not None else "" if job.options.telegram_enrichment else None
        )
        result_value = "emails_found" if emails else "no_emails"
        database.update_channel_enrichment(
            channel_id,
            emails=emails_value,
            last_updated=last_updated,
            email_gate_present=email_gate_present,
            last_enriched_at=success_time,
            last_enriched_result=result_value,
            telegram_account=telegram_value,
        )

        job.update_counts(completed=True)
        job.push_update(
            {
                "type": "channel",
                "channelId": channel_id,
                "status": "completed",
                "statusReason": None,
                "lastStatusChange": success_time,
                "emails": emails,
                "lastUpdated": last_updated,
                "emailGatePresent": email_gate_present,
                "telegramAccount": telegram_value,
                "mode": job.mode,
            }
        )

        if job.completed + job.errors >= job.total:
            job.mark_done()


manager = EnrichmentManager()

