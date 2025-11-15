"""Background enrichment manager for channel processing."""
from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import database
from .youtube import EnrichmentError, enrich_channel, enrich_channel_email_only


NO_EMAIL_RETRY_WINDOW = dt.timedelta(days=30)


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
        force_run: bool = False,
        never_reenrich: bool = False,
    ) -> EnrichmentJob:
        if mode not in {"full", "email_only"}:
            raise ValueError(f"Unsupported enrichment mode: {mode}")
        if mode == "email_only":
            channels = database.get_channels_for_email_enrichment(limit)
        else:
            channels = database.get_pending_channels(limit)
        filtered, skipped = self._filter_channels(channels, force_run=force_run, never_reenrich=never_reenrich)
        job_id = str(uuid.uuid4())
        job = EnrichmentJob(
            job_id=job_id,
            channels=filtered,
            mode=mode,
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
                    if now - last_enriched_at < NO_EMAIL_RETRY_WINDOW:
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
        if job.mode == "email_only":
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
            enriched = enrich_channel(channel)
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
        email_gate_present = enriched.get("email_gate_present")
        result_value = "emails_found" if enriched_emails else "no_emails"
        database.update_channel_enrichment(
            channel_id,
            name=enriched.get("name") or enriched.get("title") or channel.get("name") or channel.get("title"),
            subscribers=enriched.get("subscribers"),
            language=enriched.get("language"),
            language_confidence=enriched.get("language_confidence"),
            emails=emails,
            email_gate_present=email_gate_present,
            last_updated=enriched.get("last_updated") or success_time,
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
                "subscribers": enriched.get("subscribers"),
                "language": enriched.get("language"),
                "languageConfidence": enriched.get("language_confidence"),
                "emails": enriched_emails,
                "lastUpdated": enriched.get("last_updated") or success_time,
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
        should_skip = bool(stored_emails)
        if not should_skip and display_emails:
            should_skip = database.has_all_known_emails(display_emails)
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
            enriched = enrich_channel_email_only(channel)
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
        result_value = "emails_found" if emails else "no_emails"
        database.update_channel_enrichment(
            channel_id,
            emails=emails_value,
            last_updated=last_updated,
            email_gate_present=email_gate_present,
            last_enriched_at=success_time,
            last_enriched_result=result_value,
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
                "mode": job.mode,
            }
        )

        if job.completed + job.errors >= job.total:
            job.mark_done()


manager = EnrichmentManager()

