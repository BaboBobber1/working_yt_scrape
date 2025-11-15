"""FastAPI application powering the Crypto YouTube Harvester."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from . import database
from .database import (
    ChannelCategory,
    ChannelFilters,
    channel_exists,
    ensure_channel_url,
    get_discovery_keyword_state,
    update_discovery_keyword_state,
)
from .enrichment import manager
from .state import discovery_state
from .youtube import (
    ChannelResolution,
    DiscoveryMetadata,
    fetch_discovery_metadata,
    normalize_channel_reference,
    resolve_channel,
    sanitize_channel_input,
    search_channels,
    search_channels_page,
)

app = FastAPI(title="Crypto YouTube Harvester")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

DEFAULT_KEYWORDS = [
    "crypto",
    "bitcoin",
    "ethereum",
    "defi",
    "altcoin",
    "memecoin",
    "onchain",
    "crypto trading",
]


RUN_UNTIL_STOPPED_MAX_PAGES_PER_RUN = 20
RUN_UNTIL_STOPPED_MAX_NEW_CHANNELS = 200
RUN_UNTIL_STOPPED_NO_NEW_THRESHOLD = 4


def _parse_multi(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    cleaned = [value.strip() for value in values if value and value.strip()]
    return cleaned or None


def _parse_int(value: Optional[str], *, field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{field} must be an integer")


def _unwrap_single_value(value: Any) -> Any:
    if isinstance(value, list):
        for candidate in value:
            unwrapped = _unwrap_single_value(candidate)
            if unwrapped not in (None, ""):
                return unwrapped
        return None
    return value


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    candidate = str(value).strip()
    if not candidate:
        return None
    candidate = candidate.replace(",", "").replace("_", "")
    if candidate.endswith("+"):
        candidate = candidate[:-1]
    if not candidate:
        return None
    multiplier_map = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = candidate[-1].lower()
    multiplier = multiplier_map.get(suffix)
    if multiplier:
        candidate = candidate[:-1]
        try:
            return int(float(candidate) * multiplier)
        except ValueError:
            return None
    try:
        return int(candidate)
    except ValueError:
        try:
            return int(float(candidate))
        except ValueError:
            return None


def _coerce_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _collect_filters(
    *,
    q: Optional[str],
    languages: Optional[List[str]],
    statuses: Optional[List[str]],
    min_subscribers: Optional[str],
    max_subscribers: Optional[str],
    emails_only: bool,
    include_archived: bool,
    email_gate_only: bool,
    unique_emails: bool,
) -> ChannelFilters:
    language_values = [value.lower() for value in _parse_multi(languages) or []] or None
    status_values = [value.lower() for value in _parse_multi(statuses) or []] or None
    if status_values:
        allowed = {"new", "processing", "completed", "error"}
        invalid = [value for value in status_values if value not in allowed]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid status values: {', '.join(invalid)}")

    min_subs_int = _parse_int(min_subscribers, field="min_subscribers")
    max_subs_int = _parse_int(max_subscribers, field="max_subscribers")
    if (
        min_subs_int is not None
        and max_subs_int is not None
        and min_subs_int > max_subs_int
    ):
        raise HTTPException(status_code=400, detail="min_subscribers cannot exceed max_subscribers")

    return ChannelFilters(
        query_text=q.strip() if q else None,
        languages=language_values,
        statuses=status_values,
        min_subscribers=min_subs_int,
        max_subscribers=max_subs_int,
        emails_only=emails_only,
        include_archived=include_archived,
        email_gate_only=email_gate_only,
        unique_emails=emails_only and unique_emails,
    )


def _parse_iso_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _parse_category(value: Optional[str]) -> ChannelCategory:
    if value is None:
        return ChannelCategory.ACTIVE
    try:
        return ChannelCategory(value.lower())
    except ValueError as exc:
        allowed = ", ".join(category.value for category in ChannelCategory)
        raise HTTPException(status_code=400, detail=f"category must be one of: {allowed}") from exc


@dataclass
class DiscoveryProcessingContext:
    now: str
    now_dt: dt.datetime
    deny_languages: Set[str]
    last_upload_max_age_days: Optional[int]
    requires_metadata: bool
    metadata_cache: Dict[str, DiscoveryMetadata]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return False
        return normalized in {"1", "true", "yes", "on"}
    return False


def _parse_last_upload_max_age(payload: Dict[str, Any]) -> Optional[int]:
    max_age_value = _unwrap_single_value(payload.get("last_upload_max_age_days"))
    if max_age_value in (None, ""):
        max_age_value = _unwrap_single_value(payload.get("lastUploadMaxAgeDays"))
    if isinstance(max_age_value, str):
        max_age_value = max_age_value.strip()
    if max_age_value == "":
        max_age_value = None
    if max_age_value is None:
        return None
    try:
        parsed = int(max_age_value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail="last_upload_max_age_days must be an integer"
        )
    if parsed < 0:
        raise HTTPException(
            status_code=400, detail="last_upload_max_age_days cannot be negative"
        )
    return parsed


def _parse_deny_languages(payload: Dict[str, Any]) -> Set[str]:
    deny_languages_raw = payload.get("deny_languages")
    if deny_languages_raw is None:
        deny_languages_raw = payload.get("denyLanguages")
    deny_languages: Set[str] = set()
    if isinstance(deny_languages_raw, str):
        candidates = [segment.strip() for segment in deny_languages_raw.split(",")]
        deny_languages = {value.lower() for value in candidates if value}
    elif isinstance(deny_languages_raw, list):
        deny_languages = {
            str(value).strip().lower()
            for value in deny_languages_raw
            if isinstance(value, (str, int, float)) and str(value).strip()
        }
    return deny_languages


def _build_discovery_context(
    now_dt: dt.datetime, payload: Dict[str, Any]
) -> DiscoveryProcessingContext:
    deny_languages = _parse_deny_languages(payload)
    last_upload_max_age_days = _parse_last_upload_max_age(payload)
    requires_metadata = bool(deny_languages or last_upload_max_age_days is not None)
    return DiscoveryProcessingContext(
        now=now_dt.isoformat(),
        now_dt=now_dt,
        deny_languages=deny_languages,
        last_upload_max_age_days=last_upload_max_age_days,
        requires_metadata=requires_metadata,
        metadata_cache={},
    )


def _evaluate_discovery_candidate(
    result: "ChannelSearchResult", context: DiscoveryProcessingContext
) -> Tuple[Optional[Dict[str, Any]], bool]:
    metadata: Optional[DiscoveryMetadata] = None
    if context.requires_metadata:
        metadata = context.metadata_cache.get(result.channel_id)
        if metadata is None:
            metadata = fetch_discovery_metadata(result.channel_id)
            context.metadata_cache[result.channel_id] = metadata

    violations: List[str] = []
    if context.deny_languages and metadata and metadata.language:
        language_value = str(metadata.language).strip()
        if language_value and language_value.lower() in context.deny_languages:
            violations.append(
                f"Language '{language_value}' denied during discovery"
            )

    if (
        context.last_upload_max_age_days is not None
        and metadata
        and metadata.last_upload
    ):
        last_upload_dt = _parse_iso_datetime(metadata.last_upload)
        if last_upload_dt is not None:
            last_upload_utc = last_upload_dt.astimezone(dt.timezone.utc)
            age = context.now_dt - last_upload_utc
            if age > dt.timedelta(days=context.last_upload_max_age_days):
                violations.append(
                    "Last upload is older than "
                    f"{context.last_upload_max_age_days} days (last: {last_upload_utc.date().isoformat()})"
                )

    if violations:
        database.ensure_blacklisted_channel(
            result.channel_id,
            context.now,
            url=ensure_channel_url(result.channel_id, result.url),
            name=result.title,
            reason="; ".join(violations),
        )
        return None, True

    payload: Dict[str, Any] = {
        "channel_id": result.channel_id,
        "name": result.title,
        "url": ensure_channel_url(result.channel_id, result.url),
        "subscribers": result.subscribers,
        "created_at": context.now,
        "last_updated": None,
        "last_attempted": None,
        "needs_enrichment": True,
        "emails": None,
        "language": None,
        "language_confidence": None,
        "last_error": None,
        "status": "new",
        "status_reason": None,
        "last_status_change": context.now,
    }

    if metadata:
        if metadata.last_upload:
            payload["last_updated"] = metadata.last_upload
        if metadata.language:
            payload["language"] = metadata.language
        if metadata.language_confidence is not None:
            payload["language_confidence"] = metadata.language_confidence

    return payload, False


def _process_search_results(
    results: Iterable["ChannelSearchResult"],
    *,
    context: DiscoveryProcessingContext,
    seen_ids: Set[str],
    new_channels: List[Dict[str, Any]],
) -> Tuple[int, int, int]:
    new_count = 0
    known_count = 0
    blacklisted_count = 0

    for result in results:
        channel_id = (result.channel_id or "").strip().upper()
        if not channel_id or channel_id in seen_ids:
            continue
        seen_ids.add(channel_id)

        if database.is_blacklisted(channel_id):
            database.ensure_blacklisted_channel(
                channel_id,
                context.now,
                url=ensure_channel_url(channel_id, result.url),
                name=result.title,
            )
            known_count += 1
            continue

        if channel_exists(channel_id, include_blacklisted=False):
            known_count += 1
            continue

        payload, flagged = _evaluate_discovery_candidate(result, context)
        if flagged:
            blacklisted_count += 1
            continue
        if payload:
            new_channels.append(payload)
            new_count += 1

    return new_count, known_count, blacklisted_count


def _run_until_stopped_discovery(
    keyword: str,
    per_keyword: int,
    *,
    context: DiscoveryProcessingContext,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = get_discovery_keyword_state(keyword)
    page_index = max(0, int(state.page_index))
    next_token = state.next_page_token
    consecutive_no_new = max(0, int(state.no_new_pages))
    exhausted_flag = False
    stop_requested = False

    seen_ids: Set[str] = set()
    new_channels: List[Dict[str, Any]] = []
    total_known = 0
    total_blacklisted = 0
    pages_processed = 0

    target_new_limit = max(1, int(per_keyword))
    target_new_limit = min(target_new_limit, RUN_UNTIL_STOPPED_MAX_NEW_CHANNELS)
    max_pages = max(1, RUN_UNTIL_STOPPED_MAX_PAGES_PER_RUN)
    no_new_threshold = max(1, RUN_UNTIL_STOPPED_NO_NEW_THRESHOLD)

    current_token = next_token
    session = None
    last_run_timestamp: Optional[str] = None

    if page_index > 0 and not current_token:
        exhausted_flag = True
    else:
        try:
            initial_page = search_channels_page(keyword)
        except Exception as exc:  # pragma: no cover - network errors
            raise HTTPException(
                status_code=502,
                detail=f"Failed to search for keyword '{keyword}': {exc}",
            ) from exc
        session = initial_page.session
        if page_index == 0:
            new_in_page, known_in_page, blacklisted_in_page = _process_search_results(
                initial_page.results,
                context=context,
                seen_ids=seen_ids,
                new_channels=new_channels,
            )
            total_known += known_in_page
            total_blacklisted += blacklisted_in_page
            pages_processed += 1
            if new_in_page == 0:
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0
            page_index += 1
            current_token = initial_page.next_page_token
            if current_token is None:
                exhausted_flag = True
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            update_discovery_keyword_state(
                keyword,
                next_page_token=current_token,
                page_index=page_index,
                last_run_at=timestamp,
                exhausted=exhausted_flag,
                no_new_pages=consecutive_no_new,
            )
            last_run_timestamp = timestamp
        else:
            current_token = next_token

    while (
        not exhausted_flag
        and not stop_requested
        and current_token
        and pages_processed < max_pages
        and len(new_channels) < target_new_limit
    ):
        if session is None:
            exhausted_flag = True
            break
        try:
            page = search_channels_page(
                keyword, session=session, continuation_token=current_token
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise HTTPException(
                status_code=502,
                detail=f"Failed to continue search for keyword '{keyword}': {exc}",
            ) from exc

        new_in_page, known_in_page, blacklisted_in_page = _process_search_results(
            page.results,
            context=context,
            seen_ids=seen_ids,
            new_channels=new_channels,
        )
        total_known += known_in_page
        total_blacklisted += blacklisted_in_page
        pages_processed += 1
        if new_in_page == 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        page_index += 1
        current_token = page.next_page_token
        if current_token is None:
            exhausted_flag = True

        if consecutive_no_new >= no_new_threshold:
            exhausted_flag = True

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        update_discovery_keyword_state(
            keyword,
            next_page_token=current_token,
            page_index=page_index,
            last_run_at=timestamp,
            exhausted=exhausted_flag,
            no_new_pages=consecutive_no_new,
        )
        last_run_timestamp = timestamp

        if len(new_channels) >= target_new_limit:
            break
        if pages_processed >= max_pages:
            break
        if exhausted_flag:
            break
        if discovery_state.is_stop_requested():
            stop_requested = True
            break

    if last_run_timestamp is None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        update_discovery_keyword_state(
            keyword,
            next_page_token=current_token,
            page_index=page_index,
            last_run_at=timestamp,
            exhausted=exhausted_flag,
            no_new_pages=consecutive_no_new,
        )
        last_run_timestamp = timestamp
    elif exhausted_flag or stop_requested:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        update_discovery_keyword_state(
            keyword,
            next_page_token=current_token,
            page_index=page_index,
            last_run_at=timestamp,
            exhausted=exhausted_flag,
            no_new_pages=consecutive_no_new,
        )
        last_run_timestamp = timestamp

    inserted = database.bulk_insert_channels(new_channels)
    totals = database.get_channel_totals()

    response_payload: Dict[str, Any] = {
        "found": inserted,
        "uniqueTotal": totals["total"],
    }
    if total_blacklisted:
        response_payload["blacklisted"] = total_blacklisted
    if total_known:
        response_payload["known"] = total_known

    session_info = {
        "keyword": keyword,
        "newChannels": inserted,
        "knownChannels": total_known,
        "pagesProcessed": pages_processed,
        "nextPageIndex": page_index,
        "exhausted": exhausted_flag,
        "stopRequested": stop_requested,
    }

    return response_payload, session_info

@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/static/{path:path}")
def serve_static(path: str) -> FileResponse:
    return FileResponse(f"frontend/{path}")


@app.post("/api/discover")
def api_discover(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    keywords_raw = payload.get("keywords", DEFAULT_KEYWORDS)
    if not isinstance(keywords_raw, list):
        raise HTTPException(status_code=400, detail="Invalid payload")

    try:
        per_keyword_value = payload.get("perKeyword", 5)
        per_keyword = int(per_keyword_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="perKeyword must be an integer")
    if per_keyword <= 0:
        raise HTTPException(status_code=400, detail="perKeyword must be positive")

    keywords: List[str] = []
    for value in keywords_raw:
        if not isinstance(value, str):
            value = str(value)
        cleaned = value.strip()
        if cleaned:
            keywords.append(cleaned)
    if not keywords:
        raise HTTPException(status_code=400, detail="No keywords provided")

    now_dt = dt.datetime.now(dt.timezone.utc)
    context = _build_discovery_context(now_dt, payload)

    run_flag_payload = _coerce_bool(payload.get("run_until_stopped")) or _coerce_bool(
        payload.get("runUntilStopped")
    )
    loop_snapshot = discovery_state.snapshot()
    run_flag_state = bool(loop_snapshot.get("running")) and bool(
        loop_snapshot.get("run_until_stopped")
    )
    run_until_stopped = run_flag_payload or run_flag_state

    if run_until_stopped:
        if len(keywords) != 1:
            raise HTTPException(
                status_code=422,
                detail="Run-until-stopped mode requires exactly one keyword.",
            )
        keyword = keywords[0]
        response_payload, session_info = _run_until_stopped_discovery(
            keyword, per_keyword, context=context
        )
        discovery_state.update_session(
            keyword=keyword,
            new=session_info.get("newChannels"),
            known=session_info.get("knownChannels"),
            pages=session_info.get("nextPageIndex"),
            exhausted=session_info.get("exhausted"),
            run_until_stopped=True,
        )
        response_payload["session"] = session_info
        return JSONResponse(response_payload)

    seen_ids: Set[str] = set()
    new_channels: List[Dict[str, Any]] = []
    total_known = 0
    total_blacklisted = 0

    for keyword in keywords:
        try:
            results = search_channels(keyword, per_keyword)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to search for keyword '{keyword}': {exc}")
            continue
        new_in_keyword, known_in_keyword, blacklisted_in_keyword = _process_search_results(
            results,
            context=context,
            seen_ids=seen_ids,
            new_channels=new_channels,
        )
        total_known += known_in_keyword
        total_blacklisted += blacklisted_in_keyword

    inserted = database.bulk_insert_channels(new_channels)
    totals = database.get_channel_totals()

    response_payload: Dict[str, Any] = {
        "found": inserted,
        "uniqueTotal": totals["total"],
    }
    if total_blacklisted:
        response_payload["blacklisted"] = total_blacklisted
    if total_known:
        response_payload["known"] = total_known

    return JSONResponse(response_payload)


@app.post("/api/discovery/loop/start")
def api_discovery_loop_start(
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> JSONResponse:
    data = payload or {}
    runs = _coerce_non_negative_int(data.get("runs"))
    discovered = _coerce_non_negative_int(data.get("discovered"))
    run_flag_value = data.get("run_until_stopped")
    if run_flag_value is None:
        run_flag_value = data.get("runUntilStopped")
    run_flag = _coerce_bool(run_flag_value) if run_flag_value is not None else True
    state = discovery_state.mark_started(
        runs=runs, discovered=discovered, run_until_stopped=run_flag
    )
    return JSONResponse(state)


@app.post("/api/discovery/loop/progress")
def api_discovery_loop_progress(
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> JSONResponse:
    data = payload or {}
    runs = _coerce_non_negative_int(data.get("runs"))
    discovered = _coerce_non_negative_int(data.get("discovered"))
    state = discovery_state.update_progress(runs=runs, discovered=discovered)
    return JSONResponse(state)


@app.post("/api/discovery/loop/stop")
def api_discovery_loop_stop() -> JSONResponse:
    state = discovery_state.request_stop()
    return JSONResponse(state)


@app.post("/api/discovery/loop/complete")
def api_discovery_loop_complete(
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> JSONResponse:
    data = payload or {}
    runs = _coerce_non_negative_int(data.get("runs"))
    discovered = _coerce_non_negative_int(data.get("discovered"))
    reason_value = data.get("reason")
    reason = str(reason_value).strip().lower() if isinstance(reason_value, str) else None
    error_flag = bool(data.get("error"))
    message_value = data.get("message")
    message = str(message_value) if isinstance(message_value, str) else None
    state = discovery_state.mark_completed(
        runs=runs,
        discovered=discovered,
        reason=reason or None,
        error=error_flag,
        message=message,
    )
    return JSONResponse(state)


@app.post("/api/blacklist/import")
async def api_blacklist_import(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing CSV file")

    try:
        raw_bytes = await file.read()
    except Exception as exc:  # pragma: no cover - I/O errors are rare
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV is missing headers")

    normalized_headers = {header.strip().lower() for header in reader.fieldnames if header}
    if not {"channel_id", "url"} & normalized_headers:
        raise HTTPException(status_code=400, detail="CSV must include a 'channel_id' or 'url' column")

    timestamp = dt.datetime.utcnow().isoformat()
    seen: Set[str] = set()
    cache: Dict[str, Tuple[Optional[ChannelResolution], Optional[str]]] = {}
    created: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    processed = 0

    unresolved_messages = {
        "invalid_input": "No usable channel reference found in this row.",
        "invalid_url": "Value is not a valid YouTube channel URL or ID.",
        "network_error": "Network error while contacting YouTube.",
        "not_found": "Channel appears to be missing or unavailable.",
        "resolution_failed": "Unable to resolve a channel ID from the provided value.",
    }
    skipped_messages = {
        "duplicate_in_file": "Duplicate channel in uploaded CSV.",
        "already_blacklisted": "Channel already blacklisted.",
    }

    for row in reader:
        processed += 1
        normalized = {
            str(key).strip().lower(): (value or "").strip()
            for key, value in row.items()
            if key
        }
        metadata: Dict[str, Any] = {}
        csv_subscribers = normalized.get("subscribers") or normalized.get("subscriber_count")
        subscribers_value = _parse_optional_int(csv_subscribers)
        if subscribers_value is not None:
            metadata["subscribers"] = subscribers_value
        csv_language = normalized.get("language")
        if csv_language:
            metadata["language"] = csv_language
        csv_emails = normalized.get("emails") or normalized.get("email")
        if csv_emails:
            parsed_emails = database.parse_email_candidates(csv_emails)
            if parsed_emails:
                unique_emails = list(dict.fromkeys(email.strip() for email in parsed_emails if email.strip()))
                if unique_emails:
                    metadata["emails"] = ", ".join(unique_emails)
            elif csv_emails:
                metadata["emails"] = csv_emails
        source_column = "channel_id" if normalized.get("channel_id") else "url"
        candidate_value = normalized.get(source_column) or normalized.get("url") or ""
        row_number = reader.line_num
        original_value = candidate_value.strip()
        sanitized_value = sanitize_channel_input(candidate_value)
        if not sanitized_value:
            unresolved.append(
                {
                    "row": row_number,
                    "input": original_value,
                    "normalized": sanitized_value,
                    "reason": "invalid_input",
                    "message": unresolved_messages["invalid_input"],
                    "column": source_column,
                }
            )
            continue

        normalized_reference = normalize_channel_reference(sanitized_value)
        if not normalized_reference:
            unresolved.append(
                {
                    "row": row_number,
                    "input": original_value or sanitized_value,
                    "normalized": sanitized_value,
                    "reason": "invalid_url",
                    "message": unresolved_messages["invalid_url"],
                    "column": source_column,
                }
            )
            continue

        cache_key = normalized_reference.lower()
        if cache_key not in cache:
            cache[cache_key] = resolve_channel(normalized_reference)
        resolution, reason = cache[cache_key]
        if not resolution:
            reason_code = reason or "resolution_failed"
            unresolved.append(
                {
                    "row": row_number,
                    "input": original_value or sanitized_value,
                    "normalized": sanitized_value,
                    "reason": reason_code,
                    "message": unresolved_messages.get(
                        reason_code, unresolved_messages["resolution_failed"]
                    ),
                    "column": source_column,
                }
            )
            continue

        channel_id = resolution.channel_id.upper()
        if channel_id in seen:
            skipped.append(
                {
                    "row": row_number,
                    "channel_id": channel_id,
                    "reason": "duplicate_in_file",
                    "message": skipped_messages["duplicate_in_file"],
                    "column": source_column,
                }
            )
            continue

        seen.add(channel_id)
        if database.is_blacklisted(channel_id):
            skipped.append(
                {
                    "row": row_number,
                    "channel_id": channel_id,
                    "reason": "already_blacklisted",
                    "message": skipped_messages["already_blacklisted"],
                    "column": source_column,
                }
            )
            continue

        moved = database.blacklist_channels_by_ids([channel_id], timestamp)
        database.ensure_blacklisted_channel(
            channel_id,
            timestamp,
            url=resolution.canonical_url,
            name=resolution.title or resolution.handle,
            metadata=metadata,
        )
        record = {
            "channel_id": channel_id,
            "url": resolution.canonical_url,
            "handle": resolution.handle,
            "name": resolution.title or resolution.handle,
        }
        if metadata.get("subscribers") is not None:
            record["subscribers"] = metadata["subscribers"]
        if metadata.get("language"):
            record["language"] = metadata["language"]
        if metadata.get("emails"):
            record["emails"] = metadata["emails"]
        if moved:
            updated.append(record)
        else:
            created.append(record)

    result = {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "unresolved": unresolved,
        "counts": {
            "created": len(created),
            "updated": len(updated),
            "skipped": len(skipped),
            "unresolved": len(unresolved),
            "processed": processed,
        },
        "processedAt": timestamp,
    }

    return JSONResponse(result)


@app.post("/api/enrich")
def api_enrich(payload: Dict[str, Any] = Body(default={})) -> JSONResponse:
    limit = payload.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="limit must be an integer or null")
        if limit <= 0:
            raise HTTPException(status_code=400, detail="limit must be greater than zero")

    mode = payload.get("mode", "full")
    if mode not in {"full", "email_only"}:
        raise HTTPException(status_code=400, detail="mode must be 'full' or 'email_only'")

    force_run = bool(payload.get("forceRun"))
    never_reenrich = bool(payload.get("neverReenrich"))

    job = manager.start_job(
        limit,
        mode=mode,
        force_run=force_run,
        never_reenrich=never_reenrich,
    )
    return JSONResponse(
        {
            "jobId": job.job_id,
            "total": job.total,
            "mode": job.mode,
            "requested": job.requested,
            "skipped": job.skipped,
        }
    )


@app.get("/api/enrich/stream/{job_id}")
def api_enrich_stream(job_id: str) -> StreamingResponse:
    try:
        generator = manager.stream(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown enrichment job")
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generator, media_type="text/event-stream", headers=headers)


@app.get("/api/channels")
def api_channels(
    q: Optional[str] = Query(default=None),
    language: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    min_subscribers: Optional[str] = Query(default=None),
    max_subscribers: Optional[str] = Query(default=None),
    sort: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    emails_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    email_gate_only: bool = Query(default=False),
    unique_emails: bool = Query(default=False),
    category: Optional[str] = Query(default=ChannelCategory.ACTIVE.value),
) -> JSONResponse:
    category_value = _parse_category(category)
    filters = _collect_filters(
        q=q,
        languages=language,
        statuses=status,
        min_subscribers=min_subscribers,
        max_subscribers=max_subscribers,
        emails_only=emails_only,
        include_archived=include_archived,
        email_gate_only=email_gate_only,
        unique_emails=unique_emails,
    )
    items, total = database.get_channels(
        category_value,
        filters,
        sort=sort,
        order=order,
        limit=limit,
        offset=offset,
    )
    return JSONResponse({"items": items, "total": total})


@app.post("/api/channels/{channel_id}/archive")
def api_archive_channel(channel_id: str) -> JSONResponse:
    timestamp = dt.datetime.utcnow().isoformat()
    archived_ids = database.archive_channels_by_ids([channel_id], timestamp)
    if not archived_ids:
        raise HTTPException(status_code=404, detail="Channel not found or already archived")
    return JSONResponse({"archived": len(archived_ids), "archivedIds": archived_ids, "archivedAt": timestamp})


@app.post("/api/channels/archive_bulk")
def api_archive_bulk(
    payload: Dict[str, Any] = Body(default={}),
    q: Optional[str] = Query(default=None),
    language: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    min_subscribers: Optional[str] = Query(default=None),
    max_subscribers: Optional[str] = Query(default=None),
    sort: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    emails_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    email_gate_only: bool = Query(default=False),
    unique_emails: bool = Query(default=False),
    category: Optional[str] = Query(default=ChannelCategory.ACTIVE.value),
) -> JSONResponse:
    category_value = _parse_category(category)
    if category_value is not ChannelCategory.ACTIVE:
        raise HTTPException(status_code=400, detail="Archive bulk only supported for active channels")
    channel_ids: Optional[List[str]] = None
    if isinstance(payload, dict):
        ids = payload.get("channel_ids")
        if ids is not None:
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise HTTPException(status_code=400, detail="channel_ids must be a list of strings")
            channel_ids = ids
        filter_mode = payload.get("filter")
        if filter_mode == "emails_only":
            emails_only = True

    timestamp = dt.datetime.utcnow().isoformat()

    if channel_ids is None:
        filters = _collect_filters(
            q=q,
            languages=language,
            statuses=status,
            min_subscribers=min_subscribers,
            max_subscribers=max_subscribers,
            emails_only=emails_only,
            include_archived=include_archived,
            email_gate_only=email_gate_only,
            unique_emails=unique_emails,
        )
        items, _ = database.get_channels(
            category_value,
            filters,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
        channel_ids = [item["channel_id"] for item in items]

    archived_ids = database.archive_channels_by_ids(channel_ids or [], timestamp)
    return JSONResponse({"archived": len(archived_ids), "archivedIds": archived_ids, "archivedAt": timestamp})


@app.post("/api/channels/archive_exported")
def api_archive_exported(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    exported_at_raw = payload.get("exported_at")
    if not isinstance(exported_at_raw, str) or not exported_at_raw.strip():
        raise HTTPException(status_code=400, detail="exported_at must be provided")

    exported_at = exported_at_raw.strip()
    timestamp = dt.datetime.utcnow().isoformat()
    archived_ids = database.archive_channels_by_exported_at(exported_at, timestamp)
    return JSONResponse({"archived": len(archived_ids), "archivedIds": archived_ids, "archivedAt": timestamp})


@app.post("/api/channels/{channel_id}/blacklist")
def api_blacklist_channel(channel_id: str, category: Optional[str] = Query(default=None)) -> JSONResponse:
    timestamp = dt.datetime.utcnow().isoformat()
    sources: Optional[List[ChannelCategory]] = None
    if category:
        parsed = _parse_category(category)
        if parsed is ChannelCategory.BLACKLISTED:
            raise HTTPException(status_code=400, detail="Channel already blacklisted")
        sources = [parsed]
    blacklisted_ids = database.blacklist_channels_by_ids([channel_id], timestamp, source_categories=sources)
    database.ensure_blacklisted_channel(channel_id, timestamp)
    if not blacklisted_ids and not database.is_blacklisted(channel_id):
        raise HTTPException(status_code=404, detail="Channel not found")
    return JSONResponse(
        {
            "blacklisted": len(blacklisted_ids) or 1,
            "blacklistedIds": blacklisted_ids or [channel_id],
            "blacklistedAt": timestamp,
        }
    )


@app.post("/api/channels/blacklist_bulk")
def api_blacklist_bulk(
    payload: Dict[str, Any] = Body(default={}),
    q: Optional[str] = Query(default=None),
    language: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    min_subscribers: Optional[str] = Query(default=None),
    max_subscribers: Optional[str] = Query(default=None),
    sort: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    emails_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    email_gate_only: bool = Query(default=False),
    unique_emails: bool = Query(default=False),
    category: Optional[str] = Query(default=ChannelCategory.ACTIVE.value),
) -> JSONResponse:
    category_value = _parse_category(category)
    channel_ids: Optional[List[str]] = None
    if isinstance(payload, dict):
        ids = payload.get("channel_ids")
        if ids is not None:
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise HTTPException(status_code=400, detail="channel_ids must be a list of strings")
            channel_ids = ids
        filter_mode = payload.get("filter")
        if filter_mode == "emails_only":
            emails_only = True

    if channel_ids is None:
        filters = _collect_filters(
            q=q,
            languages=language,
            statuses=status,
            min_subscribers=min_subscribers,
            max_subscribers=max_subscribers,
            emails_only=emails_only,
            include_archived=include_archived,
            email_gate_only=email_gate_only,
            unique_emails=unique_emails,
        )
        items, _ = database.get_channels(
            category_value,
            filters,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
        channel_ids = [item["channel_id"] for item in items]

    timestamp = dt.datetime.utcnow().isoformat()
    sources: Optional[List[ChannelCategory]] = None
    if category_value is not ChannelCategory.BLACKLISTED:
        sources = [category_value]
    blacklisted_ids = database.blacklist_channels_by_ids(channel_ids or [], timestamp, source_categories=sources)
    for channel_id in channel_ids or []:
        database.ensure_blacklisted_channel(channel_id, timestamp)
    return JSONResponse(
        {
            "blacklisted": len(blacklisted_ids),
            "blacklistedIds": blacklisted_ids,
            "blacklistedAt": timestamp,
        }
    )


@app.post("/api/channels/{channel_id}/restore")
def api_restore_channel(channel_id: str) -> JSONResponse:
    timestamp = dt.datetime.utcnow().isoformat()
    restored_ids = database.restore_channels_by_ids([channel_id], timestamp)
    if not restored_ids:
        raise HTTPException(status_code=404, detail="Channel not found in archived or blacklisted tables")
    return JSONResponse({"restored": len(restored_ids), "restoredIds": restored_ids, "restoredAt": timestamp})


@app.post("/api/channels/restore_bulk")
def api_restore_bulk(
    payload: Dict[str, Any] = Body(default={}),
    q: Optional[str] = Query(default=None),
    language: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    min_subscribers: Optional[str] = Query(default=None),
    max_subscribers: Optional[str] = Query(default=None),
    sort: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    emails_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    email_gate_only: bool = Query(default=False),
    unique_emails: bool = Query(default=False),
    category: Optional[str] = Query(default=ChannelCategory.ARCHIVED.value),
) -> JSONResponse:
    category_value = _parse_category(category)
    if category_value is ChannelCategory.ACTIVE:
        raise HTTPException(status_code=400, detail="Restore requires archived or blacklisted category")
    channel_ids: Optional[List[str]] = None
    if isinstance(payload, dict):
        ids = payload.get("channel_ids")
        if ids is not None:
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise HTTPException(status_code=400, detail="channel_ids must be a list of strings")
            channel_ids = ids
        filter_mode = payload.get("filter")
        if filter_mode == "emails_only":
            emails_only = True

    if channel_ids is None:
        filters = _collect_filters(
            q=q,
            languages=language,
            statuses=status,
            min_subscribers=min_subscribers,
            max_subscribers=max_subscribers,
            emails_only=emails_only,
            include_archived=include_archived,
            email_gate_only=email_gate_only,
            unique_emails=unique_emails,
        )
        items, _ = database.get_channels(
            category_value,
            filters,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
        channel_ids = [item["channel_id"] for item in items]

    timestamp = dt.datetime.utcnow().isoformat()
    restored_ids = database.restore_channels_by_ids(channel_ids or [], timestamp, source_categories=[category_value])
    return JSONResponse({"restored": len(restored_ids), "restoredIds": restored_ids, "restoredAt": timestamp})


@app.get("/api/export/csv")
def api_export_csv(
    q: Optional[str] = Query(default=None),
    language: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    min_subscribers: Optional[str] = Query(default=None),
    max_subscribers: Optional[str] = Query(default=None),
    sort: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    emails_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    unique_emails: bool = Query(default=False),
    email_gate_only: bool = Query(default=False),
    category: Optional[str] = Query(default=ChannelCategory.ACTIVE.value),
    archive_exported: bool = Query(default=False),
) -> PlainTextResponse:
    category_value = _parse_category(category)
    filters = _collect_filters(
        q=q,
        languages=language,
        statuses=status,
        min_subscribers=min_subscribers,
        max_subscribers=max_subscribers,
        emails_only=emails_only,
        include_archived=include_archived,
        email_gate_only=email_gate_only,
        unique_emails=unique_emails,
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    exported_channel_ids: List[str] = []
    export_timestamp = dt.datetime.utcnow().isoformat()

    if emails_only and unique_emails:
        rows = database.get_unique_email_rows(filters, category=category_value)
        writer.writerow(
            [
                "Email",
                "Primary Channel Name",
                "Primary Channel URL",
                "Other Channels Count",
                "Last Updated",
            ]
        )
        for row in rows:
            channel_id = row.get("primary_channel_id")
            if channel_id:
                exported_channel_ids.append(channel_id)
            writer.writerow(
                [
                    row.get("email", ""),
                    row.get("primary_channel_name", ""),
                    row.get("primary_channel_url", ""),
                    row.get("other_channels_count", 0),
                    row.get("last_updated", ""),
                ]
            )
    else:
        items, _ = database.get_channels(
            category_value,
            filters,
            sort=sort,
            order=order,
            limit=10_000,
            offset=0,
        )
        writer.writerow(
            [
                "Channel Name",
                "URL",
                "Subscribers",
                "Language",
                "Emails",
                "Email Gate",
                "Status",
                "Last Updated",
                "Last Status Change",
                "Created At",
                "Last Attempted",
                "Exported At",
                "Archived At",
                "Error Reason",
            ]
        )
        for item in items:
            channel_id = item.get("channel_id")
            if channel_id:
                exported_channel_ids.append(channel_id)
            writer.writerow(
                [
                    item.get("name") or "",
                    item.get("url") or "",
                    item.get("subscribers") or "",
                    item.get("language") or "",
                    item.get("emails") or "",
                    "Yes"
                    if item.get("email_gate_present")
                    else ("No" if item.get("email_gate_present") == 0 else ""),
                    item.get("status") or "",
                    item.get("last_updated") or "",
                    item.get("last_status_change") or "",
                    item.get("created_at") or "",
                    item.get("last_attempted") or "",
                    item.get("exported_at") or "",
                    item.get("archived_at") or "",
                    item.get("status_reason") or item.get("last_error") or "",
                ]
            )

    csv_data = buffer.getvalue()
    if exported_channel_ids:
        try:
            database.mark_channels_exported(
                category_value,
                exported_channel_ids,
                export_timestamp,
                archive=archive_exported,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Failed to update exported rows") from exc
    headers = {"X-Export-Timestamp": export_timestamp}
    return PlainTextResponse(content=csv_data, media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/export/bundle")
def api_export_bundle() -> StreamingResponse:
    data, email_index = database.fetch_project_bundle_data()
    export_timestamp = dt.datetime.utcnow().replace(microsecond=0).isoformat()

    channel_counts = {
        category: len(records) for category, records in data.get("channels", {}).items()
    }
    meta = {
        "schemaVersion": database.PROJECT_BUNDLE_SCHEMA_VERSION,
        "exportedAt": export_timestamp,
        "channelCounts": channel_counts,
        "blacklistCount": len(data.get("blacklist", [])),
        "emailRelations": {
            "uniqueEmails": len(data.get("emails_unique", [])),
            "channelEmailLinks": len(data.get("channel_emails", [])),
        },
        "globalEmailIndex": email_index,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "data.json",
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        )
        bundle.writestr(
            "meta.json",
            json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True),
        )

    buffer.seek(0)
    filename = f"project-bundle-{export_timestamp.replace(':', '').replace('T', '_')}.zip"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Export-Timestamp": export_timestamp,
    }
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.post("/api/import/bundle")
async def api_import_bundle(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, alias="dryRun"),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing bundle archive")

    try:
        raw_bytes = await file.read()
    except Exception as exc:  # pragma: no cover - defensive I/O handling
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded bundle is empty")

    try:
        bundle_file = io.BytesIO(raw_bytes)
        with zipfile.ZipFile(bundle_file) as bundle:
            try:
                data_bytes = bundle.read("data.json")
            except KeyError as exc:
                raise HTTPException(status_code=400, detail="Bundle archive is missing data.json") from exc

            meta: Optional[Dict[str, Any]] = None
            if "meta.json" in bundle.namelist():
                try:
                    meta_bytes = bundle.read("meta.json")
                    meta = json.loads(meta_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HTTPException(status_code=400, detail=f"meta.json is invalid: {exc}") from exc

            try:
                data = json.loads(data_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=f"data.json is invalid: {exc}") from exc
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid bundle archive") from exc

    try:
        summary = database.restore_project_bundle(data, meta=meta, dry_run=dry_run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(summary)


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    totals = database.get_channel_totals()
    status_totals = database.get_channel_status_totals()
    loop_state = discovery_state.snapshot()
    enrichment_state = manager.get_job_summaries()
    enrichment_state.setdefault("activeJobs", 0)
    enrichment_state.setdefault("pendingChannels", 0)
    enrichment_state["processingChannels"] = status_totals.get("processing", 0)
    payload: Dict[str, Any] = {
        **totals,
        "statusTotals": status_totals,
        "discoveryLoop": loop_state,
        "enrichment": enrichment_state,
    }
    return JSONResponse(payload)
