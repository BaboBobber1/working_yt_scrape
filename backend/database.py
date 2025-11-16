"""Database utilities for the Crypto YouTube Harvester backend."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

DB_PATH = Path("data") / "channels.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_connection_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
    return _connection


@contextmanager
def get_cursor():
    conn = _get_connection()
    with _connection_lock:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        finally:
            cursor.close()


class ChannelCategory(str, Enum):
    """Available logical channel collections."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    BLACKLISTED = "blacklisted"


CHANNEL_TABLES = {
    ChannelCategory.ACTIVE: "channels_active",
    ChannelCategory.ARCHIVED: "channels_archived",
    ChannelCategory.BLACKLISTED: "channels_blacklisted",
}


@dataclass(frozen=True)
class DiscoveryKeywordState:
    """Persistent search position for a discovery keyword."""

    keyword: str
    next_page_token: Optional[str]
    page_index: int
    last_run_at: Optional[str]
    exhausted: bool
    no_new_pages: int
    updated_at: Optional[str]

PROJECT_BUNDLE_SCHEMA_VERSION = 1

DEFAULT_ENRICHMENT_CONFIG = {
    "emails_from_channel": True,
    "emails_from_videos": True,
    "language_basic": True,
    "language_precise": True,
    "update_metadata": True,
    "update_activity": True,
}

CHANNEL_COLUMNS = [
    "channel_id",
    "name",
    "url",
    "subscribers",
    "language",
    "language_confidence",
    "emails",
    "email_gate_present",
    "last_updated",
    "created_at",
    "last_attempted",
    "last_enriched_at",
    "last_enriched_result",
    "needs_enrichment",
    "last_error",
    "status",
    "status_reason",
    "last_status_change",
    "archived_at",
    "exported_at",
]

LEGACY_TABLE = "channels"


def _normalize_discovery_keyword(keyword: str) -> str:
    cleaned = (keyword or "").strip()
    return cleaned.lower()


def _normalize_enrichment_config(payload: Any) -> Dict[str, bool]:
    if not isinstance(payload, dict):
        return {**DEFAULT_ENRICHMENT_CONFIG}
    normalized: Dict[str, bool] = {}
    for key, default in DEFAULT_ENRICHMENT_CONFIG.items():
        normalized[key] = bool(payload.get(key, default))
    return normalized


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in cursor.fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_enrichment_settings(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            payload TEXT NOT NULL
        )
        """
    )
    cursor.execute("SELECT payload FROM enrichment_settings WHERE id = 1")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(
            "INSERT INTO enrichment_settings (id, payload) VALUES (1, ?)",
            (json.dumps(DEFAULT_ENRICHMENT_CONFIG),),
        )


def init_db() -> None:
    with get_cursor() as cursor:
        for table in CHANNEL_TABLES.values():
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL UNIQUE,
                    name TEXT,
                    url TEXT NOT NULL,
                    subscribers INTEGER,
                    language TEXT,
                    language_confidence REAL,
                    emails TEXT,
                    email_gate_present INTEGER,
                    last_updated TEXT,
                    created_at TEXT NOT NULL,
                    last_attempted TEXT,
                    last_enriched_at TEXT,
                    last_enriched_result TEXT,
                    needs_enrichment INTEGER NOT NULL DEFAULT 1,
                    last_error TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    status_reason TEXT,
                    last_status_change TEXT,
                    archived_at TEXT,
                    exported_at TEXT
                )
                """
            )
            _ensure_column(cursor, table, "email_gate_present", "INTEGER")
            _ensure_column(cursor, table, "last_enriched_at", "TEXT")
            _ensure_column(cursor, table, "last_enriched_result", "TEXT")
            _ensure_column(cursor, table, "archived_at", "TEXT")
            _ensure_column(cursor, table, "exported_at", "TEXT")
            cursor.execute(
                f"UPDATE {table} SET archived_at = last_status_change "
                "WHERE archived_at IS NULL AND status = 'archived' AND last_status_change IS NOT NULL"
            )

        _ensure_enrichment_settings(cursor)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklist (
                channel_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emails_unique (
                email TEXT PRIMARY KEY,
                first_seen_channel_id TEXT,
                last_seen_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_emails (
                channel_id TEXT NOT NULL,
                email TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(channel_id, email),
                FOREIGN KEY(email) REFERENCES emails_unique(email)
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_emails_channel_id ON channel_emails(channel_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_emails_email ON channel_emails(email)"
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS discovery_keyword_states (
                keyword TEXT PRIMARY KEY,
                next_page_token TEXT,
                page_index INTEGER NOT NULL DEFAULT 0,
                last_run_at TEXT,
                exhausted INTEGER NOT NULL DEFAULT 0,
                no_new_pages INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT
            )
            """
        )

        _migrate_legacy_channels(cursor)


def _migrate_legacy_channels(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (LEGACY_TABLE,),
    )
    if not cursor.fetchone():
        return

    cursor.execute(f"SELECT COUNT(*) AS count FROM {LEGACY_TABLE}")
    legacy_count = cursor.fetchone()["count"]
    if legacy_count == 0:
        cursor.execute(f"ALTER TABLE {LEGACY_TABLE} RENAME TO {LEGACY_TABLE}_legacy")
        return

    cursor.execute(f"SELECT * FROM {LEGACY_TABLE}")
    rows = cursor.fetchall()
    for row in rows:
        record = dict(row)
        channel_id = record.get("channel_id")
        if not channel_id:
            continue
        destination = ChannelCategory.ACTIVE
        if record.get("blacklisted"):
            destination = ChannelCategory.BLACKLISTED
        elif record.get("archived"):
            destination = ChannelCategory.ARCHIVED

        payload = {
            "channel_id": channel_id,
            "name": record.get("title"),
            "url": ensure_channel_url(channel_id, record.get("url")),
            "subscribers": record.get("subscribers"),
            "language": record.get("language"),
            "language_confidence": record.get("language_confidence"),
            "emails": record.get("emails"),
            "last_updated": record.get("last_updated"),
            "created_at": record.get("created_at") or record.get("last_updated"),
            "last_attempted": record.get("last_attempted"),
            "last_enriched_at": record.get("last_enriched_at"),
            "last_enriched_result": record.get("last_enriched_result"),
            "needs_enrichment": record.get("needs_enrichment", 1),
            "last_error": record.get("last_error"),
            "status": record.get("status", "new"),
            "status_reason": record.get("status_reason"),
            "last_status_change": record.get("last_status_change"),
        }
        _insert_or_replace(cursor, CHANNEL_TABLES[destination], payload)

    cursor.execute(f"ALTER TABLE {LEGACY_TABLE} RENAME TO {LEGACY_TABLE}_legacy")


@dataclass(frozen=True)
class ChannelFilters:
    query_text: Optional[str] = None
    languages: Optional[Sequence[str]] = None
    statuses: Optional[Sequence[str]] = None
    min_subscribers: Optional[int] = None
    max_subscribers: Optional[int] = None
    emails_only: bool = False
    include_archived: bool = False
    email_gate_only: bool = False
    unique_emails: bool = False


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

GLOBAL_DUPLICATE_CHANNELS_QUERY = """
    SELECT DISTINCT ce.channel_id
    FROM channel_emails ce
    JOIN (
        SELECT email
        FROM channel_emails
        GROUP BY email
        HAVING COUNT(DISTINCT channel_id) > 1
    ) dup ON dup.email = ce.email
"""


def _collect_duplicate_emails_for_channels(
    channel_ids: Sequence[str],
) -> Dict[str, Set[str]]:
    unique_ids = [channel_id for channel_id in dict.fromkeys(channel_ids) if channel_id]
    if not unique_ids:
        return {}

    placeholders = ",".join("?" for _ in unique_ids)
    query = f"""
        SELECT ce.channel_id, ce.email
        FROM channel_emails ce
        JOIN (
            SELECT email
            FROM channel_emails
            GROUP BY email
            HAVING COUNT(DISTINCT channel_id) > 1
        ) dup ON dup.email = ce.email
        WHERE ce.channel_id IN ({placeholders})
    """

    duplicates: Dict[str, Set[str]] = {}
    with get_cursor() as cursor:
        cursor.execute(query, unique_ids)
        for row in cursor.fetchall():
            channel_id = row["channel_id"]
            email = row["email"]
            if channel_id and email:
                duplicates.setdefault(channel_id, set()).add(email)
    return duplicates


def _normalize_email(value: str) -> Optional[str]:
    candidate = value.strip().lower()
    if not candidate or "@" not in candidate:
        return None
    match = EMAIL_PATTERN.fullmatch(candidate)
    return candidate if match else None


def parse_email_candidates(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return EMAIL_PATTERN.findall(value)


def get_discovery_keyword_state(keyword: str) -> DiscoveryKeywordState:
    """Return the persisted paging state for a discovery keyword."""

    normalized = _normalize_discovery_keyword(keyword)
    if not normalized:
        return DiscoveryKeywordState(
            keyword="",
            next_page_token=None,
            page_index=0,
            last_run_at=None,
            exhausted=False,
            no_new_pages=0,
            updated_at=None,
        )

    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT keyword, next_page_token, page_index, last_run_at, exhausted, no_new_pages, updated_at
            FROM discovery_keyword_states
            WHERE keyword = ?
            """,
            (normalized,),
        )
        row = cursor.fetchone()

    if row is None:
        return DiscoveryKeywordState(
            keyword=normalized,
            next_page_token=None,
            page_index=0,
            last_run_at=None,
            exhausted=False,
            no_new_pages=0,
            updated_at=None,
        )

    return DiscoveryKeywordState(
        keyword=row["keyword"],
        next_page_token=row["next_page_token"],
        page_index=int(row["page_index"] or 0),
        last_run_at=row["last_run_at"],
        exhausted=bool(row["exhausted"]),
        no_new_pages=int(row["no_new_pages"] or 0),
        updated_at=row["updated_at"],
    )


def update_discovery_keyword_state(
    keyword: str,
    *,
    next_page_token: Optional[str],
    page_index: int,
    last_run_at: Optional[str],
    exhausted: bool,
    no_new_pages: int,
) -> DiscoveryKeywordState:
    """Persist the paging state for a discovery keyword."""

    normalized = _normalize_discovery_keyword(keyword)
    if not normalized:
        raise ValueError("Discovery keyword cannot be empty")

    safe_page_index = max(0, int(page_index))
    safe_no_new = max(0, int(no_new_pages))
    timestamp = last_run_at

    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO discovery_keyword_states (
                keyword,
                next_page_token,
                page_index,
                last_run_at,
                exhausted,
                no_new_pages,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                next_page_token = excluded.next_page_token,
                page_index = excluded.page_index,
                last_run_at = excluded.last_run_at,
                exhausted = excluded.exhausted,
                no_new_pages = excluded.no_new_pages,
                updated_at = excluded.updated_at
            """,
            (
                normalized,
                next_page_token,
                safe_page_index,
                last_run_at,
                int(bool(exhausted)),
                safe_no_new,
                timestamp,
            ),
        )

    return DiscoveryKeywordState(
        keyword=normalized,
        next_page_token=next_page_token,
        page_index=safe_page_index,
        last_run_at=last_run_at,
        exhausted=bool(exhausted),
        no_new_pages=safe_no_new,
        updated_at=timestamp,
    )


def get_enrichment_settings() -> Dict[str, bool]:
    """Return the persisted enrichment configuration, seeding defaults if missing."""

    with get_cursor() as cursor:
        _ensure_enrichment_settings(cursor)
        cursor.execute("SELECT payload FROM enrichment_settings WHERE id = 1")
        row = cursor.fetchone()
        stored_payload: Dict[str, Any] = {}
        if row and row["payload"]:
            try:
                stored_payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                stored_payload = {}
        normalized = _normalize_enrichment_config(stored_payload)
        cursor.execute(
            "UPDATE enrichment_settings SET payload = ? WHERE id = 1",
            (json.dumps(normalized),),
        )
    return normalized


def save_enrichment_settings(options: Dict[str, Any]) -> Dict[str, bool]:
    """Persist the enrichment configuration and return the normalized state."""

    normalized = _normalize_enrichment_config(options)
    with get_cursor() as cursor:
        _ensure_enrichment_settings(cursor)
        cursor.execute(
            "INSERT OR REPLACE INTO enrichment_settings (id, payload) VALUES (1, ?)",
            (json.dumps(normalized),),
        )
    return normalized


def is_blacklisted(channel_id: str) -> bool:
    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT 1 FROM {CHANNEL_TABLES[ChannelCategory.BLACKLISTED]} WHERE channel_id = ?",
            (channel_id,),
        )
        if cursor.fetchone():
            return True
        cursor.execute(
            "SELECT 1 FROM blacklist WHERE channel_id = ?",
            (channel_id,),
        )
        return cursor.fetchone() is not None


def channel_exists(channel_id: str, *, include_blacklisted: bool = True) -> bool:
    """Return True if the channel ID exists in any project table."""

    normalized = (channel_id or "").strip().upper()
    if not normalized:
        return False

    categories: List[ChannelCategory]
    if include_blacklisted:
        categories = list(ChannelCategory)
    else:
        categories = [
            category
            for category in ChannelCategory
            if category != ChannelCategory.BLACKLISTED
        ]

    with get_cursor() as cursor:
        for category in categories:
            cursor.execute(
                f"SELECT 1 FROM {CHANNEL_TABLES[category]} WHERE channel_id = ?",
                (normalized,),
            )
            if cursor.fetchone():
                return True
        if include_blacklisted:
            cursor.execute(
                "SELECT 1 FROM blacklist WHERE channel_id = ?",
                (normalized,),
            )
            if cursor.fetchone():
                return True

    return False


def ensure_blacklisted_channel(
    channel_id: str,
    timestamp: str,
    *,
    url: Optional[str] = None,
    name: Optional[str] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool]:
    """Ensure a record exists for the channel in the blacklist tables."""

    created = False
    updated = False
    canonical_url = ensure_channel_url(channel_id, url)
    resolved_name = name.strip() if isinstance(name, str) else name
    if resolved_name == "":
        resolved_name = None

    status_reason = reason.strip() if isinstance(reason, str) else None
    if status_reason == "":
        status_reason = None

    metadata_payload: Dict[str, Any] = {}
    if metadata and isinstance(metadata, dict):
        subscribers_value = metadata.get("subscribers")
        if subscribers_value is not None:
            if isinstance(subscribers_value, bool):
                subscribers_int = int(subscribers_value)
            elif isinstance(subscribers_value, (int, float)):
                subscribers_int = int(subscribers_value)
            else:
                try:
                    subscribers_int = int(str(subscribers_value).strip())
                except (TypeError, ValueError):
                    subscribers_int = None
            if subscribers_int is not None:
                metadata_payload["subscribers"] = subscribers_int

        language_value = metadata.get("language")
        if isinstance(language_value, str):
            cleaned_language = language_value.strip()
            if cleaned_language:
                metadata_payload["language"] = cleaned_language

        emails_value = metadata.get("emails")
        normalized_emails: Optional[str] = None
        if isinstance(emails_value, str):
            cleaned_emails = emails_value.strip()
            normalized_emails = cleaned_emails or None
        elif isinstance(emails_value, (list, tuple, set)):
            ordered_emails = [str(item).strip() for item in emails_value if str(item).strip()]
            if ordered_emails:
                unique_emails = list(dict.fromkeys(ordered_emails))
                normalized_emails = ", ".join(unique_emails)
        if normalized_emails:
            metadata_payload["emails"] = normalized_emails

    with get_cursor() as cursor:
        cursor.execute(
            "SELECT channel_id FROM blacklist WHERE channel_id = ?",
            (channel_id,),
        )
        if cursor.fetchone():
            updated = True
            cursor.execute(
                "UPDATE blacklist SET updated_at = ? WHERE channel_id = ?",
                (timestamp, channel_id),
            )
        else:
            created = True
            cursor.execute(
                "INSERT INTO blacklist (channel_id, created_at, updated_at) VALUES (?, ?, ?)",
                (channel_id, timestamp, timestamp),
            )

        cursor.execute(
            f"SELECT * FROM {CHANNEL_TABLES[ChannelCategory.BLACKLISTED]} WHERE channel_id = ?",
            (channel_id,),
        )
        existing = cursor.fetchone()
        resolved_reason = status_reason or "Blacklisted"

        if existing is None:
            payload = {
                "channel_id": channel_id,
                "name": resolved_name,
                "url": canonical_url,
                "subscribers": metadata_payload.get("subscribers"),
                "language": metadata_payload.get("language"),
                "language_confidence": None,
                "emails": metadata_payload.get("emails"),
                "last_updated": None,
                "created_at": timestamp,
                "last_attempted": None,
                "needs_enrichment": 0,
                "last_error": None,
                "status": "blacklisted",
                "status_reason": resolved_reason,
                "last_status_change": timestamp,
            }
        else:
            payload = dict(existing)
            payload.update(
                name=resolved_name or existing["name"],
                url=canonical_url,
                needs_enrichment=0,
                status="blacklisted",
                status_reason=resolved_reason,
                last_status_change=existing["last_status_change"] or timestamp,
            )
            for field, value in metadata_payload.items():
                if value is None:
                    continue
                existing_value = payload.get(field)
                if existing_value in (None, ""):
                    payload[field] = value
        _insert_or_replace(cursor, CHANNEL_TABLES[ChannelCategory.BLACKLISTED], payload)
        return updated, created


def _prepare_channel_payload(channel: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for column in CHANNEL_COLUMNS:
        if column == "needs_enrichment":
            value = channel.get(column)
            if value is None:
                value = 1
            payload[column] = int(bool(value))
        elif column == "email_gate_present":
            value = channel.get(column)
            if value is None or value == "":
                payload[column] = None
            else:
                payload[column] = int(bool(value))
        else:
            payload[column] = channel.get(column)
    if payload.get("name") is None:
        payload["name"] = channel.get("title")
    if not payload.get("created_at"):
        payload["created_at"] = channel.get("created_at") or channel.get("last_updated")
    payload["url"] = ensure_channel_url(channel.get("channel_id"), payload.get("url"))
    return payload


def _insert_or_replace(cursor: sqlite3.Cursor, table: str, payload: Dict[str, Any]) -> None:
    columns = ", ".join(CHANNEL_COLUMNS)
    placeholders = ", ".join("?" for _ in CHANNEL_COLUMNS)
    updates = ", ".join(f"{column} = excluded.{column}" for column in CHANNEL_COLUMNS if column != "channel_id")
    values = [payload.get(column) for column in CHANNEL_COLUMNS]
    cursor.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(channel_id) DO UPDATE SET {updates}",
        values,
    )


def _chunked(sequence: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def insert_channel(channel: Dict[str, Any], *, category: ChannelCategory = ChannelCategory.ACTIVE) -> bool:
    """Insert a new channel. Returns True if inserted, False if duplicate or blacklisted."""

    channel_id = channel["channel_id"]
    if category != ChannelCategory.BLACKLISTED and is_blacklisted(channel_id):
        return False

    payload = _prepare_channel_payload(channel)
    with get_cursor() as cursor:
        try:
            columns = ", ".join(CHANNEL_COLUMNS)
            placeholders = ", ".join("?" for _ in CHANNEL_COLUMNS)
            values = [payload.get(column) for column in CHANNEL_COLUMNS]
            cursor.execute(
                f"INSERT INTO {CHANNEL_TABLES[category]} ({columns}) VALUES ({placeholders})",
                values,
            )
            return True
        except sqlite3.IntegrityError:
            return False


def bulk_insert_channels(
    channels: Iterable[Dict[str, Any]], *, category: ChannelCategory = ChannelCategory.ACTIVE
) -> int:
    inserted = 0
    for channel in channels:
        if insert_channel(channel, category=category):
            inserted += 1
    return inserted


def record_channel_emails(channel_id: str, emails: Iterable[str], timestamp: str) -> Set[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for email in emails:
        if not email:
            continue
        normalized_email = _normalize_email(email)
        if not normalized_email or normalized_email in seen:
            continue
        seen.add(normalized_email)
        normalized.append(normalized_email)

    if not normalized:
        return set()

    with get_cursor() as cursor:
        for email in normalized:
            cursor.execute(
                """
                INSERT INTO emails_unique (email, first_seen_channel_id, last_seen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    first_seen_channel_id = COALESCE(first_seen_channel_id, excluded.first_seen_channel_id)
                """,
                (email, channel_id, timestamp),
            )
            cursor.execute(
                """
                INSERT INTO channel_emails (channel_id, email, last_seen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id, email) DO UPDATE SET last_seen_at = excluded.last_seen_at
                """,
                (channel_id, email, timestamp),
            )

    return set(normalized)


def get_channel_email_set(channel_id: str) -> Set[str]:
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT email FROM channel_emails WHERE channel_id = ?",
            (channel_id,),
        )
        return {row[0] for row in cursor.fetchall()}


def has_all_known_emails(emails: Iterable[str]) -> bool:
    normalized: Set[str] = set()
    for email in emails:
        if not email:
            continue
        normalized_email = _normalize_email(email)
        if normalized_email:
            normalized.add(normalized_email)
    if not normalized:
        return False
    params = list(normalized)
    placeholders = ",".join("?" for _ in params)
    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT email FROM emails_unique WHERE email IN ({placeholders})",
            params,
        )
        rows = {row[0] for row in cursor.fetchall()}
    return rows == set(params)


def get_unique_email_rows(
    filters: ChannelFilters, *, category: ChannelCategory = ChannelCategory.ACTIVE
) -> List[Dict[str, Any]]:
    table = CHANNEL_TABLES[category]
    where_clause, params = _build_channel_filters(filters, table_alias="c")
    query = (
        """
        SELECT
            ce.email,
            c.channel_id,
            c.name,
            c.url,
            c.last_updated,
            c.created_at,
            eu.first_seen_channel_id,
            eu.last_seen_at
        FROM channel_emails ce
        JOIN {table} c ON c.channel_id = ce.channel_id
        LEFT JOIN emails_unique eu ON eu.email = ce.email
        {where_clause}
        ORDER BY eu.last_seen_at DESC, c.last_updated DESC, c.created_at DESC
        """.format(table=table, where_clause=where_clause)
    )

    with get_cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    grouped: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        email = row["email"]
        grouped.setdefault(email, []).append(row)

    unique_rows: List[Dict[str, Any]] = []
    for email, candidates in grouped.items():
        first_seen_channel = candidates[0]["first_seen_channel_id"] if candidates else None
        primary_row = None
        if first_seen_channel:
            for row in candidates:
                if row["channel_id"] == first_seen_channel:
                    primary_row = row
                    break
        if primary_row is None and candidates:
            primary_row = max(
                candidates,
                key=lambda row: (
                    row["last_seen_at"] or "",
                    row["last_updated"] or "",
                    row["created_at"] or "",
                ),
            )
        if primary_row is None:
            continue
        other_count = max(0, len(candidates) - 1)
        unique_rows.append(
            {
                "email": email,
                "primary_channel_id": primary_row["channel_id"],
                "primary_channel_name": primary_row["name"] or "",
                "primary_channel_url": ensure_channel_url(
                    primary_row["channel_id"], primary_row["url"]
                ),
                "other_channels_count": other_count,
                "last_updated": primary_row["last_seen_at"]
                or primary_row["last_updated"]
                or primary_row["created_at"],
            }
        )

    unique_rows.sort(key=lambda row: (row["last_updated"] or ""), reverse=True)
    return unique_rows


def update_channel_enrichment(
    channel_id: str,
    *,
    name: Optional[str] = None,
    subscribers: Optional[int] = None,
    language: Optional[str] = None,
    language_confidence: Optional[float] = None,
    emails: Optional[str] = None,
    email_gate_present: Optional[bool] = None,
    last_updated: Optional[str] = None,
    last_attempted: Optional[str] = None,
    last_enriched_at: Optional[str] = None,
    last_enriched_result: Optional[str] = None,
    needs_enrichment: Optional[bool] = None,
    last_error: Optional[str] = None,
    status: Optional[str] = None,
    status_reason: Optional[str] = None,
    last_status_change: Optional[str] = None,
) -> None:
    updates: Dict[str, Any] = {}
    if name is not None:
        updates["name"] = name
    if subscribers is not None:
        updates["subscribers"] = subscribers
    if language is not None:
        updates["language"] = language
    if language_confidence is not None:
        updates["language_confidence"] = language_confidence
    if emails is not None:
        updates["emails"] = emails
    if email_gate_present is not None:
        updates["email_gate_present"] = int(bool(email_gate_present))
    if last_updated is not None:
        updates["last_updated"] = last_updated
    if last_attempted is not None:
        updates["last_attempted"] = last_attempted
    if last_enriched_at is not None:
        updates["last_enriched_at"] = last_enriched_at
    if last_enriched_result is not None:
        updates["last_enriched_result"] = last_enriched_result
    if needs_enrichment is not None:
        updates["needs_enrichment"] = int(bool(needs_enrichment))
    if last_error is not None:
        updates["last_error"] = last_error
    if status is not None:
        updates["status"] = status
    if status_reason is not None:
        updates["status_reason"] = status_reason
    if last_status_change is not None:
        updates["last_status_change"] = last_status_change

    if not updates:
        return

    fields = ", ".join(f"{column} = ?" for column in updates)
    values = list(updates.values())

    with get_cursor() as cursor:
        for category in ChannelCategory:
            cursor.execute(
                f"UPDATE {CHANNEL_TABLES[category]} SET {fields} WHERE channel_id = ?",
                [*values, channel_id],
            )
            if cursor.rowcount:
                break


def set_channel_status(
    channel_id: str,
    status: str,
    *,
    reason: Optional[str],
    timestamp: Optional[str],
) -> None:
    last_error_value = reason
    if reason is None and status in {"new", "processing"}:
        last_error_value = ""
    update_channel_enrichment(
        channel_id,
        status=status,
        status_reason=reason,
        last_status_change=timestamp,
        needs_enrichment=None,
        last_error=last_error_value,
    )


def ensure_channel_url(channel_id: Optional[str], url: Optional[str]) -> str:
    if url:
        return url
    if not channel_id:
        return ""
    return f"https://www.youtube.com/channel/{channel_id}"


def _build_channel_filters(
    filters: ChannelFilters,
    *,
    table_alias: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    prefix = f"{table_alias}." if table_alias else ""

    if filters.query_text:
        clauses.append(f"({prefix}name LIKE ? OR {prefix}url LIKE ? OR {prefix}emails LIKE ?)")
        term = f"%{filters.query_text}%"
        params.extend([term, term, term])

    if filters.languages:
        placeholders = ",".join("?" for _ in filters.languages)
        clauses.append(f"{prefix}language IN ({placeholders})")
        params.extend(filters.languages)

    if filters.statuses:
        placeholders = ",".join("?" for _ in filters.statuses)
        clauses.append(f"{prefix}status IN ({placeholders})")
        params.extend(filters.statuses)

    if filters.min_subscribers is not None:
        clauses.append(f"({prefix}subscribers IS NOT NULL AND {prefix}subscribers >= ?)")
        params.append(filters.min_subscribers)

    if filters.max_subscribers is not None:
        clauses.append(f"({prefix}subscribers IS NOT NULL AND {prefix}subscribers <= ?)")
        params.append(filters.max_subscribers)

    if filters.emails_only:
        clauses.append(f"({prefix}emails IS NOT NULL AND TRIM({prefix}emails) != '')")

    if filters.email_gate_only:
        clauses.append(f"{prefix}email_gate_present = 1")

    if filters.unique_emails and filters.emails_only:
        clauses.append(f"{prefix}channel_id NOT IN ({GLOBAL_DUPLICATE_CHANNELS_QUERY})")

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_clause, params


def get_channels(
    category: ChannelCategory,
    filters: ChannelFilters,
    *,
    sort: str,
    order: str,
    limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], int]:
    valid_sorts = {
        "name": "name",
        "subscribers": "subscribers",
        "language": "language",
        "last_updated": "last_updated",
        "created_at": "created_at",
        "status": "status",
        "last_status_change": "last_status_change",
        "exported_at": "exported_at",
        "archived_at": "archived_at",
    }
    sort_column = valid_sorts.get(sort, "created_at")
    order_direction = "DESC" if order.lower() == "desc" else "ASC"

    table = CHANNEL_TABLES[category]
    where_clause, params = _build_channel_filters(filters)

    query = (
        f"SELECT * FROM {table} {where_clause} "
        f"ORDER BY {sort_column} {order_direction} LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    with get_cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        cursor.execute(
            f"SELECT COUNT(*) FROM {table} {where_clause}",
            params[:-2],
        )
        total_row = cursor.fetchone()
        total = total_row[0] if total_row else 0

    items: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        items.append(item)

    if items:
        channels_with_emails = [
            item.get("channel_id")
            for item in items
            if item.get("channel_id") and item.get("emails")
        ]
        duplicates_map = _collect_duplicate_emails_for_channels(channels_with_emails)
        for item in items:
            duplicate_values = sorted(duplicates_map.get(item.get("channel_id"), set()))
            item["duplicate_email_count"] = len(duplicate_values)
            item["duplicate_emails"] = ", ".join(duplicate_values)
            item["has_duplicate_emails"] = bool(duplicate_values)

    return items, total


def _fetch_channels_by_ids(
    cursor: sqlite3.Cursor,
    category: ChannelCategory,
    channel_ids: Sequence[str],
) -> List[sqlite3.Row]:
    if not channel_ids:
        return []
    placeholders = ",".join("?" for _ in channel_ids)
    cursor.execute(
        f"SELECT * FROM {CHANNEL_TABLES[category]} WHERE channel_id IN ({placeholders})",
        list(channel_ids),
    )
    return cursor.fetchall()


def _delete_channels_by_ids(
    cursor: sqlite3.Cursor,
    category: ChannelCategory,
    channel_ids: Sequence[str],
) -> None:
    if not channel_ids:
        return
    placeholders = ",".join("?" for _ in channel_ids)
    cursor.execute(
        f"DELETE FROM {CHANNEL_TABLES[category]} WHERE channel_id IN ({placeholders})",
        list(channel_ids),
    )


def _move_channels(
    channel_ids: Sequence[str],
    source: ChannelCategory,
    destination: ChannelCategory,
    *,
    timestamp: str,
    status: str,
    status_reason: str,
    needs_enrichment: int,
) -> List[str]:
    if not channel_ids:
        return []

    moved: List[str] = []
    with get_cursor() as cursor:
        rows = _fetch_channels_by_ids(cursor, source, channel_ids)
        if not rows:
            return []
        for row in rows:
            data = dict(row)
            data["status"] = status
            data["status_reason"] = status_reason
            data["last_status_change"] = timestamp
            data["needs_enrichment"] = needs_enrichment
            if destination is ChannelCategory.ARCHIVED:
                data["archived_at"] = timestamp
            else:
                data["archived_at"] = None
            _insert_or_replace(cursor, CHANNEL_TABLES[destination], data)
            moved.append(data["channel_id"])
        _delete_channels_by_ids(cursor, source, moved)
    return moved


def archive_channels_by_ids(channel_ids: Sequence[str], timestamp: str) -> List[str]:
    return _move_channels(
        channel_ids,
        ChannelCategory.ACTIVE,
        ChannelCategory.ARCHIVED,
        timestamp=timestamp,
        status="archived",
        status_reason="Archived",
        needs_enrichment=0,
    )


def archive_channels_by_exported_at(exported_at: str, timestamp: str) -> List[str]:
    if not exported_at:
        return []

    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT channel_id FROM {CHANNEL_TABLES[ChannelCategory.ACTIVE]} WHERE exported_at = ?",
            [exported_at],
        )
        rows = cursor.fetchall()

    channel_ids = [row["channel_id"] for row in rows if row["channel_id"]]
    if not channel_ids:
        return []

    return archive_channels_by_ids(channel_ids, timestamp)


def mark_channels_exported(
    category: ChannelCategory,
    channel_ids: Sequence[str],
    timestamp: str,
    *,
    archive: bool = False,
) -> List[str]:
    if not channel_ids:
        return []

    unique_ids = list(dict.fromkeys(cid for cid in channel_ids if cid))
    if not unique_ids:
        return []

    table = CHANNEL_TABLES[category]
    with get_cursor() as cursor:
        for chunk in _chunked(unique_ids, 200):
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(
                f"UPDATE {table} SET exported_at = ? WHERE channel_id IN ({placeholders})",
                [timestamp, *chunk],
            )

    archived: List[str] = []
    if archive and category is ChannelCategory.ACTIVE:
        archived = archive_channels_by_ids(unique_ids, timestamp)
    return archived


def fetch_project_bundle_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a complete snapshot of the project for bundle exports."""

    with get_cursor() as cursor:
        channels: Dict[str, List[Dict[str, Any]]] = {}
        for category, table in CHANNEL_TABLES.items():
            cursor.execute(f"SELECT * FROM {table} ORDER BY created_at ASC")
            channels[category.value] = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM blacklist ORDER BY updated_at DESC, created_at DESC")
        blacklist_rows = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM emails_unique ORDER BY last_seen_at DESC, first_seen_channel_id ASC"
        )
        emails_unique = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT * FROM channel_emails ORDER BY last_seen_at DESC, email ASC, channel_id ASC"
        )
        channel_emails = [dict(row) for row in cursor.fetchall()]

    email_index = _build_global_email_index(channel_emails, emails_unique)

    data = {
        "channels": channels,
        "blacklist": blacklist_rows,
        "emails_unique": emails_unique,
        "channel_emails": channel_emails,
    }

    return data, email_index


def _build_global_email_index(
    channel_emails: Sequence[Dict[str, Any]],
    emails_unique: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Construct a global index of email addresses and related channels."""

    index: Dict[str, Dict[str, Any]] = {}

    for relation in channel_emails:
        email = (relation.get("email") or "").strip()
        if not email:
            continue
        info = index.setdefault(email, {"channelIds": [], "lastSeenAt": None})
        channel_id = relation.get("channel_id")
        if channel_id and channel_id not in info["channelIds"]:
            info["channelIds"].append(channel_id)
        last_seen = relation.get("last_seen_at")
        if last_seen and (info.get("lastSeenAt") is None or last_seen > info["lastSeenAt"]):
            info["lastSeenAt"] = last_seen

    for entry in emails_unique:
        email = (entry.get("email") or "").strip()
        if not email:
            continue
        info = index.setdefault(email, {"channelIds": [], "lastSeenAt": None})
        first_seen_channel = entry.get("first_seen_channel_id")
        if first_seen_channel:
            info["firstSeenChannelId"] = first_seen_channel
        last_seen = entry.get("last_seen_at")
        if last_seen and (info.get("lastSeenAt") is None or last_seen > info["lastSeenAt"]):
            info["lastSeenAt"] = last_seen

    for email, info in index.items():
        channel_ids = sorted(dict.fromkeys(info.get("channelIds", [])))
        info["channelIds"] = channel_ids
        info.setdefault("firstSeenChannelId", None)
        info.setdefault("lastSeenAt", None)
        info["channelCount"] = len(channel_ids)

    return dict(sorted(index.items(), key=lambda item: item[0]))


def _coerce_optional_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return default
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return default
        try:
            return int(candidate)
        except ValueError:
            return default
    return default


def _coerce_flag(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    result = _coerce_optional_int(value, default=default)
    if result is None:
        return default
    return 1 if result else 0


def _normalize_bundle_channel_record(record: Dict[str, Any]) -> Dict[str, Any]:
    channel_id_raw = record.get("channel_id")
    channel_id = str(channel_id_raw).strip() if channel_id_raw is not None else ""
    if not channel_id:
        raise ValueError("channel record missing channel_id")

    normalized: Dict[str, Any] = {column: record.get(column) for column in CHANNEL_COLUMNS}
    normalized["channel_id"] = channel_id
    normalized["url"] = ensure_channel_url(channel_id, record.get("url"))
    normalized["subscribers"] = _coerce_optional_int(record.get("subscribers"))
    normalized["email_gate_present"] = _coerce_flag(record.get("email_gate_present"))
    normalized["needs_enrichment"] = _coerce_flag(record.get("needs_enrichment"), default=1)
    status = record.get("status")
    normalized["status"] = (status or "new").strip() or "new"
    if not normalized.get("created_at"):
        fallback_created = (
            record.get("last_updated")
            or record.get("last_status_change")
            or record.get("archived_at")
            or record.get("exported_at")
        )
        normalized["created_at"] = fallback_created
    return normalized


def restore_project_bundle(
    data: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Bundle data must be an object")

    if meta and isinstance(meta, dict):
        schema_version = meta.get("schemaVersion")
        if isinstance(schema_version, (int, float)) and int(schema_version) > PROJECT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "Bundle schema version is newer than supported version"
            )

    channels_payload = data.get("channels")
    if not isinstance(channels_payload, dict):
        raise ValueError("Bundle data is missing channel collections")

    blacklist_payload = data.get("blacklist") or []
    emails_unique_payload = data.get("emails_unique") or []
    channel_emails_payload = data.get("channel_emails") or []

    with get_cursor() as cursor:
        current_channels: Dict[str, Dict[str, Any]] = {}
        for category, table in CHANNEL_TABLES.items():
            cursor.execute(f"SELECT * FROM {table}")
            for row in cursor.fetchall():
                record = {column: row[column] for column in CHANNEL_COLUMNS}
                current_channels[row["channel_id"]] = {
                    "category": category,
                    "data": record,
                }

        cursor.execute("SELECT * FROM blacklist")
        current_blacklist = {
            row["channel_id"]: {
                "channel_id": row["channel_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in cursor.fetchall()
            if row["channel_id"]
        }

        cursor.execute("SELECT * FROM emails_unique")
        current_emails_unique = {
            row["email"]: {
                "email": row["email"],
                "first_seen_channel_id": row["first_seen_channel_id"],
                "last_seen_at": row["last_seen_at"],
            }
            for row in cursor.fetchall()
            if row["email"]
        }

        cursor.execute("SELECT * FROM channel_emails")
        current_channel_emails = {
            (row["channel_id"], row["email"]): {
                "channel_id": row["channel_id"],
                "email": row["email"],
                "last_seen_at": row["last_seen_at"],
            }
            for row in cursor.fetchall()
            if row["channel_id"] and row["email"]
        }

    channel_summary: Dict[str, Dict[str, int]] = {
        category.value: {
            "inserted": 0,
            "updated": 0,
            "movedIn": 0,
            "movedOut": 0,
            "unchanged": 0,
            "skipped": 0,
        }
        for category in ChannelCategory
    }
    channel_actions: List[Dict[str, Any]] = []

    for category_name, records in channels_payload.items():
        try:
            category = ChannelCategory(category_name)
        except ValueError as exc:
            raise ValueError(f"Unknown channel category: {category_name}") from exc

        if not isinstance(records, list):
            raise ValueError(f"Channel list for {category.value} must be an array")

        seen_ids: Set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                channel_summary[category.value]["skipped"] += 1
                continue
            try:
                normalized = _normalize_bundle_channel_record(record)
            except ValueError:
                channel_summary[category.value]["skipped"] += 1
                continue

            channel_id = normalized["channel_id"]
            if channel_id in seen_ids:
                continue
            seen_ids.add(channel_id)

            existing = current_channels.get(channel_id)
            if not existing:
                channel_summary[category.value]["inserted"] += 1
                channel_actions.append(
                    {
                        "channel_id": channel_id,
                        "category": category,
                        "data": dict(normalized),
                        "delete_from": None,
                    }
                )
                current_channels[channel_id] = {
                    "category": category,
                    "data": dict(normalized),
                }
                continue

            existing_category = existing["category"]
            existing_data = existing["data"]
            needs_update = normalized != existing_data or existing_category is not category

            if existing_category is not category:
                channel_summary[category.value]["movedIn"] += 1
                channel_summary[existing_category.value]["movedOut"] += 1
            elif needs_update:
                channel_summary[category.value]["updated"] += 1
            else:
                channel_summary[category.value]["unchanged"] += 1
                continue

            channel_actions.append(
                {
                    "channel_id": channel_id,
                    "category": category,
                    "data": dict(normalized),
                    "delete_from": existing_category if existing_category is not category else None,
                }
            )
            current_channels[channel_id] = {
                "category": category,
                "data": dict(normalized),
            }

    blacklist_summary = {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    blacklist_actions: List[Dict[str, Any]] = []

    for entry in blacklist_payload:
        if not isinstance(entry, dict):
            blacklist_summary["skipped"] += 1
            continue
        channel_id_raw = entry.get("channel_id")
        channel_id = str(channel_id_raw).strip() if channel_id_raw is not None else ""
        if not channel_id:
            blacklist_summary["skipped"] += 1
            continue
        created_at = entry.get("created_at")
        updated_at = entry.get("updated_at") or created_at
        normalized = {
            "channel_id": channel_id,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        existing = current_blacklist.get(channel_id)
        if not existing:
            blacklist_summary["inserted"] += 1
            blacklist_actions.append(normalized)
        elif existing != normalized:
            blacklist_summary["updated"] += 1
            blacklist_actions.append(normalized)
        else:
            blacklist_summary["unchanged"] += 1
        current_blacklist[channel_id] = normalized

    email_unique_summary = {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    email_relation_summary = {
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "autoCreatedUnique": 0,
    }
    email_unique_actions: List[Dict[str, Any]] = []
    channel_email_actions: List[Dict[str, Any]] = []

    seen_unique: Set[str] = set()
    for entry in emails_unique_payload:
        if not isinstance(entry, dict):
            email_unique_summary["skipped"] += 1
            continue
        email_raw = entry.get("email")
        email = _normalize_email(email_raw) if email_raw else None
        if not email:
            email_unique_summary["skipped"] += 1
            continue
        if email in seen_unique:
            continue
        seen_unique.add(email)
        normalized = {
            "email": email,
            "first_seen_channel_id": entry.get("first_seen_channel_id"),
            "last_seen_at": entry.get("last_seen_at"),
        }
        existing = current_emails_unique.get(email)
        if not existing:
            email_unique_summary["inserted"] += 1
            email_unique_actions.append(normalized)
        elif existing != normalized:
            email_unique_summary["updated"] += 1
            email_unique_actions.append(normalized)
        else:
            email_unique_summary["unchanged"] += 1
        current_emails_unique[email] = normalized

    seen_relations: Set[Tuple[str, str]] = set()
    for entry in channel_emails_payload:
        if not isinstance(entry, dict):
            email_relation_summary["skipped"] += 1
            continue
        channel_id_raw = entry.get("channel_id")
        channel_id = str(channel_id_raw).strip() if channel_id_raw is not None else ""
        email_raw = entry.get("email")
        email = _normalize_email(email_raw) if email_raw else None
        if not channel_id or not email:
            email_relation_summary["skipped"] += 1
            continue
        key = (channel_id, email)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        last_seen_at = entry.get("last_seen_at")
        normalized = {
            "channel_id": channel_id,
            "email": email,
            "last_seen_at": last_seen_at,
        }
        existing_relation = current_channel_emails.get(key)
        if not existing_relation:
            email_relation_summary["inserted"] += 1
            channel_email_actions.append(normalized)
        elif existing_relation != normalized:
            email_relation_summary["updated"] += 1
            channel_email_actions.append(normalized)
        else:
            email_relation_summary["unchanged"] += 1
        current_channel_emails[key] = normalized

        if email not in current_emails_unique:
            inferred_unique = {
                "email": email,
                "first_seen_channel_id": channel_id,
                "last_seen_at": last_seen_at,
            }
            current_emails_unique[email] = inferred_unique
            email_unique_actions.append(inferred_unique)
            email_unique_summary["inserted"] += 1
            email_relation_summary["autoCreatedUnique"] += 1

    delete_map: Dict[ChannelCategory, Set[str]] = {category: set() for category in ChannelCategory}
    if not dry_run:
        with get_cursor() as cursor:
            for action in channel_actions:
                _insert_or_replace(cursor, CHANNEL_TABLES[action["category"]], action["data"])
                source_category = action.get("delete_from")
                if source_category:
                    delete_map[source_category].add(action["channel_id"])

            for source_category, ids in delete_map.items():
                if not ids:
                    continue
                placeholders = ",".join("?" for _ in ids)
                cursor.execute(
                    f"DELETE FROM {CHANNEL_TABLES[source_category]} WHERE channel_id IN ({placeholders})",
                    list(ids),
                )

            for entry in blacklist_actions:
                cursor.execute(
                    """
                    INSERT INTO blacklist (channel_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel_id) DO UPDATE SET
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (entry["channel_id"], entry.get("created_at"), entry.get("updated_at")),
                )

            for entry in email_unique_actions:
                cursor.execute(
                    """
                    INSERT INTO emails_unique (email, first_seen_channel_id, last_seen_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(email) DO UPDATE SET
                        first_seen_channel_id = excluded.first_seen_channel_id,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (
                        entry["email"],
                        entry.get("first_seen_channel_id"),
                        entry.get("last_seen_at"),
                    ),
                )

            for entry in channel_email_actions:
                cursor.execute(
                    """
                    INSERT INTO channel_emails (channel_id, email, last_seen_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel_id, email) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at
                    """,
                    (
                        entry["channel_id"],
                        entry["email"],
                        entry.get("last_seen_at"),
                    ),
                )

    channel_counts: Dict[str, int] = {category.value: 0 for category in ChannelCategory}
    for info in current_channels.values():
        channel_counts[info["category"].value] += 1

    email_index = _build_global_email_index(
        list(current_channel_emails.values()),
        list(current_emails_unique.values()),
    )

    summary = {
        "dryRun": dry_run,
        "channelSummary": channel_summary,
        "channelCounts": channel_counts,
        "blacklistSummary": blacklist_summary,
        "emailSummary": {
            "unique": email_unique_summary,
            "relations": email_relation_summary,
        },
        "globalEmailIndex": email_index,
    }

    if meta and isinstance(meta, dict):
        summary["meta"] = {
            "schemaVersion": meta.get("schemaVersion"),
            "bundleExportedAt": meta.get("exportedAt"),
        }
    else:
        summary["meta"] = {
            "schemaVersion": PROJECT_BUNDLE_SCHEMA_VERSION,
            "bundleExportedAt": None,
        }

    return summary


def restore_channels_by_ids(
    channel_ids: Sequence[str],
    timestamp: str,
    *,
    source_categories: Optional[Sequence[ChannelCategory]] = None,
) -> List[str]:
    if not channel_ids:
        return []
    categories = list(source_categories) if source_categories else [
        ChannelCategory.ARCHIVED,
        ChannelCategory.BLACKLISTED,
    ]
    restored: List[str] = []
    remaining = list(channel_ids)
    for category in categories:
        if not remaining:
            break
        moved = _move_channels(
            remaining,
            category,
            ChannelCategory.ACTIVE,
            timestamp=timestamp,
            status="new",
            status_reason="Restored",
            needs_enrichment=1,
        )
        restored.extend(moved)
        remaining = [cid for cid in remaining if cid not in moved]
    return restored


def blacklist_channels_by_ids(
    channel_ids: Sequence[str],
    timestamp: str,
    *,
    source_categories: Optional[Sequence[ChannelCategory]] = None,
) -> List[str]:
    if not channel_ids:
        return []
    categories = list(source_categories) if source_categories else [
        ChannelCategory.ACTIVE,
        ChannelCategory.ARCHIVED,
    ]
    blacklisted: List[str] = []
    remaining = list(channel_ids)
    for category in categories:
        if not remaining:
            break
        moved = _move_channels(
            remaining,
            category,
            ChannelCategory.BLACKLISTED,
            timestamp=timestamp,
            status="blacklisted",
            status_reason="Blacklisted",
            needs_enrichment=0,
        )
        blacklisted.extend(moved)
        remaining = [cid for cid in remaining if cid not in moved]
    for cid in blacklisted:
        ensure_blacklisted_channel(cid, timestamp)
    return blacklisted


def get_pending_channels(limit: Optional[int]) -> List[Dict[str, Any]]:
    limit_clause = "LIMIT ?" if limit is not None else ""
    params: Tuple[Any, ...] = (limit,) if limit is not None else tuple()
    table = CHANNEL_TABLES[ChannelCategory.ACTIVE]
    query = (
        f"SELECT * FROM {table} WHERE status IN ('new', 'error') "
        f"ORDER BY last_attempted IS NULL DESC, last_attempted ASC "
        + limit_clause
    )
    with get_cursor() as cursor:
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_channels_for_email_enrichment(limit: Optional[int]) -> List[Dict[str, Any]]:
    limit_clause = "LIMIT ?" if limit is not None else ""
    params: Tuple[Any, ...] = (limit,) if limit is not None else tuple()
    table = CHANNEL_TABLES[ChannelCategory.ACTIVE]
    query = (
        f"SELECT * FROM {table} "
        f"ORDER BY last_updated IS NULL DESC, last_updated ASC "
        + limit_clause
    )
    with get_cursor() as cursor:
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_enrichment_candidates(
    email_only_mode: bool,
    *,
    scope: str,
    category: ChannelCategory,
    filters: Optional[ChannelFilters],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if scope != "filtered" or filters is None:
        if email_only_mode:
            return get_channels_for_email_enrichment(limit)
        return get_pending_channels(limit)

    limit_clause = "LIMIT ?" if limit is not None else ""
    params: List[Any] = []
    table = CHANNEL_TABLES[category]

    statuses = filters.statuses
    if statuses is None and not email_only_mode:
        statuses = ("new", "error")

    filtered = ChannelFilters(
        query_text=filters.query_text,
        languages=filters.languages,
        statuses=statuses,
        min_subscribers=filters.min_subscribers,
        max_subscribers=filters.max_subscribers,
        emails_only=filters.emails_only,
        include_archived=False,
        email_gate_only=filters.email_gate_only,
        unique_emails=filters.unique_emails,
    )

    where_clause, filter_params = _build_channel_filters(filtered)
    params.extend(filter_params)
    order_by = "last_updated IS NULL DESC, last_updated ASC" if email_only_mode else "last_attempted IS NULL DESC, last_attempted ASC"
    query = f"SELECT * FROM {table} {where_clause} ORDER BY {order_by} {limit_clause}"
    if limit is not None:
        params.append(limit)

    with get_cursor() as cursor:
        cursor.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def get_channel_status_totals() -> Dict[str, int]:
    table = CHANNEL_TABLES[ChannelCategory.ACTIVE]
    totals: Dict[str, int] = {status: 0 for status in ("new", "processing", "completed", "error")}
    extras: Dict[str, int] = {}
    with get_cursor() as cursor:
        cursor.execute(
            f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status"
        )
        for row in cursor.fetchall():
            status = (row["status"] or "").strip().lower()
            count = row["count"] if row and row["count"] is not None else 0
            if not status:
                continue
            if status in totals:
                totals[status] = count
            else:
                extras[status] = count
    return {**totals, **extras}


def get_channel_totals() -> Dict[str, int]:
    totals: Dict[str, int] = {}
    with get_cursor() as cursor:
        for category in ChannelCategory:
            cursor.execute(
                f"SELECT COUNT(*) AS count FROM {CHANNEL_TABLES[category]}",
            )
            row = cursor.fetchone()
            totals[category.value] = row["count"] if row and row["count"] is not None else 0
        cursor.execute("SELECT COUNT(*) AS count FROM emails_unique")
        email_row = cursor.fetchone()
        totals["unique_emails"] = email_row["count"] if email_row and email_row["count"] is not None else 0
    totals["total"] = totals.get(ChannelCategory.ACTIVE.value, 0)
    return totals
