"""Utilities for interacting with YouTube without requiring official APIs."""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree as ET

import requests
from langdetect import DetectorFactory, LangDetectException, detect_langs

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

DetectorFactory.seed = 0

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }
)

CHANNEL_ID_PATTERN = re.compile(r"(UC[\w-]{22})", re.IGNORECASE)
HYPERLINK_RE = re.compile(r"=HYPERLINK\(\s*([\"'])([^\"']+?)\1", re.IGNORECASE)
HANDLE_PATTERN = re.compile(r"@([A-Za-z0-9._-]{3,})$")


def _normalize_candidate(value: str) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    match = HYPERLINK_RE.search(cleaned)
    if match:
        cleaned = match.group(2).strip()
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = (
        cleaned.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
    )
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    cleaned = cleaned.strip()
    cleaned = cleaned.strip("<>\"'()")
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    if " " in cleaned:
        candidate, _, _ = cleaned.partition(" ")
        if candidate:
            cleaned = candidate
    cleaned = cleaned.rstrip(",;)")
    cleaned = cleaned.split("#", 1)[0]
    cleaned = cleaned.split("?", 1)[0]
    return cleaned.rstrip("/")


def sanitize_channel_input(value: Optional[str]) -> str:
    """Public helper returning a cleaned channel reference candidate."""

    return _normalize_candidate(value)


def _normalize_path(path: str) -> str:
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return "/"
    first = segments[0]
    if first.lower() == "channel" and len(segments) > 1:
        return f"/channel/{segments[1].upper()}"
    if first.startswith("@"):
        return f"/{first}"
    if first.lower() in {"c", "user"} and len(segments) > 1:
        return f"/{first}/{segments[1]}"
    return f"/{first}"


def _ensure_absolute_url(value: str) -> Optional[str]:
    candidate = _normalize_candidate(value)
    if not candidate:
        return None
    upper_candidate = candidate.upper()
    if CHANNEL_ID_PATTERN.fullmatch(upper_candidate):
        return f"https://www.youtube.com/channel/{upper_candidate}"
    if candidate.startswith("@"):
        candidate = candidate.split("/")[0]
        return f"https://www.youtube.com/{candidate}"
    if candidate.startswith("/"):
        candidate = f"https://www.youtube.com{candidate}"
    if candidate.lower().startswith("youtube.com"):
        candidate = f"https://{candidate}"
    if not re.match(r"^https?://", candidate, re.IGNORECASE):
        candidate = f"https://www.youtube.com/{candidate}"
    parsed = urlparse(candidate)
    netloc = parsed.netloc.lower()
    if "youtube.com" not in netloc:
        return None
    normalized = parsed._replace(
        scheme="https",
        netloc="www.youtube.com",
        path=_normalize_path(parsed.path),
        params="",
        query="",
        fragment="",
    )
    return urlunparse(normalized)


def _extract_channel_id_from_html(html_text: str) -> Optional[str]:
    canonical = re.search(
        r'<link[^>]+rel="canonical"[^>]+href="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
        html_text,
        re.IGNORECASE,
    )
    if canonical:
        return canonical.group(1).upper()

    ytcfg_match = re.search(r'"CHANNEL_ID"\s*:\s*"(UC[\w-]{22})"', html_text)
    if ytcfg_match:
        return ytcfg_match.group(1).upper()

    channel_match = re.search(r'"channelId"\s*:\s*"(UC[\w-]{22})"', html_text)
    if channel_match:
        return channel_match.group(1).upper()

    browse_match = re.search(r'"browseId"\s*:\s*"(UC[\w-]{22})"', html_text)
    if browse_match:
        return browse_match.group(1).upper()

    matches = CHANNEL_ID_PATTERN.findall(html_text)
    if matches:
        return matches[0].upper()
    return None


def _extract_canonical_channel_url(html_text: str) -> Optional[str]:
    patterns = [
        r'<link[^>]+rel="canonical"[^>]+href="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
        r'<meta[^>]+property="og:url"[^>]+content="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
        r'<meta[^>]+itemprop="channelId"[^>]+content="(UC[\w-]{22})"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            channel_id = match.group(1).upper()
            return f"https://www.youtube.com/channel/{channel_id}"
    return None


def _normalize_handle(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip()
    candidate = candidate.replace("\\/", "/")
    candidate = candidate.split("/", 1)[0]
    if not candidate:
        return None
    if not candidate.startswith("@"):
        candidate = f"@{candidate.lstrip('@')}"
    match = HANDLE_PATTERN.fullmatch(candidate)
    if not match:
        return None
    return candidate


def _extract_handle_from_html(html_text: str) -> Optional[str]:
    patterns = [
        r'<link[^>]+rel="canonical"[^>]+href="https://www\.youtube\.com/(@[^"/?#]+)"',
        r'<meta[^>]+property="og:url"[^>]+content="https://www\.youtube\.com/(@[^"/?#]+)"',
        r'"canonicalBaseUrl"\s*:\s*"\\?/?(@[^"\\]+)"',
        r'"channelHandle"\s*:\s*"(@[^"\\]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            handle = _normalize_handle(match.group(1))
            if handle:
                return handle
    return None


def _extract_handle_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    for segment in [segment for segment in parsed.path.split("/") if segment]:
        if segment.startswith("@"):
            handle = _normalize_handle(segment)
            if handle:
                return handle
    return None


def _extract_channel_title(html_text: str) -> Optional[str]:
    patterns = [
        r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"',
        r'<meta[^>]+name="title"[^>]+content="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text)
        if match:
            title = html.unescape(match.group(1)).strip()
            if title:
                return title
    return None


def extract_channel_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = _normalize_candidate(value)
    if not candidate:
        return None
    match = CHANNEL_ID_PATTERN.search(candidate)
    if match:
        return match.group(1).upper()
    if candidate.upper().startswith("UC") and len(candidate) == 24:
        return candidate.upper()
    return None


def resolve_channel_id(value: Optional[str], *, timeout: int = 8) -> Optional[str]:
    resolution, _ = resolve_channel(value, timeout=timeout)
    if resolution:
        return resolution.channel_id
    return None


class RateLimiter:
    """Simple thread-safe rate limiter based on a minimum interval."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_time = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_time = self.min_interval - (now - self._last_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_time = time.monotonic()


RATE_LIMITER = RateLimiter(min_interval=0.35)  # ~3 requests per second globally

RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
MEDIA_NS = "{http://search.yahoo.com/mrss/}"


class EnrichmentError(RuntimeError):
    """Raised when enrichment of a channel fails due to data issues."""



@dataclass
class ChannelSearchResult:
    channel_id: str
    title: str
    url: str
    subscribers: Optional[int]


@dataclass
class ChannelSearchSession:
    api_key: Optional[str]
    context: Dict[str, Any]


@dataclass
class ChannelSearchPage:
    results: List[ChannelSearchResult]
    next_page_token: Optional[str]
    session: Optional[ChannelSearchSession] = None


@dataclass
class DiscoveryMetadata:
    last_upload: Optional[str] = None
    language: Optional[str] = None
    language_confidence: Optional[float] = None


@dataclass(frozen=True)
class ChannelResolution:
    channel_id: str
    canonical_url: str
    handle: Optional[str] = None
    title: Optional[str] = None


def normalize_channel_reference(value: Optional[str]) -> str:
    candidate = sanitize_channel_input(value)
    if not candidate:
        return ""
    upper_candidate = candidate.upper()
    if CHANNEL_ID_PATTERN.fullmatch(upper_candidate):
        return upper_candidate
    absolute = _ensure_absolute_url(candidate)
    return absolute or ""


def resolve_channel(value: Optional[str], *, timeout: int = 8) -> Tuple[Optional[ChannelResolution], Optional[str]]:
    base_value = sanitize_channel_input(value)
    if not base_value:
        return None, "invalid_input"
    normalized = normalize_channel_reference(base_value)
    if not normalized:
        return None, "invalid_url"
    is_channel_id = normalized.upper().startswith("UC") and len(normalized) == 24 and "/" not in normalized
    fetch_url = (
        f"https://www.youtube.com/channel/{normalized}"
        if is_channel_id
        else normalized
    )
    try:
        RATE_LIMITER.wait()
        response = SESSION.get(fetch_url, timeout=timeout, allow_redirects=True)
    except requests.Timeout:
        return None, "network_error"
    except requests.RequestException:
        return None, "network_error"

    status = response.status_code
    if status == 404:
        return None, "not_found"
    if status >= 500:
        return None, "network_error"
    if status >= 400:
        return None, "resolution_failed"

    html_text = response.text
    channel_id = _extract_channel_id_from_html(html_text)
    if not channel_id:
        channel_id = extract_channel_id(response.url)
    if not channel_id and is_channel_id:
        channel_id = normalized
    if not channel_id:
        return None, "resolution_failed"

    canonical_url = _extract_canonical_channel_url(html_text)
    if not canonical_url:
        canonical_url = f"https://www.youtube.com/channel/{channel_id}"

    handle = _extract_handle_from_html(html_text)
    if not handle:
        handle = _extract_handle_from_url(response.url)

    title = _extract_channel_title(html_text)

    return (
        ChannelResolution(
            channel_id=channel_id,
            canonical_url=canonical_url,
            handle=handle,
            title=title,
        ),
        None,
    )


def _extract_ytinitialdata(html: str) -> Optional[Dict]:
    patterns = [
        r"ytInitialData\s*=\s*(\{.*?\});",
        r"var ytInitialData\s*=\s*(\{.*?\});",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    return None


def _find_channel_renderers(data: Dict) -> Iterable[Dict]:
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "channelRenderer" in node:
                yield node["channelRenderer"]
            else:
                stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _parse_subscriber_count(text: str) -> Optional[int]:
    text = text.replace(" subscribers", "").strip()
    multiplier = 1
    if text.endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]
    text = text.replace(",", "")
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return None


def _extract_ytcfg(html: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for match in re.finditer(r"ytcfg\.set\((\{.*?\})\);", html, re.DOTALL):
        snippet = match.group(1)
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            config.update(payload)
    return config


def _build_channel_search_session(config: Dict[str, Any]) -> ChannelSearchSession:
    api_key = config.get("INNERTUBE_API_KEY")
    context_payload: Dict[str, Any]
    context_value = config.get("INNERTUBE_CONTEXT")
    if isinstance(context_value, dict):
        try:
            context_payload = json.loads(json.dumps(context_value))
        except (TypeError, ValueError):
            context_payload = {}
    else:
        context_payload = {}

    if not context_payload:
        client_version = config.get("INNERTUBE_CLIENT_VERSION") or "2.20240624.00.00"
        client_name = config.get("INNERTUBE_CLIENT_NAME") or "WEB"
        context_payload = {
            "client": {
                "clientName": client_name,
                "clientVersion": client_version,
                "hl": "en",
                "gl": "US",
            }
        }

    return ChannelSearchSession(api_key=api_key, context=context_payload)


def _collect_channel_results(data: Dict) -> List[ChannelSearchResult]:
    results: List[ChannelSearchResult] = []
    if not isinstance(data, dict):
        return results
    for renderer in _find_channel_renderers(data):
        channel_id = renderer.get("channelId")
        if not channel_id:
            continue
        title_runs = renderer.get("title", {}).get("runs", [])
        title = (
            title_runs[0]["text"]
            if title_runs
            else renderer.get("title", {}).get("simpleText", "")
        )
        nav = renderer.get("navigationEndpoint", {}).get("browseEndpoint", {})
        canonical = nav.get("canonicalBaseUrl")
        if canonical:
            url = f"https://www.youtube.com{canonical}"
        else:
            url = f"https://www.youtube.com/channel/{channel_id}"
        sub_text_obj = renderer.get("subscriberCountText", {})
        if "simpleText" in sub_text_obj:
            subscribers = _parse_subscriber_count(sub_text_obj["simpleText"])
        else:
            runs = sub_text_obj.get("runs", [])
            subscribers = (
                _parse_subscriber_count(runs[0]["text"])
                if runs
                else None
            )
        results.append(
            ChannelSearchResult(
                channel_id=channel_id,
                title=title,
                url=url,
                subscribers=subscribers,
            )
        )
    return results


def _extract_next_token(data: Dict) -> Optional[str]:
    stack: List[Any] = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "nextContinuationData" in node:
                next_data = node["nextContinuationData"]
                if isinstance(next_data, dict):
                    token = next_data.get("continuation")
                    if token:
                        return token
            if "continuationCommand" in node:
                command = node["continuationCommand"]
                if isinstance(command, dict):
                    token = command.get("token")
                    if token:
                        return token
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _search_initial_page(keyword: str) -> ChannelSearchPage:
    params = {
        "search_query": keyword,
        "sp": "EgIQAg%3D%3D",  # channel filter
    }
    RATE_LIMITER.wait()
    response = SESSION.get("https://www.youtube.com/results", params=params, timeout=10)
    response.raise_for_status()
    html_text = response.text
    data = _extract_ytinitialdata(html_text)
    config = _extract_ytcfg(html_text)
    session = _build_channel_search_session(config)
    results = _collect_channel_results(data or {})
    next_token = _extract_next_token(data or {}) if data else None
    return ChannelSearchPage(results=results, next_page_token=next_token, session=session)


def _search_continuation_page(
    session: ChannelSearchSession, continuation_token: str
) -> ChannelSearchPage:
    if not session.api_key or not session.context:
        return ChannelSearchPage(results=[], next_page_token=None, session=session)

    payload = {"context": session.context, "continuation": continuation_token}
    params = {"key": session.api_key}
    RATE_LIMITER.wait()
    response = SESSION.post(
        "https://www.youtube.com/youtubei/v1/search",
        params=params,
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = _collect_channel_results(data if isinstance(data, dict) else {})
    next_token = _extract_next_token(data) if isinstance(data, dict) else None
    return ChannelSearchPage(results=results, next_page_token=next_token, session=session)


def search_channels_page(
    keyword: str,
    *,
    session: Optional[ChannelSearchSession] = None,
    continuation_token: Optional[str] = None,
) -> ChannelSearchPage:
    if continuation_token and session:
        return _search_continuation_page(session, continuation_token)
    return _search_initial_page(keyword)


def search_channels(keyword: str, limit: int) -> List[ChannelSearchResult]:
    page = search_channels_page(keyword)
    if not page.results:
        return []
    if limit <= 0:
        return []
    return page.results[:limit]


def fetch_discovery_metadata(channel_id: str) -> DiscoveryMetadata:
    """Return lightweight metadata useful during discovery filtering."""

    try:
        _, feed_description, video = _fetch_rss(channel_id)
    except EnrichmentError:
        return DiscoveryMetadata()
    except requests.RequestException:
        return DiscoveryMetadata()

    video_id = video.get("video_id") if isinstance(video, dict) else None
    watch_data: Dict[str, Optional[str]] = {}
    if video_id:
        try:
            watch_data = _fetch_watch_details(video_id)
        except EnrichmentError:
            watch_data = {}
        except requests.RequestException:
            watch_data = {}

    combined_texts = [
        video.get("title", "") if isinstance(video, dict) else "",
        video.get("description", "") if isinstance(video, dict) else "",
        feed_description or "",
        watch_data.get("description", "") if isinstance(watch_data, dict) else "",
    ]
    lang_result = detect_language("\n".join(filter(None, combined_texts)))

    language = None
    confidence: Optional[float] = None
    if lang_result:
        language = lang_result.get("language")
        confidence = lang_result.get("confidence")
    elif isinstance(watch_data, dict):
        language = watch_data.get("language") or None

    last_upload = None
    if isinstance(watch_data, dict):
        last_upload = watch_data.get("upload_date") or video.get("timestamp")
    elif isinstance(video, dict):
        last_upload = video.get("timestamp")

    return DiscoveryMetadata(
        last_upload=last_upload,
        language=language,
        language_confidence=confidence,
    )


def detect_language(text: str) -> Optional[Dict[str, float]]:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        langs = detect_langs(cleaned)
    except LangDetectException:
        return None
    if not langs:
        return None
    best = langs[0]
    return {"language": best.lang, "confidence": float(best.prob)}


def extract_emails(texts: Iterable[str]) -> List[str]:
    pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    found: List[str] = []
    for text in texts:
        if not text:
            continue
        found.extend(pattern.findall(text))
    unique = []
    seen = set()
    for email in found:
        email_lower = email.lower()
        if email_lower not in seen:
            unique.append(email)
            seen.add(email_lower)
    return unique


def _clean_telegram_username(username: str) -> Optional[str]:
    candidate = username.strip().lstrip("@")
    if not candidate:
        return None
    if "." in candidate:
        return None
    if candidate.lower() in {"keyframes"}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]{3,}", candidate):
        return None
    return candidate


def _contact_weight(text: str, index: int) -> int:
    window = text[max(0, index - 20) : index + 20].lower()
    keywords = ["contact", "business", "dm", "message", "inquiry"]
    return 2 if any(keyword in window for keyword in keywords) else 1


def extract_telegram_account(texts: Iterable[str]) -> Optional[str]:
    link_pattern = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9_]{3,})", re.IGNORECASE)
    handle_pattern = re.compile(r"(?<![\w.])@([A-Za-z0-9_]{3,})")

    candidates: List[Tuple[int, int, str]] = []
    for text in texts:
        if not text:
            continue
        for match in link_pattern.finditer(text):
            username = _clean_telegram_username(match.group(1))
            if not username:
                continue
            weight = _contact_weight(text, match.start())
            candidates.append((weight, match.start(), username))
        for match in handle_pattern.finditer(text):
            username = _clean_telegram_username(match.group(1))
            if not username:
                continue
            tail = text[match.end() : match.end() + 2]
            if tail.startswith("."):
                continue
            weight = _contact_weight(text, match.start())
            candidates.append((weight, match.start(), username))

    if not candidates:
        return None

    candidates.sort(key=lambda entry: (-entry[0], entry[1]))
    return candidates[0][2]


def _fetch_rss(channel_id: str, timeout: int = 8) -> Tuple[str, Optional[str], Dict[str, Optional[str]]]:
    RATE_LIMITER.wait()
    response = SESSION.get(RSS_TEMPLATE.format(channel_id=channel_id), timeout=timeout)
    if response.status_code == 404:
        raise EnrichmentError("Channel feed not available")
    response.raise_for_status()

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:  # pragma: no cover - network artifact
        raise EnrichmentError(f"Malformed channel feed: {exc}")

    title = root.findtext("atom:title", default="", namespaces=ATOM_NS)
    description = root.findtext(f"atom:subtitle", default="", namespaces=ATOM_NS)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise EnrichmentError("No public videos found in feed")

    video_id = entry.findtext("yt:videoId", default="", namespaces=ATOM_NS)
    if not video_id:
        raise EnrichmentError("Unable to read latest video id")

    media_group = entry.find(f"{MEDIA_NS}group")
    video_title = media_group.findtext(f"{MEDIA_NS}title", default="") if media_group is not None else ""
    video_description = (
        media_group.findtext(f"{MEDIA_NS}description", default="") if media_group is not None else ""
    )
    updated = entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)

    return title or "", description or None, {
        "video_id": video_id,
        "title": video_title.strip(),
        "description": video_description.strip(),
        "timestamp": updated or published,
    }


def _extract_json_blob(html_text: str, marker: str) -> Optional[Dict]:
    pattern = re.compile(rf"{marker}\s*=\s*(\{{.*?\}});", re.DOTALL)
    match = pattern.search(html_text)
    if not match:
        return None
    json_text = match.group(1)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def _find_first(node: object, key: str) -> Optional[Dict]:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_first(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first(item, key)
            if found is not None:
                return found
    return None


def _fetch_watch_details(video_id: str, timeout: int = 10) -> Dict[str, Optional[str]]:
    RATE_LIMITER.wait()
    response = SESSION.get(
        "https://www.youtube.com/watch",
        params={"v": video_id},
        timeout=timeout,
    )
    if response.status_code == 429:
        raise EnrichmentError("Rate limited by YouTube")
    if response.status_code == 410:
        raise EnrichmentError("Video is no longer available")
    response.raise_for_status()

    html_text = response.text
    player = _extract_json_blob(html_text, "ytInitialPlayerResponse")
    data = _extract_json_blob(html_text, "ytInitialData")
    if not player:
        raise EnrichmentError("Unable to parse video metadata")

    video_details = player.get("videoDetails", {}) if isinstance(player, dict) else {}
    short_description = html.unescape(video_details.get("shortDescription", ""))

    microformat = player.get("microformat", {}) if isinstance(player, dict) else {}
    micro_renderer = microformat.get("playerMicroformatRenderer", {})
    language_hint = micro_renderer.get("language")
    upload_date = micro_renderer.get("uploadDate")

    owner_renderer = _find_first(data, "videoOwnerRenderer") if data else None
    subscriber_count = None
    if owner_renderer:
        sub_text = owner_renderer.get("subscriberCountText", {})
        if isinstance(sub_text, dict):
            if "simpleText" in sub_text:
                subscriber_count = _parse_subscriber_count(sub_text["simpleText"])
            else:
                runs = sub_text.get("runs", [])
                if runs:
                    subscriber_count = _parse_subscriber_count(runs[0].get("text", ""))

    return {
        "description": short_description,
        "language": language_hint,
        "upload_date": upload_date,
        "subscribers": subscriber_count,
    }


def enrich_channel(channel: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    channel_id = channel.get("channel_id")
    if not channel_id:
        raise EnrichmentError("Missing channel id")

    feed_title, feed_description, video = _fetch_rss(channel_id)
    watch = _fetch_watch_details(video["video_id"])

    combined_description = watch.get("description") or video.get("description") or ""
    combined_texts = [video.get("title", ""), combined_description, feed_description or ""]
    lang_result = detect_language("\n".join(filter(None, combined_texts)))

    emails = extract_emails([combined_description])
    if feed_description:
        emails.extend(extract_emails([feed_description]))
    # Deduplicate again after combining feed and watch descriptions.
    unique_emails: List[str] = []
    seen = set()
    for email in emails:
        lower = email.lower()
        if lower not in seen:
            unique_emails.append(email)
            seen.add(lower)
        if len(unique_emails) >= 5:
            break

    email_gate_present: Optional[bool] = False if unique_emails else None
    if not unique_emails:
        about_emails, gate_present = _fetch_about_emails(channel)
        if about_emails:
            unique_emails = about_emails
            email_gate_present = False
        else:
            email_gate_present = gate_present

    return {
        "name": feed_title or channel.get("name") or channel.get("title"),
        "subscribers": watch.get("subscribers"),
        "language": lang_result["language"] if lang_result else (watch.get("language") or None),
        "language_confidence": lang_result["confidence"] if lang_result else None,
        "emails": unique_emails,
        "last_updated": watch.get("upload_date") or video.get("timestamp"),
        "email_gate_present": email_gate_present,
    }


def _option_enabled(options: object, key: str, default: bool) -> bool:
    if hasattr(options, key):
        try:
            return bool(getattr(options, key))
        except Exception:
            return default
    if isinstance(options, dict):
        return bool(options.get(key, default))
    return default


def enrich_channel_with_options(
    channel: Dict[str, Optional[str]], options: object
) -> Dict[str, Optional[str]]:
    channel_id = channel.get("channel_id")
    if not channel_id:
        raise EnrichmentError("Missing channel id")

    emails_from_channel = _option_enabled(options, "emails_from_channel", True)
    emails_from_videos = _option_enabled(options, "emails_from_videos", True)
    language_basic = _option_enabled(options, "language_basic", True)
    language_precise = _option_enabled(options, "language_precise", True)
    update_metadata = _option_enabled(options, "update_metadata", True)
    update_activity = _option_enabled(options, "update_activity", True)
    telegram_enabled = _option_enabled(options, "telegram_enrichment", True)

    needs_feed = any(
        [emails_from_videos, language_basic, language_precise, update_metadata, update_activity]
    )

    feed_title: Optional[str] = None
    feed_description: Optional[str] = None
    video: Dict[str, Optional[str]] = {}
    watch: Dict[str, Optional[str]] = {}
    if needs_feed:
        feed_title, feed_description, video = _fetch_rss(channel_id)
        watch = _fetch_watch_details(video["video_id"])

    combined_description = ""
    combined_texts: List[str] = []
    if video:
        combined_description = watch.get("description") or video.get("description") or ""
        combined_texts = [video.get("title", ""), combined_description, feed_description or ""]

    emails: List[str] = []
    email_gate_present: Optional[bool] = None
    about_text: Optional[str] = None

    if emails_from_videos and combined_description:
        emails.extend(extract_emails([combined_description]))
        if feed_description:
            emails.extend(extract_emails([feed_description]))
    elif emails_from_channel and feed_description:
        emails.extend(extract_emails([feed_description]))

    if emails_from_channel or telegram_enabled:
        about_result = _fetch_about_emails(channel, return_text=telegram_enabled)
        if telegram_enabled:
            about_emails, gate_present, about_text = about_result  # type: ignore[misc]
        else:
            about_emails, gate_present = about_result  # type: ignore[misc]
        if gate_present and email_gate_present is None:
            email_gate_present = gate_present
        emails.extend(about_emails)

    unique_emails: List[str] = []
    seen: Set[str] = set()
    for email in emails:
        lower = email.lower()
        if lower in seen:
            continue
        unique_emails.append(email)
        seen.add(lower)
        if len(unique_emails) >= 5:
            break

    if unique_emails:
        email_gate_present = False

    telegram_account: Optional[str] = None

    language: Optional[str] = None
    language_confidence: Optional[float] = None
    if language_precise and combined_texts:
        lang_result = detect_language("\n".join(filter(None, combined_texts)))
        if lang_result:
            language = lang_result.get("language")
            language_confidence = lang_result.get("confidence")
    if language is None and language_basic and watch:
        language = watch.get("language") or None

    last_updated = None
    if update_activity:
        last_updated = watch.get("upload_date") if watch else None
        if not last_updated and video:
            last_updated = video.get("timestamp")

    if telegram_enabled:
        telegram_sources = list(combined_texts)
        if about_text:
            telegram_sources.append(about_text)
        telegram_account = extract_telegram_account(telegram_sources)

    return {
        "name": feed_title or channel.get("name") or channel.get("title"),
        "subscribers": watch.get("subscribers") if update_metadata and watch else None,
        "language": language,
        "language_confidence": language_confidence,
        "emails": unique_emails if (emails_from_channel or emails_from_videos) else [],
        "last_updated": last_updated,
        "email_gate_present": email_gate_present,
        "telegram_account": telegram_account,
    }


def _resolve_about_url(channel: Dict[str, Optional[str]]) -> str:
    url = channel.get("url") or ""
    if url:
        base = url.split("?")[0].rstrip("/")
        if base.endswith("/about"):
            return base
        return f"{base}/about"
    channel_id = channel.get("channel_id")
    if not channel_id:
        raise EnrichmentError("Missing channel id")
    return f"https://www.youtube.com/channel/{channel_id}/about"


def _fetch_about_emails(
    channel: Dict[str, Optional[str]], timeout: int = 5, *, return_text: bool = False
) -> Tuple[List[str], bool] | Tuple[List[str], bool, str]:
    about_url = _resolve_about_url(channel)
    RATE_LIMITER.wait()
    try:
        response = SESSION.get(about_url, timeout=timeout)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise EnrichmentError(f"Failed to load channel about page: {exc}")
    if response.status_code == 404:
        return [], False
    response.raise_for_status()
    page_text = html.unescape(response.text)
    cleaned_text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", " ", page_text, flags=re.DOTALL | re.IGNORECASE
    )
    found_emails = extract_emails([page_text])
    unique_emails: List[str] = []
    seen: Set[str] = set()
    for email in found_emails:
        lower = email.lower()
        if lower in seen:
            continue
        seen.add(lower)
        unique_emails.append(email)
        if len(unique_emails) >= 5:
            break
    gate_present = False
    if not unique_emails and "view email address" in page_text.lower():
        gate_present = True
    if return_text:
        return unique_emails, gate_present, cleaned_text
    return unique_emails, gate_present


def _fetch_latest_video_metadata(channel_id: str) -> Optional[Dict[str, Optional[str]]]:
    try:
        _, feed_description, video = _fetch_rss(channel_id, timeout=5)
    except EnrichmentError:
        return None
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise EnrichmentError(f"Failed to load channel feed: {exc}")

    try:
        watch = _fetch_watch_details(video["video_id"], timeout=6)
    except EnrichmentError:
        watch = {}
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise EnrichmentError(f"Failed to load latest video: {exc}")

    description = watch.get("description") or video.get("description") or ""
    return {
        "title": video.get("title", ""),
        "description": description,
        "feed_description": feed_description or "",
        "timestamp": watch.get("upload_date") or video.get("timestamp"),
    }


def enrich_channel_email_only(
    channel: Dict[str, Optional[str]], *, telegram_enabled: bool = True
) -> Dict[str, Optional[str]]:
    channel_id = channel.get("channel_id")
    if not channel_id:
        raise EnrichmentError("Missing channel id")

    emails: List[str] = []
    telegram_account: Optional[str] = None

    about_text: Optional[str] = None
    about_emails, about_gate, about_text = _fetch_about_emails(channel, return_text=True)
    email_gate_present: Optional[bool] = about_gate
    if about_emails:
        emails.extend(about_emails)
        email_gate_present = False

    video = _fetch_latest_video_metadata(channel_id)
    last_updated = None
    candidate_texts: List[str] = []
    if video:
        candidate_texts = [video.get("title", ""), video.get("description", ""), video.get("feed_description", "")]
        emails.extend(extract_emails(candidate_texts))
        last_updated = video.get("timestamp")

    unique_emails: List[str] = []
    seen = set()
    for email in emails:
        lower = email.lower()
        if lower in seen:
            continue
        unique_emails.append(email)
        seen.add(lower)
        if len(unique_emails) >= 5:
            break

    if not last_updated:
        last_updated = dt.datetime.utcnow().isoformat()

    if unique_emails:
        email_gate_present = False

    telegram_sources = list(candidate_texts)
    if about_text:
        telegram_sources.append(about_text)
    if telegram_enabled:
        telegram_account = extract_telegram_account(telegram_sources)
    else:
        telegram_account = None

    return {
        "emails": unique_emails,
        "last_updated": last_updated,
        "email_gate_present": email_gate_present,
        "telegram_account": telegram_account,
    }
