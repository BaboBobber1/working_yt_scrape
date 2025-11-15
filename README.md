# Crypto YouTube Harvester

Crypto YouTube Harvester is a fully local web application that discovers crypto-focused YouTube channels, enriches them with metadata and exports the collected dataset as CSV.

## Features

- 🔍 **Discover**: search YouTube for real channel pages by keyword with de-duplication.
- 🧠 **Enrich**: batch enrichment to fetch subscribers, detect language via video titles/descriptions and extract visible e-mails.
- 📄 **List & Export**: searchable, sortable table with pagination and a UTF-8 CSV export that opens cleanly in Google Sheets.
- 🖥️ **Single-page UI**: dark mode dashboard with live stats polling and responsive controls.
- 🛠️ **Local-first**: runs without API keys or ffmpeg. Optional dependencies (like ffmpeg) are not required.

## Quick start (≈ 60 seconds)

### macOS / Linux

```bash
./start.sh
```

### Windows

```bat
start.bat
```

Both scripts create a virtual environment (if missing), install dependencies and launch the backend + frontend server at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Usage

1. Open the app in your browser.
2. Adjust the keyword list and per-keyword limit if needed.
3. Click **Discover** to load new channels. Duplicates are ignored automatically.
4. Click **Enrich** to process pending channels in batches (limit of 25 per click by default).
5. Use the search box and sorting options to browse the dataset.
6. Click **Export CSV** to download a Google Sheets-ready CSV snapshot.

The status line shows the latest action, while the progress line refreshes every five seconds with total channels and pending enrichment counts.

## API reference

The backend exposes the following REST endpoints:

- `POST /api/discover` – body: `{ "keywords": string[], "perKeyword": number }` → `{ "found": number, "uniqueTotal": number }`
- `POST /api/enrich` – body: `{ "limit": number | null }` → `{ "processed": number }`
- `GET /api/channels?search=&sort=&order=&limit=&offset=` → `{ "items": Channel[], "total": number }`
- `GET /api/export/csv` → CSV download
- `GET /api/stats` → `{ "total": number, "pending_enrichment": number }`

## Development notes

- Data is stored in `data/channels.db` (SQLite). Remove the file to reset the database.
- Discovery relies on public YouTube search pages and works without API keys. Network failures are handled gracefully and simply skip failed keywords.
- Enrichment uses [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for metadata retrieval. If enrichment for a specific channel fails, the error is recorded and the rest of the batch continues.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `yt_dlp` fails due to outdated certificates | Update `certifi` inside the virtual environment: `python -m pip install --upgrade certifi` |
| No ffmpeg installed | All core features work without ffmpeg. Video/audio sampling is skipped automatically. |
| No channels discovered | Ensure the machine has network access to youtube.com and try different keywords. |

## License

This project is provided as-is for demonstration purposes.
