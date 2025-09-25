import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
import tldextract

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path("newsData")
ENV_FILE = BASE_DIR / "finnhubCred.env"
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AAPL", "GOOGL", "MSFT"]   # change or extend anytime
DAYS_BACK = 30
SLEEP_BETWEEN_CALLS = 0.25            # polite pacing (seconds)

FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"

# -----------------------------
# Helpers
# -----------------------------
def iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def fqdn_of(url: str) -> str | None:
    try:
        if not url:
            return None
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    except Exception:
        pass
    # fallback
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None

def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len([w for w in text.strip().split() if w])

def write_csv(path: Path, rows: list[dict]):
    fields = [
        "symbol","id","datetime_iso","headline","summary","source",
        "domain","url","image","category","related","word_count"
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # Excel-friendly UTF-8
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def fetch_company_news(api_key: str, symbol: str, start_date: str, end_date: str) -> list[dict]:
    params = {"symbol": symbol, "from": start_date, "to": end_date, "token": api_key}
    r = requests.get(FINNHUB_COMPANY_NEWS, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Finnhub error {r.status_code}: {r.text[:300]}")
    return r.json() or []

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv(ENV_FILE)
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing FINNHUB_API_KEY in {ENV_FILE}")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    start_str, end_str = to_date_str(start_dt), to_date_str(end_dt)
    stamp = end_dt.strftime("%Y%m%d")
    print(f"Collecting company news for {SYMBOLS} from {start_str} to {end_str}...")

    all_rows = []
    seen = set()  # dedupe by (symbol,id)

    for sym in SYMBOLS:
        try:
            items = fetch_company_news(api_key, sym, start_str, end_str)
            print(f"{sym}: {len(items)} articles")
            for it in items:
                # Finnhub fields: category, datetime (unix), headline, id, image, related, source, summary, url
                dt_iso = datetime.fromtimestamp(it.get("datetime", 0), tz=timezone.utc).isoformat().replace("+00:00","Z")
                url = it.get("url") or ""
                key = (sym, it.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "symbol": sym,
                    "id": it.get("id"),
                    "datetime_iso": dt_iso,
                    "headline": it.get("headline", ""),
                    "summary": it.get("summary", ""),
                    "source": it.get("source", ""),
                    "domain": fqdn_of(url),
                    "url": url,
                    "image": it.get("image") or "",
                    "category": it.get("category") or "",
                    "related": it.get("related") or "",
                    "word_count": word_count(it.get("summary", "")),
                }
                all_rows.append(row)
            time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

    # Sort by time desc
    all_rows.sort(key=lambda r: r["datetime_iso"], reverse=True)

    base = OUT_DIR / f"finnhub_news_bigtech_{stamp}_last{DAYS_BACK}d"
    csv_path = base.with_suffix(".csv")
    jsonl_path = base.with_suffix(".jsonl")

    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)

    print(f"\n✅ Saved {len(all_rows)} articles")
    print(f"CSV  → {csv_path}")
    print(f"JSONL→ {jsonl_path}")
    print("\nTip: Finnhub also has /news-sentiment and real-time websockets if you expand later.")

if __name__ == "__main__":
    main()
