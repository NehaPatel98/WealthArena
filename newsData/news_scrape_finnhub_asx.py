import os, json, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv

BASE_DIR = Path("newsData")
ENV_FILE = BASE_DIR / "finnhubCred.env"
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS_BACK = 30
SLEEP_BETWEEN_CALLS = 0.25
FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"
SYMBOLS_FALLBACK = ["AAPL", "GOOGL", "MSFT"]

def load_symbols():
    sym_path = BASE_DIR / "symbols.txt"
    if not sym_path.exists():
        return SYMBOLS_FALLBACK
    text = sym_path.read_text(encoding="utf-8")
    parts = [p.strip().upper() for p in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def fetch_company_news(api_key: str, symbol: str, start_date: str, end_date: str) -> list[dict]:
    r = requests.get(FINNHUB_COMPANY_NEWS, params={
        "symbol": symbol, "from": start_date, "to": end_date, "token": api_key
    }, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Finnhub error {r.status_code}: {r.text[:300]}")
    return r.json() or []

def main():
    load_dotenv(ENV_FILE)
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing FINNHUB_API_KEY in {ENV_FILE}")

    symbols = load_symbols()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    start_str, end_str = to_date_str(start_dt), to_date_str(end_dt)
    stamp = end_dt.strftime("%Y%m%d")

    raw_path = OUT_DIR / f"finnhub_news_raw_{stamp}_last{DAYS_BACK}d.jsonl"
    print(f"Scraping Finnhub company news for {len(symbols)} symbols from {start_str} to {end_str}…")
    written = 0
    with raw_path.open("w", encoding="utf-8") as out:
        for sym in symbols:
            try:
                items = fetch_company_news(api_key, sym, start_str, end_str)
                print(f"{sym}: {len(items)} articles")
                for it in items:
                    # keep Finnhub payload as-is + add symbol for joining
                    rec = {"symbol": sym, **it}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"[WARN] {sym}: {e}")

    print(f"✅ Saved RAW JSONL: {raw_path} ({written} rows)")
    print("Tip: Run news_enrich_finnhub.py next.")

if __name__ == "__main__":
    main()
