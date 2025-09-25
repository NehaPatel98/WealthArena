import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
import praw
from dotenv import load_dotenv
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path("redditData")
ENV_FILE = BASE_DIR / "redditCred.env"
ISSUERS_CSV = BASE_DIR / "issuers.csv"   # now inside redditData/
OUT_DIR = BASE_DIR                       # save outputs here too

DAYS_BACK = 30
SLEEP_EVERY = 400
SLEEP_SECS = 1.0

# US-focused subs (+ company subs)
SUBREDDITS = [
    # broad US markets
    "stocks", "investing", "StockMarket", "options", "thetagang",
    "Daytrading", "algotrading", "quant", "quantfinance",
    "financialindependence", "DividendInvesting", "dividends",
    "SecurityAnalysis", "valueinvesting", "wallstreetbets",
    # tech / news
    "technology", "technews",
    # company subs
    "apple", "Google", "Microsoft",
]

# Regex for tokens like $AAPL, AAPL
TICKER_TOKEN = re.compile(r"\$?[A-Z]{1,6}")

# -----------------------------
# Helpers
# -----------------------------
def load_issuers(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing issuer file: {csv_path}\n"
            "Create CSV with columns: symbol,name"
        )
    df = pd.read_csv(csv_path)
    if not set(df.columns).issuperset({"symbol", "name"}):
        raise ValueError("Issuer CSV must contain 'symbol' and 'name' columns")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    symbols = sorted(set(df["symbol"].tolist()))
    names = set()
    for _, row in df.iterrows():
        nm = row["name"]
        names.add(nm.upper())
        if "APPLE" in nm.upper(): names.add("APPLE")
        if "ALPHABET" in nm.upper(): names.add("ALPHABET"); names.add("GOOGLE")
        if "MICROSOFT" in nm.upper(): names.add("MICROSOFT")
    return symbols, names

def build_aliases(symbols):
    aliases = set()
    for s in symbols:
        sU = s.upper()
        aliases.add(sU)
        aliases.add(f"${sU}")
    return aliases

def detect_tickers(blob_upper: str, aliases: set):
    hits = set()
    for token in TICKER_TOKEN.findall(blob_upper):
        if token in aliases or token.strip("$") in aliases:
            hits.add(token.strip("$"))
    return sorted(hits)

def is_target_post(title: str, selftext: str, aliases: set, names_set: set):
    blob_upper = f"{title}\n{selftext or ''}".upper()
    tickers = detect_tickers(blob_upper, aliases)
    if tickers:
        return True, tickers
    for nm in names_set:
        if len(nm) > 2 and nm in blob_upper:
            return True, []
    return False, []

def mk_row(post, sub_name: str, tickers_list):
    created = datetime.fromtimestamp(getattr(post, "created_utc", 0), tz=timezone.utc)
    is_self = bool(getattr(post, "is_self", False))
    body_text = (getattr(post, "selftext", "") or "").replace("\r", " ").strip()
    if len(body_text) > 1200:
        body_text = body_text[:1200] + " ..."
    flair = getattr(post, "link_flair_text", None)
    return {
        "id": getattr(post, "id", ""),
        "subreddit": sub_name,
        "title": (getattr(post, "title", "") or "").strip(),
        "author": str(getattr(post, "author", "")),
        "content_type": "text" if is_self else "link",
        "score_likes": int(getattr(post, "score", 0)),
        "num_comments": int(getattr(post, "num_comments", 0)),
        "created_utc": created.isoformat(),
        "permalink": f"https://reddit.com{getattr(post, 'permalink', '')}",
        "external_url": None if is_self else getattr(post, "url", None),
        "over_18": bool(getattr(post, "over_18", False)),
        "flair": None if not flair or flair == "None" else flair,
        "tickers": ",".join(sorted(set(tickers_list))) if tickers_list else "",
        "text_preview": body_text,
    }

def write_csv(path: Path, rows: dict):
    fields = [
        "id","subreddit","title","author","content_type","score_likes","num_comments",
        "created_utc","permalink","external_url","over_18","flair","tickers","text_preview"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)

def write_jsonl(path: Path, rows: dict):
    with path.open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv(ENV_FILE)
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "wealtharena/0.1 by u/wealthArena"),
    )

    symbols, names_set = load_issuers(ISSUERS_CSV)
    aliases = build_aliases(symbols)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    start_epoch = int(start_dt.timestamp())
    stamp = end_dt.strftime("%Y%m%d")

    rows_all_by_id = {}
    rows_filtered_by_id = {}
    metrics = defaultdict(lambda: {"total": 0, "filtered_hits": 0})

    print(f"Scanning {len(SUBREDDITS)} subreddits for last {DAYS_BACK} days...")
    count = 0
    for sub_name in sorted(set(SUBREDDITS)):
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.new(limit=None):
                created_utc = getattr(post, "created_utc", 0)
                if created_utc < start_epoch:
                    break
                title = getattr(post, "title", "") or ""
                selftext = getattr(post, "selftext", "") or ""
                hit, tickers = is_target_post(title, selftext, aliases, names_set)
                row = mk_row(post, sub_name, tickers)
                pid = row["id"]
                rows_all_by_id[pid] = row
                metrics[sub_name]["total"] += 1
                if hit:
                    rows_filtered_by_id[pid] = row
                    metrics[sub_name]["filtered_hits"] += 1
                count += 1
                if count % SLEEP_EVERY == 0:
                    time.sleep(SLEEP_SECS)
        except Exception as e:
            print(f"[WARN] Failed scanning r/{sub_name}: {e}")

    print(f"Collected total posts: {len(rows_all_by_id)}; Filtered (AAPL/GOOGL/MSFT): {len(rows_filtered_by_id)}")

    base = f"reddit_us_bigtech_{stamp}_last{DAYS_BACK}d"
    write_csv(OUT_DIR / f"{base}_all.csv", rows_all_by_id)
    write_jsonl(OUT_DIR / f"{base}_all.jsonl", rows_all_by_id)
    write_csv(OUT_DIR / f"{base}_filtered.csv", rows_filtered_by_id)
    write_jsonl(OUT_DIR / f"{base}_filtered.jsonl", rows_filtered_by_id)

    with (OUT_DIR / f"{base}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subreddit", "total_posts", "filtered_hits"])
        for s in sorted(metrics.keys()):
            w.writerow([s, metrics[s]["total"], metrics[s]["filtered_hits"]])

    print("✅ Saved outputs into:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
