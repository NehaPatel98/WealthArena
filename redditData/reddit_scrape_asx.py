# redditData/reddit_scrape_asx_raw.py
import os, csv, json, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import praw
from dotenv import load_dotenv

BASE_DIR = Path("redditData")
ENV_FILE = BASE_DIR / "redditCred.env"
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS_BACK = 30
SLEEP_EVERY = 400
SLEEP_SECS = 1.0

# AU-only subs (trim/extend as you like)
SUBREDDITS = [
    "ASX", "AusFinance", "ASX_Bets", "AusStocks"
]

def mk_row(p, sub_name: str):
    created = datetime.fromtimestamp(getattr(p, "created_utc", 0), tz=timezone.utc)
    is_self = bool(getattr(p, "is_self", False))
    body = (getattr(p, "selftext", "") or "").replace("\r", " ").strip()
    if len(body) > 1200:
        body = body[:1200] + " ..."
    flair = getattr(p, "link_flair_text", None)
    return {
        "id": getattr(p, "id", ""),
        "subreddit": sub_name,
        "title": (getattr(p, "title", "") or "").strip(),
        "author": str(getattr(p, "author", "")),
        "content_type": "text" if is_self else "link",
        "score_likes": int(getattr(p, "score", 0)),
        "num_comments": int(getattr(p, "num_comments", 0)),
        "created_utc": created.isoformat(),
        "permalink": f"https://reddit.com{getattr(p, 'permalink', '')}",
        "external_url": None if is_self else getattr(p, "url", None),
        "over_18": bool(getattr(p, "over_18", False)),
        "flair": None if not flair or flair == "None" else flair,
        "text_preview": body,
    }

def write_csv(path: Path, rows: list[dict]):
    fields = ["id","subreddit","title","author","content_type","score_likes","num_comments",
              "created_utc","permalink","external_url","over_18","flair","text_preview"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    load_dotenv(ENV_FILE)
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "wealtharena/0.1 by u/wealthArena"),
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    start_epoch = int(start_dt.timestamp())
    stamp = end_dt.strftime("%Y%m%d")

    rows = []
    count = 0
    print(f"Scraping last {DAYS_BACK}d from {len(SUBREDDITS)} AU subreddits...")
    for sub_name in sorted(set(SUBREDDITS)):
        try:
            for p in reddit.subreddit(sub_name).new(limit=None):
                if getattr(p, "created_utc", 0) < start_epoch:
                    break
                rows.append(mk_row(p, sub_name))
                count += 1
                if count % SLEEP_EVERY == 0:
                    time.sleep(SLEEP_SECS)
        except Exception as e:
            print(f"[WARN] r/{sub_name}: {e}")

    base = OUT_DIR / f"reddit_asx_raw_{stamp}_last{DAYS_BACK}d"
    write_csv(base.with_suffix(".csv"), rows)
    write_jsonl(base.with_suffix(".jsonl"), rows)
    print(f"✅ Saved raw posts: {len(rows)} → {base.name}.csv/.jsonl")

if __name__ == "__main__":
    main()
