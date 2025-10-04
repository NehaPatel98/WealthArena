import os, csv, json, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import praw
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient, ContentSettings


# ============ CONFIG ============
BASE_DIR = Path("redditData")
ENV_FILE_REDDIT = BASE_DIR / "redditCred.env"
ENV_FILE_AZURE  = BASE_DIR / "azureCred.env"
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS_BACK = 30
SLEEP_EVERY = 400
SLEEP_SECS = 1.0

SUBREDDITS = ["ASX", "AusFinance", "ASX_Bets", "AusStocks"]
# =================================


# ---------- Azure helpers (ENV ONLY) ----------
def get_blob_service_client_from_env() -> BlobServiceClient:
    """
    Build BlobServiceClient using only env-provided secrets.
    Supported:
      (A) AZURE_STORAGE_CONNECTION_STRING
      (B) AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_SAS   (SAS without leading '?')
      (C) AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY
    """
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        print("🔑 Using connection string for Azure Storage auth")
        return BlobServiceClient.from_connection_string(conn)

    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    if not account:
        raise RuntimeError(
            "No Azure creds found. Provide AZURE_STORAGE_CONNECTION_STRING, or "
            "AZURE_STORAGE_ACCOUNT + (AZURE_STORAGE_SAS | AZURE_STORAGE_KEY)."
        )

    base_url = f"https://{account}.blob.core.windows.net"
    sas = os.getenv("AZURE_STORAGE_SAS")
    key = os.getenv("AZURE_STORAGE_KEY")

    if sas:
        print("🧩 Using Account + SAS for Azure Storage auth")
        url = f"{base_url}?{sas.lstrip('?')}"
        return BlobServiceClient(account_url=url)

    if key:
        print("🧰 Using Account + Key for Azure Storage auth")
        return BlobServiceClient(account_url=base_url, credential=key)

    raise RuntimeError("AZURE_STORAGE_ACCOUNT provided but neither AZURE_STORAGE_SAS nor AZURE_STORAGE_KEY set.")


def ensure_container(svc: BlobServiceClient, name: str):
    try:
        svc.create_container(name)
        print(f"📦 Created container: {name}")
    except Exception:
        pass  # likely already exists


def upload_file(svc, container: str, blob_path: str, local_path: Path, content_type: str):
    with local_path.open("rb") as f:
        svc.get_container_client(container).upload_blob(
            name=blob_path,
            data=f,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    print(f"☁️  Uploaded to: {container}/{blob_path}")


# ---------- Reddit logic ----------
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
        "_ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_source": "reddit",
    }


def write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0].keys()) if rows else [
        "id","subreddit","title","author","content_type","score_likes","num_comments",
        "created_utc","permalink","external_url","over_18","flair","text_preview",
        "_ingested_at_utc","_source"
    ]
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
    load_dotenv(ENV_FILE_REDDIT)
    load_dotenv(ENV_FILE_AZURE)

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "wealtharena/0.1 by u/wealthArena"),
    )

    end_dt = datetime.now(timezone.utc)
    start_epoch = int((end_dt - timedelta(days=DAYS_BACK)).timestamp())
    stamp = end_dt.strftime("%Y%m%d")

    rows = []
    count = 0
    print(f"Scraping last {DAYS_BACK}d from {len(SUBREDDITS)} subreddits...")
    for sub_name in SUBREDDITS:
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
    jsonl_path = base.with_suffix(".jsonl")
    csv_path   = base.with_suffix(".csv")

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    print(f"✅ Saved locally: {len(rows)} posts")

    # ---------- Optional Azure upload (env-only auth) ----------
    if (os.getenv("AZURE_UPLOAD") or "").lower() == "true":
        container = os.getenv("AZURE_STORAGE_CONTAINER", "raw")
        svc = get_blob_service_client_from_env()
        ensure_container(svc, container)

        prefix = f"reddit/year={end_dt.year}/month={end_dt.month:02d}/day={end_dt.day:02d}/"
        upload_file(svc, container, prefix + jsonl_path.name, jsonl_path, "application/json")
        upload_file(svc, container, prefix + csv_path.name,   csv_path,   "text/csv; charset=utf-8")
    else:
        print("🛑 AZURE_UPLOAD not enabled; skipping upload.")


if __name__ == "__main__":
    main()
