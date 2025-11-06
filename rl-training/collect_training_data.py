#!/usr/bin/env python3
"""
WealthArena Multi-Asset Data Collection System

Comprehensive data collection for all financial instruments:
- Stocks (ASX, US, International)
- ETFs (US, International)
- Cryptocurrencies (Major coins)
- Currency Pairs (Major, Minor, Exotic)
- Commodities (Metals, Energy, Agriculture)
- Options (Basic chains)
- Economic Indicators (FRED data)

Features:
- Multi-source data collection (Yahoo Finance, Alpha Vantage, IEX, FRED, Crypto APIs)
- Real-time and historical data support
- Technical indicators and fundamental data
- Data quality monitoring and validation
- Comprehensive error handling and retry logic
- Batch processing with rate limiting
- Data normalization and standardization

Run examples:
  # Download all assets from all exchanges
  python data_collection_for_training.py --all-assets --all-exchanges
  
  # Download ASX stocks and ETFs (default)
  python data_collection_for_training.py --stocks --etfs
  
  # Download from multiple exchanges
  python data_collection_for_training.py --stocks --exchanges ASX NYSE NASDAQ
  
  # Download crypto and forex
  python data_collection_for_training.py --crypto --forex
  
  # Custom date range and limits
  python data_collection_for_training.py --stocks --start-date 2020-01-01 --end-date 2024-12-31 --max-stocks-symbols 100
"""

import io
import re
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import the shared download helper
from src.data.market_data import download_with_retry_and_validation

# Helper function to safely print Unicode strings on Windows
def safe_print(text: str) -> None:
    """Print text safely, handling encoding errors on Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace Unicode emojis with ASCII alternatives if encoding fails
        text = text.replace('🚀', '[STARTING]')
        text = text.replace('✅', '[OK]')
        text = text.replace('❌', '[FAILED]')
        text = text.replace('📊', '[SUMMARY]')
        text = text.replace('📅', '[DATE]')
        text = text.replace('📁', '[PATH]')
        text = text.replace('🧾', '[FILE]')
        text = text.replace('📈', '[BREAKDOWN]')
        text = text.replace('🎉', '[SUCCESS]')
        try:
            print(text)
        except UnicodeEncodeError:
            # If still failing, encode with errors='replace'
            encoding = sys.stdout.encoding or 'utf-8'
            print(text.encode(encoding, errors='replace').decode(encoding, errors='replace'))
from enum import Enum
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ccxt
from fredapi import Fred
import alpha_vantage

warnings.filterwarnings("ignore")

# ----------------------------
# TA-Lib availability (once)
# ----------------------------
try:
    import talib as _talib  # type: ignore
except Exception:
    _talib = None

# ----------------------------
# Asset Types and Data Sources
# ----------------------------
class AssetType(Enum):
    STOCKS = "stocks"
    ETFS = "etfs"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    OPTIONS = "options"
    ECONOMIC = "economic"

class DataSource(Enum):
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX = "iex"
    FRED = "fred"
    CCXT = "ccxt"
    COINGECKO = "coingecko"

@dataclass
class AssetConfig:
    """Configuration for different asset types"""
    symbols: List[str]
    data_source: DataSource
    yahoo_suffix: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

# ----------------------------
# Exchange and Asset Symbol Definitions
# ----------------------------
EXCHANGE_CONFIGS = {
    "ASX": {
        "name": "Australian Securities Exchange",
        "url": "https://www.asx.com.au/asx/research/ASXListedCompanies.csv",
        "yahoo_suffix": ".AX",
        "asset_types": [AssetType.STOCKS, AssetType.ETFS]
    },
    "NYSE": {
        "name": "New York Stock Exchange", 
        "url": "https://www.nyse.com/api/nyse-listed",
        "yahoo_suffix": "",
        "asset_types": [AssetType.STOCKS, AssetType.ETFS]
    },
    "NASDAQ": {
        "name": "NASDAQ",
        "url": "https://api.nasdaq.com/api/screener/stocks",
        "yahoo_suffix": "",
        "asset_types": [AssetType.STOCKS, AssetType.ETFS]
    }
}

# Dynamic data source configurations for all asset types
DYNAMIC_DATA_SOURCES = {
    AssetType.CRYPTO: {
        "name": "Cryptocurrency Markets",
        "sources": [
            {
                "name": "CoinGecko",
                "url": "https://api.coingecko.com/api/v3/coins/markets",
                "params": {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250},
                "symbol_field": "symbol",
                "yahoo_suffix": "-USD"
            },
            {
                "name": "CoinMarketCap",
                "url": "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                "params": {"limit": 200, "convert": "USD"},
                "symbol_field": "symbol",
                "yahoo_suffix": "-USD"
            }
        ]
    },
    AssetType.FOREX: {
        "name": "Foreign Exchange Markets",
        "sources": [
            {
                "name": "Yahoo Finance Forex",
                "url": "https://query1.finance.yahoo.com/v1/finance/screener",
                "params": {"formatted": "true", "lang": "en-US", "region": "US", "scrIds": "forex_major"},
                "symbol_field": "symbol",
                "yahoo_suffix": "=X"
            }
        ]
    },
    AssetType.COMMODITIES: {
        "name": "Commodity Markets",
        "sources": [
            {
                "name": "Yahoo Finance Commodities",
                "url": "https://query1.finance.yahoo.com/v1/finance/screener",
                "params": {"formatted": "true", "lang": "en-US", "region": "US", "scrIds": "commodities"},
                "symbol_field": "symbol",
                "yahoo_suffix": "=F"
            }
        ]
    },
    AssetType.ECONOMIC: {
        "name": "Economic Indicators",
        "sources": [
            {
                "name": "FRED API",
                "url": "https://api.stlouisfed.org/fred/series/search",
                "params": {"search_text": "economic", "limit": 100},
                "symbol_field": "id",
                "yahoo_suffix": ""
            }
        ]
    }
}

# Fallback static symbols for when dynamic fetching fails
FALLBACK_SYMBOLS = {
    AssetType.CRYPTO: {
        "major_coins": [
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOT-USD",
            "DOGE-USD", "AVAX-USD", "SHIB-USD", "MATIC-USD", "LTC-USD", "UNI-USD", "LINK-USD",
            "ATOM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "TRX-USD", "XLM-USD",
            "ETC-USD", "HBAR-USD", "NEAR-USD", "FTM-USD", "SAND-USD", "MANA-USD", "AXS-USD"
        ]
    },
    AssetType.FOREX: {
        "major_pairs": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
            "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X", "GBPJPY=X",
            "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X", "AUDJPY=X", "AUDCHF=X", "AUDCAD=X",
            "AUDNZD=X", "CADJPY=X", "CADCHF=X", "NZDJPY=X", "NZDCHF=X", "NZDCAD=X", "CHFJPY=X"
        ]
    },
    AssetType.COMMODITIES: {
        "metals": [
            "GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "ALI=F", "ZN=F", "NI=F", "SN=F", "PB=F",
            "GLD", "SLV", "PPLT", "PALL", "COPX", "JJC", "JJN", "JJT", "JJU", "LD"
        ],
        "energy": [
            "CL=F", "NG=F", "RB=F", "HO=F", "BZ=F", "QS=F", "USO", "UNG", "UCO", "SCO",
            "BNO", "KOLD", "BOIL", "HEAT", "GUSH", "DRIP", "ERX", "ERY", "TAN", "ICLN"
        ],
        "agriculture": [
            "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F", "SB=F", "JO=F", "LBS=F", "OZ=F",
            "CORN", "SOYB", "WEAT", "CANE", "NIB", "COW", "DBA", "DJP", "JJA", "JJG"
        ]
    },
    AssetType.ECONOMIC: {
        "fred_indicators": [
            "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10", "DGS2", "DGS3MO", "DGS5",
            "DGS30", "T10Y2Y", "T10Y3M", "T10YIE", "VIXCLS", "DEXUSEU", "DEXJPUS", "DEXUSUK",
            "DEXCAUS", "DEXUSAL", "DEXUSNZ", "DEXCHUS", "DEXSZUS", "DEXNOUS", "DEXSDUS",
            "DEXDNUS", "DEXUSUK", "DEXUSAL", "DEXUSNZ", "DEXCHUS", "DEXSZUS", "DEXNOUS"
        ]
    }
}

# ----------------------------
# Paths & Logging
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
for p in (LOG_DIR, RAW_DIR, PROCESSED_DIR, REFERENCE_DIR):
    p.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "data_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()]
)
logger = logging.getLogger("asx_10yr")

# ----------------------------
# ASX CSV Download (online)
# ----------------------------
PRIMARY_CSV = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
UA_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/csv,application/octet-stream,application/json,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.asx.com.au/",
}

# Known rename(s)
RENAMES = {"WPL": "WDS"}  # Woodside Petroleum -> Woodside Energy

# Constants
ASX_CODE_COLUMN_NAMES = ["asx code", "code", "ticker", "symbol"]

# ----------------------------
# HTTP helpers & CSV parsing
# ----------------------------
def http_get(url: str, timeout=30, retries=3, backoff=1.4) -> bytes:
    last = None
    s = requests.Session()
    s.headers.update(UA_HEADERS)
    for i in range(1, retries + 1):
        try:
            r = s.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            logger.warning(f"GET failed ({url}) attempt {i}/{retries}: {e}")
            if i < retries:
                time.sleep(backoff ** i)
    raise RuntimeError(f"GET failed for {url}: {last}")

def _find_header_index(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines[:100]):
        L = line.strip().lower()
        if "company name" in L and "asx code" in L:
            return i
    for i, line in enumerate(lines[:100]):
        L = line.strip().lower()
        if "asx code" in L and "gics" in L and "industry" in L:
            return i
    return -1

def _parse_asx_csv_bytes(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="replace")
    idx = _find_header_index(text)
    if idx == -1:
        raise RuntimeError("Couldn't locate CSV header in ASX file.")
    trimmed = "\n".join(text.splitlines()[idx:])
    df = pd.read_csv(io.StringIO(trimmed))
    if df.shape[1] < 2:
        raise RuntimeError("CSV parsed with too few columns.")
    return df

def _canonical_company_rows(df: pd.DataFrame) -> pd.DataFrame:
    # pick the code column (fallback to first)
    code_col = None
    for c in df.columns:
        if str(c).strip().lower() in ASX_CODE_COLUMN_NAMES:
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]

    df = df.copy()
    df["__code__"] = (
        df[code_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # drop header-like/blank values
    bad_values = {"", "ASX CODE", "CODE", "SYMBOL", "TICKER"}
    df = df[~df["__code__"].isin(bad_values)]

    # keep only canonical company codes (matches official "companies" CSV)
    df = df[df["__code__"].str.fullmatch(r"[A-Z0-9]{1,6}")]

    # de-dup just in case
    df = df.drop_duplicates(subset=["__code__"]).reset_index(drop=True)
    return df

def _classify_from_name(name: Optional[str]) -> str:
    if not name:
        return "Equity"
    n = " " + str(name).upper() + " "
    if any(k in n for k in [" ETF", "ETF ", " ETF)", "(ETF", "ETFS", "VANGUARD", "ISHARES",
                             "BETASHARES", "VANECK", "SPDR", "GLOBAL X"]):
        return "ETF"
    if "TRUST" in n:
        return "Trust"
    if "FUND" in n or "MANAGED FUND" in n:
        return "Fund"
    return "Equity"

def download_exchange_companies_csv(exchange: str, save_path: Path) -> pd.DataFrame:
    """Download company list from any supported exchange"""
    if exchange not in EXCHANGE_CONFIGS:
        raise ValueError(f"Unsupported exchange: {exchange}")
    
    config = EXCHANGE_CONFIGS[exchange]
    logger.info(f"Fetching {config['name']} company list...")
    
    if exchange == "ASX":
        return download_asx_companies_csv(save_path)
    elif exchange == "NYSE":
        return download_nyse_companies(save_path)
    elif exchange == "NASDAQ":
        return download_nasdaq_companies(save_path)
    else:
        raise ValueError(f"Exchange {exchange} not implemented yet")

def download_asx_companies_csv(save_path: Path) -> pd.DataFrame:
    """Download ASX companies using the original method"""
    logger.info("Fetching official ASX CSV…")
    content = http_get(PRIMARY_CSV, timeout=30, retries=3)
    logger.info(f"ASX CSV bytes: {len(content)}")
    df_raw = _parse_asx_csv_bytes(content)
    logger.info(f"Parsed ASX frame: {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")

    # Canonical company rows only
    df_codes = _canonical_company_rows(df_raw)

    # Map optional columns by fuzzy names
    def pick(cols, *names):
        names = [n.lower() for n in names]
        for c in cols:
            if str(c).strip().lower() in names:
                return c
        return None

    cols = list(df_raw.columns)
    name_col = pick(cols, "company name", "company", "name")
    gics_col = None
    for c in cols:
        cl = str(c).strip().lower()
        if "gics" in cl and "industry" in cl:
            gics_col = c; break
    list_col = pick(cols, "listing date", "listing", "list date")
    mcap_col = pick(cols, "market cap", "market capitalisation", "market capitalization")

    out = pd.DataFrame({"asx_code": df_codes["__code__"]})
    if name_col is not None:
        out["company_name"] = (
            df_raw.loc[df_codes.index, name_col].astype(str).str.strip().values
        )
    if gics_col is not None:
        out["gics_industry_group"] = (
            df_raw.loc[df_codes.index, gics_col].astype(str).str.strip().values
        )
    if list_col is not None:
        out["listing_date"] = df_raw.loc[df_codes.index, list_col].values
    if mcap_col is not None:
        out["market_cap"] = df_raw.loc[df_codes.index, mcap_col].values

    # Normalize to Yahoo ticker + known renames
    base = out["asx_code"].map(lambda x: RENAMES.get(x, x))
    out["ticker_yf"] = base + ".AX"

    # Quick type guess from name
    out["security_type"] = out.get("company_name", pd.Series([""]*len(out))).map(_classify_from_name)
    out["is_etf"] = out["security_type"].eq("ETF")

    # Save the reference CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved ASX reference CSV -> {save_path} ({len(out)} rows)")
    return out

def download_nyse_companies(save_path: Path) -> pd.DataFrame:
    """Download NYSE companies (placeholder - would need actual NYSE API)"""
    logger.warning("NYSE download not implemented yet - using static list")
    # For now, return a basic structure - would need actual NYSE API implementation
    nyse_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "UNH", "JNJ",
        "V", "PG", "JPM", "XOM", "HD", "CVX", "MA", "PFE", "ABBV", "BAC", "KO", "AVGO",
        "PEP", "TMO", "COST", "WMT", "MRK", "ABT", "ACN", "DHR", "VZ", "ADBE", "NFLX"
    ]
    
    out = pd.DataFrame({
        "nyse_code": nyse_stocks,
        "ticker_yf": nyse_stocks,
        "company_name": [f"Company {code}" for code in nyse_stocks],
        "security_type": ["Equity"] * len(nyse_stocks),
        "is_etf": [False] * len(nyse_stocks)
    })
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved NYSE reference CSV -> {save_path} ({len(out)} rows)")
    return out

def download_nasdaq_companies(save_path: Path) -> pd.DataFrame:
    """Download NASDAQ companies (placeholder - would need actual NASDAQ API)"""
    logger.warning("NASDAQ download not implemented yet - using static list")
    # For now, return a basic structure - would need actual NASDAQ API implementation
    nasdaq_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "INTC", "AMD", "PYPL",
        "CMCSA", "TXN", "NKE", "QCOM", "ADBE", "NFLX", "CRM", "T", "UNP", "HON"
    ]
    
    out = pd.DataFrame({
        "nasdaq_code": nasdaq_stocks,
        "ticker_yf": nasdaq_stocks,
        "company_name": [f"Company {code}" for code in nasdaq_stocks],
        "security_type": ["Equity"] * len(nasdaq_stocks),
        "is_etf": [False] * len(nasdaq_stocks)
    })
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved NASDAQ reference CSV -> {save_path} ({len(out)} rows)")
    return out

def get_exchange_symbols(exchange: str, asset_type: AssetType, max_symbols: int = 0) -> List[str]:
    """Get symbols for a specific exchange and asset type"""
    if exchange not in EXCHANGE_CONFIGS:
        raise ValueError(f"Unsupported exchange: {exchange}")
    
    config = EXCHANGE_CONFIGS[exchange]
    if asset_type not in config["asset_types"]:
        logger.warning(f"Asset type {asset_type.value} not supported for {exchange}")
        return []
    
    # Download exchange company list
    ref_path = REFERENCE_DIR / f"{exchange.lower()}_companies.csv"
    try:
        df = download_exchange_companies_csv(exchange, ref_path)
        
        # Filter by asset type
        if asset_type == AssetType.STOCKS:
            symbols = df[~df["is_etf"]]["ticker_yf"].tolist()
        elif asset_type == AssetType.ETFS:
            symbols = df[df["is_etf"]]["ticker_yf"].tolist()
        else:
            symbols = df["ticker_yf"].tolist()
        
        # Apply symbol limit
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        
        logger.info(f"Retrieved {len(symbols)} {asset_type.value} symbols from {exchange}")
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to get {exchange} symbols: {e}")
        return []

def get_dynamic_asset_symbols(asset_type: AssetType, max_symbols: int = 0) -> List[str]:
    """Get symbols for non-exchange asset types using dynamic APIs"""
    if asset_type not in DYNAMIC_DATA_SOURCES:
        logger.warning(f"No dynamic sources configured for {asset_type.value}")
        return []
    
    config = DYNAMIC_DATA_SOURCES[asset_type]
    all_symbols = []
    
    for source in config["sources"]:
        try:
            logger.info(f"Fetching {asset_type.value} symbols from {source['name']}...")
            
            if source["name"] == "CoinGecko":
                symbols = fetch_coingecko_symbols(source, max_symbols)
            elif source["name"] == "CoinMarketCap":
                symbols = fetch_coinmarketcap_symbols(source, max_symbols)
            elif source["name"] == "Yahoo Finance Forex":
                symbols = fetch_yahoo_forex_symbols(source, max_symbols)
            elif source["name"] == "Yahoo Finance Commodities":
                symbols = fetch_yahoo_commodities_symbols(source, max_symbols)
            elif source["name"] == "FRED API":
                symbols = fetch_fred_symbols(source, max_symbols)
            else:
                logger.warning(f"Unknown source: {source['name']}")
                continue
            
            if symbols:
                all_symbols.extend(symbols)
                logger.info(f"Retrieved {len(symbols)} symbols from {source['name']}")
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source['name']}: {e}")
            continue
    
    # Remove duplicates and apply limit
    all_symbols = list(dict.fromkeys(all_symbols))  # Preserve order, remove duplicates
    if max_symbols > 0:
        all_symbols = all_symbols[:max_symbols]
    
    if not all_symbols:
        logger.warning(f"No symbols retrieved for {asset_type.value}, using fallback")
        return get_fallback_symbols(asset_type, max_symbols)
    
    logger.info(f"Retrieved {len(all_symbols)} total {asset_type.value} symbols")
    return all_symbols

def fetch_coingecko_symbols(source: dict, max_symbols: int) -> List[str]:
    """Fetch cryptocurrency symbols from CoinGecko API"""
    try:
        response = requests.get(source["url"], params=source["params"], timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        for coin in data:
            symbol = coin[source["symbol_field"]].upper()
            yahoo_symbol = f"{symbol}{source['yahoo_suffix']}"
            symbols.append(yahoo_symbol)
        
        return symbols[:max_symbols] if max_symbols > 0 else symbols
    except Exception as e:
        logger.error(f"CoinGecko API error: {e}")
        return []

def fetch_coinmarketcap_symbols(source: dict, max_symbols: int) -> List[str]:
    """Fetch cryptocurrency symbols from CoinMarketCap API"""
    try:
        headers = {"X-CMC_PRO_API_KEY": "your-api-key-here"}  # Would need actual API key
        response = requests.get(source["url"], params=source["params"], headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        for coin in data["data"]:
            symbol = coin["symbol"]
            yahoo_symbol = f"{symbol}{source['yahoo_suffix']}"
            symbols.append(yahoo_symbol)
        
        return symbols[:max_symbols] if max_symbols > 0 else symbols
    except Exception as e:
        logger.error(f"CoinMarketCap API error: {e}")
        return []

def fetch_yahoo_forex_symbols(source: dict, max_symbols: int) -> List[str]:
    """Fetch forex symbols from Yahoo Finance screener"""
    try:
        response = requests.get(source["url"], params=source["params"], timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        if "finance" in data and "result" in data["finance"]:
            for item in data["finance"]["result"]:
                symbol = item.get("symbol", "")
                if symbol and "=X" in symbol:
                    symbols.append(symbol)
        
        return symbols[:max_symbols] if max_symbols > 0 else symbols
    except Exception as e:
        logger.error(f"Yahoo Finance Forex API error: {e}")
        return []

def fetch_yahoo_commodities_symbols(source: dict, max_symbols: int) -> List[str]:
    """Fetch commodity symbols from Yahoo Finance screener"""
    try:
        response = requests.get(source["url"], params=source["params"], timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        if "finance" in data and "result" in data["finance"]:
            for item in data["finance"]["result"]:
                symbol = item.get("symbol", "")
                if symbol and ("=F" in symbol or symbol in ["GLD", "SLV", "USO", "UNG"]):
                    symbols.append(symbol)
        
        return symbols[:max_symbols] if max_symbols > 0 else symbols
    except Exception as e:
        logger.error(f"Yahoo Finance Commodities API error: {e}")
        return []

def fetch_fred_symbols(source: dict, max_symbols: int) -> List[str]:
    """Fetch economic indicator symbols from FRED API"""
    try:
        params = source["params"].copy()
        params["api_key"] = "your-fred-api-key-here"  # Would need actual API key
        
        response = requests.get(source["url"], params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbols = []
        if "seriess" in data:
            for series in data["seriess"]:
                symbol = series.get("id", "")
                if symbol:
                    symbols.append(symbol)
        
        return symbols[:max_symbols] if max_symbols > 0 else symbols
    except Exception as e:
        logger.error(f"FRED API error: {e}")
        return []

def get_fallback_symbols(asset_type: AssetType, max_symbols: int) -> List[str]:
    """Get fallback symbols when dynamic fetching fails"""
    if asset_type not in FALLBACK_SYMBOLS:
        return []
    
    all_symbols = []
    for category, symbols in FALLBACK_SYMBOLS[asset_type].items():
        all_symbols.extend(symbols)
    
    return all_symbols[:max_symbols] if max_symbols > 0 else all_symbols

# ----------------------------
# Global Yahoo Finance Availability Flag
# ----------------------------
YAHOO_FINANCE_AVAILABLE = True  # Set to False after first failure to bypass completely

# ----------------------------
# Multi-Asset Data Collection System
# ----------------------------
class MultiAssetDataCollector:
    """Comprehensive data collection system for all financial instruments"""
    
    def __init__(self, config: Dict[str, Any], show_talib_fallback: bool = False):
        self.config = config
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        # Reduce batch size and workers to avoid Yahoo Finance rate limits (more conservative defaults)
        self.batch_size = int(config.get("batch_size", 5))  # Reduced from 20 to 5
        self.sleep_between = float(config.get("sleep_between", 5.0))  # Increased from 2.0 to 5.0
        self.max_workers = int(config.get("max_workers", 2))  # Reduced from 5 to 2 for Yahoo Finance
        self.raw_dir = RAW_DIR
        self.processed_dir = PROCESSED_DIR
        self._show_talib_fallback = bool(show_talib_fallback)
        
        # Configure yfinance with better error handling - clear cache to avoid stale data
        try:
            import yfinance.utils as yf_utils
            yf_utils.cache.clear()
        except:
            pass
        
        # Initialize data sources
        self.fred = None
        self.alpha_vantage = None
        self.ccxt_exchanges = {}
        self._init_data_sources()
        
        # Asset configurations
        self.asset_configs = self._build_asset_configs()
        
        logger.info(f"MultiAssetDataCollector initialized | {self.start_date} → {self.end_date}")
    
    def _init_data_sources(self):
        """Initialize external data source APIs"""
        try:
            # FRED API (requires API key)
            fred_key = self.config.get("fred_api_key")
            if fred_key:
                self.fred = Fred(api_key=fred_key)
                logger.info("FRED API initialized")
        except Exception as e:
            logger.warning(f"FRED API initialization failed: {e}")
        
        try:
            # Alpha Vantage API (requires API key)
            av_key = self.config.get("alpha_vantage_api_key")
            if av_key:
                self.alpha_vantage = alpha_vantage.AlphaVantage(api_key=av_key)
                logger.info("Alpha Vantage API initialized")
        except Exception as e:
            logger.warning(f"Alpha Vantage API initialization failed: {e}")
        
        try:
            # CCXT exchanges for crypto
            self.ccxt_exchanges = {
                'binance': ccxt.binance(),
                'coinbase': ccxt.coinbase(),
                'kraken': ccxt.kraken(),
                'bitfinex': ccxt.bitfinex()
            }
            logger.info("CCXT exchanges initialized")
        except Exception as e:
            logger.warning(f"CCXT initialization failed: {e}")
    
    def _build_asset_configs(self) -> Dict[AssetType, List[AssetConfig]]:
        """Build asset configurations based on selected asset types and exchanges"""
        configs = {}
        selected_assets = self.config.get("asset_types", [AssetType.STOCKS])
        selected_exchanges = self.config.get("exchanges", ["ASX"])  # Default to ASX
        
        for asset_type in selected_assets:
            asset_configs = []
            
            # Handle exchange-based assets (stocks, ETFs)
            if asset_type in [AssetType.STOCKS, AssetType.ETFS]:
                for exchange in selected_exchanges:
                    if exchange in EXCHANGE_CONFIGS:
                        max_symbols = self.config.get(f"max_{asset_type.value}_symbols", 0)
                        # Check if we should use verified symbols for ASX stocks
                        use_verified = self.config.get("use_verified_asx", False)
                        if asset_type == AssetType.STOCKS and exchange == "ASX" and use_verified:
                            from src.data.asx.asx_symbols import get_verified_symbols
                            symbols = get_verified_symbols(limit=max_symbols or 50)
                            logger.info(f"Using verified ASX symbols (limit={max_symbols or 50}): {len(symbols)} symbols")
                        else:
                            symbols = get_exchange_symbols(exchange, asset_type, max_symbols)
                            logger.info(f"Using exchange list for {exchange}: {len(symbols)} symbols")
                        
                        if symbols:
                            config = AssetConfig(
                                symbols=symbols,
                                data_source=DataSource.YAHOO_FINANCE,
                                yahoo_suffix=EXCHANGE_CONFIGS[exchange]["yahoo_suffix"]
                            )
                            asset_configs.append(config)
                            logger.info(f"Added {len(symbols)} {asset_type.value} symbols from {exchange}")
            
            # Handle dynamic assets (crypto, forex, commodities, economic)
            elif asset_type in DYNAMIC_DATA_SOURCES:
                max_symbols = self.config.get(f"max_{asset_type.value}_symbols", 0)
                symbols = get_dynamic_asset_symbols(asset_type, max_symbols)
                
                if symbols:
                    if asset_type == AssetType.ECONOMIC:
                        config = AssetConfig(
                            symbols=symbols,
                            data_source=DataSource.FRED,
                            additional_params={"frequency": "daily"}
                        )
                    elif asset_type == AssetType.CRYPTO:
                        config = AssetConfig(
                            symbols=symbols,
                            data_source=DataSource.YAHOO_FINANCE,
                            additional_params={"vs_currency": "USD"}
                        )
                    else:
                        config = AssetConfig(
                            symbols=symbols,
                            data_source=DataSource.YAHOO_FINANCE
                        )
                    
                    asset_configs.append(config)
                    logger.info(f"Added {len(symbols)} {asset_type.value} symbols from dynamic sources")
            
            if asset_configs:
                configs[asset_type] = asset_configs
            else:
                logger.warning(f"No symbols found for {asset_type.value}")
        
        return configs

    def _normalize_df(self, df: pd.DataFrame, asset_type: AssetType = None) -> Optional[pd.DataFrame]:
        """Normalize DataFrame to standard format"""
        if df is None or df.empty:
            return None
        
        # Standardize column names
        rename_map = {c: c.title() for c in df.columns}
        df = df.rename(columns=rename_map)
        
        # Handle different asset types
        if asset_type == AssetType.ECONOMIC:
            # Economic data typically has Date and Value columns
            if "Value" in df.columns:
                df = df.rename(columns={"Value": "Close"})
            if "Date" not in df.columns and df.index.name != "Date":
                df["Date"] = df.index
            return df.reset_index(drop=True)
        
        # Standard OHLCV format
        expected = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in expected):
            # Try alternative column names
            alt_mapping = {
                "Adj Close": "Close",
                "Adj_Close": "Close",
                "AdjClose": "Close"
            }
            for old, new in alt_mapping.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            
            # Check again
            if not all(c in df.columns for c in expected):
                logger.warning(f"Missing required columns: {expected}")
                return None
        
        df = df[expected].copy()
        df["Date"] = df.index
        return df.reset_index(drop=True)
    
    def validate_symbols_before_download(self, symbols: List[str], sample_size: int = 10, timeout: int = 5) -> List[str]:
        """
        Pre-validate symbols before attempting downloads.
        Uses shared quick_symbol_health_check from market_data.py to avoid duplication.
        
        Args:
            symbols: List of symbols to validate
            sample_size: Number of symbols to test as a sample (default 10)
            timeout: Timeout in seconds for each validation attempt (not used, kept for compatibility)
            
        Returns:
            List of symbols that are likely to succeed
        """
        if not symbols:
            return []
        
        # Use shared health check function from market_data.py
        from src.data.market_data import quick_symbol_health_check
        logger.info(f"Pre-validating {len(symbols)} symbols using shared health check...")
        valid_symbols = quick_symbol_health_check(symbols, sample_size=sample_size)
        logger.info(f"Validation complete: {len(valid_symbols)}/{len(symbols)} symbols passed")
        return valid_symbols
    
    def is_symbol_likely_valid(self, symbol: str) -> bool:
        """
        Perform basic checks to determine if a symbol is likely valid.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if symbol is likely valid, False otherwise
        """
        if not symbol:
            return False
        
        # Check symbol format (ASX pattern: 3-4 letters/numbers + .AX)
        if symbol.endswith('.AX'):
            base = symbol[:-3]
            # ASX codes: 1-6 alphanumeric characters
            if not base or len(base) > 6:
                return False
            # Should not start with certain patterns that are often invalid
            if base.startswith('0') and len(base) == 1:
                return False
        
        # Known delisted/problematic symbols (expandable blacklist)
        blacklisted_patterns = ['14D', '29M', '3PL', '4DS', '4DX', '88E', '8CO', '8IH', 
                               'T3D', 'TGP', 'TCF', 'TOT', 'TDO', '5EA', '5GN', 
                               'ABG', 'ASK', 'ABX', 'AKG', 'AX8']
        base = symbol.replace('.AX', '').replace('-USD', '').replace('=X', '').replace('=F', '')
        if base in blacklisted_patterns:
            return False
        
        return True
    
    def download_yahoo_data(self, symbol: str, asset_type: AssetType = None) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance using the shared download helper"""
        global YAHOO_FINANCE_AVAILABLE
        
        # EMERGENCY: Bypass Yahoo Finance completely if previously failed
        if not YAHOO_FINANCE_AVAILABLE:
            logger.debug(f"{symbol}: Skipping Yahoo Finance (previously blocked)")
            return None
        
        # Get max_retries from config if available, otherwise use default
        max_retries = self.config.get('max_download_retries', 3)
        
        # Call the shared download helper
        try:
            df = download_with_retry_and_validation(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                max_retries=max_retries,
                asset_type=asset_type.value if asset_type else None
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Check if this is the "Expecting value: line 1 column 1" error indicating complete blockage
            if "expecting value" in error_msg or "line 1 column 1" in error_msg:
                logger.warning(f"Yahoo Finance API appears blocked (JSON parse error). Disabling further attempts.")
                YAHOO_FINANCE_AVAILABLE = False
                return None
            raise
        
        # If download succeeded, normalize the data
        if df is not None and not df.empty:
            norm = self._normalize_df(df, asset_type)
            if norm is not None and not norm.empty:
                # Sanity check: positive prices (skip for economic data)
                if asset_type != AssetType.ECONOMIC:
                    mask = (norm[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
                    norm = norm.loc[mask].copy()
                    if norm.empty:
                        logger.warning(f"{symbol}: filtered to 0 rows after sanity check")
                        return None
                
                logger.info(f"{symbol}: {len(norm)} rows downloaded successfully")
                return norm
        
        # Download failed or returned empty - check if this indicates complete blockage
        # Log a single test to see if Yahoo Finance is completely blocked
        if YAHOO_FINANCE_AVAILABLE:
            # Test with a single symbol to determine if Yahoo Finance is blocked
            try:
                test_ticker = yf.Ticker("AAPL")
                test_df = test_ticker.history(period="5d")
                if test_df.empty:
                    logger.warning("Yahoo Finance appears blocked (test download failed). Disabling further attempts.")
                    YAHOO_FINANCE_AVAILABLE = False
            except Exception as e:
                error_msg = str(e).lower()
                if "expecting value" in error_msg or "line 1 column 1" in error_msg:
                    logger.warning("Yahoo Finance API is completely blocked (JSON parse error). Disabling further attempts.")
                    YAHOO_FINANCE_AVAILABLE = False
        
        logger.warning(f"Failed to fetch {symbol} in {self.start_date}→{self.end_date}")
        return None
    
    def download_fred_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download economic data from FRED"""
        if not self.fred:
            logger.warning("FRED API not available")
            return None
        
        try:
            df = self.fred.get_series(symbol, start=self.start_date, end=self.end_date)
            if df is not None and not df.empty:
                df = df.to_frame(name="Value")
                df["Date"] = df.index
                df = df.reset_index(drop=True)
                logger.info(f"{symbol}: {len(df)} rows from FRED")
                return df
            else:
                logger.warning(f"No FRED data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"FRED download failed for {symbol}: {e}")
            return None
    
    def download_crypto_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download cryptocurrency data using multiple sources"""
        # Try Yahoo Finance first
        yahoo_data = self.download_yahoo_data(symbol, AssetType.CRYPTO)
        if yahoo_data is not None:
            return yahoo_data
        
        # Try CCXT exchanges as fallback
        for exchange_name, exchange in self.ccxt_exchanges.items():
            try:
                # Convert symbol format (e.g., BTC-USD -> BTC/USD)
                ccxt_symbol = symbol.replace("-USD", "/USDT").replace("-", "/")
                
                # Get historical data
                ohlcv = exchange.fetch_ohlcv(ccxt_symbol, '1d', 
                                           exchange.parse8601(f"{self.start_date}T00:00:00Z"),
                                           exchange.parse8601(f"{self.end_date}T23:59:59Z"))
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.drop('timestamp', axis=1)
                    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    logger.info(f"{symbol}: {len(df)} rows from {exchange_name}")
                    return df
                    
            except Exception as e:
                logger.warning(f"{exchange_name} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All crypto sources failed for {symbol}")
        return None
    
    def download_asset_data(self, symbol: str, asset_config: AssetConfig, asset_type: AssetType) -> Optional[pd.DataFrame]:
        """Download data for a specific asset using appropriate data source"""
        if asset_config.data_source == DataSource.YAHOO_FINANCE:
            return self.download_yahoo_data(symbol, asset_type)
        elif asset_config.data_source == DataSource.FRED:
            return self.download_fred_data(symbol)
        elif asset_config.data_source == DataSource.CCXT:
            return self.download_crypto_data(symbol)
        else:
            logger.warning(f"Unsupported data source: {asset_config.data_source}")
            return None

    def add_technical_indicators(self, data: pd.DataFrame, symbol: str, asset_type: AssetType = None) -> pd.DataFrame:
        """Add comprehensive technical indicators to the data"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Skip technical indicators for economic data
        if asset_type == AssetType.ECONOMIC:
            return df
        
        # Ensure we have the required columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            logger.warning(f"Missing required columns for technical indicators: {symbol}")
            return df
        
        if _talib is not None:
            # TA-Lib implementation (comprehensive)
            try:
                # Moving Averages
                df['SMA_5'] = _talib.SMA(df['Close'], timeperiod=5)
                df['SMA_10'] = _talib.SMA(df['Close'], timeperiod=10)
                df['SMA_20'] = _talib.SMA(df['Close'], timeperiod=20)
                df['SMA_50'] = _talib.SMA(df['Close'], timeperiod=50)
                df['SMA_100'] = _talib.SMA(df['Close'], timeperiod=100)
                df['SMA_200'] = _talib.SMA(df['Close'], timeperiod=200)
                
                # Exponential Moving Averages
                df['EMA_12'] = _talib.EMA(df['Close'], timeperiod=12)
                df['EMA_26'] = _talib.EMA(df['Close'], timeperiod=26)
                df['EMA_50'] = _talib.EMA(df['Close'], timeperiod=50)
                df['EMA_200'] = _talib.EMA(df['Close'], timeperiod=200)
                
                # Momentum Indicators
                df['RSI'] = _talib.RSI(df['Close'], timeperiod=14)
                df['RSI_21'] = _talib.RSI(df['Close'], timeperiod=21)
                df['STOCH_K'], df['STOCH_D'] = _talib.STOCH(df['High'], df['Low'], df['Close'])
                df['WILLR'] = _talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
                df['CCI'] = _talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
                
                # MACD
                macd, macd_sig, macd_hist = _talib.MACD(df['Close'])
                df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd, macd_sig, macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = _talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = bb_upper, bb_middle, bb_lower
                df['BB_width'] = (bb_upper - bb_lower) / bb_middle
                df['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
                
                # Volatility Indicators
                df['ATR'] = _talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
                df['NATR'] = _talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
                df['TRANGE'] = _talib.TRANGE(df['High'], df['Low'], df['Close'])
                
                # Volume Indicators
                df['AD'] = _talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
                df['ADOSC'] = _talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
                df['OBV'] = _talib.OBV(df['Close'], df['Volume'])
                
                # Price Patterns
                df['DOJI'] = _talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
                df['HAMMER'] = _talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
                df['ENGULFING'] = _talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
                df['HARAMI'] = _talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
                
                # Trend Indicators
                df['ADX'] = _talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
                df['AROON_UP'], df['AROON_DOWN'] = _talib.AROON(df['High'], df['Low'], timeperiod=14)
                df['AROONOSC'] = _talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
                
                # Support/Resistance
                df['SAR'] = _talib.SAR(df['High'], df['Low'])
                df['SAREXT'] = _talib.SAREXT(df['High'], df['Low'])
                
            except Exception as e:
                logger.warning(f"TA-Lib indicators failed for {symbol}: {e}")
                # Fall back to simplified indicators
                self._add_simplified_indicators(df)
        else:
            # Simplified implementation using pandas
            if self._show_talib_fallback:
                logger.info("TA-Lib not available; using simplified indicators")
            self._add_simplified_indicators(df)
        
        # Additional custom indicators
        self._add_custom_indicators(df)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # Remove warmup period for stable indicators
        if len(df) > 200:
            df = df.iloc[200:].reset_index(drop=True)
        
        return df
    
    def _add_simplified_indicators(self, df: pd.DataFrame):
        """Add simplified technical indicators using pandas"""
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_std_val = df['Close'].rolling(bb_period).std()
        df['BB_upper'] = bb_middle + (bb_std_val * bb_std)
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_middle - (bb_std_val * bb_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # RSI (simplified)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (simplified)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
    
    def _add_custom_indicators(self, df: pd.DataFrame):
        """Add custom technical indicators"""
        # Returns and Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Volatility_5'] = df['Returns'].rolling(5).std() * np.sqrt(252)
        
        # Momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Price_Trend'] = (df['Volume'] * df['Returns']).rolling(20).sum()
        
        # Price position indicators
        df['Price_Position_20'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        df['Price_Position_50'] = (df['Close'] - df['Close'].rolling(50).min()) / (df['Close'].rolling(50).max() - df['Close'].rolling(50).min())
        
        # Trend strength
        df['Trend_Strength_20'] = df['Close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['Trend_Strength_50'] = df['Close'].rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Support and Resistance levels
        df['Support_20'] = df['Low'].rolling(20).min()
        df['Resistance_20'] = df['High'].rolling(20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_20']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance_20'] - df['Close']) / df['Close']

    def generate_synthetic_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic OHLCV stock data for training.
        
        Uses geometric Brownian motion with realistic correlations between stocks.
        This provides a fallback when Yahoo Finance is unavailable.
        """
        logger.info(f"Generating synthetic stock data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        dates = dates[dates.weekday < 5]  # Business days only
        n_days = len(dates)
        
        # Get parameters from config or use defaults
        daily_return_mean = self.config.get('synthetic_data_params', {}).get('daily_return_mean', 0.0005)
        daily_volatility = self.config.get('synthetic_data_params', {}).get('daily_volatility', 0.02)
        correlation = self.config.get('synthetic_data_params', {}).get('correlation', 0.3)
        starting_price = self.config.get('synthetic_data_params', {}).get('starting_price', 100.0)
        volume_mean = self.config.get('synthetic_data_params', {}).get('volume_mean', 1000000)
        
        # Generate correlated random returns using Cholesky decomposition
        n_assets = len(symbols)
        # Create correlation matrix
        corr_matrix = np.eye(n_assets) * (1 - correlation) + np.ones((n_assets, n_assets)) * correlation
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Cholesky decomposition for correlated random numbers
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use uncorrelated
            L = np.eye(n_assets)
            logger.warning("Correlation matrix not positive definite, using uncorrelated data")
        
        # Generate returns
        random_returns = np.random.randn(n_days, n_assets)
        correlated_returns = random_returns @ L.T
        
        # Apply daily return and volatility
        daily_returns = daily_return_mean + correlated_returns * daily_volatility
        
        # Generate prices using geometric Brownian motion
        all_data = {}
        
        for i, symbol in enumerate(symbols):
            # Generate log prices
            log_prices = np.log(starting_price) + np.cumsum(daily_returns[:, i])
            prices = np.exp(log_prices)
            
            # Generate OHLC from Close prices with realistic spread
            close_prices = prices
            # High = Close * (1 + random_positive)
            highs = close_prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
            # Low = Close * (1 - random_positive)
            lows = close_prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
            # Open = previous Close with some gap
            opens = np.roll(close_prices, 1)
            opens[0] = starting_price
            opens = opens * (1 + np.random.randn(n_days) * 0.005)
            
            # Ensure OHLC consistency
            highs = np.maximum(highs, np.maximum(opens, close_prices))
            lows = np.minimum(lows, np.minimum(opens, close_prices))
            
            # Generate volume (log-normal distribution)
            volumes = np.random.lognormal(
                mean=np.log(volume_mean),
                sigma=0.5,
                size=n_days
            ).astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': close_prices,
                'Volume': volumes
            })
            
            # Add realistic bid-ask spread effect
            spread = close_prices * 0.0001  # 1bp spread
            df['Bid'] = close_prices - spread / 2
            df['Ask'] = close_prices + spread / 2
            
            all_data[symbol] = df
            logger.info(f"Generated {len(df)} rows of synthetic data for {symbol}")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'parameters': {
                'daily_return_mean': daily_return_mean,
                'daily_volatility': daily_volatility,
                'correlation': correlation,
                'starting_price': starting_price,
                'volume_mean': volume_mean
            },
            'symbols': symbols,
            'date_range': {'start': start_date, 'end': end_date},
            'num_days': n_days
        }
        
        metadata_path = self.processed_dir / 'synthetic_data_metadata.json'
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
        logger.info(f"Saved synthetic data metadata to {metadata_path}")
        
        return all_data

    def save_data(self, data: pd.DataFrame, symbol: str, asset_type: AssetType, data_type: str = "processed"):
        """Save data to appropriate directory structure"""
        asset_dir = self.raw_dir if data_type == "raw" else self.processed_dir
        asset_dir = asset_dir / asset_type.value
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = asset_dir / f"{symbol}_{data_type}.csv"
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {data_type} -> {file_path}")
    
    def collect_asset_type_data(self, asset_type: AssetType) -> Dict[str, pd.DataFrame]:
        """Collect data for a specific asset type with progress indicators and enhanced error handling"""
        logger.info(f"Collecting data for {asset_type.value}...")
        
        all_data = {}
        asset_configs = self.asset_configs.get(asset_type, [])
        
        # Circuit breaker: track consecutive failures
        consecutive_failures = 0
        max_consecutive_failures = 10
        circuit_breaker_cooldown = 60
        
        # Statistics tracking
        stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "download_times": []
        }
        
        for config in asset_configs:
            logger.info(f"Processing {len(config.symbols)} {asset_type.value} symbols from {config.data_source.value}")
            
            # Pre-validate symbols before downloading
            validated_symbols = self.validate_symbols_before_download(config.symbols, sample_size=10)
            if not validated_symbols:
                logger.warning(f"No validated symbols for {asset_type.value}, using original list with pattern filtering")
                validated_symbols = [s for s in config.symbols if self.is_symbol_likely_valid(s)]
            
            logger.info(f"Validated {len(validated_symbols)}/{len(config.symbols)} symbols")
            
            # Process symbols in batches
            total_batches = (len(validated_symbols) + self.batch_size - 1) // self.batch_size
            for batch_idx in range(0, len(validated_symbols), self.batch_size):
                batch = validated_symbols[batch_idx:batch_idx + self.batch_size]
                batch_num = batch_idx // self.batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} symbols")
                
                # Progress indicator: show current symbol being processed
                start_time = time.time()
                
                # Use ThreadPoolExecutor for parallel downloads
                batch_success = 0
                batch_failed = 0
                
                # Single-attempt download (retry logic is handled by download_with_retry_and_validation)
                def download_single(symbol: str) -> Optional[pd.DataFrame]:
                    try:
                        download_start = time.time()
                        raw_data = self.download_asset_data(symbol, config, asset_type)
                        download_time = time.time() - download_start
                        stats["download_times"].append(download_time)
                        return raw_data
                    except Exception as e:
                        logger.warning(f"  {symbol}: Download failed - {str(e)[:100]}")
                        return None
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_symbol = {
                        executor.submit(download_single, symbol): symbol
                        for symbol in batch
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        stats["total_attempted"] += 1
                        try:
                            raw_data = future.result()
                            if raw_data is not None and not raw_data.empty:
                                # Save raw data
                                self.save_data(raw_data, symbol, asset_type, "raw")
                                
                                # Add technical indicators
                                processed_data = self.add_technical_indicators(raw_data, symbol, asset_type)
                                
                                # Save processed data
                                self.save_data(processed_data, symbol, asset_type, "processed")
                                
                                all_data[symbol] = processed_data
                                batch_success += 1
                                stats["successful"] += 1
                                consecutive_failures = 0  # Reset on success
                                
                                logger.info(f"[OK] {symbol}: {len(processed_data)} rows, {len(processed_data.columns)} features")
                            else:
                                batch_failed += 1
                                stats["failed"] += 1
                                consecutive_failures += 1
                                logger.warning(f"[FAILED] {symbol}: No data available")
                        except Exception as e:
                            batch_failed += 1
                            stats["failed"] += 1
                            consecutive_failures += 1
                            logger.error(f"[FAILED] {symbol}: {e}")
                
                # Real-time success/failure ratio
                batch_total = batch_success + batch_failed
                if batch_total > 0:
                    success_rate = (batch_success / batch_total) * 100
                    logger.info(f"Batch {batch_num} summary: {batch_success}/{batch_total} successful ({success_rate:.1f}%)")
                
                # Circuit breaker: pause if too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Circuit breaker triggered: {consecutive_failures} consecutive failures. Cooling down for {circuit_breaker_cooldown}s...")
                    time.sleep(circuit_breaker_cooldown)
                    consecutive_failures = 0  # Reset after cooldown
                
                # Enhanced sleep between batches with jitter
                if batch_idx + self.batch_size < len(validated_symbols):
                    # Add jitter to avoid synchronized requests
                    import random
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = self.sleep_between * jitter
                    
                    # Exponential backoff every 3 batches
                    if (batch_num % 3) == 0:
                        sleep_time *= 2  # Double sleep time every 3 batches
                        logger.info(f"Cooling period: Sleeping {sleep_time:.1f}s (batch {batch_num} of 3)...")
                    
                    # Estimate time remaining
                    elapsed = time.time() - start_time
                    remaining_batches = total_batches - batch_num
                    if batch_num > 0:
                        avg_time_per_batch = elapsed / batch_num
                        estimated_remaining = avg_time_per_batch * remaining_batches
                        logger.info(f"Sleeping {sleep_time:.1f}s between batches... (Est. {estimated_remaining/60:.1f} min remaining)")
                    
                    time.sleep(sleep_time)
        
        # Log summary statistics
        logger.info(f"Completed {asset_type.value}: {stats['successful']}/{stats['total_attempted']} symbols successful")
        if stats["download_times"]:
            avg_time = sum(stats["download_times"]) / len(stats["download_times"])
            logger.info(f"Average download time per symbol: {avg_time:.2f}s")
        
        return all_data
    
    def collect_all_data(self) -> Dict[AssetType, Dict[str, pd.DataFrame]]:
        """Collect data for all selected asset types with fallback mechanism"""
        logger.info("Starting comprehensive data collection...")
        
        all_results = {}
        total_symbols = 0
        successful_symbols = 0
        
        for asset_type in self.asset_configs.keys():
            safe_print(f"\n[STARTING] Collecting {asset_type.value.replace('_', ' ').title()} Data...")
            
            try:
                asset_data = self.collect_asset_type_data(asset_type)
                all_results[asset_type] = asset_data
                
                successful = len(asset_data)
                total = sum(len(config.symbols) for config in self.asset_configs[asset_type])
                total_symbols += total
                successful_symbols += successful
                
                safe_print(f"[OK] {asset_type.value}: {successful}/{total} symbols successful")
                
            except Exception as e:
                logger.error(f"Error collecting {asset_type.value}: {e}")
                safe_print(f"[FAILED] {asset_type.value}: Collection failed - {e}")
                all_results[asset_type] = {}
        
        # Fallback mechanism: if no data collected, try cached data or minimal dataset
        # Check if any asset type has data in the nested dict structure
        has_data = any(asset_dict for asset_dict in all_results.values() if asset_dict)
        if not has_data or successful_symbols == 0:
            logger.warning("[WARNING] No data collected, attempting fallback mechanisms...")
            
            # Try to load cached data from previous runs
            cached_data = self._load_cached_data()
            if cached_data:
                logger.info(f"[FALLBACK] Loaded {len(cached_data)} symbols from cache")
                # Merge cached data into results
                for asset_type, data in cached_data.items():
                    if asset_type not in all_results or not all_results[asset_type]:
                        all_results[asset_type] = data
                successful_symbols = sum(len(d) for d in all_results.values())
            
            # If still no data, use minimal verified symbols (configurable exchange and count)
            has_data_after_cache = any(asset_dict for asset_dict in all_results.values() if asset_dict)
            if not has_data_after_cache:
                logger.warning("[FALLBACK] No cached data available, using minimal verified symbol list...")
                try:
                    # Read fallback configuration from config file or use defaults
                    import yaml
                    config_path = Path("config/production_config.yaml")
                    fallback_exchange = "ASX"
                    fallback_count = 10
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config_data = yaml.safe_load(f)
                                data_collection_config = config_data.get("data_collection", {})
                                fallback_exchange = data_collection_config.get("minimal_verified_exchange", "ASX")
                                fallback_count = data_collection_config.get("minimal_verified_count", 10)
                                logger.info(f"[FALLBACK] Using configured fallback: exchange={fallback_exchange}, count={fallback_count}")
                        except Exception as e:
                            logger.warning(f"[FALLBACK] Failed to load config, using defaults: {e}")
                    
                    if fallback_exchange == "ASX":
                        from src.data.asx.asx_symbols import get_verified_symbols
                        verified_symbols = get_verified_symbols(limit=fallback_count)
                        logger.info(f"[FALLBACK] Using {len(verified_symbols)} verified {fallback_exchange} symbols (limit={fallback_count})")
                    else:
                        logger.warning(f"[FALLBACK] Unsupported exchange for verified fallback: {fallback_exchange}, using ASX")
                        from src.data.asx.asx_symbols import get_verified_symbols
                        verified_symbols = get_verified_symbols(limit=fallback_count)
                        logger.info(f"[FALLBACK] Using {len(verified_symbols)} verified ASX symbols (limit={fallback_count})")
                    
                    # Try to download verified symbols
                    if verified_symbols:
                        # Create minimal config
                        from dataclasses import dataclass
                        minimal_config = AssetConfig(
                            symbols=verified_symbols,
                            data_source=DataSource.YAHOO_FINANCE,
                            yahoo_suffix=".AX"
                        )
                        
                        # Try downloading with very conservative settings
                        old_batch_size = self.batch_size
                        old_sleep = self.sleep_between
                        old_workers = self.max_workers
                        
                        self.batch_size = 2  # Very small batches
                        self.sleep_between = 10.0  # Very long sleep
                        self.max_workers = 1  # Single worker
                        
                        try:
                            for symbol in verified_symbols[:5]:  # Try just first 5
                                data = self.download_asset_data(symbol, minimal_config, AssetType.STOCKS)
                                if data is not None and not data.empty:
                                    if AssetType.STOCKS not in all_results:
                                        all_results[AssetType.STOCKS] = {}
                                    all_results[AssetType.STOCKS][symbol] = data
                                    successful_symbols += 1
                                    logger.info(f"[FALLBACK] Successfully downloaded {symbol}")
                        finally:
                            # Restore original settings
                            self.batch_size = old_batch_size
                            self.sleep_between = old_sleep
                            self.max_workers = old_workers
                except Exception as e:
                    logger.error(f"[FALLBACK] Failed to use verified symbols: {e}")
        
        logger.info(f"Data collection completed: {successful_symbols}/{total_symbols} symbols successful")
        
        # Print detailed summary statistics
        self._print_summary_statistics(all_results, total_symbols, successful_symbols)
        
        return all_results
    
    def _print_summary_statistics(self, all_results: Dict[AssetType, Dict[str, pd.DataFrame]], total_symbols: int, successful_symbols: int):
        """Print detailed summary statistics"""
        failed_symbols = []
        download_times = []
        
        # Collect statistics from each asset type
        for asset_type, asset_data in all_results.items():
            total_for_type = sum(len(config.symbols) for config in self.asset_configs.get(asset_type, []))
            successful_for_type = len(asset_data)
            failed_for_type = total_for_type - successful_for_type
            
            if failed_for_type > 0:
                # Try to identify failed symbols
                all_symbols_for_type = []
                for config in self.asset_configs.get(asset_type, []):
                    all_symbols_for_type.extend(config.symbols)
                
                for symbol in all_symbols_for_type:
                    if symbol not in asset_data:
                        failed_symbols.append((asset_type.value, symbol))
            
            logger.info(f"{asset_type.value}: {successful_for_type}/{total_for_type} successful, {failed_for_type} failed")
        
        # Print summary
        logger.info("="*80)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total symbols attempted: {total_symbols}")
        logger.info(f"Successful downloads: {successful_symbols}")
        logger.info(f"Failed downloads: {total_symbols - successful_symbols}")
        if total_symbols > 0:
            success_rate = (successful_symbols / total_symbols) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # List top 10 failed symbols
        if failed_symbols:
            logger.info(f"\nTop 10 Failed Symbols:")
            for i, (asset_type, symbol) in enumerate(failed_symbols[:10], 1):
                logger.info(f"  {i}. {symbol} ({asset_type})")
        
        # Provide recommendations based on failure patterns
        if total_symbols > 0:
            failure_rate = ((total_symbols - successful_symbols) / total_symbols) * 100
            if failure_rate > 50:
                logger.warning("\nHigh failure rate suggests:")
                logger.warning("  - Yahoo Finance API rate limiting (try reducing batch size)")
                logger.warning("  - Many delisted/invalid symbols (use --use-verified-symbols)")
                logger.warning("  - Network connectivity issues (check internet connection)")
                logger.warning("  - API service outage (try again later)")
    
    def _load_cached_data(self) -> Dict[AssetType, Dict[str, pd.DataFrame]]:
        """Load previously cached data from processed directory"""
        cached_data = {}
        
        try:
            # Check processed directory for existing CSVs
            processed_dir = Path(self.processed_dir)
            if not processed_dir.exists():
                return cached_data
            
            # Scan for CSV files organized by asset type
            for asset_type_value in [e.value for e in AssetType]:
                asset_dir = processed_dir / asset_type_value
                if not asset_dir.exists():
                    continue
                
                csv_files = list(asset_dir.glob("*_processed.csv"))
                if not csv_files:
                    continue
                
                # Check file age (use cache if < 7 days old)
                from datetime import timedelta
                cutoff_date = datetime.now() - timedelta(days=7)
                
                asset_data = {}
                for csv_file in csv_files:
                    try:
                        # Check file modification time
                        mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
                        if mtime < cutoff_date:
                            logger.debug(f"Skipping old cache file: {csv_file.name} (age: {datetime.now() - mtime})")
                            continue
                        
                        symbol = csv_file.stem.replace("_processed", "")
                        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        
                        # Normalize: ensure Date column exists if index is datetime
                        # If index is datetime but Date column is missing, add it
                        if isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
                            df = df.reset_index()
                            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                            logger.debug(f"Normalized cached data for {symbol}: added Date column from index")
                        elif not isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
                            # If neither index nor Date column exists, create Date from row number
                            logger.warning(f"No date information found for {symbol}, using row numbers")
                            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                        
                        if not df.empty:
                            asset_data[symbol] = df
                            logger.debug(f"Loaded cached data: {symbol} ({len(df)} rows)")
                    except Exception as e:
                        logger.warning(f"Failed to load cached file {csv_file}: {e}")
                        continue
                
                if asset_data:
                    # Map asset_type string to enum
                    for at in AssetType:
                        if at.value == asset_type_value:
                            cached_data[at] = asset_data
                            break
                    
                    logger.info(f"Loaded {len(asset_data)} cached symbols for {asset_type_value}")
        
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
        
        return cached_data
    
    def create_comprehensive_summary(self, all_data: Dict[AssetType, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Create comprehensive summary of all collected data"""
        summary = {
            "collection_date": datetime.now().isoformat(),
            "date_range": {"start": self.start_date, "end": self.end_date},
            "asset_types": {},
            "total_symbols": 0,
            "successful_symbols": 0,
            "data_quality": {}
        }
        
        for asset_type, asset_data in all_data.items():
            asset_summary = {
                "symbols": list(asset_data.keys()),
                "count": len(asset_data),
                "categories": {}
            }
            
            # Categorize symbols by source
            for symbol, df in asset_data.items():
                if not df.empty:
                    category = "unknown"
                    for config in self.asset_configs.get(asset_type, []):
                        if symbol in config.symbols:
                            category = config.data_source.value
                            break
                    
                    if category not in asset_summary["categories"]:
                        asset_summary["categories"][category] = []
                    asset_summary["categories"][category].append(symbol)
            
            summary["asset_types"][asset_type.value] = asset_summary
            summary["total_symbols"] += len(asset_data)
            summary["successful_symbols"] += len(asset_data)
        
        return summary
    
    def validate_data_quality(self, all_data: Dict[AssetType, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Validate data quality across all asset types"""
        report = {
            "overall_quality": "good",
            "asset_quality": {},
            "issues": [],
            "recommendations": []
        }
        
        for asset_type, asset_data in all_data.items():
            asset_issues = []
            quality_scores = []
            
            for symbol, df in asset_data.items():
                if df.empty:
                    asset_issues.append(f"{symbol}: Empty dataset")
                    quality_scores.append(0)
                    continue
                
                # Data quality checks
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                if missing_pct > 10:
                    asset_issues.append(f"{symbol}: High missing data ({missing_pct:.1f}%)")
                
                # Check for extreme price movements
                extreme_moves = 0  # Initialize before use
                if 'Close' in df.columns and len(df) > 1:
                    rets = df['Close'].pct_change()
                    extreme_moves = int((rets.abs() > 0.5).sum())
                    if extreme_moves > 0:
                        asset_issues.append(f"{symbol}: {extreme_moves} extreme price moves")
                
                # Calculate quality score
                score = max(0, 100 - missing_pct - extreme_moves)
                quality_scores.append(score)
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            report["asset_quality"][asset_type.value] = {
                "average_quality": avg_quality,
                "issues": asset_issues,
                "symbols_checked": len(asset_data)
            }
            
            if avg_quality < 70:
                report["issues"].extend(asset_issues)
        
        # Overall quality assessment
        all_scores = [q["average_quality"] for q in report["asset_quality"].values()]
        overall_avg = np.mean(all_scores) if all_scores else 0
        
        if overall_avg < 60:
            report["overall_quality"] = "poor"
        elif overall_avg < 80:
            report["overall_quality"] = "fair"
        else:
            report["overall_quality"] = "excellent"
        
        return report


    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Legacy compatibility method for single-asset dictionaries.
        Refactored to use new multi-asset architecture.
        
        DEPRECATED: Use collect_all_data() for new code.
        This method is maintained for backward compatibility with test files.
        """
        logger.warning("[DEPRECATED] download_all_data() is a legacy method. Consider using collect_all_data() instead.")
        
        # Use the new multi-asset collection method
        multi_asset_data = self.collect_all_data()
        
        # Convert multi-asset structure to legacy single-asset dictionary format
        # Default to STOCKS if available, otherwise flatten all asset types
        all_data: Dict[str, pd.DataFrame] = {}
        success = 0
        
        # Prefer STOCKS data, otherwise flatten all asset types
        if AssetType.STOCKS in multi_asset_data and multi_asset_data[AssetType.STOCKS]:
            all_data = multi_asset_data[AssetType.STOCKS].copy()
            success = len(all_data)
        else:
            # Flatten all asset types into single dictionary
            for asset_type, asset_dict in multi_asset_data.items():
                if asset_dict:
                    all_data.update(asset_dict)
            success = len(all_data)
        
        logger.info(f"Legacy download_all_data() completed: {success} symbols successful")
        return all_data

    def create_data_summary(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summary = {
            "download_date": datetime.now().isoformat(),
            "symbols": list(all_data.keys()),
            "num_symbols": len(all_data),
            "date_range": {"start": self.start_date, "end": self.end_date},
            "symbol_details": {}
        }
        for sym, df in all_data.items():
            if not df.empty:
                summary["symbol_details"][sym] = {
                    "records": int(len(df)),
                    "features": int(len(df.columns)),
                    "date_range": {
                        "start": str(pd.to_datetime(df['Date']).min().date()),
                        "end": str(pd.to_datetime(df['Date']).max().date())
                    },
                    "price_range": {
                        "min": float(df['Close'].min()),
                        "max": float(df['Close'].max()),
                        "mean": float(df['Close'].mean())
                    },
                    "volume_range": {
                        "min": float(df['Volume'].min()),
                        "max": float(df['Volume'].max()),
                        "mean": float(df['Volume'].mean())
                    }
                }
        return summary

    def validate_symbol_data_quality(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Legacy standalone validation function for single-asset dictionaries.
        Note: The multi-asset version validate_data_quality() should be used for new code.
        This function is kept for backward compatibility but may be deprecated.
        """
        report = {"overall_quality": "good", "issues": [], "symbol_quality": {}}
        for sym, df in all_data.items():
            issues = []
            if df.empty:
                report["symbol_quality"][sym] = "poor"
                issues.append("Empty dataset")
                continue
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if missing_pct > 5:
                issues.append(f"High missing data: {missing_pct:.1f}%")
            extreme_moves = 0  # Initialize before use
            if len(df) > 1:
                rets = df['Close'].pct_change()
                extreme_moves = int((rets.abs() > 0.5).sum())
                if extreme_moves > 0:
                    issues.append(f"Extreme price moves: {extreme_moves}")
            if issues:
                report["symbol_quality"][sym] = "good" if len(issues) <= 2 else "poor"
                report["issues"].extend([f"{sym}: {x}" for x in issues])
            else:
                report["symbol_quality"][sym] = "excellent"
        poor = sum(1 for q in report["symbol_quality"].values() if q == "poor")
        if poor > len(all_data) * 0.3:
            report["overall_quality"] = "poor"
        elif poor > 0:
            report["overall_quality"] = "fair"
        return report

# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    global YAHOO_FINANCE_AVAILABLE

    parser = argparse.ArgumentParser(description="WealthArena Multi-Asset Data Collection System")
    
    # Asset type selection
    parser.add_argument("--all-assets", action="store_true",
                        help="Download all asset types")
    parser.add_argument("--stocks", action="store_true",
                        help="Download stocks from selected exchanges")
    parser.add_argument("--etfs", action="store_true",
                        help="Download ETFs from selected exchanges")
    parser.add_argument("--crypto", action="store_true",
                        help="Download cryptocurrencies")
    parser.add_argument("--forex", action="store_true",
                        help="Download currency pairs")
    parser.add_argument("--commodities", action="store_true",
                        help="Download commodities")
    parser.add_argument("--economic", action="store_true",
                        help="Download economic indicators")
    
    # Exchange selection
    parser.add_argument("--exchanges", nargs="+", default=["ASX"],
                        choices=list(EXCHANGE_CONFIGS.keys()),
                        help="Exchanges to download from (default: ASX)")
    parser.add_argument("--all-exchanges", action="store_true",
                        help="Download from all supported exchanges")
    
    # Configuration options
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (default: today-5y)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for downloads (default: 5, conservative to avoid rate limits)")
    parser.add_argument("--sleep-between", type=float, default=5.0,
                        help="Sleep time between batches in seconds (default: 5.0, conservative to avoid rate limits)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Maximum parallel workers (default: 2, conservative to avoid rate limits)")
    parser.add_argument("--conservative", action="store_true", default=False,
                        help="Use very conservative preset values (batch=3, sleep=10s, workers=1)")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Use faster settings (batch=20, sleep=2s, workers=5) - may trigger rate limits")
    
    # Symbol limits
    parser.add_argument("--max-stocks-symbols", type=int, default=0,
                        help="Limit number of stock symbols (0 = no limit)")
    parser.add_argument("--max-etfs-symbols", type=int, default=0,
                        help="Limit number of ETF symbols (0 = no limit)")
    parser.add_argument("--max-crypto-symbols", type=int, default=0,
                        help="Limit number of crypto symbols (0 = no limit)")
    parser.add_argument("--max-forex-symbols", type=int, default=0,
                        help="Limit number of forex symbols (0 = no limit)")
    parser.add_argument("--max-commodities-symbols", type=int, default=0,
                        help="Limit number of commodity symbols (0 = no limit)")
    
    # API keys
    parser.add_argument("--fred-api-key", default=None,
                        help="FRED API key for economic data")
    parser.add_argument("--alpha-vantage-api-key", default=None,
                        help="Alpha Vantage API key")
    
    # Other options
    parser.add_argument("--show-talib-fallback", action="store_true",
                        help="Show TA-Lib fallback warnings")
    parser.add_argument("--use-verified-asx", action="store_true", default=False,
                        help="Use verified ASX symbols list instead of exchange list (default: False)")
    parser.add_argument("--use-synthetic-data", action="store_true", default=False,
                        help="Use synthetic data generation instead of live download (default: False, but auto-enabled if Yahoo Finance blocked)")
    parser.add_argument("--prefer-synthetic", action="store_true", default=False,
                        help="Prefer synthetic data generation even when cached/live data is available (default: False)")
    parser.add_argument("--force-live-download", action="store_true", default=False,
                        help="Force live download even if synthetic data is available (default: False)")
    
    args = parser.parse_args()

    # Determine asset types to collect
    if args.all_assets:
        asset_types = list(AssetType)
    else:
        asset_types = []
        if args.stocks:
            asset_types.append(AssetType.STOCKS)
        if args.etfs:
            asset_types.append(AssetType.ETFS)
        if args.crypto:
            asset_types.append(AssetType.CRYPTO)
        if args.forex:
            asset_types.append(AssetType.FOREX)
        if args.commodities:
            asset_types.append(AssetType.COMMODITIES)
        if args.economic:
            asset_types.append(AssetType.ECONOMIC)
    
    if not asset_types:
        asset_types = [AssetType.STOCKS]  # Default to stocks only
        print("No asset types specified, defaulting to stocks only")

    # Determine exchanges to use
    if args.all_exchanges:
        exchanges = list(EXCHANGE_CONFIGS.keys())
    else:
        exchanges = args.exchanges

    # Date range: default = last ~5 years
    today = datetime.now().date()
    end_date = args.end_date or today.isoformat()
    if args.start_date:
        start_date = args.start_date
    else:
        # ~5y = 1825 days
        start_date = (today - timedelta(days=1825)).isoformat()

    # Apply conservative or fast presets if specified
    batch_size = args.batch_size
    sleep_between = args.sleep_between
    max_workers = args.max_workers
    
    if args.conservative:
        batch_size = 3
        sleep_between = 10.0
        max_workers = 1
        logger.info("Using conservative preset: batch=3, sleep=10s, workers=1")
    elif args.fast:
        batch_size = 20
        sleep_between = 2.0
        max_workers = 5
        logger.info("Using fast preset: batch=20, sleep=2s, workers=5")
    
    # Build configuration
    config = {
        "asset_types": asset_types,
        "exchanges": exchanges,
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": batch_size,
        "sleep_between": sleep_between,
        "max_workers": max_workers,
        "fred_api_key": args.fred_api_key,
        "alpha_vantage_api_key": args.alpha_vantage_api_key,
        "max_stocks_symbols": args.max_stocks_symbols,
        "max_etfs_symbols": args.max_etfs_symbols,
        "max_crypto_symbols": args.max_crypto_symbols,
        "max_forex_symbols": args.max_forex_symbols,
        "max_commodities_symbols": args.max_commodities_symbols,
        "use_verified_asx": args.use_verified_asx,
        "use_synthetic_data": args.use_synthetic_data,
        "prefer_synthetic": args.prefer_synthetic,
        "force_live_download": args.force_live_download,
        # Synthetic data parameters
        "synthetic_data_params": {
            "daily_return_mean": 0.0005,  # 0.05% per day, ~13% annual
            "daily_volatility": 0.02,  # 2% per day, ~32% annual
            "correlation": 0.3,  # Moderate correlation
            "starting_price": 100.0,
            "volume_mean": 1000000
        }
    }

    logger.info(f"Asset types to collect: {[at.value for at in asset_types]}")
    logger.info(f"Exchanges to use: {exchanges}")
    logger.info(f"Date range: {start_date} → {end_date}")
    if args.use_synthetic_data:
        logger.info("EMERGENCY: Using synthetic data generation (Yahoo Finance bypassed)")

    # Initialize collector
    collector = MultiAssetDataCollector(config, show_talib_fallback=args.show_talib_fallback)

    try:
        # EMERGENCY FALLBACK CHAIN:
        # 1. Try to load cached data from data/processed/
        # 2. If no cache, generate synthetic data (if --use-synthetic-data is set or Yahoo Finance is blocked)
        # 3. Only attempt live download if user explicitly requests with --force-live-download
        # 4. Never exit with error - always provide SOME data for training
        
        all_data = {}
        
        # Step 1: Check for cached data
        cached_data = collector._load_cached_data()
        if cached_data:
            logger.info(f"[CACHE] Loaded {sum(len(d) for d in cached_data.values())} symbols from cache")
            all_data = cached_data
        else:
            logger.info("[CACHE] No cached data found")
        
        # Step 2: Determine data collection strategy
        force_live = args.force_live_download
        # When force_live is True, always set use_synthetic to False to ensure live attempt
        use_synthetic = (args.use_synthetic_data or not YAHOO_FINANCE_AVAILABLE) if not force_live else False
        
        # Check if we have any data yet
        has_data = any(asset_dict for asset_dict in all_data.values() if asset_dict)
        cached_data = has_data  # Store whether we have cached data
        
        # Step 3: Attempt live download first if explicitly requested (force-live mode)
        if force_live:
            logger.info("[FORCE-LIVE] Force-live mode: attempting live data download (bypassing synthetic flag)...")
            # Temporarily set YAHOO_FINANCE_AVAILABLE to True to bypass early-return guard
            original_yahoo_flag = YAHOO_FINANCE_AVAILABLE
            YAHOO_FINANCE_AVAILABLE = True
            try:
                live_data = collector.collect_all_data()
                # Merge live data with existing data
                for asset_type, asset_dict in live_data.items():
                    if asset_type not in all_data:
                        all_data[asset_type] = {}
                    all_data[asset_type].update(asset_dict)
                
                # Check if live attempt returned any data
                live_has_data = any(asset_dict for asset_dict in live_data.values() if asset_dict)
                if not live_has_data:
                    logger.warning("[FORCE-LIVE] Live download returned no data; falling back to synthetic.")
            finally:
                # Restore original YAHOO_FINANCE_AVAILABLE flag
                YAHOO_FINANCE_AVAILABLE = original_yahoo_flag
        
        # Update has_data after live attempt if force_live was used
        if force_live:
            has_data = any(asset_dict for asset_dict in all_data.values() if asset_dict)
        
        # Step 4: Use synthetic data only if NO cached/live data exists AND synthetic is requested
        # Only generate synthetic when no cached/live data is available, unless --prefer-synthetic is set
        # In force-live mode, fall back to synthetic if live attempt yields no data
        prefer_synthetic = args.prefer_synthetic
        # Generate synthetic if: (no data exists AND (synthetic requested OR force-live mode)) OR (prefer-synthetic flag is set)
        should_generate_synthetic = (not has_data and (use_synthetic or force_live)) or prefer_synthetic
        if should_generate_synthetic:
            if prefer_synthetic:
                logger.info("[SYNTHETIC] Generating synthetic data for training (--prefer-synthetic flag set, ignoring cached/live data)...")
            elif force_live and not has_data:
                logger.info("[SYNTHETIC] Generating synthetic data for training (force-live mode: live returned no data, falling back to synthetic)...")
            else:
                logger.info("[SYNTHETIC] Generating synthetic data for training (no cached/live data available)...")
            # Generate symbols for synthetic data
            synthetic_symbols = []
            if AssetType.STOCKS in asset_types:
                # Get symbols from config or generate default
                if args.max_stocks_symbols > 0:
                    # Create synthetic symbols
                    for i in range(args.max_stocks_symbols):
                        synthetic_symbols.append(f"SYNTH_{i+1}.AX")
                else:
                    # Default to 20 synthetic symbols
                    for i in range(20):
                        synthetic_symbols.append(f"SYNTH_{i+1}.AX")
            
            if synthetic_symbols:
                synthetic_data = collector.generate_synthetic_stock_data(synthetic_symbols, start_date, end_date)
                # Process synthetic data with technical indicators
                for symbol, df in synthetic_data.items():
                    processed_df = collector.add_technical_indicators(df, symbol, AssetType.STOCKS)
                    collector.save_data(processed_df, symbol, AssetType.STOCKS, "processed")
                
                # Add to all_data
                if AssetType.STOCKS not in all_data:
                    all_data[AssetType.STOCKS] = {}
                for symbol, df in synthetic_data.items():
                    processed_df = collector.add_technical_indicators(df, symbol, AssetType.STOCKS)
                    all_data[AssetType.STOCKS][symbol] = processed_df
                
                logger.info(f"[SYNTHETIC] Generated {len(synthetic_data)} synthetic symbols")
        elif not force_live and not has_data and not use_synthetic:
            # Try live download if no other data available and not in force-live mode
            logger.info("[LIVE] Attempting live data download (fallback)...")
            live_data = collector.collect_all_data()
            for asset_type, asset_dict in live_data.items():
                if asset_type not in all_data:
                    all_data[asset_type] = {}
                all_data[asset_type].update(asset_dict)
        
        # Check if any asset type has data in the nested dict structure
        # all_data is Dict[AssetType, Dict[str, pd.DataFrame]]
        has_data = any(asset_dict for asset_dict in all_data.values() if asset_dict)
        
        if not has_data:
            logger.warning("[WARNING] No data collected successfully. Training may proceed with partial/cached data.")
            # Don't exit - allow training to proceed with whatever data is available
            # Generate minimal summary even if no data was collected
            if not all_data:
                # Create empty summary structure
                summary = {
                    "collection_date": datetime.now().isoformat(),
                    "date_range": {"start": start_date, "end": end_date},
                    "asset_types": {},
                    "total_symbols": 0,
                    "successful_symbols": 0,
                    "data_quality": {"overall_quality": "poor", "issues": ["No data collected"]}
                }
                quality = {
                    "overall_quality": "poor",
                    "asset_quality": {},
                    "issues": ["No data collected - training may fail"],
                    "recommendations": [
                        "Check internet connection",
                        "Verify Yahoo Finance API is accessible",
                        "Try using --use-verified-symbols flag",
                        "Reduce --max-stocks-symbols to a smaller number"
                    ]
                }
                
                # Save reports
                summary_path = BASE_DIR / "multi_asset_data_summary.json"
                quality_path = BASE_DIR / "multi_asset_quality_report.json"
                summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
                quality_path.write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
                
                logger.warning("[WARNING] Training may fail due to insufficient data. Check logs for details.")
                return  # Exit gracefully instead of sys.exit(1)

        # Create summaries
        summary = collector.create_comprehensive_summary(all_data)
        quality = collector.validate_data_quality(all_data)

        # Save reports
        summary_path = BASE_DIR / "multi_asset_data_summary.json"
        quality_path = BASE_DIR / "multi_asset_quality_report.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        quality_path.write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")

        # Print comprehensive summary
        print("\n" + "="*80)
        safe_print("[SUMMARY] WEALTHARENA MULTI-ASSET DATA COLLECTION SUMMARY")
        print("="*80)
        safe_print(f"[OK] Total symbols collected: {summary['successful_symbols']}")
        safe_print(f"[DATE] Date range: {start_date} → {end_date}")
        safe_print(f"[PATH] Raw data: {RAW_DIR}")
        safe_print(f"[PATH] Processed data: {PROCESSED_DIR}")
        safe_print(f"[FILE] Summary: {summary_path.name}")
        safe_print(f"[FILE] Quality: {quality_path.name}")
        
        safe_print(f"\n[BREAKDOWN] ASSET TYPE BREAKDOWN:")
        for asset_type, data in all_data.items():
            if data:
                print(f"  {asset_type.value.replace('_', ' ').title()}: {len(data)} symbols")
        
        safe_print(f"\n[SUMMARY] DATA QUALITY: {quality['overall_quality'].upper()}")
        for asset_type, quality_info in quality['asset_quality'].items():
            print(f"  {asset_type}: {quality_info['average_quality']:.1f}% quality")
        
        safe_print("\n[SUCCESS] Multi-Asset Data Collection Completed!")

    except Exception as e:
        logger.error(f"Data collection failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()