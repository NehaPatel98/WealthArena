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
# Multi-Asset Data Collection System
# ----------------------------
class MultiAssetDataCollector:
    """Comprehensive data collection system for all financial instruments"""
    
    def __init__(self, config: Dict[str, Any], show_talib_fallback: bool = False):
        self.config = config
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.batch_size = int(config.get("batch_size", 50))
        self.sleep_between = float(config.get("sleep_between", 1.0))
        self.max_workers = int(config.get("max_workers", 10))
        self.raw_dir = RAW_DIR
        self.processed_dir = PROCESSED_DIR
        self._show_talib_fallback = bool(show_talib_fallback)
        
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
                        symbols = get_exchange_symbols(exchange, asset_type, max_symbols)
                        
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
    
    def download_yahoo_data(self, symbol: str, asset_type: AssetType = None) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance"""
        attempts = [
            ("history-1", lambda: yf.Ticker(symbol).history(
                start=self.start_date, end=self.end_date, interval="1d",
                auto_adjust=True, back_adjust=True)),
            ("history-2", lambda: yf.Ticker(symbol).history(
                start=self.start_date, end=self.end_date, interval="1d",
                auto_adjust=True, back_adjust=False)),
            ("download", lambda: yf.download(
                tickers=symbol, start=self.start_date, end=self.end_date,
                interval="1d", auto_adjust=True, progress=False, group_by="column")),
        ]
        
        last_err = None
        for tag, fn in attempts:
            try:
                df = fn()
                norm = self._normalize_df(df, asset_type)
                if norm is not None and not norm.empty:
                    # Sanity check: positive prices (skip for economic data)
                    if asset_type != AssetType.ECONOMIC:
                        mask = (norm[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
                        norm = norm.loc[mask].copy()
                        if norm.empty:
                            logger.warning(f"{symbol}: filtered to 0 rows after sanity check")
                            continue
                    
                    logger.info(f"{symbol}: {len(norm)} rows via {tag}")
                    return norm
                else:
                    logger.warning(f"{symbol}: empty via {tag}")
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "possibly delisted" in msg or "no timezone" in msg:
                    logger.warning(f"{symbol}: {e}")
                else:
                    logger.warning(f"{symbol}: {tag} failed: {e}")
            time.sleep(0.25)
        
        if last_err:
            logger.error(f"Failed to fetch {symbol}: {last_err}")
        else:
            logger.error(f"No data for {symbol} in {self.start_date}→{self.end_date}")
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

    def save_data(self, data: pd.DataFrame, symbol: str, asset_type: AssetType, data_type: str = "processed"):
        """Save data to appropriate directory structure"""
        asset_dir = self.raw_dir if data_type == "raw" else self.processed_dir
        asset_dir = asset_dir / asset_type.value
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = asset_dir / f"{symbol}_{data_type}.csv"
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {data_type} -> {file_path}")
    
    def collect_asset_type_data(self, asset_type: AssetType) -> Dict[str, pd.DataFrame]:
        """Collect data for a specific asset type"""
        logger.info(f"Collecting data for {asset_type.value}...")
        
        all_data = {}
        asset_configs = self.asset_configs.get(asset_type, [])
        
        for config in asset_configs:
            logger.info(f"Processing {len(config.symbols)} {asset_type.value} symbols from {config.data_source.value}")
            
            # Process symbols in batches
            for i in range(0, len(config.symbols), self.batch_size):
                batch = config.symbols[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} symbols")
                
                # Use ThreadPoolExecutor for parallel downloads
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_symbol = {
                        executor.submit(self.download_asset_data, symbol, config, asset_type): symbol
                        for symbol in batch
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
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
                                logger.info(f"✅ {symbol}: {len(processed_data)} rows, {len(processed_data.columns)} features")
                            else:
                                logger.warning(f"❌ {symbol}: No data available")
                        except Exception as e:
                            logger.error(f"❌ {symbol}: {e}")
                
                # Sleep between batches
                if i + self.batch_size < len(config.symbols):
                    logger.info(f"Sleeping {self.sleep_between:.1f}s between batches...")
                    time.sleep(self.sleep_between)
        
        logger.info(f"Completed {asset_type.value}: {len(all_data)} symbols successful")
        return all_data
    
    def collect_all_data(self) -> Dict[AssetType, Dict[str, pd.DataFrame]]:
        """Collect data for all selected asset types"""
        logger.info("Starting comprehensive data collection...")
        
        all_results = {}
        total_symbols = 0
        successful_symbols = 0
        
        for asset_type in self.asset_configs.keys():
            print(f"\n🚀 Collecting {asset_type.value.replace('_', ' ').title()} Data...")
            
            try:
                asset_data = self.collect_asset_type_data(asset_type)
                all_results[asset_type] = asset_data
                
                successful = len(asset_data)
                total = sum(len(config.symbols) for config in self.asset_configs[asset_type])
                total_symbols += total
                successful_symbols += successful
                
                print(f"✅ {asset_type.value}: {successful}/{total} symbols successful")
                
            except Exception as e:
                logger.error(f"Error collecting {asset_type.value}: {e}")
                print(f"❌ {asset_type.value}: Collection failed - {e}")
                all_results[asset_type] = {}
        
        logger.info(f"Data collection completed: {successful_symbols}/{total_symbols} symbols successful")
        return all_results
    
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
        logger.info(f"Starting downloads for {len(self.symbols)} symbols (batch={self.batch_size}, sleep={self.sleep_between}s)")
        all_data: Dict[str, pd.DataFrame] = {}
        success = 0
        for i in range(0, len(self.symbols), self.batch_size):
            batch = self.symbols[i:i + self.batch_size]
            logger.info(f"Batch {i//self.batch_size+1}: {len(batch)} symbols")
            for sym in batch:
                try:
                    raw = self.download_symbol_data(sym)
                    if raw is None or raw.empty:
                        continue
                    self.save_data(raw, sym, "raw")
                    proc = self.add_technical_indicators(raw, sym)
                    self.save_data(proc, sym, "processed")
                    all_data[sym] = proc
                    success += 1
                    logger.info(f"✅ {sym}: {len(proc)} rows, {len(proc.columns)} features")
                except Exception as e:
                    logger.error(f"❌ {sym}: {e}")
            if i + self.batch_size < len(self.symbols):
                logger.info(f"Sleeping {self.sleep_between:.1f}s between batches…")
                time.sleep(self.sleep_between)
        logger.info(f"Completed: {success}/{len(self.symbols)} symbols successful")
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

    def validate_data_quality(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for downloads")
    parser.add_argument("--sleep-between", type=float, default=1.0,
                        help="Sleep time between batches (seconds)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Maximum parallel workers")
    
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

    # Build configuration
    config = {
        "asset_types": asset_types,
        "exchanges": exchanges,
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": args.batch_size,
        "sleep_between": args.sleep_between,
        "max_workers": args.max_workers,
        "fred_api_key": args.fred_api_key,
        "alpha_vantage_api_key": args.alpha_vantage_api_key,
        "max_stocks_symbols": args.max_stocks_symbols,
        "max_etfs_symbols": args.max_etfs_symbols,
        "max_crypto_symbols": args.max_crypto_symbols,
        "max_forex_symbols": args.max_forex_symbols,
        "max_commodities_symbols": args.max_commodities_symbols,
    }

    logger.info(f"Asset types to collect: {[at.value for at in asset_types]}")
    logger.info(f"Exchanges to use: {exchanges}")
    logger.info(f"Date range: {start_date} → {end_date}")

    # Initialize collector
    collector = MultiAssetDataCollector(config, show_talib_fallback=args.show_talib_fallback)

    try:
        # Collect all data
        all_data = collector.collect_all_data()
        
        if not any(all_data.values()):
            logger.error("❌ No data collected successfully.")
            sys.exit(1)

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
        print("📊 WEALTHARENA MULTI-ASSET DATA COLLECTION SUMMARY")
        print("="*80)
        print(f"✅ Total symbols collected: {summary['successful_symbols']}")
        print(f"📅 Date range: {start_date} → {end_date}")
        print(f"📁 Raw data: {RAW_DIR}")
        print(f"📁 Processed data: {PROCESSED_DIR}")
        print(f"🧾 Summary: {summary_path.name}")
        print(f"🧾 Quality: {quality_path.name}")
        
        print(f"\n📈 ASSET TYPE BREAKDOWN:")
        for asset_type, data in all_data.items():
            if data:
                print(f"  {asset_type.value.replace('_', ' ').title()}: {len(data)} symbols")
        
        print(f"\n📊 DATA QUALITY: {quality['overall_quality'].upper()}")
        for asset_type, quality_info in quality['asset_quality'].items():
            print(f"  {asset_type}: {quality_info['average_quality']:.1f}% quality")
        
        print("\n🎉 Multi-Asset Data Collection Completed!")

    except Exception as e:
        logger.error(f"Data collection failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()