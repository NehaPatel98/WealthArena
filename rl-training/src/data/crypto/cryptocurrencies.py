"""
Cryptocurrency Database for WealthArena Trading System

This module contains major global cryptocurrencies for crypto trading.
"""

# Major cryptocurrencies (top 20 by market cap)
MAJOR_CRYPTOCURRENCIES = [
    "BTC",   # Bitcoin
    "ETH",   # Ethereum
    "BNB",   # Binance Coin
    "XRP",   # Ripple
    "ADA",   # Cardano
    "SOL",   # Solana
    "DOGE",  # Dogecoin
    "DOT",   # Polkadot
    "AVAX",  # Avalanche
    "SHIB",  # Shiba Inu
    "MATIC", # Polygon
    "LTC",   # Litecoin
    "UNI",   # Uniswap
    "LINK",  # Chainlink
    "ATOM",  # Cosmos
    "ALGO",  # Algorand
    "VET",   # VeChain
    "FIL",   # Filecoin
    "TRX",   # TRON
    "ETC"    # Ethereum Classic
]

# DeFi tokens
DEFI_TOKENS = [
    "UNI",   # Uniswap
    "AAVE",  # Aave
    "COMP",  # Compound
    "MKR",   # Maker
    "SNX",   # Synthetix
    "YFI",   # Yearn Finance
    "CRV",   # Curve
    "1INCH", # 1inch
    "SUSHI", # SushiSwap
    "CAKE"   # PancakeSwap
]

# Layer 1 blockchains
LAYER_1_TOKENS = [
    "BTC",   # Bitcoin
    "ETH",   # Ethereum
    "ADA",   # Cardano
    "SOL",   # Solana
    "DOT",   # Polkadot
    "AVAX",  # Avalanche
    "ATOM",  # Cosmos
    "ALGO",  # Algorand
    "NEAR",  # NEAR Protocol
    "FTM"    # Fantom
]

# Layer 2 solutions
LAYER_2_TOKENS = [
    "MATIC", # Polygon
    "OP",    # Optimism
    "ARB",   # Arbitrum
    "LRC",   # Loopring
    "IMX",   # Immutable X
    "ZKS",   # zkSync
    "BOBA",  # Boba Network
    "METIS"  # Metis
]

# Meme coins
MEME_COINS = [
    "DOGE",  # Dogecoin
    "SHIB",  # Shiba Inu
    "PEPE",  # Pepe
    "FLOKI", # Floki
    "BONK",  # Bonk
    "WIF",   # dogwifhat
    "BABYDOGE", # Baby Doge
    "ELON"   # Dogelon Mars
]

# Stablecoins
STABLECOINS = [
    "USDT",  # Tether
    "USDC",  # USD Coin
    "BUSD",  # Binance USD
    "DAI",   # Dai
    "FRAX",  # Frax
    "TUSD",  # TrueUSD
    "USDP",  # Pax Dollar
    "GUSD"   # Gemini Dollar
]

# Exchange tokens
EXCHANGE_TOKENS = [
    "BNB",   # Binance Coin
    "FTT",   # FTX Token
    "KCS",   # KuCoin Token
    "HT",    # Huobi Token
    "OKB",   # OKB
    "LEO",   # LEO Token
    "CRO",   # Cronos
    "GT"     # GateToken
]

# Crypto categories
CRYPTO_CATEGORIES = {
    "Major_Cryptocurrencies": MAJOR_CRYPTOCURRENCIES,
    "DeFi_Tokens": DEFI_TOKENS,
    "Layer_1_Tokens": LAYER_1_TOKENS,
    "Layer_2_Tokens": LAYER_2_TOKENS,
    "Meme_Coins": MEME_COINS,
    "Stablecoins": STABLECOINS,
    "Exchange_Tokens": EXCHANGE_TOKENS
}

# Cryptocurrency characteristics
CRYPTO_INFO = {
    "BTC": {"name": "Bitcoin", "category": "Store of Value", "volatility": "High", "market_cap_rank": 1},
    "ETH": {"name": "Ethereum", "category": "Smart Contract", "volatility": "High", "market_cap_rank": 2},
    "BNB": {"name": "Binance Coin", "category": "Exchange Token", "volatility": "High", "market_cap_rank": 3},
    "XRP": {"name": "Ripple", "category": "Payment", "volatility": "High", "market_cap_rank": 4},
    "ADA": {"name": "Cardano", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 5},
    "SOL": {"name": "Solana", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 6},
    "DOGE": {"name": "Dogecoin", "category": "Meme Coin", "volatility": "Extreme", "market_cap_rank": 7},
    "DOT": {"name": "Polkadot", "category": "Interoperability", "volatility": "Very High", "market_cap_rank": 8},
    "AVAX": {"name": "Avalanche", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 9},
    "SHIB": {"name": "Shiba Inu", "category": "Meme Coin", "volatility": "Extreme", "market_cap_rank": 10},
    "MATIC": {"name": "Polygon", "category": "Layer 2", "volatility": "Very High", "market_cap_rank": 11},
    "LTC": {"name": "Litecoin", "category": "Payment", "volatility": "High", "market_cap_rank": 12},
    "UNI": {"name": "Uniswap", "category": "DeFi", "volatility": "Very High", "market_cap_rank": 13},
    "LINK": {"name": "Chainlink", "category": "Oracle", "volatility": "Very High", "market_cap_rank": 14},
    "ATOM": {"name": "Cosmos", "category": "Interoperability", "volatility": "Very High", "market_cap_rank": 15},
    "ALGO": {"name": "Algorand", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 16},
    "VET": {"name": "VeChain", "category": "Supply Chain", "volatility": "Very High", "market_cap_rank": 17},
    "FIL": {"name": "Filecoin", "category": "Storage", "volatility": "Very High", "market_cap_rank": 18},
    "TRX": {"name": "TRON", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 19},
    "ETC": {"name": "Ethereum Classic", "category": "Smart Contract", "volatility": "Very High", "market_cap_rank": 20}
}

# Volatility tiers
VOLATILITY_TIERS = {
    "Low_Volatility": ["BTC", "ETH", "LTC"],
    "Medium_Volatility": ["BNB", "XRP", "DOT", "LINK"],
    "High_Volatility": ["ADA", "SOL", "AVAX", "UNI", "ATOM", "ALGO", "VET", "FIL", "TRX", "ETC"],
    "Extreme_Volatility": ["DOGE", "SHIB", "MATIC"]
}

# Market cap tiers
MARKET_CAP_TIERS = {
    "Large_Cap": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL"],
    "Mid_Cap": ["DOGE", "DOT", "AVAX", "SHIB", "MATIC", "LTC", "UNI", "LINK"],
    "Small_Cap": ["ATOM", "ALGO", "VET", "FIL", "TRX", "ETC"]
}

def get_cryptocurrencies_by_category(category: str) -> list:
    """Get cryptocurrencies by category"""
    return CRYPTO_CATEGORIES.get(category, [])

def get_cryptocurrencies_by_volatility(tier: str) -> list:
    """Get cryptocurrencies by volatility tier"""
    return VOLATILITY_TIERS.get(tier, [])

def get_cryptocurrencies_by_market_cap(tier: str) -> list:
    """Get cryptocurrencies by market cap tier"""
    return MARKET_CAP_TIERS.get(tier, [])

def get_major_cryptocurrencies() -> list:
    """Get major cryptocurrencies"""
    return MAJOR_CRYPTOCURRENCIES.copy()

def get_crypto_info(symbol: str) -> dict:
    """Get information about a cryptocurrency"""
    return CRYPTO_INFO.get(symbol, {})
