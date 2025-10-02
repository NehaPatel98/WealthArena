"""
Currency Pairs Database for WealthArena Trading System

This module contains major currency pairs for forex trading.
"""

# Major currency pairs (8 currencies = 28 pairs)
MAJOR_CURRENCIES = ["USD", "GBP", "CHF", "AUD", "JPY", "NZD", "CAD", "EUR"]

# All possible pairs
CURRENCY_PAIRS = [
    # USD pairs
    "USDGBP", "USDCHF", "USDAUD", "USDJPY", "USDNZD", "USDCAD", "USDEUR",
    # GBP pairs
    "GBPCHF", "GBPAUD", "GBPJPY", "GBPNZD", "GBPCAD", "GBPEUR",
    # CHF pairs
    "CHFAUD", "CHFJPY", "CHFNZD", "CHFCAD", "CHFEUR",
    # AUD pairs
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDEUR",
    # JPY pairs
    "JPYNZD", "JPYCAD", "JPYEUR",
    # NZD pairs
    "NZDCAD", "NZDEUR",
    # CAD pairs
    "CADEUR"
]

# Major pairs (most liquid)
MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
]

# Minor pairs
MINOR_PAIRS = [
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "CHFJPY", "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDCAD", "NZDCHF", "NZDJPY", "CADCHF", "CADJPY"
]

# Exotic pairs (less liquid)
EXOTIC_PAIRS = [
    "USDTRY", "USDZAR", "USDMXN", "USDBRL", "USDRUB", "USDINR",
    "EURTRY", "EURZAR", "EURMXN", "EURBRL", "EURRUB", "EURINR"
]

# Currency categories
CURRENCY_CATEGORIES = {
    "Major_Pairs": MAJOR_PAIRS,
    "Minor_Pairs": MINOR_PAIRS,
    "Exotic_Pairs": EXOTIC_PAIRS,
    "All_Pairs": CURRENCY_PAIRS
}

# Currency characteristics
CURRENCY_INFO = {
    "USD": {"name": "US Dollar", "region": "North America", "volatility": "Medium"},
    "GBP": {"name": "British Pound", "region": "Europe", "volatility": "High"},
    "CHF": {"name": "Swiss Franc", "region": "Europe", "volatility": "Low"},
    "AUD": {"name": "Australian Dollar", "region": "Oceania", "volatility": "High"},
    "JPY": {"name": "Japanese Yen", "region": "Asia", "volatility": "Medium"},
    "NZD": {"name": "New Zealand Dollar", "region": "Oceania", "volatility": "High"},
    "CAD": {"name": "Canadian Dollar", "region": "North America", "volatility": "Medium"},
    "EUR": {"name": "Euro", "region": "Europe", "volatility": "Medium"}
}

def get_currency_pairs_by_category(category: str) -> list:
    """Get currency pairs by category"""
    return CURRENCY_CATEGORIES.get(category, [])

def get_major_currencies() -> list:
    """Get major currencies"""
    return MAJOR_CURRENCIES.copy()

def get_currency_info(currency: str) -> dict:
    """Get information about a currency"""
    return CURRENCY_INFO.get(currency, {})
