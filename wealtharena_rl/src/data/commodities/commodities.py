"""
Major Commodities Data

This module defines major global commodities for trading.
"""

# Major Commodities by Category
PRECIOUS_METALS = [
    "GOLD",      # Gold
    "SILVER",    # Silver
    "PLATINUM",  # Platinum
    "PALLADIUM"  # Palladium
]

ENERGY_COMMODITIES = [
    "CRUDE_OIL",     # WTI Crude Oil
    "BRENT_OIL",     # Brent Crude Oil
    "NATURAL_GAS",   # Natural Gas
    "HEATING_OIL",   # Heating Oil
    "GASOLINE"       # Gasoline
]

AGRICULTURAL_COMMODITIES = [
    "WHEAT",         # Wheat
    "CORN",          # Corn
    "SOYBEANS",      # Soybeans
    "SUGAR",         # Sugar
    "COFFEE",        # Coffee
    "COTTON",        # Cotton
    "COCOA"          # Cocoa
]

INDUSTRIAL_METALS = [
    "COPPER",        # Copper
    "ALUMINUM",      # Aluminum
    "ZINC",          # Zinc
    "NICKEL",        # Nickel
    "LEAD",          # Lead
    "TIN"            # Tin
]

LIVESTOCK = [
    "CATTLE",        # Live Cattle
    "HOGS",          # Lean Hogs
    "FEEDER_CATTLE"  # Feeder Cattle
]

# All major commodities
MAJOR_COMMODITIES = (
    PRECIOUS_METALS + 
    ENERGY_COMMODITIES + 
    AGRICULTURAL_COMMODITIES + 
    INDUSTRIAL_METALS + 
    LIVESTOCK
)

# Commodity categories for organization
COMMODITY_CATEGORIES = {
    "Precious_Metals": PRECIOUS_METALS,
    "Energy": ENERGY_COMMODITIES,
    "Agricultural": AGRICULTURAL_COMMODITIES,
    "Industrial_Metals": INDUSTRIAL_METALS,
    "Livestock": LIVESTOCK
}

# Commodity characteristics for trading
COMMODITY_CHARACTERISTICS = {
    # Precious Metals
    "GOLD": {
        "category": "Precious_Metals",
        "volatility": "Medium",
        "correlation": "Inverse to USD",
        "trading_hours": "24/7",
        "contract_size": "100 oz",
        "tick_size": 0.1,
        "margin_requirement": "High"
    },
    "SILVER": {
        "category": "Precious_Metals", 
        "volatility": "High",
        "correlation": "Similar to Gold",
        "trading_hours": "24/7",
        "contract_size": "5000 oz",
        "tick_size": 0.001,
        "margin_requirement": "High"
    },
    "PLATINUM": {
        "category": "Precious_Metals",
        "volatility": "High", 
        "correlation": "Industrial + Precious",
        "trading_hours": "24/7",
        "contract_size": "50 oz",
        "tick_size": 0.1,
        "margin_requirement": "High"
    },
    "PALLADIUM": {
        "category": "Precious_Metals",
        "volatility": "Very High",
        "correlation": "Auto Industry",
        "trading_hours": "24/7", 
        "contract_size": "100 oz",
        "tick_size": 0.05,
        "margin_requirement": "Very High"
    },
    
    # Energy
    "CRUDE_OIL": {
        "category": "Energy",
        "volatility": "Very High",
        "correlation": "Economic Growth",
        "trading_hours": "24/7",
        "contract_size": "1000 barrels",
        "tick_size": 0.01,
        "margin_requirement": "High"
    },
    "BRENT_OIL": {
        "category": "Energy",
        "volatility": "Very High", 
        "correlation": "Similar to WTI",
        "trading_hours": "24/7",
        "contract_size": "1000 barrels",
        "tick_size": 0.01,
        "margin_requirement": "High"
    },
    "NATURAL_GAS": {
        "category": "Energy",
        "volatility": "Extreme",
        "correlation": "Weather/Seasonal",
        "trading_hours": "24/7",
        "contract_size": "10000 MMBtu",
        "tick_size": 0.001,
        "margin_requirement": "High"
    },
    
    # Agricultural
    "WHEAT": {
        "category": "Agricultural",
        "volatility": "High",
        "correlation": "Weather/Supply",
        "trading_hours": "Limited",
        "contract_size": "5000 bushels",
        "tick_size": 0.25,
        "margin_requirement": "Medium"
    },
    "CORN": {
        "category": "Agricultural",
        "volatility": "High",
        "correlation": "Ethanol/Weather",
        "trading_hours": "Limited",
        "contract_size": "5000 bushels", 
        "tick_size": 0.25,
        "margin_requirement": "Medium"
    },
    "SOYBEANS": {
        "category": "Agricultural",
        "volatility": "High",
        "correlation": "China Demand",
        "trading_hours": "Limited",
        "contract_size": "5000 bushels",
        "tick_size": 0.25,
        "margin_requirement": "Medium"
    },
    "COFFEE": {
        "category": "Agricultural",
        "volatility": "Very High",
        "correlation": "Weather/Brazil",
        "trading_hours": "Limited",
        "contract_size": "37500 lbs",
        "tick_size": 0.05,
        "margin_requirement": "High"
    },
    
    # Industrial Metals
    "COPPER": {
        "category": "Industrial_Metals",
        "volatility": "High",
        "correlation": "Economic Growth",
        "trading_hours": "24/7",
        "contract_size": "25000 lbs",
        "tick_size": 0.0005,
        "margin_requirement": "Medium"
    },
    "ALUMINUM": {
        "category": "Industrial_Metals",
        "volatility": "Medium",
        "correlation": "China/Manufacturing",
        "trading_hours": "24/7",
        "contract_size": "25000 lbs",
        "tick_size": 0.0001,
        "margin_requirement": "Medium"
    }
}

def get_commodities_by_category(category: str) -> list:
    """Get commodities by category"""
    return COMMODITY_CATEGORIES.get(category, [])

def get_major_commodities() -> list:
    """Get all major commodities"""
    return MAJOR_COMMODITIES

def get_commodity_characteristics(symbol: str) -> dict:
    """Get characteristics for a specific commodity"""
    return COMMODITY_CHARACTERISTICS.get(symbol, {})

def get_high_volatility_commodities() -> list:
    """Get high volatility commodities for active trading"""
    high_vol = []
    for symbol, char in COMMODITY_CHARACTERISTICS.items():
        if char.get("volatility") in ["High", "Very High", "Extreme"]:
            high_vol.append(symbol)
    return high_vol

def get_24_7_trading_commodities() -> list:
    """Get commodities that trade 24/7"""
    continuous = []
    for symbol, char in COMMODITY_CHARACTERISTICS.items():
        if char.get("trading_hours") == "24/7":
            continuous.append(symbol)
    return continuous
