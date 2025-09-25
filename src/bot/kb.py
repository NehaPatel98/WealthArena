"""
WealthArena Trading Bot - Knowledge Base
Educational content for the chatbot
"""

# Knowledge Base Sections
KNOWLEDGE_BASE = {
    "basics": {
        "title": "Trading Basics",
        "content": """
        Trading is the act of buying and selling financial instruments like stocks, bonds, or cryptocurrencies with the goal of making a profit.
        
        Market Types:
        - Stock Market: Where shares of companies are traded
        - Forex Market: Where currencies are exchanged  
        - Crypto Market: Where digital currencies are traded
        
        Order Types:
        - Market Order: Buy/sell at current market price
        - Limit Order: Buy/sell at a specific price or better
        - Stop Loss: Automatically sell if price drops below a level
        
        Risk Management:
        Never risk more than you can afford to lose. Use position sizing and stop losses to protect your capital.
        """,
        "keywords": ["trading", "basics", "market", "order", "risk", "profit", "stocks", "crypto", "forex"]
    },
    
    "indicators": {
        "title": "Technical Indicators",
        "content": """
        Moving Averages:
        Moving averages smooth out price data to identify trends. SMA (Simple) and EMA (Exponential) are common types.
        - SMA(20): Simple Moving Average over 20 periods
        - EMA(12): Exponential Moving Average over 12 periods
        
        RSI (Relative Strength Index):
        RSI measures momentum on a scale of 0-100. Values below 30 indicate oversold conditions, above 70 indicate overbought.
        - RSI < 30: Oversold, potential buying opportunity
        - RSI > 70: Overbought, potential selling opportunity
        - RSI 40-60: Neutral zone
        
        Support and Resistance:
        Support is a price level where buying interest is strong. Resistance is where selling pressure is strong.
        
        Trend Analysis:
        Identify the overall direction of price movement using trend lines and moving averages.
        """,
        "keywords": ["rsi", "sma", "ema", "moving", "average", "indicator", "technical", "analysis", "trend", "support", "resistance", "oversold", "overbought"]
    },
    
    "risk": {
        "title": "Risk Management",
        "content": """
        Position Sizing:
        Only risk a small percentage of your account per trade (typically 1-2%). This protects against large losses.
        
        Stop Losses:
        Always set a stop loss to limit potential losses. Never let a losing trade run indefinitely.
        
        Diversification:
        Don't put all your money in one asset. Spread risk across different investments.
        
        Risk-Reward Ratios:
        Only take trades where potential profit is at least 2x the potential loss.
        
        Key Rules:
        - Never risk more than 1-2% of account per trade
        - Always use stop losses
        - Diversify your portfolio
        - Maintain proper risk-reward ratios
        """,
        "keywords": ["risk", "management", "position", "sizing", "stop", "loss", "diversification", "portfolio", "reward", "ratio", "1%", "2%"]
    },
    
    "paper_trading": {
        "title": "Paper Trading",
        "content": """
        Paper Trading:
        Practice trading with virtual money before risking real capital. This is essential for learning.
        
        Benefits:
        - Learn without financial risk
        - Test strategies safely
        - Understand market behavior
        - Practice risk management
        
        Always start with paper trading before using real money.
        """,
        "keywords": ["paper", "trading", "virtual", "practice", "learn", "safe", "risk", "free"]
    }
}

def get_knowledge_sections():
    """Get all knowledge base sections."""
    return KNOWLEDGE_BASE

def search_knowledge_base(query: str, top_k: int = 3):
    """
    Search knowledge base for relevant sections based on keyword overlap.
    
    Args:
        query: User's question
        top_k: Number of top sections to return
        
    Returns:
        List of relevant knowledge sections
    """
    query_lower = query.lower()
    scored_sections = []
    
    for section_id, section in KNOWLEDGE_BASE.items():
        score = 0
        keywords = section["keywords"]
        
        # Count keyword matches
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        
        # Also check for partial matches in content
        content_lower = section["content"].lower()
        for word in query_lower.split():
            if len(word) > 3 and word in content_lower:
                score += 0.5
        
        if score > 0:
            scored_sections.append((score, section_id, section))
    
    # Sort by score and return top_k
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    return [section for _, _, section in scored_sections[:top_k]]

def get_system_prompt():
    """Get the system prompt for LLM providers."""
    return """
    You are a trading educator for WealthArena, an educational trading platform. Your role is to teach trading concepts clearly and safely.
    
    Guidelines:
    - Be educational and informative, never give financial advice
    - Explain "why" behind trading concepts using examples
    - Always mention risks and suggest paper trading first
    - Use simple, clear language
    - If asked about specific assets, provide educational analysis only
    - Never guarantee profits or specific price targets
    - Always suggest proper risk management (1-2% position sizing, stop losses)
    
    Prohibited:
    - Real financial advice or guarantees
    - Specific buy/sell recommendations
    - Promises of profits
    - Real money trading suggestions
    
    Focus on education, risk awareness, and paper trading practice.
    """
