"""
WealthArena Chatbot Integration
Connects to the existing wealtharena_chatbot service for LLM capabilities
"""

import httpx
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

class ChatbotIntegration:
    """Integration with the existing WealthArena chatbot service"""
    
    def __init__(self, chatbot_url: str = "http://localhost:8000"):
        self.chatbot_url = chatbot_url
        self.logger = logging.getLogger(__name__)
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text using the existing chatbot service
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.chatbot_url}/v1/chat",
                    json={
                        "message": f"analyze: {text}",
                        "user_id": "rl_system",
                        "context": "sentiment_analysis"
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                "reply": "Sentiment analysis unavailable",
                "tools_used": [],
                "trace_id": "error"
            }
    
    async def explain_trade_rationale(self, symbol: str, indicators: Dict[str, Any]) -> str:
        """
        Get explainable trade rationale using the existing chatbot service
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators data
            
        Returns:
            Explanatory text for the trade rationale
        """
        try:
            # Format indicators for the chatbot
            indicators_text = ", ".join([f"{k}: {v}" for k, v in indicators.items()])
            question = f"What does the technical analysis for {symbol} indicate? Indicators: {indicators_text}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.chatbot_url}/v1/explain",
                    json={
                        "question": question,
                        "k": 3
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("answer", "Unable to generate explanation")
        except Exception as e:
            self.logger.error(f"Trade rationale explanation failed: {e}")
            return f"Based on technical analysis of {symbol}, the indicators suggest market conditions that warrant careful consideration."
    
    async def get_trade_setup(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get structured trade setup using the existing chatbot service
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Trade setup card or None
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.chatbot_url}/v1/chat",
                    json={
                        "message": f"/setup for {symbol}",
                        "user_id": "rl_system",
                        "context": "trade_setup"
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("card")
        except Exception as e:
            self.logger.error(f"Trade setup generation failed: {e}")
            return None
    
    async def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base using the existing chatbot service
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.chatbot_url}/v1/search",
                    params={"q": query, "k": k}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def get_market_data(self, symbol: str, period: str = "1d") -> Optional[Dict[str, Any]]:
        """
        Get market data using the existing chatbot service
        
        Args:
            symbol: Trading symbol
            period: Data period
            
        Returns:
            Market data or None
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get OHLC data
                ohlc_response = await client.get(
                    f"{self.chatbot_url}/v1/market/ohlc",
                    params={"symbol": symbol, "period": period}
                )
                ohlc_response.raise_for_status()
                ohlc_data = ohlc_response.json()
                
                # Get quote data
                quote_response = await client.get(
                    f"{self.chatbot_url}/v1/market/quote",
                    params={"symbol": symbol}
                )
                quote_response.raise_for_status()
                quote_data = quote_response.json()
                
                return {
                    "ohlc": ohlc_data,
                    "quote": quote_data
                }
        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            return None
    
    async def generate_news_insights(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate insights from news articles using sentiment analysis
        
        Args:
            news_articles: List of news articles
            
        Returns:
            Dictionary with news insights
        """
        try:
            insights = {
                "total_articles": len(news_articles),
                "sentiment_scores": [],
                "overall_sentiment": "neutral",
                "key_insights": []
            }
            
            # Analyze sentiment for each article
            for article in news_articles:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text_to_analyze = f"{title} {summary}".strip()
                
                if text_to_analyze:
                    sentiment_result = await self.analyze_sentiment(text_to_analyze)
                    insights["sentiment_scores"].append({
                        "title": title,
                        "sentiment": sentiment_result.get("reply", ""),
                        "url": article.get("url", "")
                    })
            
            # Calculate overall sentiment
            if insights["sentiment_scores"]:
                positive_count = sum(1 for s in insights["sentiment_scores"] if "positive" in s["sentiment"].lower())
                negative_count = sum(1 for s in insights["sentiment_scores"] if "negative" in s["sentiment"].lower())
                
                if positive_count > negative_count:
                    insights["overall_sentiment"] = "positive"
                elif negative_count > positive_count:
                    insights["overall_sentiment"] = "negative"
            
            # Generate key insights
            insights["key_insights"] = [
                f"Analyzed {len(news_articles)} news articles",
                f"Overall market sentiment: {insights['overall_sentiment']}",
                "Consider sentiment in your trading decisions"
            ]
            
            return insights
            
        except Exception as e:
            self.logger.error(f"News insights generation failed: {e}")
            return {
                "total_articles": len(news_articles),
                "sentiment_scores": [],
                "overall_sentiment": "neutral",
                "key_insights": ["Unable to analyze news sentiment"]
            }

# Global instance for reuse
_chatbot_integration = None

def get_chatbot_integration() -> ChatbotIntegration:
    """Get or create the global chatbot integration instance"""
    global _chatbot_integration
    if _chatbot_integration is None:
        _chatbot_integration = ChatbotIntegration()
    return _chatbot_integration
