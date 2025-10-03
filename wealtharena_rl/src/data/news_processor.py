"""
News Processing and NLP Pipeline for WealthArena

This module handles news sentiment analysis, event extraction, and text embeddings
for integration with trading models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import requests
import json
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class NewsConfig:
    """Configuration for news processing"""
    # News sources
    news_sources: List[str] = None
    update_frequency_minutes: int = 15
    
    # NLP settings
    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 512
    embedding_dim: int = 768
    
    # Sentiment analysis
    sentiment_threshold: float = 0.1
    confidence_threshold: float = 0.7
    
    # Event extraction
    entity_types: List[str] = None
    event_keywords: List[str] = None
    
    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.marketwatch.com/marketwatch/topstories/",
                "https://feeds.bloomberg.com/markets/news.rss"
            ]
        
        if self.entity_types is None:
            self.entity_types = ["PERSON", "ORG", "GPE", "MONEY", "PERCENT"]
        
        if self.event_keywords is None:
            self.event_keywords = [
                "earnings", "merger", "acquisition", "dividend", "split",
                "guidance", "forecast", "upgrade", "downgrade", "initiate",
                "upgrade", "downgrade", "initiate", "coverage", "target",
                "beat", "miss", "exceed", "disappoint", "surprise"
            ]


class NewsProcessor:
    """News processing and NLP pipeline for WealthArena"""
    
    def __init__(self, config: NewsConfig = None):
        self.config = config or NewsConfig()
        
        # Initialize NLP models
        self._initialize_models()
        
        # News cache
        self.news_cache = {}
        self.last_update = None
        
        logger.info("News processor initialized")
    
    def _initialize_models(self):
        """Initialize NLP models for text processing"""
        try:
            # Load pre-trained model for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.eval()
            
            # Initialize TF-IDF vectorizer for keyword extraction
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info(f"NLP models loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            # Fallback to simple text processing
            self.model = None
            self.tokenizer = None
    
    def fetch_news(self, symbols: List[str] = None, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch recent news articles"""
        news_articles = []
        
        try:
            # Simulate news fetching (in production, use real news APIs)
            news_articles = self._simulate_news_fetch(symbols, hours_back)
            
            logger.info(f"Fetched {len(news_articles)} news articles")
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _simulate_news_fetch(self, symbols: List[str], hours_back: int) -> List[Dict[str, Any]]:
        """Simulate news fetching for demonstration"""
        # This would be replaced with real news API calls
        sample_news = [
            {
                "title": "Tech Stocks Rally on Strong Earnings",
                "content": "Major technology companies reported better-than-expected earnings, driving stock prices higher across the sector.",
                "timestamp": datetime.now() - timedelta(hours=2),
                "source": "Financial News",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "sentiment": 0.7,
                "relevance": 0.9
            },
            {
                "title": "Federal Reserve Hints at Interest Rate Cuts",
                "content": "The Federal Reserve indicated potential interest rate cuts in the coming months, affecting bond yields and currency markets.",
                "timestamp": datetime.now() - timedelta(hours=4),
                "source": "Market Watch",
                "symbols": ["^TNX", "USD"],
                "sentiment": 0.3,
                "relevance": 0.8
            },
            {
                "title": "Oil Prices Surge on Supply Concerns",
                "content": "Crude oil prices jumped 5% following reports of supply disruptions in major oil-producing regions.",
                "timestamp": datetime.now() - timedelta(hours=6),
                "source": "Energy News",
                "symbols": ["CL=F", "XLE"],
                "sentiment": 0.6,
                "relevance": 0.7
            }
        ]
        
        return sample_news
    
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract text embeddings using pre-trained model"""
        if self.model is None or self.tokenizer is None:
            # Fallback to simple TF-IDF embeddings
            return self._extract_tfidf_embeddings(texts)
        
        try:
            embeddings = []
            
            for text in texts:
                # Tokenize and encode
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True,
                    padding=True
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    embeddings.append(embedding.numpy())
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return self._extract_tfidf_embeddings(texts)
    
    def _extract_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF embeddings"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
        except Exception as e:
            logger.error(f"Error extracting TF-IDF embeddings: {e}")
            # Return random embeddings as last resort
            return np.random.randn(len(texts), 100)
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of news texts"""
        sentiments = []
        
        for text in texts:
            # Simple sentiment analysis based on keywords
            sentiment_score = self._calculate_sentiment_score(text)
            
            sentiments.append({
                "sentiment": sentiment_score,
                "confidence": abs(sentiment_score),
                "positive": sentiment_score > self.config.sentiment_threshold,
                "negative": sentiment_score < -self.config.sentiment_threshold
            })
        
        return sentiments
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score based on keywords"""
        positive_words = [
            "positive", "strong", "growth", "increase", "rise", "up", "beat",
            "exceed", "surprise", "optimistic", "bullish", "gain", "profit"
        ]
        
        negative_words = [
            "negative", "weak", "decline", "decrease", "fall", "down", "miss",
            "disappoint", "pessimistic", "bearish", "loss", "crash", "drop"
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract named entities from texts"""
        entities_list = []
        
        for text in texts:
            entities = self._extract_entities_simple(text)
            entities_list.append(entities)
        
        return entities_list
    
    def _extract_entities_simple(self, text: str) -> List[Dict[str, Any]]:
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Extract money amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?[BMK]?'
        for match in re.finditer(money_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "MONEY",
                "start": match.start(),
                "end": match.end()
            })
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(percent_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "PERCENT",
                "start": match.start(),
                "end": match.end()
            })
        
        # Extract company names (simple heuristic)
        company_keywords = ["Inc", "Corp", "Ltd", "LLC", "Company"]
        for keyword in company_keywords:
            pattern = r'\b[A-Z][a-zA-Z\s]+' + keyword + r'\b'
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group(),
                    "label": "ORG",
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities
    
    def extract_events(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract financial events from texts"""
        events_list = []
        
        for text in texts:
            events = self._extract_events_simple(text)
            events_list.append(events)
        
        return events_list
    
    def _extract_events_simple(self, text: str) -> List[Dict[str, Any]]:
        """Simple event extraction using keyword matching"""
        events = []
        text_lower = text.lower()
        
        for keyword in self.config.event_keywords:
            if keyword in text_lower:
                # Find context around the keyword
                start = max(0, text_lower.find(keyword) - 50)
                end = min(len(text), text_lower.find(keyword) + len(keyword) + 50)
                context = text[start:end]
                
                events.append({
                    "event_type": keyword,
                    "context": context,
                    "confidence": 0.8,  # Simple confidence score
                    "timestamp": datetime.now()
                })
        
        return events
    
    def process_news_batch(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of news articles"""
        if not news_articles:
            return {}
        
        # Extract texts
        titles = [article.get("title", "") for article in news_articles]
        contents = [article.get("content", "") for article in news_articles]
        combined_texts = [f"{title} {content}" for title, content in zip(titles, contents)]
        
        # Process texts
        embeddings = self.extract_embeddings(combined_texts)
        sentiments = self.analyze_sentiment(combined_texts)
        entities = self.extract_entities(combined_texts)
        events = self.extract_events(combined_texts)
        
        # Combine results
        processed_news = []
        for i, article in enumerate(news_articles):
            processed_article = {
                **article,
                "embedding": embeddings[i],
                "sentiment": sentiments[i],
                "entities": entities[i],
                "events": events[i],
                "processed_at": datetime.now()
            }
            processed_news.append(processed_article)
        
        return {
            "articles": processed_news,
            "summary": {
                "total_articles": len(processed_news),
                "avg_sentiment": np.mean([s["sentiment"] for s in sentiments]),
                "total_events": sum(len(e) for e in events),
                "total_entities": sum(len(e) for e in entities)
            }
        }
    
    def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get overall market sentiment for given symbols"""
        # Fetch recent news
        news_articles = self.fetch_news(symbols, hours_back=24)
        
        if not news_articles:
            return {}
        
        # Process news
        processed_news = self.process_news_batch(news_articles)
        
        # Calculate sentiment by symbol
        symbol_sentiments = {}
        for article in processed_news["articles"]:
            article_sentiment = article["sentiment"]["sentiment"]
            article_symbols = article.get("symbols", [])
            
            for symbol in article_symbols:
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = []
                symbol_sentiments[symbol].append(article_sentiment)
        
        # Average sentiment per symbol
        avg_sentiments = {}
        for symbol, sentiments in symbol_sentiments.items():
            avg_sentiments[symbol] = np.mean(sentiments)
        
        return avg_sentiments


def create_news_processor(config: NewsConfig = None) -> NewsProcessor:
    """Create a news processor instance"""
    return NewsProcessor(config)


def main():
    """Test the news processor"""
    logging.basicConfig(level=logging.INFO)
    
    print("📰 Testing News Processor...")
    print("="*40)
    
    # Create processor
    config = NewsConfig()
    processor = NewsProcessor(config)
    
    # Test with sample symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    print(f"\n🔍 Fetching news for symbols: {symbols}")
    news_articles = processor.fetch_news(symbols, hours_back=24)
    
    if news_articles:
        print(f"✅ Fetched {len(news_articles)} articles")
        
        # Process news
        print("\n🔄 Processing news...")
        processed_news = processor.process_news_batch(news_articles)
        
        print(f"✅ Processed {processed_news['summary']['total_articles']} articles")
        print(f"📊 Average sentiment: {processed_news['summary']['avg_sentiment']:.3f}")
        print(f"🎯 Total events: {processed_news['summary']['total_events']}")
        print(f"🏷️  Total entities: {processed_news['summary']['total_entities']}")
        
        # Test market sentiment
        print(f"\n📈 Market sentiment by symbol:")
        market_sentiment = processor.get_market_sentiment(symbols)
        for symbol, sentiment in market_sentiment.items():
            print(f"  {symbol}: {sentiment:.3f}")
    
    else:
        print("❌ No news articles fetched")
    
    print("\n✅ News processor test completed!")


if __name__ == "__main__":
    main()
