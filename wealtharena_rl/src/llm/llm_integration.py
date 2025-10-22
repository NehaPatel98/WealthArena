"""
LLM Integration Module for WealthArena RL

This module provides LLM-powered capabilities including:
- Event extraction from news and financial data
- Sentiment analysis and reasoning
- Explainable trade rationales
- Named entity recognition and linking
- Cross-modal signal fusion
- Natural language strategy descriptions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import re
import requests
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

class EventType(Enum):
    EARNINGS = "earnings"
    MERGER = "merger"
    DIVIDEND = "dividend"
    GUIDANCE = "guidance"
    REGULATORY = "regulatory"
    MARKET_EVENT = "market_event"
    NEWS = "news"

@dataclass
class Event:
    """Financial event representation"""
    event_id: str
    event_type: EventType
    title: str
    description: str
    entities: List[str]
    sentiment: float
    confidence: float
    timestamp: datetime
    source: str
    impact_score: float = 0.0

@dataclass
class TradeRationale:
    """Trade rationale explanation"""
    trade_id: str
    symbol: str
    action: str  # buy, sell, hold
    quantity: float
    price: float
    rationale: str
    supporting_evidence: List[str]
    risk_factors: List[str]
    confidence: float
    timestamp: datetime

class LLMConfig:
    """Configuration for LLM integration"""
    def __init__(self, 
                 provider: LLMProvider = LLMProvider.OPENAI,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 enable_caching: bool = True):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_caching = enable_caching

class EventExtractor:
    """Extract financial events from text using LLMs"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = {} if config.enable_caching else None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM models"""
        if self.config.provider == LLMProvider.HUGGINGFACE:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="ProsusAI/finbert")
                self.ner_pipeline = pipeline("ner", 
                                           model="dbmdz/bert-large-cased-finetuned-conll03-english")
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace models: {e}")
                self.sentiment_analyzer = None
                self.ner_pipeline = None
        else:
            self.sentiment_analyzer = None
            self.ner_pipeline = None
    
    def extract_events(self, 
                      text: str, 
                      source: str = "unknown",
                      timestamp: Optional[datetime] = None) -> List[Event]:
        """Extract financial events from text"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check cache first
        if self.cache and text in self.cache:
            return self.cache[text]
        
        events = []
        
        # Extract events using LLM
        if self.config.provider == LLMProvider.OPENAI:
            events = self._extract_events_openai(text, source, timestamp)
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            events = self._extract_events_huggingface(text, source, timestamp)
        else:
            events = self._extract_events_fallback(text, source, timestamp)
        
        # Cache results
        if self.cache is not None:
            self.cache[text] = events
        
        return events
    
    def _extract_events_openai(self, text: str, source: str, timestamp: datetime) -> List[Event]:
        """Extract events using OpenAI API"""
        try:
            prompt = f"""
            Extract financial events from the following text. For each event, provide:
            1. Event type (earnings, merger, dividend, guidance, regulatory, market_event, news)
            2. Title (brief summary)
            3. Description (detailed description)
            4. Entities (company names, tickers, people mentioned)
            5. Sentiment (-1 to 1, where -1 is very negative, 1 is very positive)
            6. Confidence (0 to 1)
            7. Impact score (0 to 1, how significant this event is)
            
            Text: {text}
            
            Return as JSON array of events.
            """
            
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            events_data = json.loads(response.choices[0].message.content)
            
            events = []
            for i, event_data in enumerate(events_data):
                event = Event(
                    event_id=f"{source}_{timestamp}_{i}",
                    event_type=EventType(event_data.get('event_type', 'news')),
                    title=event_data.get('title', ''),
                    description=event_data.get('description', ''),
                    entities=event_data.get('entities', []),
                    sentiment=float(event_data.get('sentiment', 0)),
                    confidence=float(event_data.get('confidence', 0.5)),
                    timestamp=timestamp,
                    source=source,
                    impact_score=float(event_data.get('impact_score', 0.5))
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"OpenAI event extraction failed: {e}")
            return []
    
    def _extract_events_huggingface(self, text: str, source: str, timestamp: datetime) -> List[Event]:
        """Extract events using HuggingFace models"""
        events = []
        
        # Basic sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_result = self.sentiment_analyzer(text)
                sentiment_score = sentiment_result[0]['score']
                if sentiment_result[0]['label'] == 'NEGATIVE':
                    sentiment_score = -sentiment_score
            except:
                sentiment_score = 0
        else:
            sentiment_score = 0
        
        # Named entity recognition
        entities = []
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                entities = [entity['word'] for entity in ner_results if entity['entity'] in ['B-ORG', 'I-ORG', 'B-PER', 'I-PER']]
            except:
                entities = []
        
        # Simple event detection based on keywords
        event_type = self._detect_event_type(text)
        
        event = Event(
            event_id=f"{source}_{timestamp}_0",
            event_type=event_type,
            title=text[:100] + "..." if len(text) > 100 else text,
            description=text,
            entities=entities,
            sentiment=sentiment_score,
            confidence=0.7,
            timestamp=timestamp,
            source=source,
            impact_score=0.5
        )
        events.append(event)
        
        return events
    
    def _extract_events_fallback(self, text: str, source: str, timestamp: datetime) -> List[Event]:
        """Fallback event extraction using simple rules"""
        events = []
        
        # Simple keyword-based event detection
        event_type = self._detect_event_type(text)
        
        # Basic sentiment analysis using keywords
        sentiment = self._calculate_sentiment_keywords(text)
        
        # Extract potential entities using regex
        entities = self._extract_entities_regex(text)
        
        event = Event(
            event_id=f"{source}_{timestamp}_0",
            event_type=event_type,
            title=text[:100] + "..." if len(text) > 100 else text,
            description=text,
            entities=entities,
            sentiment=sentiment,
            confidence=0.5,
            timestamp=timestamp,
            source=source,
            impact_score=0.5
        )
        events.append(event)
        
        return events
    
    def _detect_event_type(self, text: str) -> EventType:
        """Detect event type using keyword matching"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual']):
            return EventType.EARNINGS
        elif any(word in text_lower for word in ['merger', 'acquisition', 'takeover', 'buyout']):
            return EventType.MERGER
        elif any(word in text_lower for word in ['dividend', 'payout', 'yield']):
            return EventType.DIVIDEND
        elif any(word in text_lower for word in ['guidance', 'forecast', 'outlook', 'projection']):
            return EventType.GUIDANCE
        elif any(word in text_lower for word in ['regulation', 'regulatory', 'sec', 'fda', 'approval']):
            return EventType.REGULATORY
        elif any(word in text_lower for word in ['market', 'trading', 'price', 'stock', 'shares']):
            return EventType.MARKET_EVENT
        else:
            return EventType.NEWS
    
    def _calculate_sentiment_keywords(self, text: str) -> float:
        """Calculate sentiment using keyword matching"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'gain', 'up', 'rise', 'increase']
        negative_words = ['bad', 'terrible', 'negative', 'loss', 'decline', 'down', 'fall', 'decrease', 'drop', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1, min(1, sentiment))  # Clamp to [-1, 1]
    
    def _extract_entities_regex(self, text: str) -> List[str]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Stock tickers (3-5 uppercase letters)
        tickers = re.findall(r'\b[A-Z]{3,5}\b', text)
        entities.extend(tickers)
        
        # Company names (capitalized words)
        company_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(company_names)
        
        # Dollar amounts
        dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        entities.extend(dollar_amounts)
        
        return list(set(entities))  # Remove duplicates

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial text"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.sentence_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of financial text"""
        if self.config.provider == LLMProvider.OPENAI:
            return self._analyze_sentiment_openai(text)
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            return self._analyze_sentiment_huggingface(text)
        else:
            return self._analyze_sentiment_fallback(text)
    
    def _analyze_sentiment_openai(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using OpenAI"""
        try:
            prompt = f"""
            Analyze the sentiment of this financial text. Provide:
            1. Overall sentiment (-1 to 1)
            2. Confidence (0 to 1)
            3. Market impact sentiment (-1 to 1)
            4. Risk sentiment (-1 to 1)
            
            Text: {text}
            
            Return as JSON.
            """
            
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'sentiment': float(result.get('sentiment', 0)),
                'confidence': float(result.get('confidence', 0.5)),
                'market_impact': float(result.get('market_impact', 0)),
                'risk_sentiment': float(result.get('risk_sentiment', 0))
            }
            
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            return self._analyze_sentiment_fallback(text)
    
    def _analyze_sentiment_huggingface(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using HuggingFace models"""
        # This would use financial-specific models like FinBERT
        # For now, return basic analysis
        return self._analyze_sentiment_fallback(text)
    
    def _analyze_sentiment_fallback(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis"""
        # Simple keyword-based sentiment analysis
        sentiment = self._calculate_sentiment_keywords(text)
        
        return {
            'sentiment': sentiment,
            'confidence': 0.6,
            'market_impact': sentiment * 0.8,
            'risk_sentiment': -abs(sentiment)  # Higher absolute sentiment = higher risk
        }
    
    def _calculate_sentiment_keywords(self, text: str) -> float:
        """Calculate sentiment using keyword matching"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'gain', 'up', 'rise', 'increase', 'strong', 'robust', 'outperform']
        negative_words = ['bad', 'terrible', 'negative', 'loss', 'decline', 'down', 'fall', 'decrease', 'drop', 'crash', 'weak', 'poor', 'underperform']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1, min(1, sentiment))

class TradeRationaleGenerator:
    """Generate explainable trade rationales using LLMs"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate_rationale(self, 
                          trade_data: Dict[str, Any],
                          market_data: Dict[str, Any],
                          events: List[Event],
                          signals: Dict[str, float]) -> TradeRationale:
        """Generate trade rationale"""
        if self.config.provider == LLMProvider.OPENAI:
            return self._generate_rationale_openai(trade_data, market_data, events, signals)
        else:
            return self._generate_rationale_fallback(trade_data, market_data, events, signals)
    
    def _generate_rationale_openai(self, 
                                 trade_data: Dict[str, Any],
                                 market_data: Dict[str, Any],
                                 events: List[Event],
                                 signals: Dict[str, float]) -> TradeRationale:
        """Generate rationale using OpenAI"""
        try:
            prompt = f"""
            Generate a trade rationale for the following trade:
            
            Trade Data: {json.dumps(trade_data, indent=2)}
            Market Data: {json.dumps(market_data, indent=2)}
            Recent Events: {[event.title for event in events[-5:]]}
            Technical Signals: {json.dumps(signals, indent=2)}
            
            Provide:
            1. Clear rationale for the trade decision
            2. Supporting evidence from market data and events
            3. Risk factors to consider
            4. Confidence level (0 to 1)
            
            Return as JSON.
            """
            
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return TradeRationale(
                trade_id=trade_data.get('trade_id', 'unknown'),
                symbol=trade_data.get('symbol', 'unknown'),
                action=trade_data.get('action', 'hold'),
                quantity=trade_data.get('quantity', 0),
                price=trade_data.get('price', 0),
                rationale=result.get('rationale', 'No rationale available'),
                supporting_evidence=result.get('supporting_evidence', []),
                risk_factors=result.get('risk_factors', []),
                confidence=float(result.get('confidence', 0.5)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"OpenAI rationale generation failed: {e}")
            return self._generate_rationale_fallback(trade_data, market_data, events, signals)
    
    def _generate_rationale_fallback(self, 
                                   trade_data: Dict[str, Any],
                                   market_data: Dict[str, Any],
                                   events: List[Event],
                                   signals: Dict[str, float]) -> TradeRationale:
        """Fallback rationale generation"""
        symbol = trade_data.get('symbol', 'unknown')
        action = trade_data.get('action', 'hold')
        
        # Simple rule-based rationale
        rationale_parts = []
        
        if action == 'buy':
            rationale_parts.append(f"Buy signal for {symbol} based on technical analysis")
        elif action == 'sell':
            rationale_parts.append(f"Sell signal for {symbol} based on technical analysis")
        else:
            rationale_parts.append(f"Hold position in {symbol} - no clear signal")
        
        # Add signal information
        if signals:
            strong_signals = [k for k, v in signals.items() if abs(v) > 0.5]
            if strong_signals:
                rationale_parts.append(f"Strong signals: {', '.join(strong_signals)}")
        
        # Add event information
        if events:
            recent_events = [e for e in events if e.impact_score > 0.5]
            if recent_events:
                rationale_parts.append(f"Recent events: {', '.join([e.title for e in recent_events[:3]])}")
        
        rationale = ". ".join(rationale_parts)
        
        return TradeRationale(
            trade_id=trade_data.get('trade_id', 'unknown'),
            symbol=symbol,
            action=action,
            quantity=trade_data.get('quantity', 0),
            price=trade_data.get('price', 0),
            rationale=rationale,
            supporting_evidence=[],
            risk_factors=["Market volatility", "Liquidity risk"],
            confidence=0.6,
            timestamp=datetime.now()
        )

class CrossModalFusion:
    """Fuse numerical and textual signals"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.sentence_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models for cross-modal fusion"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def fuse_signals(self, 
                    numerical_signals: Dict[str, float],
                    textual_signals: List[str],
                    events: List[Event]) -> Dict[str, float]:
        """Fuse numerical and textual signals"""
        # Convert textual signals to embeddings
        if self.sentence_model and textual_signals:
            text_embeddings = self.sentence_model.encode(textual_signals)
            text_signal_strength = np.mean(np.linalg.norm(text_embeddings, axis=1))
        else:
            text_signal_strength = 0.5
        
        # Calculate event impact
        event_impact = self._calculate_event_impact(events)
        
        # Fuse signals
        fused_signals = {}
        
        for signal_name, signal_value in numerical_signals.items():
            # Combine numerical signal with textual and event signals
            fused_value = signal_value * 0.7 + text_signal_strength * 0.2 + event_impact * 0.1
            fused_signals[signal_name] = fused_value
        
        # Add cross-modal signals
        fused_signals['text_sentiment'] = text_signal_strength
        fused_signals['event_impact'] = event_impact
        fused_signals['cross_modal_confidence'] = self._calculate_fusion_confidence(
            numerical_signals, textual_signals, events
        )
        
        return fused_signals
    
    def _calculate_event_impact(self, events: List[Event]) -> float:
        """Calculate overall event impact"""
        if not events:
            return 0.0
        
        # Weight events by impact score and recency
        now = datetime.now()
        total_impact = 0.0
        total_weight = 0.0
        
        for event in events:
            # Recency weight (more recent = higher weight)
            time_diff = (now - event.timestamp).total_seconds() / 3600  # hours
            recency_weight = np.exp(-time_diff / 24)  # Decay over 24 hours
            
            # Combined weight
            weight = event.impact_score * recency_weight
            total_impact += event.sentiment * weight
            total_weight += weight
        
        return total_impact / total_weight if total_weight > 0 else 0.0
    
    def _calculate_fusion_confidence(self, 
                                   numerical_signals: Dict[str, float],
                                   textual_signals: List[str],
                                   events: List[Event]) -> float:
        """Calculate confidence in cross-modal fusion"""
        # Base confidence on signal consistency
        confidence = 0.5
        
        # Numerical signal consistency
        if numerical_signals:
            signal_values = list(numerical_signals.values())
            signal_std = np.std(signal_values)
            if signal_std < 0.1:  # Low variance = high consistency
                confidence += 0.2
        
        # Textual signal availability
        if textual_signals:
            confidence += 0.1
        
        # Event relevance
        if events:
            relevant_events = [e for e in events if e.impact_score > 0.3]
            if relevant_events:
                confidence += 0.2
        
        return min(1.0, confidence)

class LLMIntegrationManager:
    """Main LLM integration manager"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.event_extractor = EventExtractor(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.rationale_generator = TradeRationaleGenerator(config)
        self.cross_modal_fusion = CrossModalFusion(config)
    
    def process_market_data(self, 
                          market_data: Dict[str, Any],
                          news_data: List[str],
                          events: List[Event]) -> Dict[str, Any]:
        """Process market data with LLM capabilities"""
        # Extract events from news
        extracted_events = []
        for news_item in news_data:
            news_events = self.event_extractor.extract_events(news_item, "news")
            extracted_events.extend(news_events)
        
        # Analyze sentiment
        sentiment_analysis = {}
        for news_item in news_data:
            sentiment = self.sentiment_analyzer.analyze_sentiment(news_item)
            sentiment_analysis[news_item[:50]] = sentiment
        
        # Fuse signals
        numerical_signals = market_data.get('signals', {})
        textual_signals = news_data
        fused_signals = self.cross_modal_fusion.fuse_signals(
            numerical_signals, textual_signals, events + extracted_events
        )
        
        return {
            'extracted_events': extracted_events,
            'sentiment_analysis': sentiment_analysis,
            'fused_signals': fused_signals,
            'processed_timestamp': datetime.now()
        }
    
    def generate_trade_explanation(self, 
                                 trade_data: Dict[str, Any],
                                 market_context: Dict[str, Any]) -> TradeRationale:
        """Generate comprehensive trade explanation"""
        events = market_context.get('events', [])
        signals = market_context.get('signals', {})
        
        return self.rationale_generator.generate_rationale(
            trade_data, market_context, events, signals
        )
    
    def analyze_portfolio_risk(self, 
                             portfolio_data: Dict[str, Any],
                             market_events: List[Event]) -> Dict[str, Any]:
        """Analyze portfolio risk using LLM insights"""
        # Analyze sentiment of portfolio holdings
        holdings_sentiment = {}
        for symbol, holding_data in portfolio_data.get('holdings', {}).items():
            if 'news' in holding_data:
                sentiment = self.sentiment_analyzer.analyze_sentiment(holding_data['news'])
                holdings_sentiment[symbol] = sentiment
        
        # Analyze event impact on portfolio
        portfolio_impact = self._analyze_portfolio_event_impact(portfolio_data, market_events)
        
        # Generate risk insights
        risk_insights = self._generate_risk_insights(holdings_sentiment, portfolio_impact)
        
        return {
            'holdings_sentiment': holdings_sentiment,
            'portfolio_impact': portfolio_impact,
            'risk_insights': risk_insights,
            'analysis_timestamp': datetime.now()
        }
    
    def _analyze_portfolio_event_impact(self, 
                                      portfolio_data: Dict[str, Any],
                                      events: List[Event]) -> Dict[str, float]:
        """Analyze impact of events on portfolio"""
        holdings = portfolio_data.get('holdings', {})
        impact_scores = {}
        
        for symbol, holding_data in holdings.items():
            symbol_impact = 0.0
            
            for event in events:
                # Check if event mentions this symbol
                if symbol in event.entities or symbol.lower() in event.description.lower():
                    symbol_impact += event.impact_score * event.sentiment
            
            impact_scores[symbol] = symbol_impact
        
        return impact_scores
    
    def _generate_risk_insights(self, 
                              holdings_sentiment: Dict[str, Dict[str, float]],
                              portfolio_impact: Dict[str, float]) -> List[str]:
        """Generate risk insights from sentiment and event analysis"""
        insights = []
        
        # Analyze sentiment trends
        negative_sentiment_count = sum(1 for sentiment in holdings_sentiment.values() 
                                     if sentiment.get('sentiment', 0) < -0.3)
        
        if negative_sentiment_count > len(holdings_sentiment) * 0.3:
            insights.append("High concentration of negative sentiment in portfolio holdings")
        
        # Analyze event impact
        high_impact_symbols = [symbol for symbol, impact in portfolio_impact.items() 
                             if abs(impact) > 0.5]
        
        if high_impact_symbols:
            insights.append(f"High event impact detected for: {', '.join(high_impact_symbols)}")
        
        # Risk concentration
        if len(holdings_sentiment) < 5:
            insights.append("Low diversification - consider adding more holdings")
        
        return insights
