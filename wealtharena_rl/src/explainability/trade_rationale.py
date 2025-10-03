"""
Trade Rationale and Explainability Module for WealthArena

This module provides explainable AI capabilities for trading decisions,
including trade rationales, decision provenance, and audit trails.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import json
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Decision type enumeration"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TradeRationale:
    """Trade rationale and explanation"""
    trade_id: str
    timestamp: datetime
    symbol: str
    action: DecisionType
    quantity: float
    price: float
    
    # Rationale components
    primary_reason: str
    supporting_evidence: List[str]
    technical_signals: Dict[str, float]
    sentiment_signals: Dict[str, float]
    risk_factors: List[str]
    
    # Confidence and metadata
    confidence_level: ConfidenceLevel
    confidence_score: float
    model_version: str
    data_snapshot_id: str
    
    # Performance tracking
    expected_return: float
    expected_risk: float
    position_size_rationale: str


@dataclass
class DecisionProvenance:
    """Decision provenance tracking"""
    decision_id: str
    timestamp: datetime
    agent_id: str
    model_version: str
    
    # Input data
    market_data_snapshot: Dict[str, Any]
    news_data_snapshot: Dict[str, Any]
    technical_indicators: Dict[str, float]
    sentiment_scores: Dict[str, float]
    
    # Model internals
    feature_importance: Dict[str, float]
    attention_weights: Optional[Dict[str, float]]
    hidden_states: Optional[List[float]]
    
    # Decision process
    decision_factors: List[str]
    alternative_actions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]


class TradeRationaleGenerator:
    """Generates trade rationales and explanations"""
    
    def __init__(self):
        self.rationale_templates = self._load_rationale_templates()
        self.technical_explanations = self._load_technical_explanations()
        self.sentiment_explanations = self._load_sentiment_explanations()
        
        logger.info("Trade rationale generator initialized")
    
    def _load_rationale_templates(self) -> Dict[str, str]:
        """Load rationale templates for different scenarios"""
        return {
            "technical_buy": "Technical analysis indicates strong bullish momentum with {indicator} showing {value}, suggesting potential upward price movement.",
            "technical_sell": "Technical indicators suggest bearish conditions with {indicator} at {value}, indicating potential downward pressure.",
            "sentiment_buy": "Positive market sentiment of {sentiment_score:.2f} based on recent news and analyst reports supports a bullish outlook.",
            "sentiment_sell": "Negative market sentiment of {sentiment_score:.2f} from recent developments suggests caution is warranted.",
            "momentum_buy": "Strong momentum signals with {indicator} trending upward, indicating continued price appreciation potential.",
            "momentum_sell": "Momentum indicators show weakening trend with {indicator} declining, suggesting potential reversal.",
            "risk_management": "Risk management rules triggered: {risk_factor} exceeds threshold of {threshold}, requiring position adjustment.",
            "rebalancing": "Portfolio rebalancing required to maintain target allocation of {target_allocation}% in {asset_class}."
        }
    
    def _load_technical_explanations(self) -> Dict[str, str]:
        """Load technical indicator explanations"""
        return {
            "rsi": "RSI (Relative Strength Index) measures momentum. Values above 70 indicate overbought conditions, below 30 indicate oversold.",
            "macd": "MACD (Moving Average Convergence Divergence) shows trend changes. Positive values suggest bullish momentum, negative values bearish.",
            "bollinger": "Bollinger Bands show price volatility. Prices near upper band suggest overbought, near lower band suggest oversold.",
            "sma": "Simple Moving Average shows trend direction. Price above SMA suggests uptrend, below suggests downtrend.",
            "volume": "Volume confirms price movements. High volume with price increase suggests strong buying pressure.",
            "atr": "ATR (Average True Range) measures volatility. High ATR suggests high volatility, low ATR suggests stability."
        }
    
    def _load_sentiment_explanations(self) -> Dict[str, str]:
        """Load sentiment analysis explanations"""
        return {
            "news_sentiment": "News sentiment analysis based on recent headlines and articles mentioning the asset.",
            "social_sentiment": "Social media sentiment from Twitter, Reddit, and other platforms discussing the asset.",
            "analyst_sentiment": "Analyst sentiment from research reports, upgrades, downgrades, and price target changes.",
            "earnings_sentiment": "Earnings sentiment based on recent earnings reports and guidance updates.",
            "market_sentiment": "Overall market sentiment affecting all assets in the portfolio."
        }
    
    def generate_trade_rationale(self,
                               trade_data: Dict[str, Any],
                               market_data: Dict[str, Any],
                               technical_signals: Dict[str, float],
                               sentiment_signals: Dict[str, float],
                               model_metadata: Dict[str, Any]) -> TradeRationale:
        """Generate comprehensive trade rationale"""
        
        try:
            # Extract trade information
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = DecisionType(trade_data.get('action', 'hold'))
            quantity = trade_data.get('quantity', 0.0)
            price = trade_data.get('price', 0.0)
            
            # Generate primary reason
            primary_reason = self._generate_primary_reason(
                action, technical_signals, sentiment_signals
            )
            
            # Generate supporting evidence
            supporting_evidence = self._generate_supporting_evidence(
                technical_signals, sentiment_signals, market_data
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                technical_signals, sentiment_signals, market_data
            )
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(
                technical_signals, sentiment_signals, market_data
            )
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Generate position size rationale
            position_size_rationale = self._generate_position_size_rationale(
                quantity, technical_signals, sentiment_signals
            )
            
            # Create rationale
            rationale = TradeRationale(
                trade_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                primary_reason=primary_reason,
                supporting_evidence=supporting_evidence,
                technical_signals=technical_signals,
                sentiment_signals=sentiment_signals,
                risk_factors=risk_factors,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                model_version=model_metadata.get('model_version', 'unknown'),
                data_snapshot_id=model_metadata.get('data_snapshot_id', 'unknown'),
                expected_return=self._calculate_expected_return(technical_signals, sentiment_signals),
                expected_risk=self._calculate_expected_risk(technical_signals, sentiment_signals),
                position_size_rationale=position_size_rationale
            )
            
            logger.info(f"Generated trade rationale for {symbol} {action.value}")
            return rationale
            
        except Exception as e:
            logger.error(f"Error generating trade rationale: {e}")
            raise
    
    def _generate_primary_reason(self,
                               action: DecisionType,
                               technical_signals: Dict[str, float],
                               sentiment_signals: Dict[str, float]) -> str:
        """Generate primary reason for the trade decision"""
        
        # Find strongest technical signal
        strongest_technical = max(technical_signals.items(), key=lambda x: abs(x[1])) if technical_signals else None
        strongest_sentiment = max(sentiment_signals.items(), key=lambda x: abs(x[1])) if sentiment_signals else None
        
        reasons = []
        
        # Technical reasons
        if strongest_technical:
            indicator, value = strongest_technical
            if action == DecisionType.BUY and value > 0:
                reasons.append(self.rationale_templates["technical_buy"].format(
                    indicator=indicator.upper(), value=f"{value:.3f}"
                ))
            elif action == DecisionType.SELL and value < 0:
                reasons.append(self.rationale_templates["technical_sell"].format(
                    indicator=indicator.upper(), value=f"{value:.3f}"
                ))
        
        # Sentiment reasons
        if strongest_sentiment:
            sentiment_type, score = strongest_sentiment
            if action == DecisionType.BUY and score > 0:
                reasons.append(self.rationale_templates["sentiment_buy"].format(
                    sentiment_score=score
                ))
            elif action == DecisionType.SELL and score < 0:
                reasons.append(self.rationale_templates["sentiment_sell"].format(
                    sentiment_score=score
                ))
        
        # Default reason if no specific signals
        if not reasons:
            if action == DecisionType.BUY:
                reasons.append("Portfolio optimization suggests increasing position based on current market conditions.")
            elif action == DecisionType.SELL:
                reasons.append("Risk management suggests reducing position based on current market conditions.")
            else:
                reasons.append("Current market conditions suggest maintaining current position.")
        
        return " ".join(reasons)
    
    def _generate_supporting_evidence(self,
                                    technical_signals: Dict[str, float],
                                    sentiment_signals: Dict[str, float],
                                    market_data: Dict[str, Any]) -> List[str]:
        """Generate supporting evidence for the trade decision"""
        
        evidence = []
        
        # Technical evidence
        for indicator, value in technical_signals.items():
            if abs(value) > 0.1:  # Significant signal
                explanation = self.technical_explanations.get(indicator, f"{indicator} shows {value:.3f}")
                evidence.append(f"{indicator.upper()}: {explanation}")
        
        # Sentiment evidence
        for sentiment_type, score in sentiment_signals.items():
            if abs(score) > 0.1:  # Significant sentiment
                explanation = self.sentiment_explanations.get(sentiment_type, f"{sentiment_type} sentiment: {score:.3f}")
                evidence.append(explanation)
        
        # Market data evidence
        if 'volume' in market_data and market_data['volume'] > 0:
            evidence.append(f"Trading volume: {market_data['volume']:,.0f} shares")
        
        if 'volatility' in market_data:
            evidence.append(f"Price volatility: {market_data['volatility']:.3f}")
        
        return evidence
    
    def _identify_risk_factors(self,
                             technical_signals: Dict[str, float],
                             sentiment_signals: Dict[str, float],
                             market_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors for the trade"""
        
        risk_factors = []
        
        # Technical risks
        if 'rsi' in technical_signals and technical_signals['rsi'] > 70:
            risk_factors.append("RSI indicates overbought conditions")
        elif 'rsi' in technical_signals and technical_signals['rsi'] < 30:
            risk_factors.append("RSI indicates oversold conditions")
        
        if 'volatility' in market_data and market_data['volatility'] > 0.05:
            risk_factors.append("High volatility increases price risk")
        
        # Sentiment risks
        if sentiment_signals:
            avg_sentiment = np.mean(list(sentiment_signals.values()))
            if avg_sentiment < -0.3:
                risk_factors.append("Negative sentiment may impact price")
            elif avg_sentiment > 0.3:
                risk_factors.append("Overly positive sentiment may indicate overvaluation")
        
        # Market risks
        if 'volume' in market_data and market_data['volume'] < 1000000:
            risk_factors.append("Low trading volume may impact liquidity")
        
        return risk_factors
    
    def _calculate_confidence(self,
                            technical_signals: Dict[str, float],
                            sentiment_signals: Dict[str, float],
                            market_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the trade decision"""
        
        confidence_factors = []
        
        # Technical confidence
        if technical_signals:
            technical_strength = np.mean([abs(v) for v in technical_signals.values()])
            confidence_factors.append(min(technical_strength * 2, 1.0))
        
        # Sentiment confidence
        if sentiment_signals:
            sentiment_consistency = 1.0 - np.std(list(sentiment_signals.values()))
            confidence_factors.append(max(sentiment_consistency, 0.0))
        
        # Data quality confidence
        data_quality = 1.0
        if 'volume' in market_data and market_data['volume'] < 100000:
            data_quality *= 0.8
        confidence_factors.append(data_quality)
        
        # Calculate overall confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default confidence
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_position_size_rationale(self,
                                        quantity: float,
                                        technical_signals: Dict[str, float],
                                        sentiment_signals: Dict[str, float]) -> str:
        """Generate rationale for position size"""
        
        # Calculate signal strength
        technical_strength = np.mean([abs(v) for v in technical_signals.values()]) if technical_signals else 0
        sentiment_strength = np.mean([abs(v) for v in sentiment_signals.values()]) if sentiment_signals else 0
        
        total_strength = (technical_strength + sentiment_strength) / 2
        
        if total_strength > 0.7:
            return f"Large position size ({quantity:.0f} shares) justified by strong signals (strength: {total_strength:.2f})"
        elif total_strength > 0.4:
            return f"Moderate position size ({quantity:.0f} shares) based on moderate signal strength ({total_strength:.2f})"
        else:
            return f"Small position size ({quantity:.0f} shares) due to weak signals (strength: {total_strength:.2f})"
    
    def _calculate_expected_return(self,
                                 technical_signals: Dict[str, float],
                                 sentiment_signals: Dict[str, float]) -> float:
        """Calculate expected return based on signals"""
        
        # Simple expected return calculation
        technical_return = np.mean(list(technical_signals.values())) if technical_signals else 0
        sentiment_return = np.mean(list(sentiment_signals.values())) if sentiment_signals else 0
        
        return (technical_return + sentiment_return) / 2
    
    def _calculate_expected_risk(self,
                               technical_signals: Dict[str, float],
                               sentiment_signals: Dict[str, float]) -> float:
        """Calculate expected risk based on signals"""
        
        # Risk based on signal volatility
        technical_risk = np.std(list(technical_signals.values())) if technical_signals else 0.1
        sentiment_risk = np.std(list(sentiment_signals.values())) if sentiment_signals else 0.1
        
        return (technical_risk + sentiment_risk) / 2


class AuditTrail:
    """Audit trail for tracking all trading decisions and model outputs"""
    
    def __init__(self, audit_file: str = "audit_trail.jsonl"):
        self.audit_file = audit_file
        self.audit_entries = []
        
        logger.info("Audit trail initialized")
    
    def log_decision(self, decision_provenance: DecisionProvenance):
        """Log a decision to the audit trail"""
        
        try:
            audit_entry = {
                "timestamp": decision_provenance.timestamp.isoformat(),
                "decision_id": decision_provenance.decision_id,
                "agent_id": decision_provenance.agent_id,
                "model_version": decision_provenance.model_version,
                "decision_factors": decision_provenance.decision_factors,
                "feature_importance": decision_provenance.feature_importance,
                "risk_assessment": decision_provenance.risk_assessment,
                "data_snapshot": {
                    "market_data_keys": list(decision_provenance.market_data_snapshot.keys()),
                    "news_data_keys": list(decision_provenance.news_data_snapshot.keys()),
                    "technical_indicators": decision_provenance.technical_indicators,
                    "sentiment_scores": decision_provenance.sentiment_scores
                }
            }
            
            self.audit_entries.append(audit_entry)
            
            # Write to file
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            logger.info(f"Logged decision {decision_provenance.decision_id} to audit trail")
            
        except Exception as e:
            logger.error(f"Error logging decision to audit trail: {e}")
    
    def log_trade_rationale(self, rationale: TradeRationale):
        """Log trade rationale to the audit trail"""
        
        try:
            audit_entry = {
                "timestamp": rationale.timestamp.isoformat(),
                "trade_id": rationale.trade_id,
                "symbol": rationale.symbol,
                "action": rationale.action.value,
                "quantity": rationale.quantity,
                "price": rationale.price,
                "primary_reason": rationale.primary_reason,
                "confidence_level": rationale.confidence_level.value,
                "confidence_score": rationale.confidence_score,
                "model_version": rationale.model_version,
                "data_snapshot_id": rationale.data_snapshot_id,
                "expected_return": rationale.expected_return,
                "expected_risk": rationale.expected_risk,
                "supporting_evidence": rationale.supporting_evidence,
                "risk_factors": rationale.risk_factors
            }
            
            self.audit_entries.append(audit_entry)
            
            # Write to file
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            logger.info(f"Logged trade rationale {rationale.trade_id} to audit trail")
            
        except Exception as e:
            logger.error(f"Error logging trade rationale to audit trail: {e}")
    
    def get_audit_entries(self, 
                         start_date: datetime = None,
                         end_date: datetime = None,
                         agent_id: str = None) -> List[Dict[str, Any]]:
        """Get audit entries with optional filtering"""
        
        filtered_entries = self.audit_entries
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if datetime.fromisoformat(e['timestamp']) >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries if datetime.fromisoformat(e['timestamp']) <= end_date]
        
        if agent_id:
            filtered_entries = [e for e in filtered_entries if e.get('agent_id') == agent_id]
        
        return filtered_entries
    
    def generate_audit_report(self, 
                            start_date: datetime = None,
                            end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        entries = self.get_audit_entries(start_date, end_date)
        
        if not entries:
            return {"message": "No audit entries found for the specified period"}
        
        # Analyze entries
        total_decisions = len(entries)
        agents = list(set(e.get('agent_id', 'unknown') for e in entries))
        
        # Count by decision type
        decision_types = {}
        for entry in entries:
            if 'action' in entry:
                action = entry['action']
                decision_types[action] = decision_types.get(action, 0) + 1
        
        # Calculate average confidence
        confidence_scores = [e.get('confidence_score', 0) for e in entries if 'confidence_score' in e]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "period": {
                "start": start_date.isoformat() if start_date else "beginning",
                "end": end_date.isoformat() if end_date else "now"
            },
            "summary": {
                "total_decisions": total_decisions,
                "unique_agents": len(agents),
                "agents": agents,
                "decision_types": decision_types,
                "average_confidence": avg_confidence
            },
            "entries": entries
        }


def create_trade_rationale_generator() -> TradeRationaleGenerator:
    """Create a trade rationale generator"""
    return TradeRationaleGenerator()


def create_audit_trail(audit_file: str = "audit_trail.jsonl") -> AuditTrail:
    """Create an audit trail instance"""
    return AuditTrail(audit_file)


def main():
    """Test the explainability system"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testing Trade Rationale and Explainability...")
    print("="*50)
    
    # Create components
    rationale_generator = create_trade_rationale_generator()
    audit_trail = create_audit_trail("test_audit_trail.jsonl")
    
    # Test data
    trade_data = {
        'symbol': 'AAPL',
        'action': 'buy',
        'quantity': 100,
        'price': 150.0
    }
    
    market_data = {
        'volume': 50000000,
        'volatility': 0.02,
        'price_change': 0.015
    }
    
    technical_signals = {
        'rsi': 65.0,
        'macd': 0.5,
        'bollinger_position': 0.7,
        'sma_20': 148.0,
        'volume_ratio': 1.2
    }
    
    sentiment_signals = {
        'news_sentiment': 0.3,
        'social_sentiment': 0.2,
        'analyst_sentiment': 0.4,
        'earnings_sentiment': 0.1
    }
    
    model_metadata = {
        'model_version': 'v1.2.3',
        'data_snapshot_id': 'snapshot_20240101_120000'
    }
    
    # Generate trade rationale
    print("\n📝 Generating trade rationale...")
    rationale = rationale_generator.generate_trade_rationale(
        trade_data, market_data, technical_signals, sentiment_signals, model_metadata
    )
    
    print(f"✅ Trade rationale generated:")
    print(f"   Trade ID: {rationale.trade_id}")
    print(f"   Symbol: {rationale.symbol}")
    print(f"   Action: {rationale.action.value}")
    print(f"   Primary Reason: {rationale.primary_reason}")
    print(f"   Confidence: {rationale.confidence_level.value} ({rationale.confidence_score:.3f})")
    print(f"   Expected Return: {rationale.expected_return:.3f}")
    print(f"   Expected Risk: {rationale.expected_risk:.3f}")
    
    print(f"\n📊 Supporting Evidence:")
    for evidence in rationale.supporting_evidence:
        print(f"   • {evidence}")
    
    print(f"\n⚠️  Risk Factors:")
    for risk in rationale.risk_factors:
        print(f"   • {risk}")
    
    # Test audit trail
    print(f"\n📋 Testing audit trail...")
    audit_trail.log_trade_rationale(rationale)
    
    # Generate audit report
    report = audit_trail.generate_audit_report()
    print(f"✅ Audit report generated:")
    print(f"   Total decisions: {report['summary']['total_decisions']}")
    print(f"   Average confidence: {report['summary']['average_confidence']:.3f}")
    
    print(f"\n✅ Explainability test completed!")


if __name__ == "__main__":
    main()
