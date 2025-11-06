"""
Integration Test for WealthArena Complete System

This script tests the integration of all new components:
- News embeddings and NLP pipeline
- Signal fusion system
- Historical fast-forward game
- Explainability and audit trails
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.news_processor import NewsProcessor, NewsConfig
from src.data.signal_fusion import SignalFusion, SignalFusionConfig
from src.game.historical_game import HistoricalGame, GameConfig
from src.explainability.trade_rationale import TradeRationaleGenerator, AuditTrail

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_news_processor():
    """Test news processing and NLP pipeline"""
    print("\n" + "="*60)
    print("📰 TESTING NEWS PROCESSOR & NLP PIPELINE")
    print("="*60)
    
    try:
        # Create news processor
        config = NewsConfig()
        processor = NewsProcessor(config)
        
        # Test news fetching
        print("\n🔍 Testing news fetching...")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        news_articles = processor.fetch_news(symbols, hours_back=24)
        print(f"✅ Fetched {len(news_articles)} news articles")
        
        # Test news processing
        print("\n🔄 Testing news processing...")
        processed_news = processor.process_news_batch(news_articles)
        print(f"✅ Processed {processed_news['summary']['total_articles']} articles")
        print(f"📊 Average sentiment: {processed_news['summary']['avg_sentiment']:.3f}")
        
        # Test market sentiment
        print("\n📈 Testing market sentiment...")
        market_sentiment = processor.get_market_sentiment(symbols)
        for symbol, sentiment in market_sentiment.items():
            print(f"  {symbol}: {sentiment:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ News processor test failed: {e}")
        return False


def test_signal_fusion():
    """Test signal fusion system"""
    print("\n" + "="*60)
    print("🔄 TESTING SIGNAL FUSION SYSTEM")
    print("="*60)
    
    try:
        # Create signal fusion
        config = SignalFusionConfig()
        signal_fusion = SignalFusion(config)
        
        # Create sample market data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='D')
        market_data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 100 + 1000,
            'high': np.random.randn(len(dates)) * 100 + 1050,
            'low': np.random.randn(len(dates)) * 100 + 950,
            'close': np.random.randn(len(dates)) * 100 + 1000,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        print("\n📊 Testing signal extraction...")
        try:
            results = signal_fusion.process_market_data_with_signals(market_data, symbols)
            
            print(f"✅ Signal extraction results:")
            for signal_type, data in results.items():
                if not data.empty:
                    print(f"  {signal_type}: {data.shape} features")
                else:
                    print(f"  {signal_type}: No data")
        except Exception as e:
            print(f"❌ Signal extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if 'trading' in results and not results['trading'].empty:
            print(f"\n🎯 Trading signals created:")
            print(f"  Columns: {list(results['trading'].columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal fusion test failed: {e}")
        return False


def test_historical_game():
    """Test historical fast-forward game"""
    print("\n" + "="*60)
    print("🎮 TESTING HISTORICAL FAST-FORWARD GAME")
    print("="*60)
    
    try:
        # Create game
        config = GameConfig(
            episode_duration_days=30,
            initial_capital=100000.0,
            available_instruments=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        )
        game = HistoricalGame(config)
        
        # Create episode
        print("\n📅 Creating historical episode...")
        episode = game.create_episode(
            start_date="2024-01-01",
            end_date="2024-01-31",
            instruments=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        )
        print(f"✅ Episode created: {episode.episode_id}")
        
        # Create players
        players = [
            {'player_id': 'human_1', 'player_type': 'human', 'name': 'Human Player'},
            {'player_id': 'agent_1', 'player_type': 'agent', 'name': 'RL Agent'},
            {'player_id': 'benchmark_1', 'player_type': 'benchmark', 'name': 'S&P 500 Benchmark'}
        ]
        
        # Start game
        print(f"\n🚀 Starting game with {len(players)} players...")
        game_id = game.start_game(episode, players)
        print(f"✅ Game started: {game_id}")
        
        # Simulate a few turns
        print(f"\n🎯 Simulating game turns...")
        for turn in range(3):
            print(f"\n--- Turn {turn + 1} ---")
            
            # Human player action
            human_actions = {
                'buy': {'AAPL': 10, 'MSFT': 5},
                'sell': {}
            }
            result = game.execute_turn('human_1', human_actions)
            print(f"Human player: Portfolio value = ${result['portfolio_value']:,.2f}")
            
            # Agent player action
            agent_actions = {
                'buy': {'GOOGL': 8, 'SPY': 20},
                'sell': {}
            }
            result = game.execute_turn('agent_1', agent_actions)
            print(f"RL Agent: Portfolio value = ${result['portfolio_value']:,.2f}")
            
            # Benchmark player (buy and hold)
            if turn == 0:
                benchmark_actions = {
                    'buy': {'SPY': 100},
                    'sell': {}
                }
                result = game.execute_turn('benchmark_1', benchmark_actions)
                print(f"Benchmark: Portfolio value = ${result['portfolio_value']:,.2f}")
            else:
                result = game.execute_turn('benchmark_1', {'buy': {}, 'sell': {}})
                print(f"Benchmark: Portfolio value = ${result['portfolio_value']:,.2f}")
        
        # Get final results
        print(f"\n🏆 Final Results:")
        leaderboard = game.get_leaderboard()
        for player in leaderboard:
            print(f"  {player['rank']}. {player['name']}: {player['total_return']:.2%} return")
        
        return True
        
    except Exception as e:
        print(f"❌ Historical game test failed: {e}")
        return False


def test_explainability():
    """Test explainability and audit trails"""
    print("\n" + "="*60)
    print("🔍 TESTING EXPLAINABILITY & AUDIT TRAILS")
    print("="*60)
    
    try:
        # Create components
        rationale_generator = TradeRationaleGenerator()
        audit_trail = AuditTrail("test_audit_trail.jsonl")
        
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
        print("\n📝 Testing trade rationale generation...")
        rationale = rationale_generator.generate_trade_rationale(
            trade_data, market_data, technical_signals, sentiment_signals, model_metadata
        )
        
        print(f"✅ Trade rationale generated:")
        print(f"   Symbol: {rationale.symbol}")
        print(f"   Action: {rationale.action.value}")
        print(f"   Confidence: {rationale.confidence_level.value} ({rationale.confidence_score:.3f})")
        print(f"   Primary Reason: {rationale.primary_reason}")
        
        # Test audit trail
        print("\n📋 Testing audit trail...")
        audit_trail.log_trade_rationale(rationale)
        
        # Generate audit report
        report = audit_trail.generate_audit_report()
        print(f"✅ Audit report generated:")
        print(f"   Total decisions: {report['summary']['total_decisions']}")
        print(f"   Average confidence: {report['summary']['average_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Explainability test failed: {e}")
        return False


def test_integration():
    """Test complete system integration"""
    print("\n" + "="*60)
    print("🔗 TESTING COMPLETE SYSTEM INTEGRATION")
    print("="*60)
    
    try:
        # Test all components together
        print("\n🔄 Testing integrated workflow...")
        
        # 1. Process news and generate sentiment
        news_processor = NewsProcessor(NewsConfig())
        symbols = ["AAPL", "MSFT", "GOOGL"]
        market_sentiment = news_processor.get_market_sentiment(symbols)
        
        # 2. Generate market data and fuse signals
        signal_fusion = SignalFusion(SignalFusionConfig())
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='D')
        market_data = pd.DataFrame({
            'open': np.random.randn(len(dates)) * 100 + 1000,
            'high': np.random.randn(len(dates)) * 100 + 1050,
            'low': np.random.randn(len(dates)) * 100 + 950,
            'close': np.random.randn(len(dates)) * 100 + 1000,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        signal_results = signal_fusion.process_market_data_with_signals(market_data, symbols)
        
        # Handle empty results
        if not signal_results:
            signal_results = {
                "technical": pd.DataFrame(),
                "news": pd.DataFrame(),
                "fundamental": pd.DataFrame(),
                "macro": pd.DataFrame(),
                "fused": pd.DataFrame(),
                "trading": pd.DataFrame()
            }
        
        # 3. Create game with integrated signals
        game_config = GameConfig(
            episode_duration_days=30,
            initial_capital=100000.0,
            available_instruments=symbols
        )
        game = HistoricalGame(game_config)
        
        episode = game.create_episode(
            start_date="2024-01-01",
            end_date="2024-01-31",
            instruments=symbols
        )
        
        players = [
            {'player_id': 'human_1', 'player_type': 'human', 'name': 'Human Player'},
            {'player_id': 'agent_1', 'player_type': 'agent', 'name': 'RL Agent'}
        ]
        
        game_id = game.start_game(episode, players)
        
        # 4. Generate explainable trade decisions
        rationale_generator = TradeRationaleGenerator()
        audit_trail = AuditTrail("integration_audit_trail.jsonl")
        
        # Simulate a trade with full explainability
        trade_data = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 50,
            'price': 150.0
        }
        
        technical_signals = signal_results.get('technical', pd.DataFrame()).iloc[-1].to_dict() if not signal_results.get('technical', pd.DataFrame()).empty else {}
        sentiment_signals = signal_results.get('news', pd.DataFrame()).iloc[-1].to_dict() if not signal_results.get('news', pd.DataFrame()).empty else {}
        
        rationale = rationale_generator.generate_trade_rationale(
            trade_data, market_data.iloc[-1].to_dict(), technical_signals, sentiment_signals, {'model_version': 'v1.0.0'}
        )
        
        audit_trail.log_trade_rationale(rationale)
        
        print(f"✅ Integrated workflow completed successfully!")
        print(f"   News sentiment processed: {len(market_sentiment)} symbols")
        print(f"   Signals fused: {len(signal_results)} signal types")
        print(f"   Game created: {game_id}")
        print(f"   Trade rationale generated: {rationale.trade_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 WEALTHARENA COMPLETE SYSTEM INTEGRATION TEST")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run individual component tests
    test_results['news_processor'] = test_news_processor()
    test_results['signal_fusion'] = test_signal_fusion()
    test_results['historical_game'] = test_historical_game()
    test_results['explainability'] = test_explainability()
    test_results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*80)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
