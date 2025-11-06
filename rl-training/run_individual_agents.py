#!/usr/bin/env python3
"""
Individual Agent Training Script

This script allows you to train and evaluate individual financial instrument agents
one by one to see their performance.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.asx_stocks_trainer import main as run_asx_stocks
from src.training.currency_pairs_trainer import main as run_currency_pairs
from src.training.cryptocurrency_trainer import main as run_cryptocurrency
from src.training.etf_trainer import main as run_etf
from src.training.master_trainer import main as run_master

def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("WEALTHARENA INDIVIDUAL AGENT TRAINING")
    print("="*60)
    print("\nSelect an agent to train and evaluate:")
    print("1. ASX Stocks Agent (30 stocks)")
    print("2. Currency Pairs Agent (10 major pairs)")
    print("3. Cryptocurrency Agent (12 major cryptos)")
    print("4. ETF Agent (20 major ETFs)")
    print("5. Run All Agents (Master Comparison)")
    print("6. Exit")
    print("-" * 60)

def run_agent(choice: int):
    """Run the selected agent"""
    
    if choice == 1:
        print("\n🚀 Starting ASX Stocks Agent Training...")
        print("="*50)
        try:
            trainer, report = run_asx_stocks()
            print("\n✅ ASX Stocks Agent completed successfully!")
            return trainer, report
        except Exception as e:
            print(f"\n❌ ASX Stocks Agent failed: {e}")
            return None, None
    
    elif choice == 2:
        print("\n🚀 Starting Currency Pairs Agent Training...")
        print("="*50)
        try:
            trainer, report = run_currency_pairs()
            print("\n✅ Currency Pairs Agent completed successfully!")
            return trainer, report
        except Exception as e:
            print(f"\n❌ Currency Pairs Agent failed: {e}")
            return None, None
    
    elif choice == 3:
        print("\n🚀 Starting Cryptocurrency Agent Training...")
        print("="*50)
        try:
            trainer, report = run_cryptocurrency()
            print("\n✅ Cryptocurrency Agent completed successfully!")
            return trainer, report
        except Exception as e:
            print(f"\n❌ Cryptocurrency Agent failed: {e}")
            return None, None
    
    elif choice == 4:
        print("\n🚀 Starting ETF Agent Training...")
        print("="*50)
        try:
            trainer, report = run_etf()
            print("\n✅ ETF Agent completed successfully!")
            return trainer, report
        except Exception as e:
            print(f"\n❌ ETF Agent failed: {e}")
            return None, None
    
    elif choice == 5:
        print("\n🚀 Starting Master Training (All Agents)...")
        print("="*50)
        try:
            master_trainer, results = run_master()
            print("\n✅ Master Training completed successfully!")
            return master_trainer, results
        except Exception as e:
            print(f"\n❌ Master Training failed: {e}")
            return None, None
    
    else:
        print("\n❌ Invalid choice. Please select 1-6.")
        return None, None

def main():
    """Main function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Welcome to WealthArena Individual Agent Training!")
    print("This tool allows you to train and evaluate each financial instrument agent individually.")
    
    while True:
        print_menu()
        
        try:
            choice = int(input("\nEnter your choice (1-6): "))
            
            if choice == 6:
                print("\n👋 Goodbye! Thank you for using WealthArena.")
                break
            
            if choice in [1, 2, 3, 4, 5]:
                trainer, report = run_agent(choice)
                
                if trainer is not None:
                    print(f"\n📊 Agent training and evaluation completed!")
                    print(f"📁 Results have been saved to the respective results directory.")
                    
                    # Ask if user wants to continue
                    continue_choice = input("\nWould you like to train another agent? (y/n): ").lower()
                    if continue_choice not in ['y', 'yes']:
                        print("\n👋 Goodbye! Thank you for using WealthArena.")
                        break
                else:
                    print(f"\n❌ Agent training failed. Please check the logs for details.")
                    continue_choice = input("\nWould you like to try another agent? (y/n): ").lower()
                    if continue_choice not in ['y', 'yes']:
                        print("\n👋 Goodbye! Thank you for using WealthArena.")
                        break
            else:
                print("\n❌ Invalid choice. Please select 1-6.")
                continue
                
        except ValueError:
            print("\n❌ Invalid input. Please enter a number between 1-6.")
            continue
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using WealthArena.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
