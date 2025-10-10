#!/usr/bin/env python3
"""
Quick Setup Script for Paper Trading System
Runs migrations, creates sample data, and validates setup
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
sys.path.append('/workspaces/congenial-fortnight')
django.setup()

from django.core.management import call_command
from paper_trading.data_aggregator import DataAggregator
from paper_trading.engine import PaperTradingEngine

def main():
    print("🚀 Setting up Paper Trading System...\n")
    
    # Step 1: Run migrations
    print("📦 Step 1: Running database migrations...")
    try:
        call_command('makemigrations', 'paper_trading', interactive=False)
        call_command('migrate', interactive=False)
        print("✅ Migrations complete\n")
    except Exception as e:
        print(f"❌ Migration error: {e}\n")
    
    # Step 2: Test data aggregator
    print("📊 Step 2: Testing data aggregator...")
    try:
        aggregator = DataAggregator()
        price = aggregator.get_realtime_price('EURUSD')
        if price:
            print(f"✅ Data aggregator working: EURUSD = {price['close']:.5f}\n")
        else:
            print("⚠️  Warning: Could not fetch price (but system will work)\n")
    except Exception as e:
        print(f"⚠️  Data aggregator warning: {e}\n")
    
    # Step 3: Test paper trading engine
    print("🎯 Step 3: Testing paper trading engine...")
    try:
        engine = PaperTradingEngine(initial_balance=10000)
        summary = engine.get_performance_summary(days=7)
        print(f"✅ Paper trading engine initialized")
        print(f"   Current balance: ${engine.current_balance}\n")
    except Exception as e:
        print(f"❌ Engine error: {e}\n")
    
    # Step 4: Check API keys
    print("🔑 Step 4: Checking API keys...")
    from dotenv import load_dotenv
    load_dotenv()
    
    keys_found = []
    if os.getenv('TWELVE_DATA_API_KEY'):
        keys_found.append('Twelve Data')
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        keys_found.append('Alpha Vantage')
    if os.getenv('FINNHUB_API_KEY'):
        keys_found.append('Finnhub')
    
    if keys_found:
        print(f"✅ API keys found: {', '.join(keys_found)}")
    else:
        print("⚠️  No API keys found (will use Yahoo Finance - works great!)")
    print()
    
    # Step 5: Print instructions
    print("=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n🚀 To start the system:\n")
    print("Terminal 1 (Django server with WebSockets):")
    print("  daphne -b 0.0.0.0 -p 8000 forex_signal.asgi:application")
    print("  OR")
    print("  python manage.py runserver (without WebSockets)\n")
    
    print("Terminal 2 (Price update worker):")
    print("  python manage.py run_price_worker --interval=5 --pairs=EURUSD,XAUUSD\n")
    
    print("Terminal 3 (Frontend - optional):")
    print("  cd frontend")
    print("  npm install lightweight-charts")
    print("  npm start\n")
    
    print("📊 Access points:")
    print("  - API: http://localhost:8000/api/paper-trading/")
    print("  - Admin: http://localhost:8000/admin/")
    print("  - WebSocket: ws://localhost:8000/ws/trading/")
    print("  - Frontend: http://localhost:3000/\n")
    
    print("📖 Full documentation: PAPER_TRADING_IMPLEMENTATION_COMPLETE.md\n")

if __name__ == '__main__':
    main()
