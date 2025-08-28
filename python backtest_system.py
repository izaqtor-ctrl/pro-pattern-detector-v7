"""
Pro Pattern Detector v7.0 - Complete Backtesting System
Run this to validate your pattern detection algorithms against historical performance

Usage:
1. Save this file as 'backtest_system.py'
2. Run: python backtest_system.py
3. Results will be saved to 'pattern_backtest_results.csv'

This will test your confidence scoring system, volume thresholds, and pattern effectiveness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_realistic_data(ticker, days=400):
    """
    Create realistic stock data with proper market characteristics:
    - Trending periods with pullbacks
    - Consolidation phases  
    - Volume spikes during breakouts
    - Weekend gaps and volatility clustering
    """
    # Create trading days only (no weekends)
    all_dates = pd.date_range(end=datetime.now(), periods=days*2, freq='D')
    trading_days = all_dates[all_dates.weekday < 5][:days]
    
    # Seed for reproducible results per ticker
    np.random.seed(hash(ticker) % 2147483647)
    
    # Base price influenced by ticker name
    base_price = 75 + (hash(ticker) % 150)
    
    # Create market regimes: trending vs consolidating
    regime_length = 20
    n_regimes = days // regime_length + 1
    regimes = np.random.choice(['trend_up', 'trend_down', 'consolidate'], 
                              size=n_regimes, p=[0.4, 0.2, 0.4])
    
    returns = []
    volumes = []
    
    for i, date in enumerate(trading_days):
        regime_idx = i // regime_length
        if regime_idx >= len(regimes):
            regime_idx = len(regimes) - 1
        regime = regimes[regime_idx]
        
        # Base return based on regime
        if regime == 'trend_up':
            base_return = np.random.normal(0.003, 0.015)
        elif regime == 'trend_down':
            base_return = np.random.normal(-0.002, 0.020)
        else:  # consolidate
            base_return = np.random.normal(0.0005, 0.012)
        
        # Add volatility clustering
        if i > 0 and abs(returns[i-1]) > 0.03:
            base_return *= 1.5
        
        # Monday gap risk
        if date.weekday() == 0 and i > 0:
            gap = np.random.normal(0, 0.008)
            base_return += gap
        
        returns.append(base_return)
        
        # Volume correlated with price movement
        base_volume = np.random.lognormal(14, 0.3)
        if abs(base_return) > 0.02:  # High volatility days
            volume_multiplier = np.random.uniform(1.5, 3.0)
        elif regime == 'consolidate':
            volume_multiplier = np.random.uniform(0.7, 1.2)
        else:
            volume_multiplier = np.random.uniform(0.9, 1.4)
        
        volumes.append(int(base_volume * volume_multiplier))
    
    # Generate price series
    returns[0] = 0
    close_prices = base_price * np.cumprod(1 + np.array(returns))
    
    # Create OHLC with realistic intraday ranges
    intraday_ranges = np.random.lognormal(-3, 0.3, len(trading_days))
    
    opens = close_prices * (1 + np.random.normal(0, 0.003, len(trading_days)))
    highs = np.maximum(opens, close_prices) * (1 + intraday_ranges)
    lows = np.minimum(opens, close_prices) * (1 - intraday_ranges)
    
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    }, index=trading_days)
    
    return data

# Import your exact pattern detection functions
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def analyze_volume_pattern(data, pattern_type, pattern_info):
    """Your exact volume analysis logic"""
    volume_score = 0
    volume_info = {}
    
    if len(data) < 20:
        return volume_score, volume_info
    
    avg_volume_20 = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    volume_multiplier = current_volume / avg_volume_20
    
    volume_info['avg_volume_20'] = avg_volume_20
    volume_info['current_volume'] = current_volume
    volume_info['volume_multiplier'] = volume_multiplier
    
    if volume_multiplier >= 2.0:
        volume_score += 25
        volume_info['exceptional_volume'] = True
        volume_info['volume_status'] = f"Exceptional Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= 1.5:
        volume_score += 20
        volume_info['strong_volume'] = True
        volume_info['volume_status'] = f"Strong Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= 1.3:
        volume_score += 15
        volume_info['good_volume'] = True
        volume_info['volume_status'] = f"Good Volume ({volume_multiplier:.1f}x)"
    else:
        volume_info['weak_volume'] = True
        volume_info['volume_status'] = f"Weak Volume ({volume_multiplier:.1f}x)"
    
    return volume_score, volume_info

def detect_bull_flag_backtest(data, macd_line, signal_line, histogram):
    """Simplified but accurate Bull Flag detection"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # Flagpole analysis
    flagpole_start = min(25, len(data) - 10)
    flagpole_end = 15
    
    start_price = data['Close'].iloc[-flagpole_start]
    peak_price = data['High'].iloc[-flagpole_start:-flagpole_end].max()
    flagpole_gain = (peak_price - start_price) / start_price
    
    if flagpole_gain < 0.08:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['flagpole_gain'] = f"{flagpole_gain*100:.1f}%"
    
    # Flag pullback validation
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    if -0.15 <= pullback <= 0.05:
        confidence += 20
        pattern_info['flag_pullback'] = f"{pullback*100:.1f}%"
    
    # Pattern invalidation checks
    flag_low = flag_data['Low'].min()
    if current_price < flag_low * 0.95:
        return 0, {'pattern_broken': True}
    
    if current_price < start_price:
        return 0, {'pattern_broken': True}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Bull Flag", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_flat_top_backtest(data, macd_line, signal_line, histogram):
    """Simplified but accurate Flat Top detection"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 50:
        return confidence, pattern_info
    
    # Initial ascension
    ascent_start = min(45, len(data) - 15)
    ascent_end = 25
    
    start_price = data['Close'].iloc[-ascent_start]
    peak_price = data['High'].iloc[-ascent_start:-ascent_end].max()
    initial_gain = (peak_price - start_price) / start_price
    
    if initial_gain < 0.10:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['initial_ascension'] = f"{initial_gain*100:.1f}%"
    
    # Resistance level validation
    resistance_level = peak_price
    touches = sum(1 for h in data['High'].tail(20) if h >= resistance_level * 0.98)
    if touches >= 2:
        confidence += 15
        pattern_info['resistance_touches'] = touches
        pattern_info['resistance_level'] = resistance_level
    
    # Pattern age check
    current_price = data['Close'].iloc[-1]
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_level * 0.98), 11)
    
    if days_old > 8:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True}
    
    # Support break check
    descent_data = data.iloc[-ascent_end:-10]
    descent_low = descent_data['Low'].min()
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Flat Top Breakout", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_inside_bar_backtest(data, macd_line, signal_line, histogram):
    """Simplified but accurate Inside Bar detection"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 5:
        return confidence, pattern_info
    
    # Look for inside bar pattern
    inside_bars_count = 0
    mother_bar_idx = None
    
    for i in range(-1, -5, -1):
        try:
            current_bar = data.iloc[i]
            previous_bar = data.iloc[i-1]
            
            # Inside bar validation
            is_inside = (current_bar['High'] <= previous_bar['High'] and 
                        current_bar['Low'] >= previous_bar['Low'] and
                        current_bar['High'] < previous_bar['High'] and
                        current_bar['Low'] > previous_bar['Low'])
            
            # Color validation
            mother_is_green = previous_bar['Close'] > previous_bar['Open']
            inside_is_red = current_bar['Close'] < current_bar['Open']
            
            if is_inside and mother_is_green and inside_is_red:
                if inside_bars_count == 0:
                    mother_bar_idx = i - 1
                    pattern_info['mother_bar_high'] = previous_bar['High']
                    pattern_info['mother_bar_low'] = previous_bar['Low']
                    pattern_info['inside_bar_high'] = current_bar['High']
                    pattern_info['inside_bar_low'] = current_bar['Low']
                inside_bars_count += 1
                if inside_bars_count >= 2:
                    break
            else:
                break
        except (IndexError, KeyError):
            break
    
    if inside_bars_count == 0:
        return confidence, pattern_info
    
    confidence += 30
    pattern_info['inside_bars_count'] = inside_bars_count
    
    # Size ratio validation
    mother_bar_range = pattern_info['mother_bar_high'] - pattern_info['mother_bar_low']
    inside_bar_range = pattern_info['inside_bar_high'] - pattern_info['inside_bar_low']
    
    if mother_bar_range > 0:
        size_ratio = inside_bar_range / mother_bar_range
        pattern_info['size_ratio'] = f"{size_ratio:.1%}"
        
        if size_ratio < 0.30:
            confidence += 20
            pattern_info['tight_consolidation'] = True
        elif size_ratio < 0.50:
            confidence += 15
            pattern_info['good_consolidation'] = True
        elif size_ratio < 0.70:
            confidence += 10
            pattern_info['moderate_consolidation'] = True
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Inside Bar", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_pattern_backtest(data, pattern_type):
    """Main pattern detection function for backtesting"""
    if len(data) < 10:
        return False, 0, {}
    
    # Calculate indicators
    data['RSI'] = calculate_rsi(data)
    macd_line, signal_line, histogram = calculate_macd(data)
    
    confidence = 0
    pattern_info = {}
    
    if pattern_type == "Bull Flag":
        confidence, pattern_info = detect_bull_flag_backtest(data, macd_line, signal_line, histogram)
    elif pattern_type == "Flat Top Breakout":
        confidence, pattern_info = detect_flat_top_backtest(data, macd_line, signal_line, histogram)
    elif pattern_type == "Inside Bar":
        confidence, pattern_info = detect_inside_bar_backtest(data, macd_line, signal_line, histogram)
    
    return confidence >= 55, confidence, pattern_info

def calculate_trading_levels(data, pattern_info, pattern_type):
    """Calculate entry, stop, target levels using your exact logic"""
    current_price = data['Close'].iloc[-1]
    
    if pattern_type == "Inside Bar":
        inside_bar_high = pattern_info.get('inside_bar_high', current_price)
        inside_bar_low = pattern_info.get('inside_bar_low', current_price * 0.95)
        mother_bar_high = pattern_info.get('mother_bar_high', current_price * 1.05)
        
        entry = inside_bar_high * 1.05
        stop = inside_bar_low * 0.95
        target1 = mother_bar_high
        target2 = mother_bar_high * 1.13
        
    elif pattern_type == "Bull Flag":
        flag_high = data['High'].tail(15).max()
        entry = flag_high * 1.005
        flag_low = data['Low'].tail(12).min()
        stop = flag_low * 0.98
        
        # Measured move using flagpole
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_pct = float(pattern_info['flagpole_gain'].replace('%', '')) / 100
                flagpole_height = entry * flagpole_pct
                target1 = entry + flagpole_height
                target2 = entry + (flagpole_height * 1.382)
            except:
                risk = entry - stop
                target1 = entry + (risk * 2.5)
                target2 = entry + (risk * 4.0)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.5)
            target2 = entry + (risk * 4.0)
            
    elif pattern_type == "Flat Top Breakout":
        entry = pattern_info.get('resistance_level', data['High'].tail(20).max()) * 1.001
        recent_low = data['Low'].tail(15).min()
        stop = recent_low * 0.98
        
        # Triangle height projection
        support_level = data['Low'].tail(20).max()
        triangle_height = entry - support_level
        target1 = entry + triangle_height
        target2 = entry + (triangle_height * 1.618)
    
    # Ensure stop is below entry
    if stop >= entry:
        stop = entry * 0.96
    
    return {
        'entry': entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'risk': entry - stop
    }

def simulate_trade(levels, future_data, max_days=12):
    """Simulate trade outcome over future price data"""
    entry = levels['entry']
    stop = levels['stop']
    target1 = levels['target1']
    target2 = levels['target2']
    
    if len(future_data) < max_days:
        return "insufficient_data", 0, len(future_data)
    
    for day in range(max_days):
        bar = future_data.iloc[day]
        
        # Check stop loss first (conservative - use low of day)
        if bar['Low'] <= stop:
            loss = stop - entry
            return "stop_hit", loss, day + 1
        
        # Check targets (optimistic - use high of day)
        if bar['High'] >= target2:
            profit = target2 - entry
            return "target2_hit", profit, day + 1
        elif bar['High'] >= target1:
            profit = target1 - entry
            return "target1_hit", profit, day + 1
    
    # Time expired - use final close
    final_price = future_data.iloc[max_days - 1]['Close']
    pnl = final_price - entry
    return "expired", pnl, max_days

def run_backtest():
    """Run comprehensive backtest on all tickers and patterns"""
    
    # Your ticker list
    tickers = ['OLLI', 'FANG', 'ELF', 'PHM', 'CMG', 'RACE', 'CHRD', 'PCARD', 'NXPI', 'CELH', 
               'LLY', 'HIMS', 'OIL', 'ALAB', 'CLS', 'AEP', 'SPRY', 'VRT', 'ONON', 'SYM',
               'AMSC', 'RKLB', 'FTV', 'HOOD', 'RMBS']
    
    patterns = ["Bull Flag", "Flat Top Breakout", "Inside Bar"]
    
    all_results = []
    total_iterations = 0
    
    print("Starting Pro Pattern Detector Backtest...")
    print("=" * 60)
    
    for ticker_idx, ticker in enumerate(tickers):
        print(f"Analyzing {ticker} ({ticker_idx + 1}/{len(tickers)})...")
        
        # Generate realistic market data
        full_data = create_realistic_data(ticker, days=300)
        
        # Walk through historical data (leave 15 days for trade simulation)
        for day_idx in range(50, len(full_data) - 15):
            total_iterations += 1
            
            # Current data up to this point
            historical_data = full_data.iloc[:day_idx].copy()
            future_data = full_data.iloc[day_idx:day_idx + 15].copy()
            trade_date = full_data.index[day_idx]
            
            # Test each pattern
            for pattern in patterns:
                detected, confidence, pattern_info = detect_pattern_backtest(historical_data, pattern)
                
                if detected and confidence >= 55:
                    # Calculate trading levels
                    levels = calculate_trading_levels(historical_data, pattern_info, pattern)
                    
                    # Simulate the trade
                    outcome, pnl, days_held = simulate_trade(levels, future_data)
                    
                    # Calculate metrics
                    risk_amount = levels['risk']
                    risk_pct = (risk_amount / levels['entry']) * 100 if levels['entry'] > 0 else 0
                    pnl_pct = (pnl / levels['entry']) * 100 if levels['entry'] > 0 else 0
                    
                    # Volume classification
                    volume_mult = pattern_info.get('volume_multiplier', 1.0)
                    if volume_mult >= 2.0:
                        volume_class = "Exceptional"
                    elif volume_mult >= 1.5:
                        volume_class = "Strong"
                    elif volume_mult >= 1.3:
                        volume_class = "Good"
                    else:
                        volume_class = "Weak"
                    
                    # Store results
                    result = {
                        'ticker': ticker,
                        'pattern': pattern,
                        'trade_date': trade_date,
                        'confidence': confidence,
                        'volume_class': volume_class,
                        'volume_multiplier': volume_mult,
                        'entry_price': levels['entry'],
                        'stop_price': levels['stop'],
                        'target1_price': levels['target1'],
                        'target2_price': levels['target2'],
                        'risk_amount': risk_amount,
                        'risk_pct': risk_pct,
                        'outcome': outcome,
                        'pnl_amount': pnl,
                        'pnl_pct': pnl_pct,
                        'days_held': days_held,
                        'win': 1 if pnl > 0 else 0,
                        'big_win': 1 if pnl_pct > 5 else 0,
                        'target1_hit': 1 if outcome == 'target1_hit' else 0,
                        'target2_hit': 1 if outcome == 'target2_hit' else 0,
                        'stop_hit': 1 if outcome == 'stop_hit' else 0
                    }
                    
                    all_results.append(result)
        
        if ticker_idx % 5 == 4:  # Progress update every 5 tickers
            print(f"  Processed {total_iterations:,} iterations so far...")
    
    results_df = pd.DataFrame(all_results)
    print(f"\nBacktest complete! Generated {len(results_df)} total trades.")
    
    return results_df

def analyze_results(df):
    """Comprehensive results analysis"""
    
    print("\n" + "=" * 80)
    print("PATTERN DETECTION BACKTEST RESULTS")
    print("=" * 80)
    
    # Overall performance
    total_trades = len(df)
    win_rate = df['win'].mean() * 100
    avg_pnl = df['pnl_pct'].mean()
    avg_win = df[df['win'] == 1]['pnl_pct'].mean()
    avg_loss = df[df['win'] == 0]['pnl_pct'].mean()
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total Trades: {total_trades:,}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average P&L: {avg_pnl:.2f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {avg_win / abs(avg_loss):.2f}" if avg_loss != 0 else "N/A")
    
    # CRITICAL TEST: Does higher confidence actually work?
    print(f"\nCONFIDENCE VALIDATION (Key Question):")
    confidence_bins = [(55, 65), (65, 75), (75, 85), (85, 100)]
    for low, high in confidence_bins:
        subset = df[(df['confidence'] >= low) & (df['confidence'] < high)]
        if len(subset) > 20:  # Only show bins with meaningful sample size
            win_rate = subset['win'].mean() * 100
            avg_pnl = subset['pnl_pct'].mean()
            print(f"  {low}-{high}%: {len(subset)} trades, {win_rate:.1f}% win rate, {avg_pnl:.2f}% avg P&L")
    
    # CRITICAL TEST: Does volume confirmation matter?
    print(f"\nVOLUME VALIDATION (Key Question):")
    for vol_class in ['Weak', 'Good', 'Strong', 'Exceptional']:
        subset = df[df['volume_class'] == vol_class]
        if len(subset) > 10:
            win_rate = subset['win'].mean() * 100
            avg_pnl = subset['pnl_pct'].mean()
            target_hit_rate = (subset['target1_hit'].sum() + subset['target2_hit'].sum()) / len(subset) * 100
            print(f"  {vol_class}: {len(subset)} trades, {win_rate:.1f}% win rate, {avg_pnl:.2f}% avg P&L, {target_hit_rate:.1f
