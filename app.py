import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Try to import yfinance with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pro Pattern Detector v7.0", 
    layout="wide"
)

def get_market_context():
    """Get current market timing context and trading recommendations"""
    current_time = datetime.now()
    current_day = current_time.strftime('%A')
    current_hour = current_time.hour
    
    context = {
        'day': current_day,
        'hour': current_hour,
        'is_weekend': current_day in ['Saturday', 'Sunday'],
        'is_friday': current_day == 'Friday',
        'is_monday': current_day == 'Monday',
        'is_midweek': current_day in ['Tuesday', 'Wednesday', 'Thursday'],
        'market_hours': 9 <= current_hour <= 16,
        'pre_market': 4 <= current_hour < 9,
        'after_market': 16 < current_hour <= 20
    }
    
    # Generate timing recommendations
    if context['is_weekend']:
        context['warning'] = "⏰ **Weekend Analysis**: Patterns based on Friday's close. Monitor Monday gap risk."
        context['recommendation'] = "📋 **Action**: Review patterns, prepare watchlist. Wait for Monday confirmation before entry."
        context['gap_risk'] = "HIGH - Weekend news can cause significant gaps"
        context['entry_timing'] = "Wait for Monday open confirmation"
        
    elif context['is_monday']:
        if context['pre_market']:
            context['warning'] = "🌅 **Monday Pre-Market**: Watch for gaps that might invalidate weekend patterns."
            context['recommendation'] = "📊 **Action**: Check pre-market levels vs. pattern entry points."
            context['gap_risk'] = "ACTIVE - Monitor gap direction"
            context['entry_timing'] = "Wait for market open gap assessment"
        else:
            context['warning'] = "🌅 **Monday Trading**: Gap risk period. Validate patterns post-open."
            context['recommendation'] = "⚡ **Action**: Entry valid if patterns hold after gap settlement."
            context['gap_risk'] = "MEDIUM - Early session volatility"
            context['entry_timing'] = "Patterns valid if holding post-gap"
    
    elif context['is_friday']:
        if context['after_market']:
            context['warning'] = "📅 **Friday After-Hours**: Consider weekend risk for new positions."
            context['recommendation'] = "🛡️ **Action**: Avoid new breakouts. Weekend news risk."
            context['gap_risk'] = "MEDIUM - Weekend headline risk"
            context['entry_timing'] = "Avoid new positions into weekend"
        else:
            context['warning'] = "📅 **Friday Session**: Strong volume required for weekend holds."
            context['recommendation'] = "📊 **Action**: Require exceptional volume (2.0x+) for Friday entries."
            context['gap_risk'] = "MEDIUM - Weekend news risk"
            context['entry_timing'] = "High volume confirmation essential"
    
    elif context['is_midweek']:
        context['warning'] = None
        context['recommendation'] = f"🎯 **{current_day} Trading**: Optimal timing for pattern entries."
        context['gap_risk'] = "LOW - Standard trading conditions"
        context['entry_timing'] = "Patterns active for immediate consideration"
    
    else:
        context['warning'] = None
        context['recommendation'] = "📈 **Active Trading**: Standard market conditions."
        context['gap_risk'] = "LOW - Normal conditions"
        context['entry_timing'] = "Patterns active for entry"
    
    return context

def display_market_context():
    """Display market timing context prominently in the UI"""
    market_context = get_market_context()
    
    st.markdown("### 🕐 Market Timing Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Current Day**: {market_context['day']}")
        st.info(f"**Gap Risk**: {market_context['gap_risk']}")
    
    with col2:
        st.info(f"**Entry Timing**: {market_context['entry_timing']}")
    
    with col3:
        if market_context['is_weekend']:
            st.warning("🔒 **Market Closed**")
        elif market_context['market_hours']:
            st.success("🟢 **Market Open**")
        elif market_context['pre_market']:
            st.info("🟡 **Pre-Market**")
        elif market_context['after_market']:
            st.info("🟡 **After-Hours**")
        else:
            st.error("🔴 **Market Closed**")
    
    if market_context['warning']:
        st.warning(market_context['warning'])
    
    st.success(market_context['recommendation'])
    
    return market_context

def adjust_confidence_for_timing(confidence, pattern_info, market_context):
    """Adjust pattern confidence based on market timing"""
    original_confidence = confidence
    timing_adjustments = []
    
    if market_context['is_weekend']:
        confidence *= 0.95
        timing_adjustments.append("Weekend analysis (-5%)")
    
    elif market_context['is_friday']:
        volume_status = pattern_info.get('volume_status', '')
        
        if 'Exceptional' not in volume_status:
            confidence *= 0.85
            timing_adjustments.append("Friday without exceptional volume (-15%)")
            pattern_info['friday_risk'] = "High volume required for weekend hold"
        else:
            timing_adjustments.append("Friday with exceptional volume (✓)")
    
    elif market_context['is_monday']:
        timing_adjustments.append("Monday gap risk - validate post-open")
        pattern_info['monday_gap_check'] = "Verify patterns hold after gap"
    
    elif market_context['is_midweek']:
        confidence *= 1.02
        timing_adjustments.append("Mid-week optimal timing (+2%)")
    
    pattern_info['timing_adjustments'] = timing_adjustments
    pattern_info['original_confidence'] = original_confidence
    pattern_info['timing_adjusted_confidence'] = confidence
    
    return confidence, pattern_info

def create_demo_data(ticker, period):
    """Create realistic demo data when yfinance is not available"""
    if period == "1wk":
        days_map = {"1y": 52, "6mo": 26, "3mo": 13, "1mo": 4}
        freq = 'W'
    else:
        days_map = {"1y": 252, "6mo": 126, "3mo": 63, "1mo": 22}
        freq = 'D'
    
    days = days_map.get(period.replace("1wk", "1y"), 63)
    dates = pd.date_range(end=datetime.now(), periods=days, freq=freq)
    
    np.random.seed(hash(ticker) % 2147483647)
    base_price = 150 + (hash(ticker) % 100)
    
    returns = np.random.normal(0.001, 0.02, days)
    returns[0] = 0
    
    close_prices = base_price * np.cumprod(1 + returns)
    
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, days))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, days))
    open_mult = 1 + np.random.normal(0, 0.005, days)
    
    data = pd.DataFrame({
        'Open': close_prices * open_mult,
        'High': close_prices * high_mult,
        'Low': close_prices * low_mult,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    data['Low'] = np.minimum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    
    return data

def get_stock_data(ticker, period):
    """Fetch stock data with fallback to demo data"""
    if not YFINANCE_AVAILABLE:
        st.info(f"Using demo data for {ticker}")
        return create_demo_data(ticker, period)
    
    try:
        stock = yf.Ticker(ticker)
        
        # Handle weekly data
        if period == "1wk":
            data = stock.history(period="1y", interval="1wk")
        else:
            data = stock.history(period=period)
            
        if len(data) == 0:
            st.warning(f"No data for {ticker}, using demo data")
            return create_demo_data(ticker, period)
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker}, using demo data")
        return create_demo_data(ticker, period)

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def analyze_volume_pattern(data, pattern_type, pattern_info):
    """Enhanced volume analysis with breakout confirmation and confidence capping"""
    volume_score = 0
    volume_info = {}
    
    if len(data) < 20:
        return volume_score, volume_info
    
    avg_volume_20 = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    recent_volume_5 = data['Volume'].tail(5).mean()
    
    volume_multiplier = current_volume / avg_volume_20
    recent_multiplier = recent_volume_5 / avg_volume_20
    
    volume_info['avg_volume_20'] = avg_volume_20
    volume_info['current_volume'] = current_volume
    volume_info['volume_multiplier'] = volume_multiplier
    volume_info['recent_multiplier'] = recent_multiplier
    
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
    
    if pattern_type == "Bull Flag":
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_start = min(25, len(data) - 10)
                flagpole_end = 15
                
                flagpole_vol = data['Volume'].iloc[-flagpole_start:-flagpole_end].mean()
                flag_vol = data['Volume'].tail(15).mean()
                
                if flagpole_vol > flag_vol * 1.2:
                    volume_score += 20
                    volume_info['flagpole_volume_pattern'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
                elif flagpole_vol > flag_vol * 1.1:
                    volume_score += 10
                    volume_info['moderate_flagpole_volume'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
            except:
                pass
    
    elif pattern_type == "Cup Handle":
        try:
            handle_days = min(30, len(data) // 3)
            if handle_days > 5:
                cup_data = data.iloc[:-handle_days]
                handle_data = data.tail(handle_days)
                
                if len(cup_data) > 10:
                    cup_volume = cup_data['Volume'].mean()
                    handle_volume = handle_data['Volume'].mean()
                    
                    if handle_volume < cup_volume * 0.80:
                        volume_score += 20
                        volume_info['significant_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
                    elif handle_volume < cup_volume * 0.90:
                        volume_score += 15
                        volume_info['moderate_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
        except:
            pass
    
    elif pattern_type == "Flat Top Breakout":
        resistance_tests = data['Volume'].tail(20)
        avg_resistance_volume = resistance_tests.mean()
        
        if current_volume > avg_resistance_volume * 1.4:
            volume_score += 20
            volume_info['breakout_volume_surge'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
        elif current_volume > avg_resistance_volume * 1.2:
            volume_score += 15
            volume_info['moderate_breakout_volume'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
    
    elif pattern_type == "Inside Bar":
        # Inside bar specific volume analysis
        if 'inside_bar_range' in pattern_info:
            # Prefer lower volume during consolidation
            if volume_multiplier < 0.8:
                volume_score += 15
                volume_info['consolidation_volume'] = True
            elif volume_multiplier < 1.0:
                volume_score += 10
                volume_info['quiet_consolidation'] = True
            
            # Check for volume expansion on potential breakout
            if volume_multiplier >= 1.5:
                volume_score += 15
                volume_info['breakout_volume_expansion'] = True
    
    volume_trend = data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()
    if volume_trend > 1.1:
        volume_score += 5
        volume_info['increasing_volume_trend'] = True
    elif volume_trend < 0.9:
        volume_score += 5
        volume_info['decreasing_volume_trend'] = True
    
    return volume_score, volume_info

def detect_inside_bar(data, macd_line, signal_line, histogram, market_context, timeframe="daily"):
    """Detect Inside Bar pattern - buy-only with specific entry rules and color requirements"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 5:
        return confidence, pattern_info
    
    # Adjust lookback based on timeframe
    if timeframe == "1wk":
        max_lookback_range = range(-1, -7, -1)  # Look back 6 weeks maximum
        aging_threshold = -8  # Pattern stale after 8 weeks
        pattern_info['timeframe'] = 'Weekly'
    else:
        max_lookback_range = range(-1, -5, -1)  # Look back 4 days maximum  
        aging_threshold = -6  # Pattern stale after 6 days
        pattern_info['timeframe'] = 'Daily'
    
    # Look for inside bar pattern (max 2 inside bars)
    mother_bar_idx = None
    inside_bars_count = 0
    inside_bar_indices = []
    
    # Start from the most recent bar and look backwards
    for i in max_lookback_range:
        try:
            current_bar = data.iloc[i]
            previous_bar = data.iloc[i-1]
            
            # Check if current bar is inside previous bar
            is_inside = (current_bar['High'] <= previous_bar['High'] and 
                        current_bar['Low'] >= previous_bar['Low'] and
                        current_bar['High'] < previous_bar['High'] and  # Must be strictly inside
                        current_bar['Low'] > previous_bar['Low'])
            
            # Color validation: Mother bar must be green, inside bar must be red
            mother_is_green = previous_bar['Close'] > previous_bar['Open']
            inside_is_red = current_bar['Close'] < current_bar['Open']
            
            if is_inside and mother_is_green and inside_is_red:
                if inside_bars_count == 0:
                    # First inside bar found, previous bar is mother bar
                    mother_bar_idx = i - 1
                    inside_bar_indices.append(i)
                    inside_bars_count = 1
                elif inside_bars_count == 1 and i == inside_bar_indices[0] - 1:
                    # Second consecutive inside bar (must also be red)
                    inside_bar_indices.append(i)
                    inside_bars_count = 2
                    break  # Max 2 inside bars
                else:
                    break  # Not consecutive, stop looking
            else:
                break  # No valid inside bar (size or color), stop looking
        except (IndexError, KeyError):
            break
    
    if inside_bars_count == 0:
        return confidence, pattern_info
    
    # Get mother bar and inside bar(s) data
    mother_bar = data.iloc[mother_bar_idx]
    latest_inside_bar = data.iloc[inside_bar_indices[0]]  # Most recent inside bar
    
    # Validate color requirements one more time
    mother_is_green = mother_bar['Close'] > mother_bar['Open']
    inside_is_red = latest_inside_bar['Close'] < latest_inside_bar['Open']
    
    if not (mother_is_green and inside_is_red):
        return confidence, pattern_info
    
    # Base confidence for pattern formation
    base_confidence = 35 if timeframe == "1wk" else 30  # Higher base for weekly patterns
    confidence += base_confidence
    
    pattern_info['mother_bar_high'] = mother_bar['High']
    pattern_info['mother_bar_low'] = mother_bar['Low']
    pattern_info['inside_bar_high'] = latest_inside_bar['High']
    pattern_info['inside_bar_low'] = latest_inside_bar['Low']
    pattern_info['inside_bars_count'] = inside_bars_count
    pattern_info['color_validated'] = True
    pattern_info['mother_bar_color'] = 'Green'
    pattern_info['inside_bar_color'] = 'Red'
    
    # Bonus for proper color combination
    confidence += 15
    pattern_info['proper_color_combo'] = True
    
    # Prefer single inside bar over double
    if inside_bars_count == 1:
        confidence += 15
        pattern_info['single_inside_bar'] = True
    else:
        confidence += 10
        pattern_info['double_inside_bar'] = True
    
    # Calculate inside bar size relative to mother bar
    mother_bar_range = mother_bar['High'] - mother_bar['Low']
    inside_bar_range = latest_inside_bar['High'] - latest_inside_bar['Low']
    
    if mother_bar_range > 0:
        size_ratio = inside_bar_range / mother_bar_range
        pattern_info['size_ratio'] = f"{size_ratio:.1%}"
        
        # Prefer smaller inside bars (tighter consolidation)
        # Weekly patterns can tolerate slightly larger inside bars
        tight_threshold = 0.35 if timeframe == "1wk" else 0.30
        good_threshold = 0.55 if timeframe == "1wk" else 0.50
        moderate_threshold = 0.75 if timeframe == "1wk" else 0.70
        
        if size_ratio < tight_threshold:
            confidence += 20
            pattern_info['tight_consolidation'] = True
        elif size_ratio < good_threshold:
            confidence += 15
            pattern_info['good_consolidation'] = True
        elif size_ratio < moderate_threshold:
            confidence += 10
            pattern_info['moderate_consolidation'] = True
        else:
            confidence += 5
    
    # Check position within mother bar (prefer middle positioning)
    mother_bar_midpoint = (mother_bar['High'] + mother_bar['Low']) / 2
    inside_bar_midpoint = (latest_inside_bar['High'] + latest_inside_bar['Low']) / 2
    
    distance_from_middle = abs(inside_bar_midpoint - mother_bar_midpoint) / mother_bar_range
    if distance_from_middle < 0.25:
        confidence += 10
        pattern_info['centered_inside_bar'] = True
    elif distance_from_middle < 0.35:
        confidence += 5
        pattern_info['well_positioned'] = True
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_improving'] = True
    
    # Current price should be near inside bar range for valid setup
    current_price = data['Close'].iloc[-1]
    if current_price >= latest_inside_bar['Low'] * 0.98:
        confidence += 10
        pattern_info['price_in_range'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Inside Bar", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Apply volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    # Pattern age check - timeframe adjusted
    if mother_bar_idx <= aging_threshold:
        aging_penalty = 0.7 if timeframe == "1wk" else 0.8
        confidence *= aging_penalty
        pattern_info['pattern_aging'] = True
        pattern_info['age_periods'] = abs(mother_bar_idx)
    
    # Apply timing adjustments
    confidence, pattern_info = adjust_confidence_for_timing(confidence, pattern_info, market_context)
    
    return confidence, pattern_info

def detect_flat_top(data, macd_line, signal_line, histogram, market_context):
    """Detect flat top with enhanced volume and timing"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 50:
        return confidence, pattern_info
    
    ascent_start = min(45, len(data) - 15)
    ascent_end = 25
    
    start_price = data['Close'].iloc[-ascent_start]
    peak_price = data['High'].iloc[-ascent_start:-ascent_end].max()
    initial_gain = (peak_price - start_price) / start_price
    
    if initial_gain < 0.10:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['initial_ascension'] = f"{initial_gain*100:.1f}%"
    
    descent_data = data.iloc[-ascent_end:-10]
    descent_low = descent_data['Low'].min()
    pullback = (peak_price - descent_low) / peak_price
    
    if pullback < 0.08:
        return confidence, pattern_info
    
    descent_highs = descent_data['High'].rolling(3, center=True).max().dropna()
    if len(descent_highs) >= 2:
        if descent_highs.iloc[-1] < descent_highs.iloc[0] * 0.97:
            confidence += 20
            pattern_info['descending_highs'] = True
    
    current_lows = data.tail(15)['Low'].rolling(3, center=True).min().dropna()
    if len(current_lows) >= 3:
        if current_lows.iloc[-1] > current_lows.iloc[0] * 1.01:
            confidence += 25
            pattern_info['higher_lows'] = True
    
    resistance_level = peak_price
    touches = sum(1 for h in data['High'].tail(20) if h >= resistance_level * 0.98)
    if touches >= 2:
        confidence += 15
        pattern_info['resistance_level'] = resistance_level
        pattern_info['resistance_touches'] = touches
    
    current_price = data['Close'].iloc[-1]
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_level * 0.98), 11)
    
    if days_old > 8:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below support'}
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Flat Top Breakout", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    confidence, pattern_info = adjust_confidence_for_timing(confidence, pattern_info, market_context)
    
    return confidence, pattern_info

def detect_bull_flag(data, macd_line, signal_line, histogram, market_context):
    """Detect bull flag with enhanced volume analysis and timing"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    flagpole_start = min(25, len(data) - 10)
    flagpole_end = 15
    
    start_price = data['Close'].iloc[-flagpole_start]
    peak_price = data['High'].iloc[-flagpole_start:-flagpole_end].max()
    flagpole_gain = (peak_price - start_price) / start_price
    
    if flagpole_gain < 0.08:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['flagpole_gain'] = f"{flagpole_gain*100:.1f}%"
    
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    if -0.15 <= pullback <= 0.05:
        confidence += 20
        pattern_info['flag_pullback'] = f"{pullback*100:.1f}%"
        pattern_info['healthy_pullback'] = True
    
    flag_low = flag_data['Low'].min()
    if current_price < flag_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flag support'}
    
    if current_price < start_price:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flagpole start'}
    
    flag_high = flag_data['High'].max()
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] == flag_high), 11)
    
    if days_old > 10:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    pattern_info['days_since_high'] = days_old
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Bull Flag", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if current_price >= flag_high * 0.95:
        confidence += 10
        pattern_info['near_breakout'] = True
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    confidence, pattern_info = adjust_confidence_for_timing(confidence, pattern_info, market_context)
    
    return confidence, pattern_info

def detect_cup_handle(data, macd_line, signal_line, histogram, market_context):
    """Detect cup handle with enhanced volume analysis and timing"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    max_lookback = min(100, len(data) - 3)
    handle_days = min(30, max_lookback // 3)
    
    cup_data = data.iloc[-max_lookback:-handle_days] if handle_days > 0 else data.iloc[-max_lookback:]
    handle_data = data.tail(handle_days) if handle_days > 0 else data.tail(5)
    
    if len(cup_data) < 15:
        return confidence, pattern_info
    
    cup_start = cup_data['Close'].iloc[0]
    cup_bottom = cup_data['Low'].min()
    cup_right = cup_data['Close'].iloc[-1]
    cup_depth = (max(cup_start, cup_right) - cup_bottom) / max(cup_start, cup_right)
    
    if cup_depth < 0.08 or cup_depth > 0.60:
        return confidence, pattern_info
    
    if cup_right < cup_start * 0.75:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['cup_depth'] = f"{cup_depth*100:.1f}%"
    
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        if handle_depth > 0.25:
            confidence += 10
            pattern_info['deep_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.08:
            confidence += 20
            pattern_info['perfect_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.15:
            confidence += 15
            pattern_info['good_handle'] = f"{handle_depth*100:.1f}%"
        else:
            confidence += 10
            pattern_info['acceptable_handle'] = f"{handle_depth*100:.1f}%"
        
        if handle_days > 25:
            confidence *= 0.8
            pattern_info['long_handle'] = f"{handle_days} days"
        elif handle_days <= 10:
            confidence += 10
            pattern_info['short_handle'] = f"{handle_days} days"
        elif handle_days <= 20:
            confidence += 5
            pattern_info['medium_handle'] = f"{handle_days} days"
    else:
        confidence += 10
        pattern_info['forming_handle'] = "Handle forming"
    
    current_price = data['Close'].iloc[-1]
    breakout_level = max(cup_start, cup_right)
    
    if current_price < breakout_level * 0.70:
        confidence *= 0.7
        pattern_info['far_from_rim'] = True
    else:
        confidence += 5
    
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        if current_price < handle_low * 0.90:
            confidence *= 0.8
            pattern_info['below_handle'] = True
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Cup Handle", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if confidence < 35:
        return confidence, pattern_info
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    confidence, pattern_info = adjust_confidence_for_timing(confidence, pattern_info, market_context)
    
    return confidence, pattern_info

def detect_pattern(data, pattern_type, market_context, timeframe="daily"):
    """Detect patterns with enhanced volume analysis and timing awareness"""
    if len(data) < 10:
        return False, 0, {}
    
    data['RSI'] = calculate_rsi(data)
    macd_line, signal_line, histogram = calculate_macd(data)
    
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    confidence = 0
    pattern_info = {}
    
    if pattern_type == "Flat Top Breakout":
        confidence, pattern_info = detect_flat_top(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence, 100)
        
    elif pattern_type == "Bull Flag":
        confidence, pattern_info = detect_bull_flag(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence * 1.05, 100)
        
    elif pattern_type == "Cup Handle":
        confidence, pattern_info = detect_cup_handle(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence * 1.1, 100)
        
    elif pattern_type == "Inside Bar":
        confidence, pattern_info = detect_inside_bar(data, macd_line, signal_line, histogram, market_context, timeframe)
        confidence = min(confidence, 100)
    
    pattern_info['macd_line'] = macd_line
    pattern_info['signal_line'] = signal_line
    pattern_info['histogram'] = histogram
    
    return confidence >= 55, confidence, pattern_info

def calculate_levels(data, pattern_info, pattern_type):
    """Calculate entry, stop, targets using MEASURED MOVES"""
    current_price = data['Close'].iloc[-1]
    recent_range = data['High'].tail(20) - data['Low'].tail(20)
    avg_range = recent_range.mean()
    volatility_stop_distance = avg_range * 1.5
    
    if pattern_type == "Inside Bar":
        # Inside Bar specific calculations
        inside_bar_high = pattern_info.get('inside_bar_high', current_price)
        inside_bar_low = pattern_info.get('inside_bar_low', current_price * 0.95)
        mother_bar_high = pattern_info.get('mother_bar_high', current_price * 1.05)
        
        # Entry: 5% above inside bar high
        entry = inside_bar_high * 1.05
        
        # Stop: 5% below inside bar low
        stop = inside_bar_low * 0.95
        
        # Target 1: Mother bar high
        target1 = mother_bar_high
        
        # Target 2: 13% above mother bar high
        target2 = mother_bar_high * 1.13
        
        # Target 3: 21% above mother bar high
        target3 = mother_bar_high * 1.21
        
        target_method = "Inside Bar Fixed Targets"
        
        # Calculate risk/reward
        risk_amount = entry - stop
        reward1 = target1 - entry
        reward2 = target2 - entry
        reward3 = target3 - entry
        
        return {
            'entry': entry,
            'stop': stop,
            'target1': target1,
            'target2': target2,
            'target3': target3,
            'risk': risk_amount,
            'reward1': reward1,
            'reward2': reward2,
            'reward3': reward3,
            'rr_ratio1': reward1 / risk_amount if risk_amount > 0 else 0,
            'rr_ratio2': reward2 / risk_amount if risk_amount > 0 else 0,
            'rr_ratio3': reward3 / risk_amount if risk_amount > 0 else 0,
            'target_method': target_method,
            'measured_move': True,
            'volatility_adjusted': False,
            'has_target3': True
        }
    
    elif pattern_type == "Flat Top Breakout":
        entry = pattern_info.get('resistance_level', current_price * 1.01)
        recent_low = data['Low'].tail(15).min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = recent_low * 0.98
        stop = max(volatility_stop, traditional_stop)
        
        min_stop_distance = entry * 0.03
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        if 'resistance_level' in pattern_info:
            support_level = data['Low'].tail(20).max()
            triangle_height = entry - support_level
            triangle_height = max(triangle_height, entry * 0.05)
            target1 = entry + triangle_height
            target2 = entry + (triangle_height * 1.618)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.5)
        
        target_method = "Triangle Height Projection"
        
    elif pattern_type == "Bull Flag":
        flag_high = data['High'].tail(15).max()
        entry = flag_high * 1.005
        flag_low = data['Low'].tail(12).min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = flag_low * 0.98
        stop = max(volatility_stop, traditional_stop)
        
        min_stop_distance = entry * 0.04
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_pct_str = pattern_info['flagpole_gain'].replace('%', '')
                flagpole_pct = float(flagpole_pct_str) / 100
                flagpole_start_price = entry / (1 + flagpole_pct)
                flagpole_height = entry - flagpole_start_price
                flagpole_height = max(flagpole_height, entry * 0.08)
                target1 = entry + flagpole_height
                target2 = entry + (flagpole_height * 1.382)
            except (ValueError, KeyError):
                risk = entry - stop
                target1 = entry + (risk * 2.5)
                target2 = entry + (risk * 4.0)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.5)
            target2 = entry + (risk * 4.0)
        
        target_method = "Flagpole Height Projection"
        
    elif pattern_type == "Cup Handle":
        if 'cup_depth' in pattern_info:
            try:
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                estimated_rim = current_price / (1 - cup_depth_pct * 0.3)
                entry = estimated_rim * 1.005
            except (ValueError, KeyError):
                entry = current_price * 1.02
        else:
            entry = current_price * 1.02
        
        handle_low = data.tail(15)['Low'].min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = handle_low * 0.97
        stop = max(volatility_stop, traditional_stop)
        
        min_stop_distance = entry * 0.05
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        if 'cup_depth' in pattern_info:
            try:
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                cup_depth_dollars = entry * cup_depth_pct
                cup_depth_dollars = max(cup_depth_dollars, entry * 0.10)
                target1 = entry + cup_depth_dollars
                target2 = entry + (cup_depth_dollars * 1.618)
            except (ValueError, KeyError):
                risk = entry - stop
                target1 = entry + (risk * 2.0)
                target2 = entry + (risk * 3.0)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.0)
        
        target_method = "Cup Depth Projection"
    
    else:
        entry = current_price * 1.01
        stop = current_price * 0.95
        target1 = entry + (entry - stop) * 2.0
        target2 = entry + (entry - stop) * 3.0
        target_method = "Traditional 2:1 & 3:1"
    
    # For non-Inside Bar patterns, calculate standard 2-target structure
    risk_amount = entry - stop
    reward1 = target1 - entry
    reward2 = target2 - entry
    
    if risk_amount > 0:
        rr1 = reward1 / risk_amount
        rr2 = reward2 / risk_amount
        
        if rr1 < 1.5:
            target1 = entry + (risk_amount * 1.5)
            reward1 = target1 - entry
            rr1 = 1.5
        
        if rr2 < 2.5:
            target2 = entry + (risk_amount * 2.5)
            reward2 = target2 - entry
            rr2 = 2.5
    else:
        risk_amount = entry * 0.05
        stop = entry - risk_amount
        target1 = entry + (risk_amount * 2.0)
        target2 = entry + (risk_amount * 3.0)
        reward1 = target1 - entry
        reward2 = target2 - entry
        rr1 = 2.0
        rr2 = 3.0
    
    return {
        'entry': entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'risk': risk_amount,
        'reward1': reward1,
        'reward2': reward2,
        'rr_ratio1': reward1 / risk_amount if risk_amount > 0 else 0,
        'rr_ratio2': reward2 / risk_amount if risk_amount > 0 else 0,
        'target_method': target_method,
        'measured_move': True,
        'volatility_adjusted': True,
        'has_target3': False
    }

def create_chart(data, ticker, pattern_type, pattern_info, levels, market_context, timeframe):
    """Create enhanced chart with volume analysis and timing context"""
    timeframe_label = "Weekly" if timeframe == "1wk" else "Daily"
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{ticker} - {pattern_type} ({timeframe_label}) | {levels["target_method"]} | {market_context["day"]}',
            'MACD Analysis', 
            'Volume Profile (20-Period Average)'
        ),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', 
                  line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Trading levels
    fig.add_hline(y=levels['entry'], line_color="green", line_width=2,
                 annotation_text=f"Entry: ${levels['entry']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['stop'], line_color="red", line_width=2,
                 annotation_text=f"Stop: ${levels['stop']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['target1'], line_color="lime", line_width=2,
                 annotation_text=f"Target 1: ${levels['target1']:.2f} ({levels['rr_ratio1']:.1f}:1)", row=1, col=1)
    fig.add_hline(y=levels['target2'], line_color="darkgreen", line_width=1,
                 annotation_text=f"Target 2: ${levels['target2']:.2f} ({levels['rr_ratio2']:.1f}:1)", row=1, col=1)
    
    # Add Target 3 for Inside Bar patterns
    if levels.get('has_target3'):
        fig.add_hline(y=levels['target3'], line_color="purple", line_width=1,
                     annotation_text=f"Target 3: ${levels['target3']:.2f} ({levels['rr_ratio3']:.1f}:1)", row=1, col=1)
    
    # Pattern-specific annotations
    if pattern_type == "Inside Bar":
        # Highlight mother bar and inside bar ranges
        mother_bar_high = pattern_info.get('mother_bar_high')
        mother_bar_low = pattern_info.get('mother_bar_low')
        inside_bar_high = pattern_info.get('inside_bar_high')
        inside_bar_low = pattern_info.get('inside_bar_low')
        
        if mother_bar_high and mother_bar_low:
            fig.add_hline(y=mother_bar_high, line_color="blue", line_width=1, line_dash="dash",
                         annotation_text=f"Mother Bar High: ${mother_bar_high:.2f}", row=1, col=1)
            fig.add_hline(y=mother_bar_low, line_color="blue", line_width=1, line_dash="dash",
                         annotation_text=f"Mother Bar Low: ${mother_bar_low:.2f}", row=1, col=1)
        
        if inside_bar_high and inside_bar_low:
            fig.add_hline(y=inside_bar_high, line_color="yellow", line_width=1, line_dash="dot",
                         annotation_text=f"Inside Bar High: ${inside_bar_high:.2f}", row=1, col=1)
        
        # Consolidation annotation
        consolidation_info = f"Consolidation: {pattern_info.get('size_ratio', 'N/A')}"
        if pattern_info.get('inside_bars_count', 0) > 1:
            consolidation_info += f" | {pattern_info['inside_bars_count']} Inside Bars"
        
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=consolidation_info,
            showarrow=True, arrowhead=2, arrowcolor="blue",
            bgcolor="rgba(0,0,255,0.1)", bordercolor="blue"
        )
    
    # Market timing context annotation
    timing_color = 'red' if market_context['is_weekend'] else 'orange' if market_context['is_friday'] else 'yellow' if market_context['is_monday'] else 'lightgreen'
    fig.add_annotation(
        x=data.index[-15], y=levels['entry'] * 0.98,
        text=f"{market_context['entry_timing']}",
        showarrow=True, arrowhead=2, arrowcolor=timing_color,
        bgcolor=f"rgba(255,255,255,0.8)", bordercolor=timing_color,
        font=dict(color=timing_color, size=10)
    )
    
    # Volume status annotation
    volume_status = pattern_info.get('volume_status', 'Unknown Volume')
    volume_color = 'lime' if pattern_info.get('exceptional_volume') else 'orange' if pattern_info.get('strong_volume') else 'yellow' if pattern_info.get('good_volume') else 'red'
    
    fig.add_annotation(
        x=data.index[-10], y=levels['entry'] * 1.02,
        text=f"{volume_status}",
        showarrow=True, arrowhead=2, arrowcolor=volume_color,
        bgcolor=f"rgba(255,255,255,0.8)", bordercolor=volume_color,
        font=dict(color=volume_color, size=12)
    )
    
    # Pattern-specific annotations for other patterns
    if pattern_type == "Bull Flag" and 'flagpole_gain' in pattern_info:
        flagpole_height = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Measured Move: ${flagpole_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Cup Handle" and 'cup_depth' in pattern_info:
        cup_move = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Cup Depth Move: ${cup_move:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Flat Top Breakout":
        triangle_height = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Triangle Height: ${triangle_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    # MACD chart
    macd_line = pattern_info['macd_line']
    signal_line = pattern_info['signal_line']
    histogram = pattern_info['histogram']
    
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='Signal', line=dict(color='red')), row=2, col=1)
    
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color=colors, opacity=0.6), row=2, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    
    # Volume chart with color coding
    volume_colors = []
    avg_volume = data['Volume'].rolling(window=20).mean()
    
    for i, vol in enumerate(data['Volume']):
        if i >= 19:
            if vol >= avg_volume.iloc[i] * 2.0:
                volume_colors.append('darkgreen')
            elif vol >= avg_volume.iloc[i] * 1.5:
                volume_colors.append('green')
            elif vol >= avg_volume.iloc[i] * 1.3:
                volume_colors.append('lightgreen')
            else:
                volume_colors.append('red')
        else:
            volume_colors.append('blue')
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                        marker_color=volume_colors, opacity=0.7), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=avg_volume, name='20-Period Avg', 
                            line=dict(color='black', width=2, dash='dash')), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def main():
    st.title("Pro Pattern Detector v7.0")
    st.markdown("**Inside Bar Pattern & Multi-Timeframe Analysis** - Professional Pattern Recognition with Consolidation Breakouts")
    
    if not YFINANCE_AVAILABLE:
        st.warning("Demo Mode: Using simulated data (yfinance not available)")
    
    st.error("""
    DISCLAIMER: Educational purposes only. Not financial advice. 
    Trading involves substantial risk. Consult professionals before trading.
    """)
    
    # Market Timing Context Display
    market_context = display_market_context()
    
    # Info box about new features
    with st.expander("What's New in v7.0 - Inside Bar Pattern & Multi-Timeframe"):
        st.markdown("""
        ### Inside Bar Pattern Detection
        
        **Pattern Structure**:
        - **Consolidation Signal**: 1-2 inside bars within mother bar range
        - **Buy-Only Strategy**: Long positions only
        - **Conservative Entry**: 5% above inside bar high (reduces false breakouts)
        - **Fixed Stop**: 5% below inside bar low
        
        **Triple Target System**:
        - **Target 1**: Mother bar high (measured move)
        - **Target 2**: 13% above mother bar high
        - **Target 3**: 21% above mother bar high
        
        **Multi-Timeframe Support**:
        - **Daily Charts**: Standard swing trading timeframe
        - **Weekly Charts**: Position trading opportunities
        - **Enhanced Analysis**: Pattern detection across timeframes
        
        **Volume & Timing Integration**:
        - Same confidence scoring system as other patterns
        - Volume confirmation requirements
        - Market timing adjustments (weekend/Friday/Monday)
        - Prefers smaller inside bars (tighter consolidation)
        
        **Risk/Reward Characteristics**:
        - Typically 2:1 to 4:1 risk/reward ratios
        - Conservative entry reduces whipsaw trades
        - Multiple targets for position scaling
        
        This adds a fourth pattern type focused on consolidation breakouts rather than trend continuation patterns.
        """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    patterns = ["Flat Top Breakout", "Bull Flag", "Cup Handle", "Inside Bar"]
    selected_patterns = st.sidebar.multiselect(
        "Select Patterns:", patterns, default=["Flat Top Breakout", "Bull Flag", "Inside Bar"]
    )
    
    tickers = st.sidebar.text_input("Tickers:", "AAPL,MSFT,NVDA")
    
    # Updated period options with weekly support
    period_options = ["1mo", "3mo", "6mo", "1y", "1wk (Weekly)"]
    period_display = st.sidebar.selectbox("Period:", period_options, index=1)
    period = "1wk" if period_display == "1wk (Weekly)" else period_display
    
    min_confidence = st.sidebar.slider("Min Confidence:", 45, 85, 55)
    
    # Volume filter options
    st.sidebar.subheader("Volume Filters")
    require_volume = st.sidebar.checkbox("Require Volume Confirmation", value=False)
    volume_threshold = st.sidebar.selectbox("Volume Threshold:", 
                                          ["1.3x (Good)", "1.5x (Strong)", "2.0x (Exceptional)"], 
                                          index=0)
    
    # Timing filter options
    st.sidebar.subheader("Timing Filters")
    show_timing_adjustments = st.sidebar.checkbox("Show Timing Adjustments", value=True)
    
    if st.sidebar.button("Analyze", type="primary"):
        if tickers and selected_patterns:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            
            st.header("Pattern Analysis Results")
            results = []
            
            for ticker in ticker_list:
                st.subheader(f"{ticker}")
                
                data = get_stock_data(ticker, period)
                if data is not None and len(data) >= 10:
                    
                    for pattern in selected_patterns:
                        detected, confidence, info = detect_pattern(data, pattern, market_context, period)
                        
                        # Apply volume filter
                        skip_pattern = False
                        if require_volume:
                            volume_multiplier = info.get('volume_multiplier', 0)
                            threshold_map = {"1.3x (Good)": 1.3, "1.5x (Strong)": 1.5, "2.0x (Exceptional)": 2.0}
                            required_threshold = threshold_map[volume_threshold]
                            
                            if volume_multiplier < required_threshold:
                                skip_pattern = True
                                st.info(f"{pattern}: {confidence:.0f}% - Filtered by volume requirement")
                                continue
                        
                        if detected and confidence >= min_confidence:
                            levels = calculate_levels(data, info, pattern)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Enhanced confidence display with timing context
                                if confidence >= 80:
                                    st.success(f"{pattern} DETECTED")
                                elif confidence >= 70:
                                    st.success(f"{pattern} DETECTED")
                                else:
                                    st.info(f"{pattern} DETECTED")
                                
                                # Display timing-adjusted confidence
                                if show_timing_adjustments and 'timing_adjusted_confidence' in info:
                                    original_conf = info['original_confidence']
                                    adjusted_conf = info['timing_adjusted_confidence']
                                    if abs(original_conf - adjusted_conf) > 0.5:
                                        st.metric("Confidence", f"{confidence:.0f}%", 
                                                f"{adjusted_conf - original_conf:+.0f}% (timing)")
                                    else:
                                        st.metric("Confidence", f"{confidence:.0f}%")
                                else:
                                    st.metric("Confidence", f"{confidence:.0f}%")
                                
                                # Volume status display
                                volume_status = info.get('volume_status', 'Unknown')
                                if info.get('exceptional_volume'):
                                    st.success(f"{volume_status}")
                                elif info.get('strong_volume'):
                                    st.success(f"{volume_status}")
                                elif info.get('good_volume'):
                                    st.info(f"{volume_status}")
                                else:
                                    st.warning(f"{volume_status}")
                                
                                # Show confidence capping and timing adjustments
                                if info.get('confidence_capped'):
                                    st.warning(f"Capped: {info['confidence_capped']}")
                                
                                # Display timing adjustments
                                if show_timing_adjustments and 'timing_adjustments' in info:
                                    with st.expander("Timing Details"):
                                        for adjustment in info['timing_adjustments']:
                                            st.write(f"• {adjustment}")
                                
                                # Special Friday/Monday warnings
                                if info.get('friday_risk'):
                                    st.warning(f"{info['friday_risk']}")
                                if info.get('monday_gap_check'):
                                    st.info(f"{info['monday_gap_check']}")
                                
                                # Trading levels
                                st.write("**Trading Levels:**")
                                st.write(f"**Entry**: ${levels['entry']:.2f}")
                                st.write(f"**Stop**: ${levels['stop']:.2f}")
                                st.write(f"**Target 1**: ${levels['target1']:.2f}")
                                st.write(f"**Target 2**: ${levels['target2']:.2f}")
                                if levels.get('has_target3'):
                                    st.write(f"**Target 3**: ${levels['target3']:.2f}")
                                
                                st.write("**Risk/Reward:**")
                                st.write(f"**T1 R/R**: {levels['rr_ratio1']:.1f}:1")
                                st.write(f"**T2 R/R**: {levels['rr_ratio2']:.1f}:1")
                                if levels.get('has_target3'):
                                    st.write(f"**T3 R/R**: {levels['rr_ratio3']:.1f}:1")
                                
                                st.info(f"**Method**: {levels['target_method']}")
                            
                            with col2:
                                # Enhanced pattern information with timing details
                                
                                # Market timing context
                                st.write("**Market Context:**")
                                st.write(f"• **Gap Risk**: {market_context['gap_risk']}")
                                st.write(f"• **Entry Timing**: {market_context['entry_timing']}")
                                
                                # Pattern-specific information
                                if pattern == "Inside Bar":
                                    if info.get('single_inside_bar'):
                                        st.write("Single inside bar (preferred)")
                                    elif info.get('double_inside_bar'):
                                        st.write("Double inside bar")
                                    if info.get('size_ratio'):
                                        st.write(f"Consolidation: {info['size_ratio']}")
                                    if info.get('tight_consolidation'):
                                        st.success("Tight consolidation")
                                    if info.get('color_validated'):
                                        st.success("Mother Bar: Green | Inside Bar: Red")
                                    st.success("**Triple Targets**: T1 Mother Bar, T2 +13%, T3 +21%")
                                
                                elif info.get('initial_ascension'):
                                    st.write(f"🚀 Initial rise: {info['initial_ascension']}")
                                if info.get('flagpole_gain'):
                                    st.write(f"🚀 Flagpole: {info['flagpole_gain']}")
                                    st.success(f"📏 **Measured Move**: ${levels['reward1']:.2f}")
                                if info.get('cup_depth'):
                                    st.write(f"☕ Cup depth: {info['cup_depth']}")
                                    st.success(f"📏 **Measured Move**: ${levels['reward1']:.2f}")
                                
                                # Technical indicators
                                if info.get('macd_bullish'):
                                    st.write("📈 MACD bullish")
                                if info.get('momentum_recovering'):
                                    st.write("📈 Momentum recovering")
                                if info.get('near_breakout'):
                                    st.write("🎯 Near breakout")
                            
                            # Chart with timing context
                            fig = create_chart(data, ticker, pattern, info, levels, market_context, period)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to results with timing information
                            timing_status = f"{market_context['day']} ({market_context['gap_risk']} Gap Risk)"
                            
                            result_dict = {
                                'Ticker': ticker,
                                'Pattern': pattern,
                                'Timeframe': 'Weekly' if period == '1wk' else 'Daily',
                                'Confidence': f"{confidence:.0f}%",
                                'Timing': timing_status,
                                'Volume': info.get('volume_status', 'Unknown'),
                                'Entry': f"${levels['entry']:.2f}",
                                'Stop': f"${levels['stop']:.2f}",
                                'Target 1': f"${levels['target1']:.2f}",
                                'Target 2': f"${levels['target2']:.2f}",
                                'R/R 1': f"{levels['rr_ratio1']:.1f}:1",
                                'R/R 2': f"{levels['rr_ratio2']:.1f}:1",
                                'Risk': f"${levels['risk']:.2f}",
                                'Method': levels['target_method']
                            }
                            
                            # Add Target 3 for Inside Bar patterns
                            if levels.get('has_target3'):
                                result_dict['Target 3'] = f"${levels['target3']:.2f}"
                                result_dict['R/R 3'] = f"{levels['rr_ratio3']:.1f}:1"
                            
                            results.append(result_dict)
                        else:
                            if not skip_pattern:
                                st.info(f"⏸ {pattern}: {confidence:.0f}% (below threshold)")
                else:
                    st.error(f"⏸ Insufficient data for {ticker}")
            
            # Summary with timing context
            if results:
                st.header("📋 Summary")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Patterns", len(results))
                with col2:
                    scores = [int(r['Confidence'].replace('%', '')) for r in results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    st.metric("Avg Confidence", f"{avg_score:.0f}%")
                with col3:
                    if results:
                        ratios = [float(r['R/R 1'].split(':')[0]) for r in results]
                        avg_rr = sum(ratios) / len(ratios) if ratios else 0
                        st.metric("Avg R/R T1", f"{avg_rr:.1f}:1")
                with col4:
                    high_vol_count = sum(1 for r in results if 'Strong' in r['Volume'] or 'Exceptional' in r['Volume'])
                    vol_quality = (high_vol_count / len(results)) * 100 if results else 0
                    st.metric("High Volume %", f"{vol_quality:.0f}%")
                with col5:
                    low_risk_count = sum(1 for r in results if 'LOW' in r['Timing'])
                    timing_quality = (low_risk_count / len(results)) * 100 if results else 0
                    st.metric("Low Gap Risk %", f"{timing_quality:.0f}%")
                
                # Pattern distribution
                if len(results) > 1:
                    st.subheader("📊 Pattern Distribution")
                    pattern_counts = {}
                    timeframe_counts = {}
                    
                    for result in results:
                        pattern = result['Pattern']
                        timeframe = result['Timeframe']
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**By Pattern:**")
                        for pattern, count in pattern_counts.items():
                            pct = (count / len(results)) * 100
                            st.write(f"• {pattern}: {count} ({pct:.0f}%)")
                    
                    with col2:
                        st.write("**By Timeframe:**")
                        for timeframe, count in timeframe_counts.items():
                            pct = (count / len(results)) * 100
                            st.write(f"• {timeframe}: {count} ({pct:.0f}%)")
                
                # Download with timing data
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"patterns_v7_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
            else:
                st.info("📊 No patterns detected. Try lowering the confidence threshold or adjusting volume filters.")

if __name__ == "__main__":
    main()
