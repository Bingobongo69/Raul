# backtester.py (FINALE VERSION MIT PARAMETER-OPTIMIERUNG & REALISTISCHEN KOSTEN)

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Globale Konfiguration ---
SYMBOL = 'BTC-EUR'
TIMEFRAME = '1h'
START_DATE = '2022-01-01T00:00:00Z'

# --- REALISTISCHE WIRTSCHAFTSPARAMETER (ANGEPASST) ---
INITIAL_CAPITAL = 400.0
TAKER_FEE_PERCENT = 0.6  # Gebühr für den Market-Buy (Einstieg)
MAKER_FEE_PERCENT = 1.2  # Gebühr für den Limit-Sell (Ausstieg)

# --- Standard-Parameter für den Backtest ---
base_params = {
    "EMA_PERIOD": 200,
    "SWING_LOOKBACK": 15,
    "RISK_REWARD_RATIO": 2.5,
    "ATR_PERIOD": 14,
    "MIN_ATR_MULTIPLIER": 1.0,
    "MAX_ATR_MULTIPLIER": 4.0,
    "RISK_PER_TRADE_PERCENT": 5.0, # Angepasst auf ein vernünftigeres Risiko
    "TRAILING_STOP_RR": 1.5
}

# --- DATENLADE-FUNKTION ---
def fetch_historical_data(symbol, timeframe, start_date):
    print("Lade historische Daten...")
    exchange = ccxt.coinbase()
    since = exchange.parse8601(start_date)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=300)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Lade Daten ab {exchange.iso8601(since)}...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}"); break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Daten erfolgreich geladen. {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")
    return df

# --- STRATEGIE-FUNKTIONEN ---
def add_indicators(df, params):
    df[f'ema_{params["EMA_PERIOD"]}'] = df['close'].ewm(span=params["EMA_PERIOD"], adjust=False).mean()
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.ewm(span=params["ATR_PERIOD"], adjust=False).mean()
    return df

def identify_fvg(df):
    df['bull_fvg_low'] = np.nan
    df['bull_fvg_high'] = np.nan
    for i in range(2, len(df)):
        candle1_high = df['high'].iloc[i-2]
        candle3_low = df['low'].iloc[i]
        if candle3_low > candle1_high:
            df.loc[df.index[i-1], 'bull_fvg_low'] = candle1_high
            df.loc[df.index[i-1], 'bull_fvg_high'] = candle3_low
    return df

def check_for_signal(df_slice, params):
    current_candle = df_slice.iloc[-1]
    last_fvg_candle = df_slice.iloc[-2]
    entry_price, stop_loss, take_profit, trade_reason = 0, 0, 0, ""
    
    last_ema = current_candle[f'ema_{params["EMA_PERIOD"]}']
    if current_candle['close'] > last_ema and pd.notna(last_fvg_candle['bull_fvg_low']):
        fvg_high = last_fvg_candle['bull_fvg_high']
        if current_candle['low'] <= fvg_high:
            entry_price, stop_loss = fvg_high, last_fvg_candle['low']
            if entry_price > stop_loss:
                take_profit = entry_price + (entry_price - stop_loss) * params["RISK_REWARD_RATIO"]
                trade_reason = "Trend-Following (EMA)"
                
    lookback_df = df_slice.iloc[-(params["SWING_LOOKBACK"] + 3) : -3]
    if not lookback_df.empty:
        swing_low_price = lookback_df['low'].min()
        sweep_candle = df_slice.iloc[-3]
        if sweep_candle['low'] < swing_low_price and pd.notna(last_fvg_candle['bull_fvg_low']):
            fvg_high = last_fvg_candle['bull_fvg_high']
            if current_candle['low'] <= fvg_high:
                entry_price, stop_loss = fvg_high, sweep_candle['low']
                if entry_price > stop_loss:
                    take_profit = entry_price + (entry_price - stop_loss) * params["RISK_REWARD_RATIO"]
                    trade_reason = "Liquidity Sweep"

    if trade_reason:
        stop_loss_distance = entry_price - stop_loss
        current_atr = current_candle['atr']
        if stop_loss_distance > 0 and pd.notna(current_atr):
            is_large_enough = stop_loss_distance > params["MIN_ATR_MULTIPLIER"] * current_atr
            is_not_too_large = stop_loss_distance < params["MAX_ATR_MULTIPLIER"] * current_atr
            if is_large_enough and is_not_too_large:
                return {'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'reason': trade_reason}
    return None

# --- BACKTEST-FUNKTION (ANGEPASST AN NEUE KOSTENLOGIK) ---
def run_backtest(df, params):
    trades = []
    capital = INITIAL_CAPITAL
    active_trade = None
    trade_state = 0

    for i in range(params["EMA_PERIOD"], len(df)):
        if active_trade:
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Trailing Stop Logik
            if trade_state == 1:
                break_even_target = active_trade['entry'] + (active_trade['entry'] - active_trade['initial_sl'])
                if current_high >= break_even_target:
                    active_trade['sl'] = active_trade['entry']
                    trade_state = 2

            if trade_state in [1, 2] and current_high >= active_trade['tp']:
                new_sl = active_trade['entry'] + (active_trade['entry'] - active_trade['initial_sl']) * params["TRAILING_STOP_RR"]
                active_trade['sl'] = new_sl
                trade_state = 3
            
            if trade_state == 3:
                new_sl = df['low'].iloc[i-1]
                if new_sl > active_trade['sl']:
                    active_trade['sl'] = new_sl

            # Trade schließen
            if current_low <= active_trade['sl']:
                exit_price = active_trade['sl']
                revenue = exit_price * active_trade['amount']
                exit_fee = revenue * (MAKER_FEE_PERCENT / 100.0)
                net_pnl = (revenue - exit_fee) - active_trade['cost']
                capital += net_pnl
                
                status = 'T-SL' if exit_price > active_trade['entry'] else 'BE' if exit_price == active_trade['entry'] else 'SL'
                active_trade.update({'exit_price': exit_price, 'pnl': net_pnl, 'status': status, 'exit_time': df.index[i]})
                trades.append(active_trade)
                active_trade = None
                trade_state = 0
        
        if not active_trade:
            df_slice = df.iloc[:i+1]
            signal = check_for_signal(df_slice, params)
            if signal:
                risk_amount_eur = capital * (params["RISK_PER_TRADE_PERCENT"] / 100.0)
                stop_loss_distance = signal['entry'] - signal['sl']
                if stop_loss_distance > 0:
                    amount_to_buy = risk_amount_eur / stop_loss_distance
                    cost_brutto = amount_to_buy * signal['entry']
                    entry_fee = cost_brutto * (TAKER_FEE_PERCENT / 100.0)
                    total_cost = cost_brutto + entry_fee

                    if capital > total_cost:
                        capital -= total_cost # Gesamtkosten sofort vom Kapital abziehen
                        active_trade = {'entry_time': df.index[i], 'entry': signal['entry'], 'sl': signal['sl'], 
                                        'initial_sl': signal['sl'], 'tp': signal['tp'], 'reason': signal['reason'], 
                                        'amount': amount_to_buy, 'cost': total_cost, 'status': 'active'}
                        trade_state = 1

    # Kapitalverlauf korrekt berechnen
    capital_df = pd.DataFrame(index=df.index)
    capital_df['capital'] = INITIAL_CAPITAL
    
    if trades:
        # Erstelle eine Serie von PnL-Änderungen an den Ausstiegsdaten
        pnl_series = pd.Series([t['pnl'] for t in trades], index=[t['exit_time'] for t in trades])
        # Addiere die PnL-Änderungen zum Kapital
        pnl_cumsum = pnl_series.cumsum()
        capital_df['pnl_change'] = pnl_cumsum
        # Fülle die Lücken und addiere zum Startkapital
        capital_df['capital'] = capital_df['capital'] + capital_df['pnl_change'].ffill().fillna(0)
        
    return trades, capital_df

# --- ANALYSE-FUNKTION (unverändert) ---
def analyze_results(trades, capital_over_time, params):
    if not trades:
        return {"Endkapital": INITIAL_CAPITAL, "Netto Profit/Verlust": 0, "Gesamtzahl Trades": 0, "Profit Faktor": 0, "Maximaler Drawdown": 0, "Win Rate (inkl. T-SL)": 0}

    trades_df = pd.DataFrame(trades)
    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'])
    net_pnl = trades_df['pnl'].sum()
    total_trades = len(trades_df)
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    peak = capital_over_time['capital'].cummax()
    drawdown = (capital_over_time['capital'] - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    end_capital = capital_over_time['capital'].iloc[-1]
    
    return {
        "Endkapital": round(end_capital, 2), "Netto Profit/Verlust": round(net_pnl, 2),
        "Gesamtzahl Trades": total_trades, "Profit Faktor": round(profit_factor, 2),
        "Maximaler Drawdown": round(max_drawdown, 2), "Win Rate (inkl. T-SL)": round(win_rate, 2)
    }

# --- HAUPTFUNKTION mit Optimierungs-Schleife ---
if __name__ == '__main__':
    historical_df = fetch_historical_data(SYMBOL, TIMEFRAME, START_DATE)
    
    if not historical_df.empty:
        optimization_results = []
        
        rr_to_test = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        
        print("\nStarte Parameter-Optimierung für 1h-Chart (realistische Kosten)...")
        for rr_value in rr_to_test:
            print(f"Teste RR-Ratio: {rr_value}...")
            
            current_params = base_params.copy()
            current_params["RISK_REWARD_RATIO"] = rr_value
            
            df_with_indicators = add_indicators(historical_df.copy(), current_params)
            df_with_fvg = identify_fvg(df_with_indicators)
            
            trade_log, capital_log = run_backtest(df_with_fvg, current_params)
            
            metrics = analyze_results(trade_log, capital_log, current_params)
            metrics['RR_Ratio'] = rr_value
            optimization_results.append(metrics)
            
        print("\n\n" + "="*60)
        print("OPTIMIERUNG ABGESCHLOSSEN - ZUSAMMENFASSUNG (realistische Kosten)")
        print("="*60)
        
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values(by="Endkapital", ascending=False)
        
        cols_to_show = [
            'RR_Ratio', 'Endkapital', 'Netto Profit/Verlust', 'Profit Faktor', 
            'Maximaler Drawdown', 'Win Rate (inkl. T-SL)', 'Gesamtzahl Trades'
        ]
        print(results_df[cols_to_show].to_string(index=False))
