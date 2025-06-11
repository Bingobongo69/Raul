# -----------------------------------------------------------------
# DATEI: backtest-neu.py (Version 19 - Zeitzonen-Fix)
# -----------------------------------------------------------------
import os
import ccxt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

# --- 1. Konfiguration ---
START_KAPITAL = 400.0; POSITIONSGROESSE_PROZENT = 0.05; MIN_POSITIONSGROESSE_EUR = 5.0
TAKER_GEBUEHR = 0.012
SYMBOL = 'BTC/EUR'; TIMEFRAME = '1h'; START_DATUM_STR = '2021-01-01T00:00:00Z'

# --- 2. Strategie-Parameter ---
EMA_PERIOD = 200; SWING_LOOKBACK = 15; RISK_REWARD_RATIO = 2.0

# --- Deine Klassen und Funktionen ---

def daten_beschaffen(symbol, timeframe, start_date_str, file_name):
    if os.path.exists(file_name):
        print(f"Daten-Datei '{file_name}' existiert bereits. Lade aus Datei.")
        try:
            # Stelle sicher, dass die Zeitstempel als UTC gelesen werden
            return pd.read_csv(file_name, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Konnte Datei nicht lesen: {e}"); return None

    print(f"Starte Download von Coinbase..."); exchange = ccxt.coinbase()
    since = exchange.parse8601(start_date_str); all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv); since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
            print(f"  Lade Datenblock, letztes Datum: {datetime.fromtimestamp(ohlcv[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')}...")
        except Exception as e:
            print(f"Fehler: {e}"); break
    if not all_ohlcv: return None
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # *** HIER IST DIE KORREKTUR 1 ***
    # Konvertiere Timestamp und mache ihn explizit Zeitzonen-bewusst (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]; df.sort_index(inplace=True)
    
    # Filtere den DataFrame, um sicherzustellen, dass er am Zieldatum beginnt
    target_start_date = pd.to_datetime(start_date_str)
    df = df[df.index >= target_start_date]
    
    df.to_csv(file_name)
    print(f"\nDaten in '{file_name}' gespeichert."); return df

class MinRiskSizer(bt.Sizer):
    params = (('risk_percent', 0.05), ('min_trade_size_eur', 5.0),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        size_in_eur = self.broker.get_value() * self.p.risk_percent
        if size_in_eur < self.p.min_trade_size_eur: size_in_eur = self.p.min_trade_size_eur
        if size_in_eur > cash: return 0
        return size_in_eur / data.close[0]

class FvgStrategy(bt.Strategy):
    params = (('ema_period', EMA_PERIOD), ('swing_lookback', SWING_LOOKBACK), ('rr_ratio', RISK_REWARD_RATIO),)
    def __init__(self):
        self.ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.ema_period)
        self.order = None
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None
    def next(self):
        if self.position or self.order: return
        if not (self.data.close[0] > self.ema[0]): return
        for i in range(2, 7):
            if len(self) < i + 1: continue
            if self.data.low[-i+2] > self.data.high[-i]:
                fvg_high = self.data.low[-i+2]; fvg_low = self.data.high[-i]
                if self.data.low[0] > fvg_high: continue
                lookback_start = -i - self.p.swing_lookback
                if len(self) < abs(lookback_start): continue
                lookback_data = self.data.low.get(ago=lookback_start, size=self.p.swing_lookback)
                if not lookback_data: continue
                swing_low = min(lookback_data)
                stop_loss = fvg_low
                for k in range(i + 1, i + 6):
                    if len(self) < k + 1: continue
                    if self.data.low[-k] < swing_low: stop_loss = self.data.low[-k]; break
                if fvg_high > stop_loss:
                    entry_price = fvg_high
                    take_profit = entry_price + (entry_price - stop_loss) * self.p.rr_ratio
                    self.order = self.buy(sl=stop_loss, tp=take_profit); return

# --- Hauptprogramm ---
if __name__ == '__main__':
    data_file_name = f"coinbase_{SYMBOL.replace('/', '')}_{TIMEFRAME}_data.csv"
    if os.path.exists(data_file_name): os.remove(data_file_name)
    
    dataframe = daten_beschaffen(SYMBOL, TIMEFRAME, START_DATUM_STR, data_file_name)
    if dataframe is None or dataframe.empty: exit()
    
    start_date = dataframe.index[0]; end_date = dataframe.index[-1]
    cerebro = bt.Cerebro()
    cerebro.broker.setcommission(commission=TAKER_GEBUEHR)
    
    # *** HIER IST DIE KORREKTUR 2 ***
    # Wir übergeben den bereits korrekt formatierten DataFrame
    data = bt.feeds.PandasData(dataname=dataframe)

    cerebro.adddata(data); cerebro.addstrategy(FvgStrategy); cerebro.addsizer(MinRiskSizer)
    cerebro.broker.set_cash(START_KAPITAL)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"\nStarte Backtest...")
    results = cerebro.run(); strat = results[0]
    
    end_value = cerebro.broker.getvalue()
    print("\n--- Backtest Ergebnisse ---"); print(f"Endkapital: {end_value:.2f}€"); print(f"Bruttogewinn/-verlust: {end_value - START_KAPITAL:.2f}€")
    trade_info = strat.analyzers.trades.get_analysis()
    if 'total' in trade_info and trade_info.total.total > 0:
        wins = trade_info.won.total if 'won' in trade_info and 'total' in trade_info.won else 0
        losses = trade_info.lost.total if 'lost' in trade_info and 'total' in trade_info.lost else 0
        win_rate = (wins / trade_info.total.total) * 100
        avg_win = trade_info.won.pnl.average if wins > 0 else 0
        avg_loss = abs(trade_info.lost.pnl.average) if losses > 0 else 0
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        print(f"Win Rate: {win_rate:.2f}%"); print(f"Anzahl Trades: {trade_info.total.total}"); print(f"Profit Ratio: {profit_ratio:.2f}")
    else: print("Keine Trades ausgeführt.")