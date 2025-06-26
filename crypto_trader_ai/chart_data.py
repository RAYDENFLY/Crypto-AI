import ccxt
import datetime

def get_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=50):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Format data untuk chart.js atau plotly
    formatted = [
        {
            'timestamp': datetime.datetime.fromtimestamp(candle[0] / 1000).isoformat(),
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        }
        for candle in data
    ]
    
    return formatted
