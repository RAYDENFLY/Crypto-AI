from flask import Flask, render_template, request, jsonify
from datetime import datetime
import random
from collections import defaultdict
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env file

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

import traceback

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    context = data.get("context", {})

    if not message:
        return jsonify({"error": "No message provided."}), 400

    try:
        system_prompt = f"""You are a helpful crypto trading assistant.
Here is some context:
Coin: {context.get('coin')}
Timeframe: {context.get('timeframe')}
Price: {context.get('price')}
OHLC: {context.get('ohlc')}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


import requests

@app.route('/api/chart-data')
def chart_data():
    coin = request.args.get('coin', 'bitcoin')
    days = request.args.get('days', '2')  # default 2 hari
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'prices' not in data:
            return jsonify({"error": "Invalid data received from CoinGecko", "raw": data}), 500

        prices_raw = data['prices']  # [[timestamp, price], ...]

        # Build OHLC per hour
        ohlc_map = {}
        for timestamp, price in prices_raw:
            dt = datetime.fromtimestamp(timestamp / 1000)
            key = dt.replace(minute=0, second=0, microsecond=0)
            if key not in ohlc_map:
                ohlc_map[key] = {"o": price, "h": price, "l": price, "c": price}
            else:
                ohlc_map[key]["h"] = max(ohlc_map[key]["h"], price)
                ohlc_map[key]["l"] = min(ohlc_map[key]["l"], price)
                ohlc_map[key]["c"] = price  # update close continuously

        # Final OHLC format for chartjs-chart-financial
        ohlc_data = [{
            "x": key.strftime('%Y-%m-%dT%H:%M:%S'),
            "o": round(val["o"], 2),
            "h": round(val["h"], 2),
            "l": round(val["l"], 2),
            "c": round(val["c"], 2)
        } for key, val in sorted(ohlc_map.items())]

        # For line chart
        prices = [p[1] for p in prices_raw]
        times = [datetime.fromtimestamp(p[0] / 1000).strftime('%H:%M') for p in prices_raw]

        current_price = prices[-1]
        price_change = ((current_price - prices[0]) / prices[0]) * 100

        print('DEBUG ohlc_data:', ohlc_data)
        print('DEBUG prices:', prices)
        print('DEBUG times:', times)

        return jsonify({
            "times": times,
            "prices": prices,
            "ohlc": ohlc_data,
            "currentPrice": current_price,
            "changePercent": price_change
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/binance-price-history')
def binance_price_history():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    interval = request.args.get('interval', '1d')
    limit = 100

    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'

    try:
        response = requests.get(url)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text[:300])

        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.status_code}")

        raw_data = response.json()
        data = [{
            "time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5])
        } for candle in raw_data]

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
