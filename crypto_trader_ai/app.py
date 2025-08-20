from flask import Flask, render_template, request, jsonify
from datetime import datetime
import random
from collections import defaultdict
import os
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import requests
import json
import threading
import time

load_dotenv()  # Load .env file

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock')
def stock_index():
    return render_template('stock_index.html')

@app.route('/stock-standalone')
def stock_index_standalone():
    return render_template('stock_index_standalone.html')

import traceback

NEWS_API_KEY = "10e006394468438e89711e759385265d"
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_JSON_PATH = "news_cache.json"

# Topik yang mempengaruhi pasar kripto
NEWS_TOPICS = [
    "bitcoin", "ethereum", "crypto", "blockchain", "altcoin", "regulation", "perang", "war", "conflict", "inflation", "interest rate", "fed", "bank", "usd", "dollar", "china", "russia", "israel", "palestine", "ukraine", "oil", "gold", "market crash", "hacker", "scam", "exchange", "binance", "coinbase", "etf", "halving", "mining", "regulasi", "krisis", "geopolitik",
    # Saham dan ekonomi global
    "stock market", "saham", "IHSG", "Dow Jones", "Nasdaq", "S&P 500", "bursa efek", "wall street", "emiten", "dividen", "earnings", "IPO", "reksadana", "obligasi", "ekonomi global", "ekonomi indonesia", "ekonomi amerika", "ekonomi cina", "ekonomi eropa", "ekonomi jepang", "ekonomi asia", "ekonomi dunia"
]

def fetch_and_cache_news():
    all_articles = []
    for topic in NEWS_TOPICS:
        try:
            params = {
                "q": topic,
                "apiKey": NEWS_API_KEY,
                "language": "id,en",
                "sortBy": "publishedAt",
                "pageSize": 20
            }
            resp = requests.get(NEWS_API_URL, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                print(f"Fetched {len(articles)} articles for topic '{topic}'")
                for a in articles:
                    article = {
                        "title": a.get("title"),
                        "url": a.get("url"),
                        "publishedAt": a.get("publishedAt"),
                        "topic": topic,
                        "source": a.get("source", {}).get("name", "")
                    }
                    all_articles.append(article)
            else:
                pass  # NewsAPI error, skip topic
        except Exception as e:
            print(f"Exception fetching topic '{topic}': {e}")
            continue
    print(f"Total articles fetched: {len(all_articles)}")
    if all_articles:
        with open(NEWS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        print(f"News cache updated: {NEWS_JSON_PATH}")
    else:
        print("No articles fetched, cache not updated.")

# Scheduler background fetch news setiap jam

def background_news_fetcher():
    while True:
        try:
            fetch_and_cache_news()
        except Exception:
            pass
        time.sleep(3600)  # 1 jam

# Jalankan background fetcher saat server start
threading.Thread(target=background_news_fetcher, daemon=True).start()

def get_news_from_cache(query=None, max_results=10):
    try:
        with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
        if query:
            filtered = [a for a in articles if query.lower() in (a.get("title", "") + a.get("topic", "")).lower()]
        else:
            filtered = articles
        filtered = sorted(filtered, key=lambda x: x.get("publishedAt", ""), reverse=True)
        return filtered[:max_results]
    except Exception:
        return []

def get_news_headlines(query=None, max_results=5, include_finnhub=True, finnhub_symbols=None):
    articles = get_news_from_cache(query, max_results*3)
    # Tambahkan berita Finnhub jika diinginkan
    if include_finnhub:
        if finnhub_symbols is None:
            finnhub_symbols = ["AAPL", "MSFT", "TSLA"]
        finnhub_articles = []
        for symbol in finnhub_symbols:
            finnhub_articles.extend(fetch_finnhub_news(symbol))
        # Hindari duplikat
        for a in finnhub_articles:
            if not any(b.get('title') == a['title'] and b.get('url') == a['url'] for b in articles):
                articles.append(a)
    # Sortir terbaru
    articles = sorted(articles, key=lambda x: str(x.get("publishedAt", "")), reverse=True)
    articles = articles[:max_results]
    if not articles:
        return "Tidak ada berita terbaru."
    headlines = []
    for a in articles:
        headlines.append(f"- {a['title']} ({a['url']})")
    return "\n".join(headlines)

@app.route("/fetch-news")
def fetch_news_api():
    fetch_and_cache_news()
    return jsonify({"status": "ok"})

@app.route("/fetch-finnhub-news")
def fetch_finnhub_news_api():
    fetch_and_cache_finnhub_news()
    return jsonify({"status": "ok"})

@app.route("/get-news")
def get_news():
    query = request.args.get("query")
    symbol = request.args.get("symbol")  # untuk saham
    try:
        limit = int(request.args.get("limit", 10))
    except Exception:
        limit = 10
    since = request.args.get("since")  # format: YYYY-MM-DD
    try:
        # Ambil dari news_cache.json
        try:
            with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
                articles = json.load(f)
            if not isinstance(articles, list):
                articles = []
        except Exception:
            articles = []
        # Ambil dari manual_news (SQLite)
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT title, url, publishedAt, topic, source FROM manual_news")
        manual_articles = [dict(row) for row in c.fetchall()]
        conn.close()
        # Gabungkan dan filter duplikat (berdasarkan title+url)
        all_articles = articles + [a for a in manual_articles if not any(b.get('title') == a['title'] and b.get('url') == a['url'] for b in articles)]
        # Tambahkan berita Finnhub jika symbol saham diberikan
        finnhub_articles = []
        if symbol:
            finnhub_articles = fetch_finnhub_news(symbol)
            # Hindari duplikat
            all_articles += [a for a in finnhub_articles if not any(b.get('title') == a['title'] and b.get('url') == a['url'] for b in all_articles)]
        # Filter by query
        if query:
            all_articles = [a for a in all_articles if query.lower() in (a.get("title", "") + a.get("topic", "")).lower()]
        # Filter by since (jika ada)
        if since:
            all_articles = [a for a in all_articles if a.get("publishedAt") and str(a["publishedAt"])[:10] >= since]
        # Sort by publishedAt descending
        all_articles = sorted(all_articles, key=lambda x: str(x.get("publishedAt", "")), reverse=True)
        return jsonify({"status": "ok", "articles": all_articles[:limit]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    context = data.get("context", {})
    model = data.get("model", "openai")

    if not message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Jika user menulis kata kunci saham, stock, atau berita hari ini, ambil berita sesuai query
        user_query = None
        msg_lower = message.lower()
        if any(x in msg_lower for x in ["saham", "stock", "ihsg", "wall street", "nasdaq", "dow jones", "bursa", "emiten", "dividen", "obligasi", "ekonomi"]):
            user_query = "saham"
        elif "berita hari ini" in msg_lower or "berita terbaru" in msg_lower:
            user_query = None  # Ambil semua berita terbaru
        elif "kripto" in msg_lower or "crypto" in msg_lower or "bitcoin" in msg_lower or "ethereum" in msg_lower:
            user_query = "crypto"
        # Jika tidak ada kata kunci, default: None (ambil semua headline terbaru)
        news = get_news_headlines(query=user_query, max_results=5)
        system_prompt = f"""
You are a helpful financial assistant specialized in analyzing cryptocurrency, stock market, and global economic news.
Use the following context to provide trading advice, news summary, or answer user questions about crypto, stocks, or global markets.

Respond **only in Bahasa Indonesia**.

Context:
- Coin: {context.get('coin')}
- Timeframe (days): {context.get('timeframe')}
- Current price and OHLC data: {context.get('price')} | {context.get('ohlc')}
- Berita terbaru (faktor global, geopolitik, ekonomi, perang, regulasi, saham, dll):
{news}

When responding, please:
- Analyze the price trend and volatility if relevant.
- Summarize or explain news if user asks about news.
- Suggest clear TP, SL, and sell/buy strategies for trading questions.
- Use simple, concise language for the user.
- Jawaban harus mudah dimengerti oleh trader Indonesia.

User message: {message}
"""


        if model == "groq":
            import requests
            groq_api_key = os.getenv("GROQ_API_KEY", "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")
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            }
            groq_resp = requests.post(groq_url, json=payload, headers=headers, timeout=30)
            groq_resp.raise_for_status()
            groq_data = groq_resp.json()
            reply = groq_data["choices"][0]["message"]["content"].strip()
            return jsonify({"reply": reply})

        elif model == "groq_vision":
            image = data.get("image")
            if not image:
                return jsonify({"error": "Image is required for groq_vision model."}), 400
            import requests
            GROQ_API_KEY_VISION = "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"
            GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY_VISION}"
            }
            payload = {
                "model": "meta-llama/llama-4-vision-17b-instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                    ]}
                ]
            }
            resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return jsonify({"reply": reply})
        elif model == "mixtral":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                import torch
                model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                if not hasattr(ask, "mixtral_pipe"):
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                    ask.mixtral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
                prompt = system_prompt + "\n" + message
                result = ask.mixtral_pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
                reply = result[0]["generated_text"][len(prompt):].strip()
                return jsonify({"reply": reply})
            except Exception as e:
                return jsonify({"reply": f"[Mixtral-8x7b error: {str(e)}]"})
        else:
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

DB_PATH = "chats.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )
    """)
    # Tabel manual_news
    c.execute("""
    CREATE TABLE IF NOT EXISTS manual_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        url TEXT,
        publishedAt TEXT,
        topic TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route('/api/chats', methods=['GET', 'POST'])
def chats():
    conn = get_db()
    c = conn.cursor()
    if request.method == 'POST':
        data = request.get_json()
        title = data.get('title', 'New Chat')
        c.execute("INSERT INTO chats (title) VALUES (?)", (title,))
        chat_id = c.lastrowid
        conn.commit()
        c.execute("SELECT * FROM chats WHERE id=?", (chat_id,))
        chat = dict(c.fetchone())
        conn.close()
        return jsonify(chat)
    else:
        c.execute("SELECT * FROM chats ORDER BY created_at DESC")
        chats = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(chats)

@app.route('/api/chats/<int:chat_id>/messages', methods=['GET', 'POST'])
def chat_messages(chat_id):
    conn = get_db()
    c = conn.cursor()
    if request.method == 'POST':
        data = request.get_json()
        sender = data.get('sender', 'user')
        message = data.get('message', '')
        c.execute("INSERT INTO messages (chat_id, sender, message) VALUES (?, ?, ?)", (chat_id, sender, message))
        conn.commit()
        conn.close()
        return jsonify({'status': 'ok'})
    else:
        c.execute("SELECT sender, message, created_at FROM messages WHERE chat_id=? ORDER BY created_at ASC", (chat_id,))
        messages = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(messages)

@app.route('/add-news', methods=['GET', 'POST'])
def add_news():
    message = None
    if request.method == 'POST':
        title = request.form.get('title')
        url = request.form.get('url')
        publishedAt = request.form.get('publishedAt')
        topic = request.form.get('topic')
        source = request.form.get('source')
        # Simpan ke SQLite
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO manual_news (title, url, publishedAt, topic, source) VALUES (?, ?, ?, ?, ?)",
                  (title, url, publishedAt, topic, source))
        conn.commit()
        conn.close()
        # Simpan ke news_cache.json (append jika belum ada)
        try:
            with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
                articles = json.load(f)
            if not isinstance(articles, list):
                articles = []
        except Exception:
            articles = []
        # Cek duplikat berdasarkan title dan url
        exists = any(a.get('title') == title and a.get('url') == url for a in articles)
        if not exists:
            articles.append({
                "title": title,
                "url": url,
                "publishedAt": publishedAt,
                "topic": topic,
                "source": source
            })
            with open(NEWS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
        message = "Berita berhasil ditambahkan!"
    return render_template('add_news.html', message=message)

@app.route('/add-news-standalone', methods=['GET', 'POST'])
def add_news_standalone():
    message = None
    if request.method == 'POST':
        title = request.form.get('title')
        url = request.form.get('url')
        publishedAt = request.form.get('publishedAt')
        topic = request.form.get('topic')
        source = request.form.get('source')
        # Simpan ke SQLite
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO manual_news (title, url, publishedAt, topic, source) VALUES (?, ?, ?, ?, ?)",
                  (title, url, publishedAt, topic, source))
        conn.commit()
        conn.close()
        # Simpan ke news_cache.json (append jika belum ada)
        try:
            with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
                articles = json.load(f)
            if not isinstance(articles, list):
                articles = []
        except Exception:
            articles = []
        # Cek duplikat berdasarkan title dan url
        exists = any(a.get('title') == title and a.get('url') == url for a in articles)
        if not exists:
            articles.append({
                "title": title,
                "url": url,
                "publishedAt": publishedAt,
                "topic": topic,
                "source": source
            })
            with open(NEWS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
        message = "Berita berhasil ditambahkan!"
    return render_template('add_news_standalone.html', message=message)

@app.route('/list-news')
def list_news():
    # Gabungkan berita dari news_cache.json dan manual_news (seperti di /get-news)
    try:
        try:
            with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
                articles = json.load(f)
            if not isinstance(articles, list):
                articles = []
        except Exception:
            articles = []
        # Ambil dari manual_news (SQLite)
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT title, url, publishedAt, topic, source FROM manual_news")
        manual_articles = [dict(row) for row in c.fetchall()]
        conn.close()
        # Gabungkan dan filter duplikat (berdasarkan title+url)
        all_articles = articles + [a for a in manual_articles if not any(b.get('title') == a['title'] and b.get('url') == a['url'] for b in articles)]
        # Sort by publishedAt descending
        all_articles = sorted(all_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
        return render_template('list_news.html', articles=all_articles)
    except Exception as e:
        return render_template('list_news.html', articles=[], error=str(e))

FINNHUB_API_KEY = "d1f23g1r01qsg7d9tkmgd1f23g1r01qsg7d9tkn0"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_STOCK_CANDLE_URL = "https://finnhub.io/api/v1/stock/candle"

# Fungsi fetch berita dari Finnhub (default: AAPL, bisa diubah)
def fetch_finnhub_news(symbol="AAPL"):  # symbol: saham, misal AAPL, MSFT, TSLA
    from datetime import datetime, timedelta
    today = datetime.utcnow().date()
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }
    try:
        resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=10)
        data = resp.json()
        articles = []
        for a in data:
            articles.append({
                "title": a.get("headline"),
                "url": a.get("url"),
                "publishedAt": a.get("datetime"),
                "topic": symbol,
                "source": a.get("source"),
                "summary": a.get("summary")
            })
        return articles
    except Exception as e:
        print(f"Finnhub news error: {e}")
        return []

def fetch_and_cache_finnhub_news(symbols=None):
    if symbols is None:
        symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "BABA", "INTC"]
    all_articles = []
    for symbol in symbols:
        articles = fetch_finnhub_news(symbol)
        all_articles.extend(articles)
    # Gabungkan ke news_cache.json (append, hindari duplikat)
    try:
        with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
            cache_articles = json.load(f)
        if not isinstance(cache_articles, list):
            cache_articles = []
    except Exception:
        cache_articles = []
    # Hindari duplikat berdasarkan title+url
    for a in all_articles:
        if not any(b.get('title') == a['title'] and b.get('url') == a['url'] for b in cache_articles):
            cache_articles.append(a)
    with open(NEWS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_articles, f, ensure_ascii=False, indent=2)
    print(f"Finnhub news added to cache: {len(all_articles)} new articles.")

@app.route('/api/stock-chart')
def stock_chart():
    import time as t
    symbol = request.args.get('symbol', 'AAPL').upper()
    resolution = request.args.get('resolution', 'D')  # D, 1, 5, 15, 30, 60
    # Default: 30 hari terakhir
    now = int(t.time())
    from_time = now - 30 * 24 * 60 * 60
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': from_time,
        'to': now,
        'token': FINNHUB_API_KEY
    }
    try:
        resp = requests.get(FINNHUB_STOCK_CANDLE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get('s') != 'ok':
            return jsonify({'error': 'Finnhub API error', 'raw': data}), 500
        # Format untuk Chart.js financial
        ohlc = []
        for i in range(len(data['t'])):
            ohlc.append({
                'x': datetime.fromtimestamp(data['t'][i]).strftime('%Y-%m-%dT%H:%M:%S'),
                'o': data['o'][i],
                'h': data['h'][i],
                'l': data['l'][i],
                'c': data['c'][i]
            })
        return jsonify({'ohlc': ohlc, 'symbol': symbol})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
