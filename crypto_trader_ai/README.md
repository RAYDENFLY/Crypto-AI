# Crypto Trader AI

Crypto Trader AI adalah aplikasi web berbasis Flask yang berfungsi sebagai asisten analisis pasar kripto, saham, dan ekonomi global. Aplikasi ini dapat memberikan saran trading, merangkum berita terbaru, serta menjawab pertanyaan seputar crypto dan saham dalam Bahasa Indonesia.

## Fitur Utama
- **Analisis Harga Crypto & Saham**: Menampilkan data harga, grafik OHLC, dan tren pasar dari berbagai sumber (CoinGecko, Binance, Finnhub).
- **Berita Terbaru**: Mengambil dan menggabungkan berita dari NewsAPI, Finnhub, serta input manual, lalu menampilkan headline dan ringkasan berita terkait pasar keuangan.
- **Chat AI**: Fitur chat yang didukung oleh model AI (OpenAI GPT, Groq, Mixtral) untuk menjawab pertanyaan, memberikan analisis, dan saran trading.
- **Manajemen Berita**: Tambah, lihat, dan kelola berita secara manual melalui antarmuka web.
- **Database**: Menyimpan riwayat chat dan berita secara lokal menggunakan SQLite.

## Cara Menjalankan
1. Pastikan Python 3.x sudah terinstall.
2. Install dependencies dengan `pip install -r requirements.txt`.
3. Siapkan file `.env` berisi API key yang diperlukan (OpenAI, NewsAPI, Finnhub, dll).
4. Jalankan aplikasi dengan perintah:
   ```bash
   python app.py
   ```
5. Buka browser dan akses `http://localhost:5000`.

## Struktur Folder
- `app.py` : Main aplikasi Flask
- `templates/` : HTML untuk tampilan web
- `static/` : File statis (gambar, css, js)
- `news_cache.json` : Cache berita terbaru
- `chats.db` : Database chat dan berita

## API & Integrasi
- CoinGecko, Binance, Finnhub untuk data harga dan berita
- NewsAPI untuk berita global
- OpenAI, Groq, Mixtral untuk AI chat

## Kontribusi
Silakan fork dan pull request jika ingin menambah fitur atau memperbaiki bug.

## Lisensi
Aplikasi ini bersifat open source dan bebas digunakan untuk edukasi dan pengembangan.
