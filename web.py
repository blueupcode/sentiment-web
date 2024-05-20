# Mengimpor modul yang diperlukan
import io
import base64
import urllib
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
from google_play_scraper import app, Sort, reviews_all

# Mendownload lexicon 'vader' yang digunakan untuk analisis sentimen
nltk.download('vader_lexicon')

# Membuat instance Flask
app = Flask(__name__)

# Membuat instance SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Mendefinisikan route '/' untuk metode GET
@app.route('/', methods=['GET'])
def home():
    # Merender template 'index.html' saat route '/' diakses
    return render_template('index.html')

# Mendefinisikan route '/analyze' untuk metode POST
@app.route('/analyze', methods=['POST'])
def analyze():
    game_id = request.form.get('game_id')

    # Mengambil review dari aplikasi 'com.mobile.legends' menggunakan fungsi reviews_all
    game_reviews = reviews_all(
        game_id,
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=2000,
        filter_score_with= None
    )
    
    # Membuat list kosong untuk menyimpan review yang telah diproses
    reviews = []
    for review in game_reviews:
        # Membersihkan konten review dan mengubahnya menjadi huruf kecil
        clean_content = review['content'].replace(r'[^\w\s]', '')
        clean_content = clean_content.lower()
        
        # Menghitung skor sentimen dari konten review yang telah dibersihkan
        polarity_scores = sia.polarity_scores(clean_content)
        
        # Menentukan sentimen berdasarkan skor sentimen
        sentiment = 'positif' if polarity_scores['compound'] > 0 else ('negatif' if polarity_scores['compound'] < 0 else 'netral')

        # Menambahkan review dan sentimen ke dalam list reviews
        reviews.append({
            'content': review['content'],
            'sentiment': sentiment
        })
    
    # Menghitung jumlah review untuk setiap sentimen
    sentiment_counts = pd.DataFrame(reviews)['sentiment'].value_counts()

    # Membuat plot pie
    plt.figure(figsize=(6,6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')

    # Menyimpan plot sebagai gambar PNG dalam buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Mengubah gambar PNG menjadi data URL
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
    
    # Merender template 'index.html' dengan variabel reviews
    return render_template('index.html', reviews=reviews, plot_url=plot_url)

# Menjalankan aplikasi Flask dalam mode debug
if __name__ == "__main__":
	app.run(debug=True)
