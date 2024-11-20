import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# stil Seaborn
sns.set(style="whitegrid", palette="pastel")

# Pas 1: Colectăm datele istorice
historical_data = [
    {"ds": "2024-10-01", "y": 12154},
    {"ds": "2024-10-02", "y": 12150},
    {"ds": "2024-10-03", "y": 12160},
    {"ds": "2024-10-04", "y": 12140},
    {"ds": "2024-10-05", "y": 12170},
    {"ds": "2024-10-06", "y": 12160},
    {"ds": "2024-10-07", "y": 12180},
    {"ds": "2024-10-08", "y": 12150},
    {"ds": "2024-10-09", "y": 12190},
    {"ds": "2024-10-10", "y": 12200},
    {"ds": "2024-10-11", "y": 12180},
    {"ds": "2024-10-12", "y": 12160},
    {"ds": "2024-10-13", "y": 12150},
    {"ds": "2024-10-14", "y": 12140},
    {"ds": "2024-10-15", "y": 12160},
    {"ds": "2024-10-16", "y": 12180},
    {"ds": "2024-10-17", "y": 12190},
    {"ds": "2024-10-18", "y": 12170},
    {"ds": "2024-10-19", "y": 12180},
    {"ds": "2024-10-20", "y": 12160},
    {"ds": "2024-10-21", "y": 12150},
    {"ds": "2024-10-22", "y": 12140},
    {"ds": "2024-10-23", "y": 12130},
    {"ds": "2024-10-24", "y": 12120},
    {"ds": "2024-10-25", "y": 12110},
    {"ds": "2024-10-26", "y": 12100},
    {"ds": "2024-10-27", "y": 12110},
    {"ds": "2024-10-28", "y": 12120},
    {"ds": "2024-10-29", "y": 12130},
    {"ds": "2024-10-30", "y": 12140},
    {"ds": "2024-10-31", "y": 12150},
]

# Pas 2: Colectăm sentimentul din știri


def get_sentiment_from_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()  # Extragem textul
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Întoarcem polaritatea sentimentului


# URL-urile pentru știrile de sentiment
news_urls = [
    'https://tradingeconomics.com/commodity/eu-natural-gas',
    'https://radiochisinau.md/de-la-1-ianuarie-2025-intreprinderile-mari-din-republica-moldova-vor-putea-cumpara-gaze-doar-pe-piata-libera---203315.html',
    'https://www.undp.org/moldova/press-releases/republic-moldova-ready-any-gas-market-scenario',
    'https://www.romania-insider.com/romania-gas-imports-moldova-nine-months-2024'
]

# Colectăm sentimentul pentru fiecare știre
sentiments = []
for url in news_urls:
    sentiment_score = get_sentiment_from_news(url)
    sentiments.append(sentiment_score)

# Calculăm sentimentul mediu
average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

# Pasul 3: Pregătim datele istorice și scorurile de sentiment pentru antrenament
df = pd.DataFrame(historical_data)
df['ds'] = pd.to_datetime(df['ds'])
df['sentiment'] = average_sentiment  # Coloana sentiment

# Creăm laguri de prețuri pentru predicție
df['lag_1'] = df['y'].shift(1)  # Prețul de ieri
df['lag_2'] = df['y'].shift(2)  # Prețul de acum 2 zile

# Eliminăm valorile NaN (datorită lagurilor)
df.dropna(inplace=True)

# Pasul 4: Antrenăm modelul Random Forest
X = df[['sentiment', 'lag_1', 'lag_2']]  # Sentiment + Laguri
y = df['y']  # Prețul gazului

# Antrenăm modelul RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Prezicem prețurile pentru următoarele 30 de zile
future_dates = pd.date_range(
    df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
predicted_prices = []

# Variabilitate mai mare a sentimentului
for i in range(30):
    future_sentiment = average_sentiment + \
        np.random.uniform(-0.3, 0.3)  # Variabilitate în sentiment
    lag_1 = df['y'].iloc[-1]  # Ultimul preț cunoscut
    lag_2 = df['y'].iloc[-2]  # Prețul de acum 2 zile
    # Prezicem prețul folosind sentimentul și lagurile
    predicted_price = model.predict([[future_sentiment, lag_1, lag_2]])
    predicted_prices.append(predicted_price[0])

# Pasul 5: Creăm DataFrame pentru prețurile prezise
predicted_df = pd.DataFrame({
    'ds': future_dates,
    'y': predicted_prices,
    'sentiment': [average_sentiment] * 30
})

# Combinăm datele istorice cu cele prezise
df_combined = pd.concat([df, predicted_df])

# Pasul 6: Vizualizăm rezultatele
plt.figure(figsize=(10, 6))

# Prețurile
plt.plot(df_combined['ds'], df_combined['y'],
         label='Prețurile în luna Octombrie', color='#5D4E8D', linewidth=2)

# Prețurile prezise
plt.plot(predicted_df['ds'], predicted_df['y'], label='Prezicerea pentru următoarele 30 de zile',
         color='#F10C45', linestyle='--', linewidth=2)  # Linia prezicerii în roz pastelat

# Titlu și etichete
plt.title('Prezicerea prețului gazului pentru următoarele 30 de zile',
          fontsize=16, fontweight='bold', family='Poppins')
plt.xlabel('Data', fontsize=12, fontweight='bold', family='Poppins')
plt.ylabel('Prețul Gazului (Lei/1000 m³)', fontsize=12,
           fontweight='bold', family='Poppins')

# Legenda graficului
plt.legend()

# Etichetele de pe axa X
plt.xticks(rotation=45, fontsize=10, family='Poppins')

# Grila roz pastelat
plt.grid(True, linestyle='--', alpha=0.6,
         color='#F28D8D')  # Grilă roz pastelat

# Afișăm graficul
plt.tight_layout()  # Este vizibil?
plt.show()
