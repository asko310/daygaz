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

# Pas 1:datele ultimele 10 luni
historical_data = [
    {"ds": "2024-01-01", "y": 10000},
    {"ds": "2024-02-01", "y": 10150},
    {"ds": "2024-03-01", "y": 10050},
    {"ds": "2024-04-01", "y": 10200},
    {"ds": "2024-05-01", "y": 10100},
    {"ds": "2024-06-01", "y": 10350},
    {"ds": "2024-07-01", "y": 10280},
    {"ds": "2024-08-01", "y": 10450},
    {"ds": "2024-09-01", "y": 10500},
    {"ds": "2024-10-01", "y": 10650},
]  # Datele pentru ultimele 10 luni

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
df['lag_1'] = df['y'].shift(1)  # Prețul de luna trecută
df['lag_2'] = df['y'].shift(2)  # Prețul acum 2 luni

# Eliminăm valorile NaN (datorită lagurilor)
df.dropna(inplace=True)

# Pasul 4: Antrenăm modelul Random Forest
X = df[['sentiment', 'lag_1', 'lag_2']]  # Sentiment + Laguri
y = df['y']  # Prețul gazului

# Antrenăm modelul RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Prezicem prețurile pentru următoarele 30 de zile ale lunii următoare
future_dates = pd.date_range(
    df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
predicted_prices = []

# Variabilitate mai mare a sentimentului
for i in range(30):
    future_sentiment = average_sentiment + \
        np.random.uniform(-0.3, 0.3)  # Variabilitate în sentiment
    lag_1 = df['y'].iloc[-1]  # Ultimul preț cunoscut (luna curentă)
    lag_2 = df['y'].iloc[-2]  # Prețul acum 2 luni
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
         label='Prețurile în ultimele 10 luni', color='#5D4E8D', linewidth=2)

# Prețurile prezise pentru luna următoare
plt.plot(predicted_df['ds'], predicted_df['y'], label='Prezicerea pentru luna următoare',
         color='#F10C45', linestyle='--', linewidth=2)  # Linia prezicerii în roz pastelat

# Titlu și etichete
plt.title('Prezicerea prețului gazului pentru luna următoare',
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
