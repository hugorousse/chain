import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '90',
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['day'] = range(1, len(df) + 1)
    df['price'] = df['price'].astype(float)
    return df[['day', 'price']]

data = fetch_historical_data()
print("Historical data:\n", data.head())

print("Number of data points:", len(data))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['price']])

X, y = [], []
if len(scaled_data) >= 60:
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
else:
    raise ValueError("Not enough data to create sequences of 60 days.")

X, y = np.array(X), np.array(y)

print("Shape of X before reshape:", X.shape)
print("Shape of y:", y.shape)

if len(X.shape) == 2:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
else:
    raise ValueError("The shape of X is not as expected, please check the data preprocessing steps.")

print("Shape of X after reshape:", X.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=25, batch_size=32)

model.save('token_price_predictor_lstm.h5')
joblib.dump(scaler, 'scaler.pkl')
