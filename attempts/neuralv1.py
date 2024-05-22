import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def fetch_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            all_data[ticker] = data['Close']
    return all_data

def get_past_six_months_dates():
    tz = pytz.timezone("Asia/Kolkata")
    end_date = tz.localize(datetime.now())
    start_date = end_date - timedelta(days=180)
    return start_date, end_date

def prepare_data(data, time_steps=60):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps, 0])
        y.append(scaled_data[i + time_steps, 0])
    X, y = np.array(X), np.array(y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def plot_predictions(real_data, predicted_data, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(real_data, color='blue', label='Actual Stock Price')
    plt.plot(predicted_data, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date, end_date = get_ppast_six_months_dates()

    all_data = fetch_data(tickers, start_date, end_date)

    for stock_name, stock_data in all_data.items():
        stock_data = stock_data.dropna().sort_index()
        
        # Prepare data
        time_steps = 60
        X, y, scaler = prepare_data(stock_data, time_steps)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Create and train the LSTM model
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Plot predictions
        real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
        plot_predictions(real_stock_price, predictions, stock_name)

if __name__ == "__main__":
    main()
