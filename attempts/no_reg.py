import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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
    start_date = end_date - timedelta(days=720)
    return start_date, end_date

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit_sine_curve(dates, prices):
    # Convert dates to numerical values
    x_data = np.array((dates - dates.min()).days)
    y_data = prices.values

    # Scale x_data and y_data to improve fitting
    x_data_scaled = x_data / np.max(x_data)
    y_data_scaled = (y_data - np.mean(y_data)) / np.std(y_data)

    # Initial guess for the parameters
    guess_freq = 2 * np.pi  # Adjust initial guess for the frequency
    guess = [1, guess_freq, 0, 0]

    try:
        # Fit the sine curve with increased maxfev
        params, _ = curve_fit(sine_function, x_data_scaled, y_data_scaled, p0=guess, maxfev=10000)

        # Rescale the parameters
        params[0] *= np.std(y_data)
        params[3] = np.mean(y_data)
        params[1] /= np.max(x_data)
        
        # Calculate fitted values
        fitted_values = sine_function(x_data, *params)

        # Calculate R-squared value
        r_squared = r2_score(y_data, fitted_values)

        return params, r_squared
    except RuntimeError as e:
        print(f"Error fitting sine curve for data: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def graph_data(ticker, close_prices, params, r_squared):
    plt.figure(figsize=(10, 5))
    plt.plot(close_prices.index, close_prices, marker='o', linestyle='-', label='Actual Data')

    if params is not None:
        # Plot the sine fit
        x_data = np.array((close_prices.index - close_prices.index.min()).days)
        fitted_values = sine_function(x_data, *params)
        plt.plot(close_prices.index, fitted_values, label=f'Sine Fit (R²={r_squared:.2f})', color='red')

    plt.title(f'Closing Prices for {ticker} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Example tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    # Get the date range for the past six months
    start_date, end_date = get_past_six_months_dates()

    # Fetch the data for the specified tickers
    all_data = fetch_data(tickers, start_date, end_date)

    # Graph the data for each ticker
    for stock_name, stock_data in all_data.items():
        # Drop NaN values and sort the data
        stock_data = stock_data.dropna().sort_index()
        params, r_squared = fit_sine_curve(stock_data.index, stock_data)
        graph_data(stock_name, stock_data, params, r_squared)

if __name__ == "__main__":
    main()
