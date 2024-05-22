# Analyzes the ticker data
import csv
import pytz
import yfinance as yf
from yahooquery import search
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.optimize import minimize

# Get data for each company from yf
def fetch_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data is not None:
            all_data[ticker] = data['Close']
    return all_data

def get_past_six_months_dates():
    tz = pytz.timezone("Asia/Kolkata")
    end_date = tz.localize(datetime.today())
    start_date = end_date - timedelta(days=180)
    return start_date, end_date

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def regularized_cost_function(params, x_data, y_data, lambda_reg):
    A, B, C, D = params
    predictions = sine_function(x_data, A, B, C, D)
    residuals = y_data - predictions
    regularization = lambda_reg * (A**2 + B**2 + C**2 + D**2)
    return np.sum(residuals**2) + regularization

def fit_sine_curve_with_regularization(dates, prices, lambda_reg=1.0):
    # Convert dates to numerical values
    x_data = np.array((dates - dates.min()).days)
    y_data = prices.values

    # Scale x_data and y_data to improve fitting
    x_data_scaled = x_data / np.max(x_data)
    y_data_scaled = (y_data - np.mean(y_data)) / np.std(y_data)

    # Initial guess for the parameters
    guess_freq = 2 * np.pi  # Adjust initial guess for the frequency
    initial_guess = [1, guess_freq, 0, 0]

    # Minimize the regularized cost function
    result = minimize(regularized_cost_function, initial_guess, args=(x_data_scaled, y_data_scaled, lambda_reg), method='L-BFGS-B')

    if result.success:
        params = result.x
        params[0] *= np.std(y_data)
        params[3] = np.mean(y_data)
        params[1] /= np.max(x_data)

        # Calculate fitted values
        fitted_values = sine_function(x_data, *params)

        # Calculate R-squared value
        r_squared = r2_score(y_data, fitted_values)
        return params, r_squared
    else:
        print(f"Error fitting sine curve: {result.message}")
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
    # Get the data - read tickers from new csv file
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] # For testing
    # start_date = '2023-01-01' # Start and end should use get_past_six_months_dates()
    # end_date = '2023-01-31'
    start_date, end_date = get_past_six_months_dates()
    all_data = fetch_data(tickers, start_date, end_date)

    # Graph the data and fit sine curve (for testing)
    for stock_name, stock_data in all_data.items():
        stock_data = stock_data.dropna().sort_index()
        params, r_squared = fit_sine_curve_with_regularization(stock_data.index, stock_data)
        graph_data(stock_name, stock_data, params, r_squared)
    
    # Use fitment measure (probably r-square value) to see if ticker qualifies

    # return a list of tickers that qualify


if __name__ == "__main__":
    main()
