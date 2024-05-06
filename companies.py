# Gets the list of companies.

import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        stock_data[ticker] = data['Close']
    return stock_data

def main():
    # Input data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    
    # Read company closing costs into dict
    stock_data = get_stock_data(tickers, start_date, end_date)


if __name__ == "__main__":
    main()

