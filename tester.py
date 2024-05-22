import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # Generate synthetic data with added noise
# x = np.linspace(0, 10, 100)
# y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)  # Simulated data with noise

# # Define the model function (sinusoidal function)
# def sinusoidal_func(x, A, f, phi, offset):
#     return A * np.sin(2 * np.pi * f * x + phi) + offset

# # Use curve_fit to fit the model to the data
# p0 = [1, 1, 0, 0]  # Initial guess for the parameters
# params, covariance = curve_fit(sinusoidal_func, x, y, p0=p0)

# # Extract the fitted parameters
# A_fit, f_fit, phi_fit, offset_fit = params

# # Calculate R-squared value
# y_pred = sinusoidal_func(x, A_fit, f_fit, phi_fit, offset_fit)
# residuals = y - y_pred
# ss_res = np.sum(residuals ** 2)
# ss_tot = np.sum((y - np.mean(y)) ** 2)
# r_squared = 1 - (ss_res / ss_tot)

# # Plot the original data and the fitted curve
# plt.scatter(x, y, label='Data')
# plt.plot(x, y_pred, color='red', label='Fit')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.title('Sinusoidal Curve Fitting')
# plt.show()

# print("Fitted Parameters:")
# print("Amplitude:", A_fit)
# print("Frequency:", f_fit)
# print("Phase:", phi_fit)
# print("Offset:", offset_fit)

# # Calculate the covariance matrix to estimate uncertainties in the parameters
# parameter_errors = np.sqrt(np.diag(covariance))
# print("\nParameter Errors (Standard Deviation):")
# print("Amplitude:", parameter_errors[0])
# print("Frequency:", parameter_errors[1])
# print("Phase:", parameter_errors[2])
# print("Offset:", parameter_errors[3])

# print("\nR-squared value:", r_squared)

# from yahooquery import search

# def get_ticker(company_name):
#     # Perform a search using the company name
#     result = search(company_name)
    
#     # Extract the first matching ticker symbol from the search results
#     if result['quotes']:
#         ticker = result['quotes'][0]['symbol']
#         return ticker
#     else:
#         return None

# # Example usage
# company_name = "Tesla"
# ticker = get_ticker(company_name)
# if ticker:
#     print(f"The ticker symbol for {company_name} is {ticker}.")
# else:
#     print(f"Ticker symbol for {company_name} not found.")

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page to scrape
url = 'https://www1.nseindia.com/products/content/equities/indices/nifty_500.htm'

# Headers to mimic a browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Request the page
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table containing the Nifty 500 companies
    table = soup.find('table', {'id': 'constituents'})
    
    # Check if the table was found
    if table:
        # Extract the rows of the table
        rows = table.find_all('tr')
        
        # List to hold the ticker symbols
        tickers = []
        
        # Iterate over the rows, skip the header row
        for row in rows[1:]:
            # Find all the columns in the row
            cols = row.find_all('td')
            # The ticker is in the first column
            ticker = cols[0].text.strip()
            # Append the ticker to the list
            tickers.append(ticker)
        
        # Convert the list of tickers to a DataFrame
        df = pd.DataFrame(tickers, columns=['Ticker'])
        
        # Save the DataFrame to a CSV file
        df.to_csv('nifty_500_tickers.csv', index=False)
        
        print('Tickers have been successfully saved to nifty_500_tickers.csv')
    else:
        print('Failed to find the table of Nifty 500 companies')
else:
    print(f'Failed to retrieve the webpage, status code: {response.status_code}')
