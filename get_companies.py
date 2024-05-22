# Download CSV from website

import time
from selenium import webdriver
import yfinance as yf
import csv
import requests

# # Gets list of companies
# """ Testing using nseindia. """
# # Set URl
# url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# # Set options and stop browser from closing
# options = webdriver.EdgeOptions()
# prefs = {"download.default_directory" : "C:\\Users\\Adi\\Files\\Projects\\nse_stocks_pattern"}
# options.add_experimental_option("prefs", prefs)
# options.add_experimental_option("detach", True)
# driver = webdriver.Edge(options=options)

# # Open URL and login
# driver.get(url)

# # Close window
# time.sleep(10)
# driver.quit()

# # Copy company tickers into a list
# names = []
# tickers = []
# with open("ind_nifty500list.csv", "r") as file:
#     file.readline()
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         if (row[0]):
#             names.append(row[0])
#             tickers.append()

# print(tickers)

# # Create CSV of accurate tickers
# filename = "tickers.csv"
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for row in tickers:
#         writer.writerow(row)

# # Pretend that those tasks above are done

