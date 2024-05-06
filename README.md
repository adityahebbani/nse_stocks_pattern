# nse_stocks_pattern

Finds companies in NIFTY 500 that follow a sine-wave pattern. Notifies when a company is on an up trend or on a down trend. 

A csv file containing all the companies that we need to keep track of is read. 

Closing costs from the past six months is read and stored in a dataframe, then graphed. The pattern that we are looking for is a sinusoidal pattern. If the company fits this pattern, it is written to a new csv. This is repeated for each company in the first csv. 

Then, we identify the peaks and troughs for each company in the second csv. Using this, we predict the upcoming peak and trough. User is notified on the day of a predicted peak and trough.

There will be two scripts in this project:
1) The script to write the second csv. This is ran manually as often as desired. 
2) The script to identify the peaks and troughs of the companies in the second csv. This is ran every day. 



capital-mind
tradingview.com
create free id, stock charts available
candlestick pattern, look for stocks that behave in an up down pattern over years chart

install yfinance library
use nseindia.com to find table for companies
take the last 6  months time frame and look for stocks in this pattern
find the median closing price of each cycle, then find when the stocks are going up and down. Goal is to buy when price dips, then sell when it will go high.