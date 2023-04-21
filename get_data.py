import pandas as pd
import yfinance as yf

# Download the data
data = yf.download(tickers='BTC-USD', period='3000d', interval='1d')

# Store the data in a CSV file
data.to_csv('btc_data.csv')
