# Calculates and graphs predicted closing prices

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt

companies = ['HDFCBANK.NS']
start_date = input("Enter start date in format %Y-%m-%d:")
end_date = input("enter date date in format %Y-%m-%d:")

def data_weekly(tikr):
    empty_df = pd.DataFrame()
    for i in tikr:
        comp_df = yf.download(i, start=start_date, end=end_date, interval="1wk")
        comp_df["Symbol"] = i
        comp_df['Date'] = comp_df.index
        comp_df = comp_df[comp_df['Close'].notna()]
        comp_df.ta.ema(length=13,append=True)
        comp_df.ta.ema(length=20,append=True)
        comp_df.ta.ema(length=40,append=True)
        empty_df = pd.concat([empty_df,comp_df])
    return(empty_df)

comp_df = data_weekly(companies)

def trigger(data):
    data['13w_gtr_20w'] = np.where(data['EMA_13'] > data['EMA_20'],1,0)
    data['20w_gtr_40w'] = np.where(data['EMA_20'] > data['EMA_40'],1,0)
    data['20w_cd_wclose'] = np.where(((data['Close'] > data['EMA_20']) & (data['Close'].shift(1) < data['EMA_20'].shift(1))) ,1,0)
    data['wclose_cd_20w'] = np.where(((data['Close'] < data['EMA_20']) & (data['Close'].shift(1) > data['EMA_20'].shift(1))) ,1,0)
    return data
comp_df_trigger = trigger(comp_df)

def buy_trigger(data):
    data['buy'] = np.where((data['13w_gtr_20w'] == 1 ) & (data['20w_gtr_40w'] == 1) & (data['20w_cd_wclose']==1),1,0)
    return data

def sell_trigger(data):
    data['sell'] = np.where((data['wclose_cd_20w']==1),1,0)
    return data