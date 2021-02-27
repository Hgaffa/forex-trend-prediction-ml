#Library imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy

from sklearn.preprocessing import StandardScaler

#Technical Indicator Libraries
import talib as ta
import pandas_ta as new_ta

import seaborn as sns

from pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import os

import math
import json
import itertools
import requests
import time
from datetime import datetime, timedelta, timezone

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from eda import EDA

import pandas as pd

class TechnicalAnalysis:

    def __init__(self, data = None): 
          
        self.df = data.copy()

        self.ta()
        
    def ta(self):

        ta_data = self.technical_analysis(5)[30:len(self.df)-1]
        return ta_data

    #SSL Indicator Function
    def ssl_channel_indicator(self, period = 20):
        
        self.df['SMA_High'] = ta.SMA(self.df.High, timeperiod=period)
        
        self.df['SMA_Low'] = ta.SMA(self.df.Low, timeperiod=period)
        
        hlv = pd.DataFrame(columns=['HLV'], index=self.df.index)
        
        sslDown = pd.DataFrame(columns=['SSL_Down'], index=self.df.index)
        
        sslUp = pd.DataFrame(columns=['SSL_Up'], index=self.df.index)
        
        for i in range(0,len(self.df.Close)):
            
            if self.df.Close.iloc[i] > self.df.SMA_High.iloc[i]:
                
                hlv.iloc[i] = 1
                
            elif self.df.Close.iloc[i] < self.df.SMA_Low.iloc[i]:
                
                hlv.iloc[i] = -1
                
            else:
                
                hlv.iloc[i] = hlv.iloc[i-1]
                
        for i in range(0,len(hlv)):
            
            if hlv.HLV.iloc[i] <= 0:
                
                sslDown.iloc[i] = self.df.SMA_High.iloc[i]
                
            else:
                
                sslDown.iloc[i] = self.df.SMA_Low.iloc[i]
                
        for i in range(0,len(hlv)):
            
            if hlv.HLV.iloc[i] < 0:
                
                sslUp.iloc[i] = self.df.SMA_Low.iloc[i]
                
            else:
                
                sslUp.iloc[i] = self.df.SMA_High.iloc[i]     
        
        return sslDown, sslUp

    #ATR Indicator Function
    def atr_indicator(self, period = 13):
        
        atr = new_ta.atr(self.df.High, self.df.Low, self.df.Close, period)

        return atr

    #Linear WMA Indicaotr Function
    def lwma_indicator(self, period=15):
            
        lwma = pd.DataFrame(columns=['LWMA'], index=self.df.index)
        
        for i in range(0,len(self.df)):
            
            if i < (period-1):
                
                continue
            
            else:
                
                sum_prices = 0
        
                for j in range(0,period+1):
                
                    price = self.df.Close.iloc[i-j]
                    
                    sum_prices += (price * (period-j))
                    
                lwma.iloc[i] = sum_prices/(np.sum(range(1,period+1)))
            
        return lwma

    #QQE indicators function - similar to RSI indicator (shows trend strength signal)
    def qqe_indicator(self, period=10):
        
        qqe = new_ta.qqe(self.df.Close, length=period)
        
        qqe.columns = ['QQE','RSI','Long','Short']

        qqe = qqe.reset_index().drop(columns=['Long','Short'])
        
        QQE = qqe['QQE']

        RSI = qqe['RSI']
        
        return QQE.values, RSI.values

    #Volatility Index Indicator
    def volatility_index_indicator(self, period=10):
        
        volatility = pd.DataFrame(columns=['VOLATILITY'], index=self.df.index)
        
        for i in reversed(range(0,len(self.df)-1)):
            
            if self.df.Close.iloc[i] > self.df.Open.iloc[i]:
                
                volatility.iloc[i] = abs(self.df.High.iloc[i] - self.df.Low.iloc[i+1])
            
            elif self.df.Close.iloc[i] < self.df.Open.iloc[i]:
                
                volatility.iloc[i] = abs(self.df.Low.iloc[i] - self.df.High.iloc[i+1])
                
            else:
                
                volatility.iloc[i] = volatility.iloc[i+1]
            
        volatility = ta.SMA(volatility.VOLATILITY, timeperiod=5)
        
        sma_vol = ta.SMA(volatility, timeperiod=15)
                
        return volatility, sma_vol

    #Function to generate technical analysis features
    def technical_analysis(self, period):
        
        #Simple moving average
        #volatility indicators
        self.df['ATR'] = ta.ATR(self.df.High, self.df.Low, self.df.Close, timeperiod=10)
        
        #momentum indicators
        self.df['ADX'] = ta.ADX(self.df.High, self.df.Low, self.df.Close, timeperiod=10)
        
        self.df['AROON_Down'], self.df['AROON_Up'] = ta.AROON(self.df.High, self.df.Low, timeperiod = 10)
        
        self.df['CCI'] = ta.CCI(self.df.High, self.df.Low, self.df.Close, timeperiod = 10)
        
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = ta.MACD(self.df.Close, fastperiod=15, slowperiod=30)
        
        self.df['PROC12'] = ta.ROC(self.df.Close, timeperiod = 12)
        
        self.df['PROC13'] = ta.ROC(self.df.Close, timeperiod = 13)
        
        self.df['PROC14'] = ta.ROC(self.df.Close, timeperiod = 14)
        
        self.df['PROC15'] = ta.ROC(self.df.Close, timeperiod = 15)
        
        self.df['W%R6'] = ta.WILLR(self.df.High, self.df.Low, self.df.Close, timeperiod=6)
        
        self.df['W%R7'] = ta.WILLR(self.df.High, self.df.Low, self.df.Close, timeperiod=7)

        self.df['W%R8'] = ta.WILLR(self.df.High, self.df.Low, self.df.Close, timeperiod=8)

        self.df['W%R9'] = ta.WILLR(self.df.High, self.df.Low, self.df.Close, timeperiod=9)

        self.df['W%R10'] = ta.WILLR(self.df.High, self.df.Low, self.df.Close, timeperiod=10)

        self.df['STOCH_K5'], self.df['STOCH_D5'] = ta.STOCH(self.df.High, self.df.Low, self.df.Close, fastk_period=5)
        
        self.df['STOCH_K8'], self.df['STOCH_D8'] = ta.STOCH(self.df.High, self.df.Low, self.df.Close, fastk_period=8)

        self.df['STOCH_K10'], self.df['STOCH_D10'] = ta.STOCH(self.df.High, self.df.Low, self.df.Close, fastk_period=10)
        
        self.df['MOM3'] = ta.MOM(self.df.Close, timeperiod=3)
        
        self.df['MOM5'] = ta.MOM(self.df.Close, timeperiod=5)
        
        self.df['MOM8'] = ta.MOM(self.df.Close, timeperiod=8)
        
        self.df['MOM9'] = ta.MOM(self.df.Close, timeperiod=9)
        
        self.df['MOM10'] = ta.MOM(self.df.Close, timeperiod=10)
        
        self.df['BB_High'], self.df['BB_Mid'], self.df['BB_Low'] = ta.BBANDS(self.df.Close, timeperiod=15)

        self.df['DEMA10'] = ta.DEMA(self.df.Close, timeperiod=10)
        
        self.df['DEMA15'] = ta.DEMA(self.df.Close, timeperiod=15)
        
        self.df['SMA10'] = ta.SMA(self.df.Close, timeperiod=10)
        
        self.df['SMA15'] = ta.SMA(self.df.Close, timeperiod=15)
        
        self.df['EMA10'] = ta.EMA(self.df.Close, timeperiod=10)
        
        self.df['EMA15'] = ta.EMA(self.df.Close, timeperiod=15)
        
        self.df['SAR'] = ta.SAR(self.df.High, self.df.Low)
        
        self.df['ADOSC1'] = ta.ADOSC(self.df.High, self.df.Low, self.df.Close, self.df.Volume, fastperiod=2, slowperiod=10)
        
        self.df['ADOSC2'] = ta.ADOSC(self.df.High, self.df.Low, self.df.Close, self.df.Volume, fastperiod=3, slowperiod=10)

        self.df['ADOSC3'] = ta.ADOSC(self.df.High, self.df.Low, self.df.Close, self.df.Volume, fastperiod=4, slowperiod=10)

        self.df['ADOSC4'] = ta.ADOSC(self.df.High, self.df.Low, self.df.Close, self.df.Volume, fastperiod=5, slowperiod=10)

        self.df['ADOSC5'] = ta.ADOSC(self.df.High, self.df.Low, self.df.Close, self.df.Volume, fastperiod=6, slowperiod=10)

        self.df['SSL_Down'], self.df['SSL_Up'] = self.ssl_channel_indicator()
        
        self.df['RSI10'] = ta.RSI(self.df.Close, timeperiod=10)
        
        self.df['RSI15'] = ta.RSI(self.df.Close, timeperiod=15)
        
        self.df['KAMA15'] = ta.KAMA(self.df.Close, timeperiod=15)
        
        self.df['KAMA30'] = ta.KAMA(self.df.Close, timeperiod=30)
        
        self.df['Return'] = self.df.Close.pct_change()

        self.df['SSL_Down'], self.df['SSL_Up'] = self.ssl_channel_indicator()

        self.df['ATR'] = self.atr_indicator(period=13)

        self.df['LWMA'] = self.lwma_indicator(period=10)

        self.df['QQE'], self.df['RSI_MA'] = self.qqe_indicator(period=13)

        self.df['VOLA'], self.df['SMA_VOLA'] = self.volatility_index_indicator(period=13)
        
        return self.df