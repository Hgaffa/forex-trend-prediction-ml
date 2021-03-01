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


#Class for making time series stationary with ADF test
class Stationary:

    def __init__(self, data = None): 
            
        self.df = data.copy()

        self.THRESH = 0.05

        self.st()

    def st(self):

        bad_columns = self.get_non_stationary()

        stationary = self.make_stationary(bad_columns)

        return stationary

    #function to return all columns that are not stationary
    def get_non_stationary(self):
        
        non_stationary = []
            
        for column in self.df.columns:
            
            check_stationary = adfuller(self.df[column].values)

            adf_stat = check_stationary[0]

            p_val = check_stationary[1]

            crit_vals = check_stationary[4]

            if p_val < self.THRESH:
                
                continue
                
            else:
                
                non_stationary.append(column)
                
        return non_stationary
            
    #function to make non-stationary columns stationary
    def make_stationary(self, cols):
        
        for col in cols:
            
            self.df[col] = pd.DataFrame(np.log(self.df[col])-np.log(self.df[col].shift(1))).values
        
        self.df = self.df.dropna()
        
        check = self.get_non_stationary()
        
        if len(check) == 0:
            
            print("Date is now stationary!")
            
        else:
            
            print("Data still non-stationary!")

        return self.df

    #function that returns adf-test statistics
    #ADF Test
    def adf_test(self, feature):
            
        check_stationary = adfuller(feature.values)
        
        self.adf_stat = check_stationary[0]
        
        self.p_val = check_stationary[1]
        
        self.crit_vals = check_stationary[4]
        
        print("The Close Price ADF Statistic is: ", self.adf_stat)
        
        print("The P-Value is: ", self.p_val)
        
        print("Critical Values:")
        print("================")

        for key, val in self.crit_vals.items():
            
            print(f'{key}: {val}')

