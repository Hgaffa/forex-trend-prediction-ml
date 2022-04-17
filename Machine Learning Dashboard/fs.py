#Class for feature selection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Technical Indicator Libraries
import talib as ta
import pandas_ta as new_ta

import seaborn as sns
import re
from pylab import rcParams
import os
import string
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

class FeatureSelection:

    def __init__(self, data=None):

        self.df = data.copy()


    #function to make splits for time series
    def get_split(self, train_size):
    
        #Split data for time series
        X = self.df.copy().drop(columns=['Labels'])

        y = self.df.Labels

        split = int(len(X)*train_size)

        X_train = X[:split]
        X_test = X[split+1:]
        
        #Scale data for better logistic regression model optimization
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y[:split]
        y_test = y[split+1:]
        
        #Return sets
        return X_train, X_test, y_train, y_test

    #function for feature selection between range of number of features
    def feature_selection(self, num_select):

        print("Starting Feature Selection...")

        #Get data split
        X_train, X_test, y_train, y_test = self.get_split(0.7)


        #classifier to base accuracies from
        model = LogisticRegression(max_iter=1000, n_jobs=-1)
    
        #Define RFE model and fit to data
        rfe = RFE(model, n_features_to_select=num_select)

        X_train_rfe = rfe.fit_transform(X_train, y_train)

        X_test_rfe = rfe.transform(X_test)

        model.fit(X_train_rfe,y_train)

        pred = model.predict(X_test_rfe)
            
        #get accuracy
        accuracy = model.score(X_test_rfe, y_test)

        print(classification_report(pred, y_test))

        feature_list = pd.Series(rfe.support_)

        best_features = list(feature_list[feature_list==True].index)
        
        return best_features