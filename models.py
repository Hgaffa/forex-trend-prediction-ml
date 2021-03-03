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

from bs4 import BeautifulSoup
import requests
from urllib.request import Request, urlopen

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter

import os

import praw
from pushshift_py import PushshiftAPI 

import math
import json
import itertools
import requests
import time
from datetime import datetime, timedelta, timezone
import random

from sklearn.decomposition import PCA 

from sklearn.feature_selection import RFE
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

from IPython.display import clear_output

class Models:

    def __init__(self, master=None, data=None, old_returns=None):

        self.df = data
        
        self.val = self.get_val_data()

        self.old_returns = old_returns

    def get_val_data(self):

        split = int(0.9*len(self.df))

        self.df = self.df[:split]

        return self.df[split+1:]

    #Function used to evaluate the financial viability of a model
    def financial_report(self, predictions, returns):
        
        cum_returns = pd.DataFrame(index=predictions.index, columns=['Returns'])
        
        total = 1000
        
        for i in range(0,len(predictions)):
        
            total += (predictions.iloc[i]*returns.iloc[i])*total

            cum_returns.iloc[i] = (total.values[0] - 1000)/1000
            
        cum_returns = cum_returns[cum_returns['Returns'] != 0]
        
        r = cum_returns.diff()
        
        r = r.replace([np.inf, -np.inf], np.nan)
        
        r = r.dropna()
        
        sr = r.mean()/r.std() * np.sqrt(252)
        
        ind = range(0,len(cum_returns))
        
        fig2, ax2 = plt.subplots(figsize=(10,5))
        
        ax2.plot(ind, cum_returns)
        
        return sr, total

    #General model evaluation report function
    def model_report(self, model, scaled):

        og_columns = self.df.copy().drop(columns=['Labels']).columns
        
        X_train, X_test, y_train, y_test = self.get_split(0.8, scaled)
            
        fig, ax = plt.subplots(figsize=(5, 5))

        #get passed model
        clf = model

        #predict x_test
        pred = clf.predict(X_test)
        
        #print classification report
        print(classification_report(y_test, pred))
        
        #get f1 score for best model
        f1 = f1_score(y_test, pred, average='weighted')
        
        #Financial report
        sr, total = self.financial_report(pd.DataFrame(pred), self.old_returns[int(len(self.df)*0.8)+1:])
        
        print()
        print("Model Results:")
        print("====================")
        
        print(f'The sharpes ratio of this model is: {sr.values[0]}')
        
        print(f'The Equity before backtest is: 1000')
        
        print(f'The Equity after backtest is: {total.values[0]}')
        
        print(f'The total return of model is: {(total.values[0] - 1000)/1000}')
         
        #draw confusion matrix
        disp = plot_confusion_matrix(clf, X_test, y_test,                           
                                    display_labels=[-1,1],
                                    cmap=plt.cm.Blues, ax=ax)
        
        plt.show()

    #Function used to split data
    def get_split(self, train_size, scale):
    
        #Split data for time series
        X = self.df.copy().drop(columns=['Labels'])

        y = self.df.Labels

        split = int(len(X)*train_size)

        X_train = X[:split]
        X_test = X[split+1:]

        #Scale data for better logistic regression model optimization
        scaler = StandardScaler()

        if scale:
        
            X_train = scaler.fit_transform(X_train)
            
            X_test = scaler.transform(X_test)

        y_train = y[:split]
        y_test = y[split+1:]

        #Return sets
        return X_train, X_test, y_train, y_test
    

    def get_knn(self, nn, metric, alg):

        #Split data
        X_train, X_test, y_train, y_test = self.get_split(0.8, True)

        clf = KNeighborsClassifier(n_neighbors=int(nn), metric=metric, algorithm=alg, n_jobs=-1)

        clf.fit(X_train, y_train)

        self.model_report(clf, True)

