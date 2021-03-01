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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class FundamentalAnalysis:

    def __init__(self, data=None):

        self.df = data.copy()

        #Intialise sentiment analyzer model and update lexicon with financial jargon and appropriate costs
        self.analyser = SentimentIntensityAnalyzer()

        new_words = {
            'highs': 5.0,
            'lows': -5.0,
            'higher': 2.0,
            'lower': -2.0,
            'high': 5.0,
            'low': -5.0,
            'crash': -7.0,
            'crashes': -7.0,
            'bullish': 5.0,
            'bearish': -5.0,
            'bulls': 5.0,
            'bears': -5.0,
            'drop': -3.0,
            'drops': -3.0,
            'surges': 3.0,
            'surges': 3.0,
            'up': 2.0,
            'down': -2.0,
            'soars': 4.0,
            'soaring': 4.0,
            'dropping': -3
        }

        self.analyser.lexicon.update(new_words)

    #method for returning decimal value for an inputted tweet based on the positivity/negativity represented in the text
    def get_sentiment(self,tweet):
        
        score = self.analyser.polarity_scores(tweet)['compound']
        
        return score

    def fa(self):

        news = pd.read_csv("./data/demo_fa.csv")

        news['Date'] = pd.to_datetime(news.Date, format="%Y-%m-%d")

        news = news.set_index(news.Date).drop(columns=['Date.1'])

        news = news.dropna()

        words = ['USD','US Dollar','USA','AUD/USD']

        news = self.get_articles_from_list(words, news)

        news['Headline_Cleaned'] = pd.DataFrame(news.Headline).applymap(lambda x: self.clean(x))

        #Generate sentiment values
        news['Sentiment'] = pd.DataFrame(news.Headline_Cleaned).applymap(lambda x: self.get_sentiment(x))

        #Find mean sentiment for each day
        news = news.groupby('Date', as_index=True).agg({'Sentiment': ['mean']})

        news.columns = ['News']

        news_df = self.df.join(news)

        news_df['News'] = news_df['News'].interpolate(method='time', order=2)

        news_df = news_df.fillna(method = 'bfill')

        return news_df

    #Function to get headlines containing set of words
    def get_articles_from_list(self, words, articles):
    
        news = pd.DataFrame(columns=['Date','Headline'])
        
        for word in words:
        
            found = articles[articles.Headline.str.contains(word)]
            
            news = news.append(found)
                        
        news.sort_index(inplace=True)
        
        news = news.drop_duplicates()
        
        return news

    #Function to clean an inputed text string
    def clean(self, text):

        # remove hyperlinks
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

        # remove hashtags
        text = re.sub(r'#', '', text)
        
        # remove dashes
        text = re.sub(r'–', '', text)

        # remove punctuations
        text = re.sub(r'['+string.punctuation+']+', ' ', text)
        
        #remove new lines
        text = re.sub(r'\n', '', text)
        
        text = text.lower()
        
        return text