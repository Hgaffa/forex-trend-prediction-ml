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

class SentimentAnalysis:

    def __init__(self, data = None):

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

    def sa(self):

        tweets = self.get_twitter()

        reddit = self.get_reddit()

        sa_df = self.df.join(tweets)

        sa_df = sa_df.join(reddit)

        sa_df['Reddit'] = sa_df['Reddit'].interpolate(method='time', order=2)

        sa_df['Twitter'] = sa_df['Twitter'].interpolate(method='time', order=2)

        sa_df = sa_df.fillna(method = 'bfill')

        return sa_df


    def get_reddit(self):

        rd = pd.read_csv("./data/demo_sa_reddit.csv")

        rd['Date'] = pd.to_datetime(rd.Date, format="%Y-%m-%d")

        rd = rd.set_index(rd.Date).drop(columns=['Date'])

        #remove all posts who have less than 5 upvotes
        rd = rd[rd['Score'] !=0]

        #remove all posts that have below 70% upvoted ratio
        rd = rd[rd['Upvote Ratio'] != 0]

        rd['Cleaned'] = pd.DataFrame(rd.Titles).applymap(lambda x: self.clean(x))

        rd['Sentiment'] = pd.DataFrame(rd.Cleaned).applymap(lambda x: self.get_sentiment(x))

        rd = rd.groupby('Date', as_index=True).agg({'Sentiment': ['mean']})

        rd.columns = ['Reddit']

        return rd

    def get_twitter(self):

        tw = pd.read_csv("./data/demo_sa_twitter.csv")
        
        tw['Date'] = pd.to_datetime(tw.Date, format="%Y-%m-%d %H:%M:%S.%f")

        #Generate cleaned tweets and sentiment for each tweet
        tw['Cleaned'] = pd.DataFrame(tw.Tweet).applymap(lambda x: self.clean(x))

        tw['Sentiment'] = pd.DataFrame(tw.Cleaned).applymap(lambda x: self.get_sentiment(x))

        #Find mean sentiment for each day of tweets
        tw['Date'] = pd.DataFrame(tw.Date).applymap(lambda x: x.date())
        tw = tw.set_index(tw.Date).drop(columns=['Date'])

        twitter_news = tw.groupby('Date', as_index=True).agg({'Sentiment': ['mean']})

        twitter_news.columns = ['Twitter']

        return twitter_news

    def clean(self, text):

        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '', text)

        #remove hyperlinks
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

        # remove hyperlinks
        text = re.sub(r"http\S+", '', text)

        # remove hashtags
        text = re.sub(r'#', '', text)
        
        # remove dashes
        text = re.sub(r'-', '', text)

        # remove mentions
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  

        # remove punctuations
        text = re.sub(r'['+string.punctuation+']+', ' ', text)
        
        #remove new lines
        text = re.sub(r'\n', '', text)
        
        return text
