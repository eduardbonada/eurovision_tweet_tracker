"""
Script that creates a plot with a tweet timeline
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
mpl.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import seaborn as sns

# set environment
production = False

# set filenames depending on the environment
if production == True:
    sqlite_file = '/home/ebonada/tests/euro2018/db_2018_live.db'
    plots_file = '/home/ebonada/tests/euro2018/server/public/tweets.png'
else:
    sqlite_file = 'db_2018_live.db'
    plots_file = 'server/public/tweets.png'

# set plot limits
min_date_tweets_plot = datetime(2018,5,4,0,0)
hours_shown_in_tweets_per_hour = 6

# Setup sqlite to read from
connection = sqlite3.connect(sqlite_file)
db = connection.cursor()

# read from db
tweets = pd.read_sql_query("SELECT * FROM TweetsRaw", connection)

# get dataframes ready for plotting
tweets['createdAt'] = pd.to_datetime(tweets['createdAt'], format ='%a %b %d %H:%M:%S +0000 %Y') + pd.DateOffset(hours=3)
tweets.index = tweets['createdAt']
tweets = tweets[ tweets['createdAt'] > min_date_tweets_plot]
recent_tweets = tweets[ tweets['createdAt'] > (datetime.now() - timedelta(hours=hours_shown_in_tweets_per_hour))]

fig = plt.figure()

# create plot with all tweets
ax1 = fig.add_subplot(211)
ax1.set_xlabel('')
if tweets['createdAt'].count() > 0:
  ax1 = tweets.resample('H').count()['createdAt'].plot(kind='area')

# create plot with recent tweets
ax2 = fig.add_subplot(212)
ax2.set_xlabel('')
if recent_tweets['createdAt'].count() > 0:
  ax2 = recent_tweets.resample('T').count()['createdAt'].plot(color='green', kind='area')

# store in file
fig = ax1.get_figure()
fig.savefig(plots_file)

print('Plot succesfully created')