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


# Setup sqlite to read from
sqlite_file = '/home/ebonada/python/eurotweet/eurovision_final.db'
connection = sqlite3.connect(sqlite_file)
db = connection.cursor()

try:

	# read from db
	tweets = pd.read_sql_query("SELECT * FROM TweetsRaw", connection)

	# get dataframes ready for plotting
	tweets['createdAt'] = pd.to_datetime(tweets['createdAt'], format ='%a %b %d %H:%M:%S +0000 %Y') + pd.DateOffset(hours=2)
	tweets.index = tweets['createdAt']
	tweets = tweets[ tweets['createdAt'] > datetime(2017,5,12,6,0)]
	recent_tweets = tweets[ tweets['createdAt'] > (datetime.now() - timedelta(hours=-1))]

	fig = plt.figure()
	
	# create plot with all tweets
	ax1 = fig.add_subplot(211)
	ax1 = tweets.resample('H').count()['id'].plot(kind='area')
	ax1.set_xlabel('')

	# create plot with recent tweets
	ax2 = fig.add_subplot(212)
	ax2 = recent_tweets.resample('T').count()['id'].plot(color='green', kind='area')
	ax2.set_xlabel('')

	# store in file
	fig = ax2.get_figure()
	fig.savefig("/home/ebonada/python/eurotweet/server/public/tweets.png")

except:
	print("Error")