"""
Script that reads tweets from a database and processes the information to predict the results of the Eurovision Song Contest
"""

import re
import sqlite3
import pandas as pd
import numpy as np
import json
from textblob import TextBlob
from collections import Counter

"""
Aux functions
"""

def get_tweet_sentiment(tweet):
    """
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    """

    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet['tweetText']))
    
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


"""
Setup
"""

# Model coefficients
"""
model_coefs = np.array([-0.28177526, \
                        -15.09077224, 0.64080446, 3.93206855, 3.85856528,  \
                        13.44687015, 0.76969512, -6.81908351, 2.01213479])[...,None] # semi1
"""
model_coefs = np.array([ 2.22157127, -2.0677648 , -0.25456623,  0.0645284 ,  2.79439937])[...,None]; # semis


# set the features that will be used in the prediction
"""
features = ['negative_log', 'neutral_log', 'positive_log', 'tweets_log', \
            'negative_norm', 'neutral_norm', 'positive_norm', 'tweets_norm']
"""
features = ['negative_log', 'neutral_log', 'positive_log', 'tweets_log']


# Setup sqlite to read from
sqlite_file = 'eurovision_semi1.db'
connection = sqlite3.connect(sqlite_file)
db = connection.cursor()

# set country hashtags - semifinal 1
all_hashtags = ['SWE', 'GEO', 'AUS', 'ALB', 'BEL', 'MNE', 'FIN', 'AZE', 'POR', \
                'POL', 'MDA', 'ISL', 'CZE', 'CYP', 'ARM', 'SLO', 'LAT', 'GRE', \
                'AUT', 'BLR', 'DEN', 'EST', 'MKD', 'HUN', 'IRL', 'ISR', 'LTU', \
                'NOR', 'ROM', 'SMR', 'SRB', 'SUI', 'NED', 'CRO', 'BUL', 'MLT',  \
                'ITA', 'FRA', 'ESP', 'GBR', 'UKR', 'GER']
hashtags_semi1 = ['SWE', 'GEO', 'AUS', 'ALB', 'BEL', 'MNE', 'FIN', 'AZE', 'POR', 'GRE', \
                    'POL', 'MDA', 'ISL', 'CZE', 'CYP', 'ARM', 'SLO', 'LAT']
hashtags_semi2 = ['AUT', 'BLR', 'DEN', 'EST', 'MKD', 'HUN', 'IRL', 'ISR', 'LTU', 'MLT', \
                    'NOR', 'ROM', 'SMR', 'SRB', 'SUI', 'NED', 'CRO', 'BUL']
hashtags_final = ['ARM', 'AZE', 'ITA', 'MDA', 'POL', 'POR', 'UKR', 'AUS', 'BEL', 'CYP', 'FRA',\
                  'GER', 'GRE', 'ESP', 'GBR', 'SWE', 'BUL', 'BLR', 'CRO', 'HUN', 'DEN',\
                  'ISR', 'ROM', 'NOR', 'NED', 'AUT']
hashtags = hashtags_final



"""
Count tweets and analyze sentiment
"""

# read ALL tweets in english from db, evaluate sentiment, and count
all_sentiments = []
for country in hashtags:

    # get tweets from DB
    country_tweets = pd.read_sql_query("SELECT * FROM TweetsRaw WHERE language='en' AND tweetText LIKE '%#{}%'".format(country), connection)

    # count number of sentiments
    sentiments_count = Counter(country_tweets.apply(get_tweet_sentiment, axis=1))
    
    # append country to list
    all_sentiments.append({ \
                            'country': country, \
                            'positive': sentiments_count['positive'], \
                            'neutral': sentiments_count['neutral'], \
                            'negative': sentiments_count['negative'] \
                          })

# read all tweets (and just count)
all_tweet_counts = []
for country in hashtags:

    # get tweet count from DB
    db.execute("SELECT COUNT(*) AS count FROM TweetsRaw WHERE tweetText LIKE '%#{}%'".format(country))
    country_tweet_count = db.fetchone()[0]
    
    # append country to list
    all_tweet_counts.append({ \
                                'country': country, \
                                'count': country_tweet_count \
                            })

# transform to pandas dataframe from sentiments list and add total tweet count
results = pd.DataFrame(all_sentiments)
results = results.set_index(['country'])
results['tweets'] = [tc['count'] for tc in all_tweet_counts]


"""
Apply Prediction Model already trained
"""

# Feature engineering
"""
results['negative_norm'] = ( (results['negative'] - results['negative'].mean() ) / results['negative'].std() ).fillna(0).replace([np.inf, -np.inf], 0)
results['neutral_norm'] = ( (results['neutral'] - results['neutral'].mean() ) / results['neutral'].std() ).fillna(0).replace([np.inf, -np.inf], 0)
results['positive_norm'] = ( (results['positive'] - results['positive'].mean() ) / results['positive'].std() ).fillna(0).replace([np.inf, -np.inf], 0)
results['tweets_norm'] = ( (results['tweets'] - results['tweets'].mean() ) / results['tweets'].std() ).fillna(0).replace([np.inf, -np.inf], 0)

results['negative_log'] = ( np.log(results['negative']) ).fillna(0).replace([np.inf, -np.inf], 0)
results['neutral_log'] = ( np.log(results['neutral']) ).fillna(0).replace([np.inf, -np.inf], 0)
results['positive_log'] = ( np.log(results['positive']) ).fillna(0).replace([np.inf, -np.inf], 0)
results['tweets_log'] = ( np.log(results['tweets']) ).fillna(0).replace([np.inf, -np.inf], 0)
"""
results['positive_perc'] = results['positive'] / results['positive'].sum()
results['negative_perc'] = results['negative'] / results['negative'].sum()
results['neutral_perc'] = results['neutral'] / results['neutral'].sum()
results['tweets_perc'] = results['tweets'] / results['tweets'].sum()

results['negative_log'] = np.log(results['negative_perc']).fillna(0).replace([np.inf, -np.inf], 0)
results['neutral_log'] = np.log(results['neutral_perc']).fillna(0).replace([np.inf, -np.inf], 0)
results['positive_log'] = np.log(results['positive_perc']).fillna(0).replace([np.inf, -np.inf], 0)
results['tweets_log'] = np.log(results['tweets_perc']).fillna(0).replace([np.inf, -np.inf], 0)

# Apply model coeficients to data and compute 
X = results[features].values
X = np.append(np.ones(X.shape[0])[...,None] , X, axis=1)
results['predicted_score'] = np.dot(X, model_coefs)


"""
Compute ranking
"""

# Compute and log to console
ranking = results[['negative', 'neutral', 'positive', 'tweets', 'predicted_score']].sort_values(by = 'predicted_score', ascending = False)
print(ranking)

# log to file
with open('ranking.json', 'w') as f:
    f.write(ranking.to_json(orient = 'index'))
