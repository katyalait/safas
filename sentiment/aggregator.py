from data_handler.models import Article, StockPrice, Asset, WordsInArticleCount
from datetime import timedelta
from datetime import datetime
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy import signal
from plotly.offline import plot
import pandas as pd
import numpy as np
import plotly.express as px
from sentiment.models import Category
import sys
from django.db.models import Avg, Count, Min, Sum
from datetime import timedelta, date
from statsmodels.tsa.api import VAR
from scipy import stats

from data_handler.iterators import DataFrameCreator, AggregateRowIterator


def get_sentiment(article):
    words_in_article = WordsInArticleCount.objects.filter(article=article)
    print("Found {} words in article in database.".format(len(words_in_article)))
    negativ = Category.objects.filter(name="Negativ").first()
    ngtv = Category.objects.filter(name="Ngtv").first()
    neg_count = 0
    for word in words_in_article:
        if negativ.words.filter(word=word.word) or ngtv.words.filter(word=word.word):
            print(str(word.word) + " is a negative word.")
            neg_count += word.frequency
    return neg_count

def get_smarter_sentiment(contents):
    # get contents
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(contents)
    neg_count = 0
    negativ = Category.objects.filter(name="Negativ").first()
    ngtv = Category.objects.filter(name="Ngtv").first()
    lemmatizer = WordNetLemmatizer()
    for sentence in sentences:
        if len(sentence) < 1:
            continue
        word_tokens = word_tokenize(sentence)
        filtered_tokens=[token for token in word_tokens if token not in stop_words and token.isalpha()]
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in filtered_tokens]
        for token in lemmatized_tokens:
            if negativ.words.filter(word=token.upper()) or ngtv.words.filter(word=token.upper()):
                neg_count += 1
    return neg_count


def get_article_sentiments(start, end):
    articles = Article.objects.filter(date_written__range=[start, end])
    index = 0
    end = len(articles)
    for article in articles:
        index += 1
        progress(index, end, status="Creating article sentiment object ... ({}/{})".format(index, end))
        if article.contents:
            article.smarter_negative_words = get_smarter_sentiment(article.contents)
            article.save()

def correlate_rows(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = signal.correlate(a, b, 'full')
    return c


def collect_data(start, end):
    # get_article_sentiments(start, end)
    index = 0
    df = pd.DataFrame(([p.date, p.pos_sum, p.neg_sum,
                        p.length, p.neg_words, p.interday_volatility,
                        p.asset, p.pos_sent, p.neg_sent,
                        p.naive_sent]
                        for p in AggregateRowIterator(start, end)),
                        columns=['date', 'pos_sum', 'neg_sum',
                                'length', 'neg_words', 'return',
                                'asset', 'pos_sent', 'neg_sent',
                                'naive_sent'])
    print(df.head())
    return df

def autocorrelate_data():
    stocks = StockPrice.objects.all()
    dfc = DataFrameCreator(stocks)
    df = pd.DataFrame.from_dict(dfc)

    # create the VAR model
    model = VAR(endog=df)
    model_fit = model.fit()
    model.plot()


def produce_plots(start, end, assets):
    # get article sentiment by summing neg words over the course of the day and diving by length of words over the day
    df = collect_data(start, end)
    fig = go.Figure()
    dates = df['date'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent'].drop_duplicates().dropna().to_frame()
    pos_sent = df['pos_sent'].drop_duplicates().dropna().to_frame()
    fig.add_trace(go.Scatter(x=dates['date'], y=stats.zscore(neg_sent['neg_sent']), mode='lines', name='Negative Sentiment'))
    fig.add_trace(go.Scatter(x=dates['date'], y=stats.zscore(pos_sent['pos_sent']), mode='lines', name='Positive Sentiment'))
    for asset in assets:
        asset = Asset.objects.get(id=asset.id)
        stocks = StockPrice.objects.filter(asset=asset.id).filter(date__range=[start, end])
        df = pd.DataFrame(([s.date, s.interday_volatility] for s in stocks), columns=['date', 'return'])
        print(df.head())
        fig.add_trace(go.Scatter(x=df['date'], y=stats.zscore(df['return']), mode='lines', name=asset.name))
    plt_div = plot(fig, output_type='div')
    return plt_div

def produce_table(start, end):
    df = collect_data(start, end)
    return df
