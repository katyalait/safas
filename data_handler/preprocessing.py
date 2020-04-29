import pandas as pd
import numpy as np
import math
import os
from data_handler.models import Word, Article, WordsInArticleCount, Source, StockPrice, Asset
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentiment.models import Category
import json
from datetime import datetime, timedelta
import sys
from django.conf import settings
from django.utils.timezone import make_aware
from dateutil import parser
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from .iterators import StockRowIterator, RelatedIterator
from data_handler.helpers import progress
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ArticlePreprocessor(object):
    def __init__(self, filename):
        self.df = pd.read_csv(os.path.join(BASE_DIR, filename))
        self.exist_arts = pd.DataFrame(([a['date_written'], a['headline']] for a in
                                            Article.objects.all().values('headline', 'date_written').order_by('headline')),
                                            columns=['date', 'headline'])


    def create_objects(self, add_contents=True):
        articles_len = len(self.df.index)
        for index, row in self.df.iterrows():
            article = self.exist_arts.loc[self.exist_arts['headline']==row.headline]
            if not article.empty:
                progress(index, articles_len, "Parsing article {}/{}... ".format(index, articles_len))
                source = Source.objects.filter(name=row['source']).first()
                if not source:
                    country = row['country']
                    if 'IRELAND' in country:
                        country = 0
                    elif 'UNITED KINGDOM' in country or 'ENGLAND' in country:
                        country = 1
                    else:
                        print(row['country'])
                        country = None
                    source = Source(
                        name=row['source'],
                        type=row['publication_type'],
                        country=country
                    )
                    source.save()
                date = datetime.strftime(parser.parse(row['date']), '%Y-%m-%d')
                article = Article(
                    headline=row['headline'],
                    length=int(row['length'].split("words")[0]),
                    source=source,
                )
                article.save()
                if add_contents:
                    self.add_contents(article, row['content'])
                article.date_written = date
                article.save()
            else:
                progress(index, articles_len, "Article {}/{} exists ...".format(index, articles_len))

    def add_contents(self, article, contents):
        article.contents = contents
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(contents)
        lemmatizer = WordNetLemmatizer()
        filtered_tokens=[token for token in word_tokens if token not in stop_words and token.isalpha()]
        lemmatized_tokens = "".join([lemmatizer.lemmatize(token.lower()) for token in filtered_tokens])
        article.tokens = lemmatized_tokens
        article.save()

    def get_df(self):
        return self.df

class FinancialDataPreprocessor(object):
    def __init__(self, filename, date="Date", open="Open",
                    adj_close="Adj Close", close="Close", high="High",
                    low="Low", volume="Volume", adj_included=False):

        self.df = pd.read_csv(filename, header=0, index_col=None)
        self.df = self.df.fillna(self.df.mean())
        self.numeric_cols = [close, open, high, low, volume]
        self.date = date
        self.open = open
        if adj_included:
            self.adj_close = adj_close
        self.adj_included = adj_included
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume


    def set_asset(self, name, ticker):
        try:
            self.asset = Asset.objects.get(name=name, ticker=ticker)
            self.exist_stocks = pd.DataFrame(([s.asset.ticker, s.date]
                                        for s in StockPrice.objects.filter(asset=asset)),
                                        columns=['asset', 'date'])
        except:
            self.asset = Asset.objects.create(name=name, ticker=ticker)
            self.asset.save()
            self.exist_stocks = pd.DataFrame()
        return self.asset

    def create_objects(self):
        df_len = len(self.df.index)

        # for col in self.numeric_cols:
        #     if self.df[col].dtype == 'object':
        #         self.df[col] = self.df[col].apply(lambda x: float(x.replace(",", "")))
        #         if col==self.volume:
        #             self.df[col] = self.df[col].apply(self.convert_vol)

        for index, row in self.df.iterrows():
            stockprice = None
            if not self.exist_stocks.empty:
                stockprice = self.exist_stocks.loc[self.exist_stocks['date']==row[self.date]]
            if not stockprice:
                progress(index, df_len, "Creating stock price object ...")
                curday = parser.parse(row[self.date]).date()
                interday = math.log(row[self.close]/row[self.open], 10)*100
                stockprice = StockPrice(asset=self.asset, open=row[self.open],
                                        close=row[self.close], high=row[self.high],
                                        low=row[self.low], adj_close=row[self.adj_close] if self.adj_included else row[self.close],
                                        volume= 0.0 if not row[self.volume] else row[self.volume],
                                        interday_volatility=interday)
                stockprice.save()
                try:
                    stockprice.date = curday
                    stockprice.save()
                except:
                    print(curday)
                    raise Exception("Found date error")
            else:
                progress(index, df_len, "Stock price object exists ...")

    def convert_vol(self, x):
        if 'M' in x:
            x = x.replace('M', "0000")
        if 'B' in x:
            x = x.replace('B', '000000000')
        if '.' in x:
            x = x.replace('.', '')
        return x




def produce_plots(articles):
    # # TODO: change to new iterator which will call stock iterator and then
    # # article iterator and merge the two on the date
    df = pd.DataFrame(([p.date, p.pos_sent, p.neg_sent,
                        p.interday_volatility, p.asset.name]
                        for p in RelatedIterator(articles)),
                        columns=['date', 'pos_sent', 'neg_sent',
                                'return', 'asset'])

    df = df.sort_values(by=['date'])
    fig = go.Figure()

    dates = df['date'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent'].drop_duplicates().to_frame()
    pos_sent = df['pos_sent'].drop_duplicates().to_frame()
    fig.add_trace(go.Scatter(x=dates['date'], y=stats.zscore(neg_sent['neg_sent']), mode='lines', name='Negative Sentiment'))
    fig.add_trace(go.Scatter(x=dates['date'], y=stats.zscore(pos_sent['pos_sent']), mode='lines', name='Positive Sentiment'))

    assets = Asset.objects.all()
    for a in assets:
        asset = df.loc[df['asset']==a.name]
        fig.add_trace(go.Scatter(x=asset['date'], y=stats.zscore(asset['return']), mode='lines', name=a.name))
    plt_div = plot(fig, output_type='div')
    return plt_div
