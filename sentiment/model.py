from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson

import gensim
import re
import pandas as pd
from time import time
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.lang.en import English
import logging
from gensim.models.phrases import Phrases, Phraser
import sys
from data_handler.models import StockPrice, Article, Asset, Source
from sentiment.models import Category, Label, Column, Value
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.interfaces import TransformedCorpus
import os
from data_handler.helpers import progress, time_to_complete, daterange
from datetime import timedelta, date, datetime
from plotly.offline import plot
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_handler.iterators import SentimentIterator, SentimentExtractor, SentiWordNetExtractor
from statsmodels.tsa.stattools import adfuller
import hurst as hrst
from scipy import stats
import numpy as np
from django.db.models import Avg, Count, Min, Sum, Max
from scipy.stats import t
from scipy.stats import pearsonr

from rq import Queue
from worker import conn


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA = os.path.join(BASE, "cleaned_data.csv")
MODEL = os.path.join(BASE, "model2vec.model")
CORPUS = os.path.join(BASE, "corpus.mm")

class DocModelGenerator(object):
    """
    DocModelGenerator is an object which holds the predefined methods for
    creating a doc2vec model for a corpus of articles. By aggregating the
    necessary preprocessing and data cleaning methods, it reduces the overall
    complexity of desiging a doc2vec model when compared to other manual
    methods.

    Creates a doc2vec model for a given input csv file which contains headline,
    date and contents columns. Names for these columns can also be provided in
    the init but are set to 'headline', 'date' and 'content' by default.
    param file_path: a path to the .csv file containing the data
    param headline: name for the column which contains headline data
    param data: name for the column which contains date data
    param contents: name for the column which contains contents data
    """

    class Rating:
        def __init__(self, article, rating):
            self.article = article
            self.rating = rating

    class _doc_iter:
        def __init__(self, labels_list, doc_labels):
           self.doc_labels = labels_list
           self.doc_list = clean_docs
        def __iter__(self):
            for index, row in self.doc_list.iteritems():
                yield TaggedDocument(words = row.split(), tags=[self.labels_list[index]])

    class _article_iter:
        def __init__(self, similar):
            self.similar = similar

        def __iter__(self):
            for headline, rating in self.similar:
                article = Article.objects.get(headline=headline)
                yield (article, rating)



    def __init__(self):
        logging.info("Ready. Create or load a model.")

    def load_model(self, path):
        full_path  = os.path.join(BASE_DIR, path)
        self.model = Doc2Vec.load(full_path)


    def load_csv(self, file_path, headline='headline',
                    date='date', content='content'):
        self.df = pd.read_csv(filepath, usecols=[headline, date, content]).dropna()
        self.df = self.df.dropna().reset_index(drop=True).drop_duplicates()
        logging.info("Generator ready. Call create_model() to continue ...")

    def token_clean(doc):
        text = [token.lemma_ for token in doc if not token.is_stop]
        if len(text)>2:
            text = ' '.join(text)
            return text

    def preprocess(self, to_process):
        nlp = spacy.load("en_core_web_sm")
        logging.info("Prepaing data ...")
        t = time()
        index = 0
        mini_clean = (re.sub("[^A-Za-z']+", ' ', str(entry)).lower()
                        for entry in to_process)
        self.clean_docs = [token_clean(doc) for doc in
                            nlp.pipe(mini_clean,
                            disable=["tagger", "parser", "ner"])]
        print("Time to clean up everything: {}".format((time() - t)/ 60))

    def create_iterator(self):
        self.doc_labels = self.df['headline']
        self.preprocess((self.df['headline'] + ". " + self.df['content']))
        self.dociter = _doc_iter(self.doc_labels, self.clean_docs)
        return self.dociter

    def create_model(self, **kwargs):
        self.model = Doc2Vec(**kwargs)
        return self.model

    def build_vocab(self, **kwargs):
        self.model.build_vocab(self.dociter, **kwargs)
        return self.model

    def train_model(self, **kwargs):
        self.model.train(**kwargs)
        self.model.save(os.path.join(BASE_DIR, "models/doc2vec.model"))
        return self.model

    def get_related_articles(self, article, topn=20, ratings=True):
        similar = self.model.docvecs.most_similar(article.headline, topn=topn)
        return [a for a in self._article_iter(similar)]


class SentimentPriceModel(object):
    sentiments = Category.objects.all().values_list('name', flat=True)
    weights_c = {'Financial Times (London, England)': 385833/1503479, 'The Irish Times': 171288/383244, 'Irish Independent': 211956/383244,'The Times (London)': 799655/1503479,  'The Guardian(London)': 317991/1503479 }
    weights_t = {'Financial Times (London, England)': 385833/1886723, 'The Irish Times': 171288/1886723, 'Irish Independent': 211956/1886723,'The Times (London)': 799655/1886723,  'The Guardian(London)': 317991/1886723 }
    class StationaryResults():
        def __init__(self, df, column, adf, hurst):
            self.df = df[column].dropna()
            self.column = column
            if adf:
                self.adf = True
                self.result = adfuller(self.df)
                self.adf_stat = self.result[0]
                self.pval = self.result[1]
                self.critial_vals = self.result[4]
            if hurst:
                self.hurst = True
                H, c, data = hrst.compute_Hc(self.df)
                self.H = H
                self.c = c
                self.data = data
        def __repr__(self):
            result = "Testing {} for stationary".format(self.column)
            if self.adf:
                result += 'Low ADF statistic tells us that the series is non-stationary.\n'
                result += 'P-values higher than 0.05 also indicate non-stationarity.\n'
                result += 'ADF Statistic: {}\n'.format(self.adf_stat)
                result += 'p-value: {}\n'.format(self.pval)
                result += 'Critial-Values: \n'
                for key, value in self.critial_vals.items():
                    result += "\t{}%: {}\n".format(key, value)
            if self.hurst:
                    result += 'H < 0.5 is stationary\nH > 0.05 is non-stationary\n'
                    result += 'H = 0.5 is random walk/Brownian motion.\n'
                    result += 'H = {:.4f}, c = {:.4f}\n\n'.format(self.H, self.c)
            print(result)
            return result

    class ModelPlotter:
        def __init__(self, df):
            self.fig = go.Figure()
            self.df = df
            for column in self.df.columns:
                ewm = self.df.loc[:,column].ewm(span=30,adjust=False).mean()
                self.fig.add_trace(go.Scatter(x=self.df.index, y=ewm, mode='lines', name=column))

        def add_line(self, df, column):
            self.fig.add_trace(go.Scatter(x=self.df.index, y=df[column], mode='lines', name=column))

        def get_html_plot(self):
            plt_div = plot(self.fig, output_type='div')
            return plt_div

        def plot(self):
            self.fig.show()


    class VAR:
        def __init__(self, df):
            self.df = df
            self.model = VAR(df)


        def get_order_table(self, order, bootstrap=True):
            result = self.model.select_order(order)
            result = result.summary()
            html = result.as_html()
            if bootstrap:
                html = html.replace('simpletable', 'table')
            return html
        def adjust(self, val, length= 6):
            return str(val).ljust(length)

        def coint_results(self):
            results = []
            for col in self.df.columns:
                yield self.cointegration_test(self.df, col)

        def cointegration_test(self, df, key_col, alpha=0.05):
            """Perform Johanson's Cointegration Test and Report Summary"""
            out = coint_johansen(df,-1,5)
            d = {'0.90':0, '0.95':1, '0.99':2}
            traces = out.lr1
            cvts = out.cvt[:, d[str(1-alpha)]]
            print('\nName   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
            for col, trace, cvt in zip(df.columns, traces, cvts):
                if col == key_col:
                    return col, round(trace,2), cvt

        def fit_model(self, lag=1, freq=None):
            if not freq==None:
                try:
                    self.df.index = pd.DatetimeIndex(pd.to_datetime(self.df.index)).to_period(freq)
                except:
                    logging.info('Period type already set to {}'.format(self.df.index.freq))
            self.fit = self.model.fit(lag, trend='c', verbose=True)
            return self.fit

        def coefficient_df(self, lag, key_col):
            try:
                fit = self.fit
            except:
                fit = self.fit_model(lag=lag, freq="B")
            df = pd.DataFrame()
            spm2 = SentimentPriceModel()
            mv = self.df
            cols = mv.columns
            acc_df = pd.DataFrame()
            acc_df[key_col] = mv[key_col]
            acc_df.set_index(mv.index)
            counter = 1
            stop = 100
            constants = pd.DataFrame(index=["Const"])

            rsquared = pd.DataFrame(index=["R2"])
            durbin = pd.DataFrame(index=["Durbin-Watson"])
            for col in cols:
                if col == key_col:
                    continue
                model = "Model {}".format(counter)
                acc_df[col] = mv[col]
                acc_df = acc_df.dropna()
                spm2.set_df(acc_df)
                var = spm2.var()
                fit = var.model.fit(lag)
                index = fit.get_eq_index(key_col)
                df_out = pd.DataFrame(columns = [model])
                lookup = {}
                out = durbin_watson(fit.resid)
                dw = 0
                for col1, val in zip(acc_df.columns, out):
                    if col1 == col:
                        dw = round(val, 2)
                durbin[model] = pd.DataFrame({dw}, index=["Durbin-Watson"], columns=[model])[model]
                y = fit.fittedvalues[col]+fit.resid[col]
                R2 = 1 - np.sum(fit.resid[col].values**2)/np.sum((y.values-y.mean())**2)
                rsquared[model] = pd.DataFrame({R2}, index=["R2"], columns=[model])[model]

                for name in fit.names:
                    j = fit.get_eq_index(name)
                    lookup[j] = name

                for i in range(len(fit.coefs[0])):
                    row = lookup[i]
                    for j in range(len(fit.coefs)):
                        t = j+1
                        row_name = "{} t-{}".format(row, t)
                        pval = fit.pvalues[key_col]["L{}.{}".format(t, row)]
                        res = ""
                        if pval <= 0.10 and pval > 0.05:
                            res = "{:.3f}*".format(fit.coefs[j][index][i])

                        elif pval <= 0.05 and pval > 0.01:
                            res = "{:.3f}**".format(fit.coefs[j][index][i])
                        elif pval <= 0.01:
                            res = "{:.3f}***".format(fit.coefs[j][index][i])
                        else:
                            res = "{:.3f}".format(fit.coefs[j][index][i])
                        df_out.loc[row_name] = res
                pval = fit.pvalues[key_col]["const"]
                res = ""
                if pval <= 0.10 and pval > 0.05:
                    res = "{:.3f}*".format(fit.coefs_exog[0][index])

                elif pval <= 0.05 and pval > 0.01:
                    res = "{:.3f}**".format(fit.coefs_exog[0][index])
                elif pval <= 0.01:
                    res = "{:.3f}***".format(fit.coefs_exog[0][index])
                else:
                    res = "{:.3f}".format(fit.coefs_exog[0][index])

                constants[model] = pd.DataFrame({res}, index=["Const"], columns=[model])[model]

                df = pd.concat([df, df_out], axis=1, sort=False)
                counter += 1
                if counter == stop:
                    print(fit.summary())
                    break
            df = pd.concat([df, constants], sort=False)
            df = pd.concat([df, durbin], sort=False)
            df = pd.concat([df, rsquared], sort=False)
            df = df.fillna("-")
            return df

        def pvalues(self, bootstrap=True, lag=1, freq=None):
            tvals = ""
            try:
                tvals = self.fit.tvalues
            except:
                self.fit_model(lag=lag, freq=freq)
                tvals = self.fit.tvalues
            df = tvals.copy()
            for index, row in tvals.iterrows():
                for col in tvals.columns:
                    val = row[col]
                    pval = t.sf(abs(val), len(self.df)-1)*2
                    if pval < 0.05:
                        pval = "*{:.3f}$".format(pval)
                    else:
                        pval = "{:.3f}".format(pval)
                    df.loc[index,col] = pval
            html = df.to_html()
            if bootstrap:
                html = html.replace('dataframe','table table-bordered table-striped table-hover').replace('border=\"1\"', "")
            html = html.replace("*", "<b>").replace("$","</b>")
            return html

        def correlation(self, freq=None):
            corr = ""
            df = self.df
            pvals = pd.DataFrame(columns=df.columns, index=df.columns)
            for i in df.columns:
                for j in df.columns:
                    correl, pval = pearsonr(df[i], df[j])
                    correl = "{:.2f}%".format(correl*100)
                    if pval <= 0.10 and pval > 0.05:
                        pvals.loc[i][j] = "£{}$".format(correl)
                    elif pval <= 0.05 and pval > 0.01:
                        pvals.loc[i][j] = "£{}$*".format(correl)
                    elif pval <= 0.01:
                        pvals.loc[i][j] = "£{}$**".format(correl)
                    else:
                        pvals.loc[i][j] = correl

            return pvals

        def is_significant(self, col, row, lag, sig):
            tvals = self.fit.tvalues
            rows = ["L{}.{}".format(i+1, row) for i in range(lag)]
            for r in rows:
                val = tvals.loc[r,col]
                pval = t.sf(abs(val), len(self.df)-1)*2
                if pval <= sig:
                    return True
            return False

        def signif_correlation(self, freq=None, bootstrap=True):
            corr = self.correlation(freq=freq)
            for i in range(len(corr)):
                for j in range(0, i):
                    corr.iloc[i,j] = ""
            html = corr.to_html(classes="table table-bordered table-striped table-hover")
            html = html.replace('simpletable', '')
            html = html.replace('border=\"1\"', "")
            html = html.replace("£", "<b>").replace("$","</b>")
            return html


    def __init__(self):
        self.assets = Asset.objects.all()

    def load_csv(self, path):
        df = pd.read_csv(os.path.join(BASE_DIR, path), index_col='date')
        #df.index = df['date']
        self.multivariate_df = df
        return df

    def save(self, name):
        self.multivariate_df.to_csv(name)

    def load_db(self, label, set=False):
        try:
            label = Label.objects.get(model_name=label)
        except:
            raise Exception("Model does not exist.")
        columns = Column.objects.filter(label=label)
        queryset = Value.objects.filter(column__in=columns).order_by('date')
        q_df = pd.DataFrame(([q.column.name, q.date, q.value] for q in queryset), columns=['column', 'date', 'value'])
        column_list = list(q_df['column'].unique())
        df = pd.DataFrame(columns=column_list.append('date'))
        for column in q_df['column'].unique():
            cur_col = q_df.loc[q_df['column']==column]
            cur_col = cur_col.set_index('date')
            df[column] = cur_col['value']
        if set:
            print("Is na: {}".format(df.isna().sum()))
            self.multivariate_df = df.fillna(0.001)

        return df

    def slice_year(self, year):
        try:
            df = self.multivariate_df.copy()
        except:
            raise Exception("Plese set a dataframe")
        if not year == 0:
            start = "{}-01-01".format(year)
            end = "{}-01-01".format(year+1)
            startdate = pd.to_datetime(start).date()
            enddate = pd.to_datetime(end).date()
            df = df.loc[startdate:enddate].dropna()
        return df

    def get_df(self):
        return self.multivariate_df.copy()

    def set_df(self, df):
        self.multivariate_df = df


    def get_asset(self, asset=None, reset_df=True, start=None, end=None, column_name='return', zscore=True, volume=False, get_asset=True):
        unique = self.assets.values_list('ticker', flat=True)
        if asset==None:
            raise Exception('Not enough arguments provided, please pass a value for asset. The following are options for your model: \n{}'.format(unique))
        if not asset in unique:
            raise Exception('Asset not in dataset. Please provide one of the following options: {}'.format(unique))
        if not start:
            start = StockPrice.objects.filter(asset__ticker=asset).values_list('date').annotate(Min('date')).order_by('date')[0][1]
        if not end:
            end = StockPrice.objects.filter(asset__ticker=asset).values_list('date').annotate(Max('date')).order_by('-date')[0][1]

        stocks = StockPrice.objects.filter(asset__ticker=asset).filter(date__range=[start, end])
        if volume and get_asset:
            asset_df = pd.DataFrame(([s.date, s.interday_volatility, s.volume] for s in stocks), columns=['date', column_name + " Return", column_name + " Volume"]).set_index('date')
        elif not volume and get_asset:
            asset_df = pd.DataFrame(([s.date, s.interday_volatility] for s in stocks), columns=['date', column_name + " Return"]).set_index('date')
        elif volume and not get_asset:
            asset_df = pd.DataFrame(([s.date, s.volume] for s in stocks), columns=['date', column_name + " Volume"]).set_index('date')
        if zscore:
            for col in asset_df.columns:
                asset_df[col] = stats.zscore(asset_df[col])

        if reset_df:
            self.asset_name = asset
            self.dates = asset_df.index
            self.asset_df = asset_df
        return asset_df

    def get_sentiment(self, category=None, set=False, zscore=True, column_name=None, source_filter=[], sentiment_words=[], sentiwordnet=False, include_pos=False, h_contents = []):
        sentiment_df = pd.DataFrame()
        # check if they specified sentiment words
        if len(sentiment_words)==0:
            # check if they specified sentiwordnet
            if not sentiwordnet:
                # check if they set a category
                if not category:
                    raise Exception('Please specify a category name')
                else:
                    logging.info("Extracting {} sentiment from articles".format(category))
                    sentiment_words = [w.lower() for w in list(Category.objects.get(name=category).words.all().values_list('word', flat=True))]
                    if not column_name:
                        column_name = category
        if not column_name:
            raise Exception('Please set category name')

        if len(source_filter) > 0:
            sources = Source.objects.filter(id__in=source_filter).values_list('id', flat=True)
        else:
            sources = Source.objects.all().values_list('id', flat=True)
        if not sentiwordnet:
            sentiment_df = pd.DataFrame(([s['date'], s['sentiment']]
                                        for s in SentimentExtractor(self.dates,
                                                                sentiment_words,
                                                                sources,
                                                                h_contents)),
                                        columns=['date', column_name]).set_index('date')
        else:
            sentiment_df = pd.DataFrame(([s['date'], s['n_sentiment'], s['p_sentiment']]
                                        for s in SentiWordNetExtractor(self.dates,
                                                                sources,
                                                                h_contents)),
                                        columns=['date', "{} Negative".format(column_name),
                                                "{} Positive".format(column_name)]).set_index('date')

                # get the z-score
            if not include_pos:
                sentiment_df = sentiment_df.drop(["{} Positive".format(column_name)], axis=1)

        if set:
            self.sentiment_df = sentiment_df
        return sentiment_df

    def test_stationary(self, adf=True, hurst=True, df=None):
        if not df:
            try:
                df = self.multivariate_df
            except:
                raise AttributeError('Please set the asset to test by calling get_asset()')
        stationary_results = [self.StationaryResults(df, column, adf, hurst) for column in df.columns]
        return stationary_results

    def var(self, df=pd.DataFrame()):
        if df.empty:
            df = self.multivariate_df
        model = self.VAR(df)
        return model

    def stats(self, df=pd.DataFrame(), bootstrap=True):
        if df.empty:
            df = self.multivariate_df
        df2 = pd.DataFrame()
        for col in df.columns:
            df_temp = pd.DataFrame(
                                [df[col].mean(),
                                df[col].std(),
                                df[col].var(),
                                (df[col].skew()),
                                (df[col].kurtosis()),
                                (np.amax(df[col])),
                                (np.amin(df[col]))],
                                columns=[col]
                        )
            df2[col] = df_temp[col]
        df2.index = ["Mean", "S.D", "Variance", "Skew", "Kurtosis", "Max", "Min"]
        html = df2.to_html()
        if bootstrap:
            html = html.replace('border=\"1\"', "")
            html = html.replace('dataframe','table table-bordered table-striped table-hover')
        return html


    def add_asset_variable(self, left=None, asset=None, reset_df=False, set_multi=True, column_name='return', zscore=True, volume=False, get_asset=True):
        if not left:
            try:
                left = self.multivariate_df
            except:
                try:
                    left = self.asset_df
                except:
                    left = pd.DataFrame()
        if not asset:
            raise Exception('Not enough arguments provided, please pass a value for asset. The following are options for your model: \n{}'.format(self.asset_df['asset'].unique()))
        if left.empty:
            reset_df=True
        right = self.get_asset(asset=asset, reset_df=reset_df, column_name=column_name, zscore=zscore, volume=volume, get_asset=get_asset)
        if not left.empty:
            merged = left.merge(right=right, on='date')
        else:
            merged = right
        if set_multi:
            self.multivariate_df = merged
        return merged

    def add_sentiment_variable(self, left=None, category="", set=True, zscore=True, column_name=None, source_filter=[], reset_df=False, sentiment_words = [], include_pos=False,  sentiwordnet=False, h_contents=[], weighted=False, countries=False):
        if not left:
            try:
                left = self.multivariate_df
            except:
                try:
                    left = self.asset_df
                except:
                    raise Exception('Please set the asset_df variable by calling get_asset().')
        if not column_name:
            column_name = category
        sentiment_df = self.get_sentiment(category=category, zscore=zscore,
                                                column_name=column_name,
                                                source_filter=source_filter,
                                                set=reset_df,
                                                include_pos=include_pos,
                                                sentiment_words=sentiment_words,
                                                sentiwordnet=sentiwordnet,
                                                h_contents = h_contents)

        if weighted and len(source_filter) == 1:
            sourcename = Source.objects.get(id=source_filter[0]).name
            for col in sentiment_df.columns:
                print("Multiplying by weights: {}".format(col))
                if countries:
                    sentiment_df[col] *= self.weights_c[sourcename]
                else:
                    sentiment_df[col] *= self.weights_t[sourcename]
        sentiment_df = sentiment_df.fillna(0)
        if zscore:
            print("Getting zscore")

            for col in sentiment_df.columns:
                print("Zscore of {}".format(col))
                sentiment_df[col] = stats.zscore(sentiment_df[col])

        new_df = pd.concat([left, sentiment_df], axis=1, sort=False)
        # new_df = left.merge(right=sentiment_df, on='date')
        if set:
            self.multivariate_df = new_df
        return sentiment_df

    def produce_plot_model(self, df=pd.DataFrame()):
        if df.empty:
            try:
                df = self.multivariate_df
            except:
                try:
                    df = self.asset_df
                except:
                    try:
                        df = self.sentiment_df
                    except:
                        raise Exception("No dataframes set in the model.")
        self.mp = SentimentPriceModel.ModelPlotter(df=df)
        self.save('aggregate2.csv')
        return self.mp

    def remove_variable(self, column=None):
        if not column:
            raise Exception("Please provide a column name.")
        try:
            del self.multivariate_df[column]
        except:
            try:
                del self.asset_df[column]
            except:
                raise Exception("Column does not exist in the object dataframe. Please try entering an existing column.")

    def save_to_database(self, name, description, df=None):
        if not df:
            try:
                df = self.multivariate_df
            except:
                try:
                    logging.info("Multivariate DF not set. Attempting to save asset DF.")
                    df = self.asset_df
                except:
                    raise Exception("Asset DF not set. Set this before attempting to save it to database.")
        try:
            df.index = pd.DatetimeIndex(df.index.to_timestamp())
        except:
            pass
        print(df.head())
        print("Length: {},{}".format(df.shape[0], df.shape[1]))
        object_creation_count = 0
        cols, rows = df.shape
        desired_count = cols*rows + len(df.columns) + 1
        label = Label.objects.create(model_name=name, description=description)
        label.save()
        object_creation_count +=1
        for col in df.columns:
            colobj = Column.objects.create(label=label, name=col)
            colobj.save()
            object_creation_count +=1
            df_col = df.loc[:, [col]]
            for index, row in df_col.iterrows():
                object_creation_count +=1
                progress(object_creation_count, desired_count, status="Saving column: {}".format(col))
                val = Value.objects.create(column=colobj, date=index, value=row[col])
                val.save()
        logging.info("Created {}/{} required objects during save".format(object_creation_count, desired_count))

    def create_general_inquirer(self, request, cats, assets, source, h_contents, weighted, source_signals, included_countries, irish_sources, uk_sources, include):

        for asset in assets:
            obj = Asset.objects.get(ticker=asset)
            if "volume" in include:
                volume = True
            if "assets" in include:
                assets = True
            self.add_asset_variable(asset=asset, column_name=obj.name, zscore=True, volume=True)

        for cat in cats:
            if source==0:
                if weighted == 1 or source_signals == 1:
                    for s in irish_sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True)
                    if source_signals == 0:
                        df = self.get_df()
                        df["Irish {}".format(cat)] = 0
                        print(df.head())
                        for c in irish_sources:
                            df["Irish {}".format(cat)] += df["{}.{}".format(c, cat)]
                            df = df.drop(["{}.{}".format(c, cat)], axis=1)
                        self.multivariate_df = df

                    for s in uk_sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True)

                    if source_signals == 0:
                        df = self.get_df()
                        df["UK {}".format(cat)] = 0
                        for c in uk_sources:
                            df["UK {}".format(cat)] += df["{}.{}".format(c, cat)]
                            df = df.drop(["{}.{}".format(c, cat)], axis=1)
                        self.multivariate_df = df

                else:
                    self.add_sentiment_variable(category=cat, column_name="UK {}".format(cat), source_filter=uk_sources,set=True)
                    self.add_sentiment_variable(category=cat, column_name="Irish {}".format(cat), source_filter=irish_sources,set=True)

            else:
                if weighted == 1:
                    sources = list(Source.objects.all().values_list('id', flat=True))
                    for s in sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s], set=True, h_contents= h_contents, weighted=True, countries=False)
                    df = self.get_df()
                    df["{}".format(cat)] = 0
                    for c in sources:
                        df["{}".format(cat)] += df["{}.{}".format(c, cat)]
                        df = df.drop(["{}.{}".format(c, cat)], axis=1)
                    self.multivariate_df = df
                else:
                    self.add_sentiment_variable(category=cat, column_name="{}".format(cat), set=True, )
        name = request['name']
        description = request['description']
        self.save_to_database(name, description)

    def create_sentiwordnet(self, request, assets, source, source_signals, include_pos, h_contents, weighted, irish_sources, uk_sources, include):
        for asset in assets:
            obj = Asset.objects.get(ticker=asset)
            if "volume" in include:
                volume = True
            if "assets" in include:
                assets = True
            self.add_asset_variable(asset=asset, column_name=obj.name, zscore=True, volume=True)
        if source==0:
            if weighted == 1 or source_signals == 1:
                for s in irish_sources:
                    sourcename = Source.objects.get(id=s).name
                    self.add_sentiment_variable(column_name="{} SentiWordNet".format(sourcename), source_filter=[s], sentiwordnet=True, include_pos = include_pos, h_contents= h_contents, weighted=False, countries=True)
                if source_signals == 0:
                    df = self.get_df()
                    df["Irish SentiWordNet Negative"] = 0
                    if include_pos:
                        df["Irish SentiWordNet Positive"] = 0
                    for c in irish_sources:
                        sourcename = Source.objects.get(id=c).name
                        df["Irish SentiWordNet Negative"] += df["{} SentiWordNet Negative".format(sourcename)]
                        df = df.drop(["{} SentiWordNet Negative".format(sourcename)], axis=1)
                        if include_pos:
                            df["Irish SentiWordNet Positive"] += df["{} SentiWordNet Positive".format(sourcename)]
                            df = df.drop(["{} SentiWordNet Positive".format(sourcename)], axis=1)
                    self.multivariate_df = df
                for s in uk_sources:
                    sourcename = Source.objects.get(id=s).name
                    self.add_sentiment_variable(column_name="{} SentiWordNet".format(sourcename), source_filter=[s], sentiwordnet=True, include_pos = include_pos, h_contents= h_contents, weighted=False, countries=True)

                if source_signals == 0:
                    df = self.get_df()
                    df["UK SentiWordNet Negative"] = 0
                    if include_pos:
                        df["UK SentiWordNet Positive"] = 0
                    for c in uk_sources:
                        sourcename = Source.objects.get(id=c).name
                        df["UK SentiWordNet Negative"] += df["{} SentiWordNet Negative".format(sourcename)]
                        df = df.drop(["{} SentiWordNet Negative".format(sourcename)], axis=1)
                        if include_pos:
                            df["UK SentiWordNet Positive"] += df["{}.SentiWordNet Positive".format(c)]
                            df = df.drop(["{}.SentiWordNet Positive".format(c)], axis=1)
                    self.multivariate_df = df
            else:
                self.add_sentiment_variable(column_name="SentiWordNet UK", sentiwordnet=True, include_pos = include_pos, set=True, source_filter=uk_sources, h_contents= h_contents,)
                self.add_sentiment_variable(column_name="SentiWordNet Irish", sentiwordnet=True, include_pos = include_pos, set=True, source_filter=irish_sources, h_contents= h_contents, )
        else:
            if weighted == 1 or source_signals==1:
                sources = list(Source.objects.all().values_list('id', flat=True))
                for s in sources:
                    sourcename = Source.objects.get(id=s).name
                    self.add_sentiment_variable(column_name="{} SentiWordNet".format(sourcename), source_filter=[s], sentiwordnet=True, include_pos = include_pos, h_contents= h_contents, weighted=True, countries=False)
                if source_signals == 0:
                    df = self.get_df()
                    df["SentiWordNet Negative"] = 0
                    if include_pos:
                        df["SentiWordNet Positive"] = 0
                    for c in sources:
                        sourcename = Source.objects.get(id=c).name
                        df["SentiWordNet Negative"] += df["{} SentiWordNet Negative".format(sourcename)]
                        df = df.drop(["{} SentiWordNet Negative".format(sourcename)], axis=1)
                        if include_pos:
                            df["SentiWordNet Positive"] += df["{}.SentiWordNet Positive".format(c)]
                            df = df.drop(["{}.SentiWordNet Positive".format(c)], axis=1)
                    self.multivariate_df = df
            else:
                self.add_sentiment_variable(column_name="SentiWordNet", sentiwordnet=True, include_pos = include_pos, set=True, )

        name = request['name']
        description = request['description']
        self.save_to_database(name, description)

    def create_word2vec(self, request, cats, assets, include, w_cats, topn, source, weighted, source_signals, h_contents, irish_sources, uk_sources, vectors, sw):
        wordset = set()
        for asset in assets:
            obj = Asset.objects.get(ticker=asset)
            if "volume" in include:
                volume = True
            if "assets" in include:
                assets = True
            self.add_asset_variable(asset=asset, column_name=obj.name, zscore=True, volume=volume, get_asset=assets)
        for cat in cats:
            if source==0:
                if weighted == 1 or source_signals == 1:
                    for s in irish_sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True)
                    if source_signals == 0:
                        df = self.get_df()
                        df["Irish {}".format(cat)] = 0
                        print(df.head())
                        for c in irish_sources:
                            df["Irish {}".format(cat)] += df["{}.{}".format(c, cat)]
                            df = df.drop(["{}.{}".format(c, cat)], axis=1)
                        self.multivariate_df = df

                    for s in uk_sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True)
                    if source_signals == 0:
                        df = self.get_df()
                        df["UK {}".format(cat)] = 0
                        for c in uk_sources:
                            df["UK {}".format(cat)] += df["{}.{}".format(c, cat)]
                            df = df.drop(["{}.{}".format(c, cat)], axis=1)
                        self.multivariate_df = df

                else:
                    self.add_sentiment_variable(category=cat, column_name="UK {}".format(cat), source_filter=uk_sources,set=True, )
                    self.add_sentiment_variable(category=cat, column_name="Irish {}".format(cat), source_filter=irish_sources,set=True, )
            else:
                if weighted == 1:
                    sources = list(Source.objects.all().values_list('id', flat=True))
                    for s in sources:
                        self.add_sentiment_variable(category=cat, column_name="{}.{}".format(s, cat), source_filter=[s], set=True, h_contents= h_contents, weighted=True, countries=False)
                    df = self.get_df()
                    df["{}".format(cat)] = 0
                    for c in sources:
                        df["{}".format(cat)] += df["{}.{}".format(c, cat)]
                        df = df.drop(["{}.{}".format(c, cat)], axis=1)
                    self.multivariate_df = df
                else:
                    self.add_sentiment_variable(category=cat, column_name="{}".format(cat), set=True, )
        for cat in w_cats:
            words = [w.lower() for w in Category.objects.get(name=cat).words.all().values_list('word', flat=True)]
            expanded_list = set()
            index = 0
            length = len(words)
            for w in words:
                index +=1
                try:
                    if w in sw:
                        continue
                    werds = [x for (x,_) in vectors.most_similar(positive=w, topn=topn)]
                    for x in werds:
                        expanded_list.add(x)
                    progress(index, length, status="Added word")
                except Exception as e:
                    progress(index, length, status="Word not added. {}".format(e))
                expanded_list.add(w)
            if source==0:
                if weighted == 1:
                    for s in irish_sources:
                        sourcename = Source.objects.get(id=s).name
                        self.add_sentiment_variable(column_name="{} {}".format(sourcename, cat), sentiment_words=list(expanded_list), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True, )
                    if source_signals == 0:
                        df = self.get_df()
                        df["Word2Vec Irish {}".format(cat)] = 0
                        print(df.head())
                        for c in irish_sources:
                            sourcename = Source.objects.get(id=c).name
                            df["Word2Vec Irish {}".format(cat)] += df["{} {}".format(sourcename, cat)]
                            df = df.drop(["{} {}".format(sourcename, cat)], axis=1)
                        self.multivariate_df = df

                    for s in uk_sources:
                        sourcename=Source.objects.get(id=s).name
                        self.add_sentiment_variable(column_name="{} {}".format(sourcename, cat), sentiment_words=list(expanded_list), source_filter=[s],set=True, h_contents= h_contents, weighted=True, countries=True, )
                    if source_signals == 0:
                        df = self.get_df()
                        df["Word2Vec UK {}".format(cat)] = 0
                        for c in uk_sources:
                            sourcename=Source.objects.get(id=c).name
                            df["Word2Vec UK {}".format(cat)] += df["{} {}".format(sourcename, cat)]
                            df = df.drop(["{} {}".format(sourcename, cat)], axis=1)
                        self.multivariate_df = df

                else:

                    self.add_sentiment_variable(column_name="Word2Vec UK {}".format(cat), source_filter=uk_sources, sentiment_words=list(expanded_list), h_contents= h_contents, )
                    self.add_sentiment_variable(column_name="Word2Vec Irish {}".format(cat), source_filter=irish_sources, sentiment_words=list(expanded_list), h_contents= h_contents, )
            else:
                if weighted == 1 or source_signals == 1:
                    sources = list(Source.objects.all().values_list('id', flat=True))
                    for s in sources:
                        sourcename=Source.objects.get(id=s).name
                        self.add_sentiment_variable(column_name="{} {}".format(sourcename, cat), set=True, source_filter=[s], sentiment_words=list(expanded_list), h_contents= h_contents, weighted=True, countries=False, )
                    if source_signals == 0:
                        df = self.get_df()
                        df["{}".format(cat)] = 0
                        for c in sources:
                            sourcename=Source.objects.get(id=c).name
                            df["{}".format(cat)] += df["{} {}".format(sourcename, cat)]
                            df = df.drop(["{} {}".format(sourcename, cat)], axis=1)
                    self.multivariate_df = df
                else:
                    self.add_sentiment_variable(column_name="Word2Vec {}".format(cat), sentiment_words=list(expanded_list), h_contents= h_contents, )
        name = request['name']
        description = request['description']
        self.save_to_database(name, description)

class ArticleSentimentPriceModel(object):
    """
    Given a list of articles, the object finds their sentiment and the
    associated returns of markets on a given day.
    """
    class _SentiRet:
        def __init__(self, returns, sentiment):
            self.returns = returns
            self.sentiment = sentiment

    def __init__(self, articles, asset, category):
        self.articles = articles

        self.dates = list(set(articles.values_list('date_written', flat=True)))
        self.returns = StockPrice.objects.filter.filter(asset=asset).filter(date__in=self.dates).values_list('interday_volatility', flat=True)
        # get sentiement of articles

class CorrelationModel(object):
    """
    Extracts the correlation of data streams from a correlation table and plots it.
    """
    def __init__(self, model_name):
        self.spm = SentimentPriceModel()
        self.spm.load_db(model_name, set=True)
        self.df = self.spm.get_df()
        self.columns = self.df.columns

    def produce_plots(self, indicator):
        self.fig = go.Figure()
        total_df = pd.DataFrame()
        year = 2016
        for i in range(4):
            df = self.spm.slice_year(year+i)
            corr = self.correlation_df(df)
            total_df[year+i] = corr[indicator]
        total_df = total_df.T
        for col in total_df.columns:
            if col == indicator:
                continue
            if indicator == "UK Negative" and col=="UK Positive":
                continue
            if indicator == "UK Positive" and col=="UK Negative":
                continue
            arr = np.array(total_df[col])*100
            self.fig.add_trace(go.Scatter(x=total_df.index, y=arr, mode='lines', name=col))
        plt_div = plot(self.fig, output_type='div')
        return plt_div

    def produce_table(self, year):
        df = self.spm.slice_year(year)
        var = self.spm.var(df)
        html = var.signif_correlation(freq="B")
        return html

    def correlation_df(self, df):
        pvals = pd.DataFrame(columns=df.columns, index=df.columns)
        for i in df.columns:
            for j in df.columns:
                correl, pval = pearsonr(df[i], df[j])
                pvals.loc[i][j] = correl
        return pvals

class Word2VecModel(object):
    def __init__(self):
        # convert to lines
        # clean lines
        # phrase detector
        # feed to model
        # build model
        # save and store in database
        pass




def get_related_words(wv, word_list):
    dictionary = set()
    for word in word_list:
        if word in wv:
            [dictionary.add(x) for (x,y) in wv.most_similar(positive=[word], topn=5)]
            dictionary.add(word)
    new_word_list = list(dictionary)
    return new_word_list

def get_list(wv):
    neg = Category.objects.get(name="Negativ")
    neg_words = neg.words.all()

    pos = Category.objects.get(name="Positiv")
    pos_words = pos.words.all()

    neg_list = [word.word.lower() for word in neg_words]
    pos_list = [word.word.lower() for word in pos_words]

    neg_ = get_related_words(wv, neg_list)
    pos_ = get_related_words(wv, pos_list)

    subjective_words = neg_list + pos_list

    word_list = get_related_words(wv, subjective_words)

    return word_list, neg_, pos_


def clean_article(row):
    lines = []
    headline = row['headline']
    content = row['content']
    total = headline + ". " + content
    sentences = sent_tokenize(total)
    return sentences

def clean_articles_for_doc(row):
    total = row['headline'] + ". " + row["content"]
    return row

def convert_to_lines_df(input_file, headline, content):
    df = pd.read_csv(input_file)
    df = df.loc[:, [headline, content]]
    logging.info("Cleaning null values.")
    #
    logging.info(df.isnull().sum())
    df = df.dropna().reset_index(drop=True)
    logging.info(df.isnull().sum())
    #
    lines = []
    total, _ = df.shape
    for index, row in df.iterrows():
        sentences = clean_article(row)
        progress(index, total, status="Found {} sentences in article {}/{}".format(len(sentences), index, total))
        lines.extend(sentences)
    logging.info("Total lines found = {}".format(len(lines)))
    return lines

def mini_clean(text):
    return [re.sub("[^A-Za-z']+", ' ', str(text)).lower()]

def token_clean(doc):
    text = [token.lemma_ for token in doc if not token.is_stop]
    if len(text)>2:
        line = ' '.join(text)
        return line


def preprocess(lines):
    nlp = spacy.load("en_core_web_sm")
    docs = []
    t = time()
    index = 0
    mini_clean = (re.sub("[^A-Za-z']+", ' ', str(line)).lower() for line in lines)
    clean_lines = [token_clean(line) for line in nlp.pipe(mini_clean, disable=["tagger", "parser", "ner"])]
    print("Time to clean up everything: {}".format((time() - t)/ 60))
    return clean_lines

def phrase_detector(df, min_count):
    # create sentences
    sents = [line.split() for line in df['clean_docs']]
    phrases = Phrases(sents, min_count=min_count, progress_per=1000)
    bigram = Phraser(phrases)
    sentences = bigram[sents]
    logging.info("Finished replacing bigrams")
    logging.info(type(sentences))
    return sentences


def create_model(sentences, epochs, min_count, window, name="word2vec.model"):
    w2v_model = gensim.models.Word2Vec(
                        min_count=min_count,
                        window=window,
                        size=120,
                        sample=2e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=multiprocessing.cpu_count()-1
                        )
    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.save("built_vocab.model")

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)
    w2v_model.save(name)
