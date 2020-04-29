
from .models import Article, StockPrice, Asset
from gensim.models import Doc2Vec
from django.conf import settings
from django.db.models import Count
from nltk.tokenize import word_tokenize
from .helpers import progress
import heapq
from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np
from django.conf import settings
from scipy import stats

def produce_aggregate_plots(start, end):
    # get article sentiment by summing neg words over the course of the day and diving by length of words over the day
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
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
    fig = go.Figure()
    dates = df['date'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent'].drop_duplicates().to_frame()
    pos_sent = df['pos_sent'].drop_duplicates().to_frame()

    gbpeur = df.loc[df['asset']=="GBP/EUR"]
    gbpusd = df.loc[df['asset']=="GBP/USD"]

    ftse100 = df.loc[df['asset']=="FTSE 100"]
    ftse100['return'] = ftse100['return'].apply(lambda x: x/100)

    fig.add_trace(go.Scatter(x=dates['date'], y=neg_sent['neg_sent'], mode='lines', name='Negative Sentiment'))
    fig.add_trace(go.Scatter(x=dates['date'], y=pos_sent['pos_sent'], mode='lines', name='Positive Sentiment'))
    fig.add_trace(go.Scatter(x=gbpeur['date'], y=gbpeur['return'], mode='lines', name='GBP/EUR'))
    fig.add_trace(go.Scatter(x=gbpusd['date'], y=gbpusd['return'], mode='lines', name='GBP/USD'))
    fig.add_trace(go.Scatter(x=ftse100['date'], y=ftse100['return'], mode='lines', name='FTSE 100'))

    plt_div = plot(fig, output_type='div')
    return plt_div

def produce_article_plots(articles):
    length = len(articles)
    df = pd.DataFrame(([p.date, p.pos_sent, p.neg_sent,
                        p.interday_volatility, p.asset]
                        for p in RowIterator(articles)),
                        columns=['date', 'pos_sent', 'neg_sent',
                                'return', 'asset'])
    print(df)
    df = df.sort_values(by=['date'])
    fig = go.Figure()

    dates = df['dates'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent']

    fig = px.line(df, x="date", y="value", color='line', line_group='line')
    plt_div = plot(fig, output_type='div')
    return df


def produce_table(start, end):
    df = collect_data(start, end)
    return df

def produce_article_freq_plots(queryset):
    articles = queryset.values('date_written').annotate(total=Count('date_written')).order_by('-total')
    df = pd.DataFrame(([a['date_written'], a['total']] for a in articles), columns=['date', 'count'])
    df = df.sort_values(by=['date'])
    cumsum = df['count'].cumsum()
    df['cumsum'] = cumsum
    df['ewm'] = df.loc[:,'count'].ewm(span=30,adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['count'], mode='lines', name='Articles per Day'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['cumsum'], mode='lines', name='Cumulative Sum'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['ewm'], mode='lines', name='Exponential Cumulative Sum'))


    plt_div = plot(fig, output_type='div')
    return plt_div

def produce_stock_plots(queryset):
    df = pd.DataFrame(([s.date, s.asset.name, s.asset.ticker, s.open, s.volume, s.close, s.interday_volatility] for s in queryset),
                        columns=['date', 'name', 'ticker', 'open', 'volume', 'close', 'return'])
    unique_assets = df.name.unique()
    df = df.drop_duplicates()

    fig = go.Figure()
    fig1 = go.Figure()
    for asset in unique_assets:
        line = df.loc[df['name']==asset]
        line['ewm'] = line.loc[:,'return'].ewm(span=30,adjust=False).mean()
        line['avg_return'] = stats.zscore(line['return'])
        line['ewm2'] = line.loc[:,'avg_return'].ewm(span=30,adjust=False).mean()
        line['ewmv'] = line.loc[:,'volume'].ewm(span=30,adjust=False).mean()

        # fig.add_trace(go.Scatter(x=line['date'], y=line['avg_return'], mode='lines', name=asset))
        # fig.add_trace(go.Scatter(x=line['date'], y=line['return'], mode='lines', name=asset))
        # fig.add_trace(go.Scatter(x=line['date'], y=line['volume'], mode='lines', name=asset))
        fig.add_trace(go.Scatter(x=line['date'], y=line['ewm'], mode='lines', name=asset))
        fig1.add_trace(go.Scatter(x=line['date'], y=line['ewmv'], mode='lines', name=asset))




    plt_div = plot(fig, output_type='div')
    plt_div1 = plot(fig1, output_type='div')


    return plt_div, plt_div1

def get_dates(request):
    start_day = int(request.get('date_start_day'))
    start_month = int(request.get('date_start_month'))
    start_year = int(request.get('date_start_year'))

    end_day = int(request.get('date_end_day'))
    end_month = int(request.get('date_end_month'))
    end_year = int(request.get('date_end_year'))

    date_start = datetime(day=start_day, month=start_month, year=start_year).date()
    date_end = datetime(day=end_day, month=end_month, year=end_year).date()

    return date_start, date_end


def get_related_articles(article):
    dmodel = Doc2Vec.load(settings.DOC2VECMODEL)
    similar = dmodel.docvecs.most_similar(article.headline, topn=20)
    articles = []
    for headline, rating in similar:
        cur = Article.objects.get(headline=headline)
        articles.append([cur, rating])
    return articles

def get_relevant_stocks(articles):
    stocks = []
    for article, rating in articles:
        cur = StockPrice.objects.filter(date=article.date_written)
        if cur:
            stocks.append(cur)
    return stocks


def frequency_count(articles):
    total_articles = len(articles)
    word_count = {}
    index = 0
    for article in articles:
        index += 1
        progress(index, total_articles, status="")
        tokens = word_tokenize(article.tokens)
        for token in tokens:
            if not token.isalpha():
                continue
            if token in word_count:
                word_count[token][0] += 1
            else:
                word_count[token] = [1]
    print()
    df = pd.DataFrame.from_dict(word_count, orient='index', columns=['Frequency'])
    df = df.sort_values('Frequency', ascending=False)
    df = df.reset_index()
    df = df.rename(columns={'index': 'Word'})
    df = df.iloc[100:200]
    df = df.reset_index()

    counts = []

    for index, row in df.iterrows():
        progress(index, len(df), status="")

        # check each tokens appearance count in all articles
        counts.append("{:.2f}%".format(len(articles.filter(tokens__contains=row['Word']))/total_articles*100))
    df['Presence %'] = pd.DataFrame(counts)
    print(df.head())

    return df

def produce_article_freq(fig, queryset, line, name):
    articles = queryset.values('date_written').annotate(total=Count('date_written')).order_by('-total')
    df = pd.DataFrame(([a['date_written'], a['total']] for a in articles), columns=['date', 'count'])
    df = df.sort_values(by=['date'])
    cumsum = df['count'].cumsum()
    df['cumsum'] = cumsum
    df['ewm'] = df.loc[:,'count'].ewm(span=30,adjust=False).mean()
    if line=="count":
        fig.add_trace(go.Scatter(x=df['date'], y=df['count'], mode='lines', name=name))
    elif line=="cumsum":
        fig.add_trace(go.Scatter(x=df['date'], y=df['cumsum'], mode='lines', name=name))
    else:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ewm'], mode='lines', name=name))

    return fig

def produce_source_article_freq_plots(sources):
    count = go.Figure()
    cumsum = go.Figure()

    ewm = go.Figure()
    for source in sources:

        queryset = Article.objects.filter(source=source.id)
        count = produce_article_freq(count, queryset, "count", source.name)
        cumsum = produce_article_freq(cumsum, queryset, "cumsum", source.name)

        ewm = produce_article_freq(ewm, queryset, "ewm", source.name)


    plt_div = plot(count, output_type='div')
    plt_div1 = plot(cumsum, output_type='div')

    plt_div2 = plot(ewm, output_type='div')

    return plt_div, plt_div1, plt_div2



def produce_stock_freq(fig, queryset, line, name):
    stocks = queryset.values('date', 'close', 'interday_volatility')
    df = pd.DataFrame(([a['date'], a['close'], a['interday_volatility']] for a in stocks), columns=['date', 'price', 'return'])
    df = df.sort_values(by=['date'])
    df['ewm'] = df.loc[:,'return'].ewm(span=30,adjust=False).mean()
    if line=="price":
        fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines', name=name))
    elif line=="return":
        fig.add_trace(go.Scatter(x=df['date'], y=df['return'], mode='lines', name=name))
    else:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ewm'], mode='lines', name=name))

    return fig

def produce_stock_freq_plots(assets, stocks):
    return_f = go.Figure()
    price = go.Figure()

    ewm = go.Figure()
    for asset in assets:
        asset = Asset.objects.get(id=asset)
        queryset = stocks.filter(asset=asset.id)
        return_f = produce_stock_freq(return_f, queryset, "return", asset.name)
        price = produce_stock_freq(price, queryset, "price", asset.name)

        ewm = produce_stock_freq(ewm, queryset, "ewm", asset.name)


    plt_div = plot(return_f, output_type='div')
    plt_div1 = plot(price, output_type='div')

    plt_div2 = plot(price, output_type='div')

    return plt_div, plt_div1, plt_div2
