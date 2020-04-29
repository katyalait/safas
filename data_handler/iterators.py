from .models import Article, StockPrice
from datetime import datetime
from .helpers import daterange
from django.db.models import Avg, Count, Min, Sum, Max
from nltk.tokenize import word_tokenize
from data_handler.helpers import progress
import pandas as pd
from nltk.corpus import sentiwordnet as swn

class StockRow(object):
    def __init__(self, stock):
        self.date = stock.date
        self.asset = stock.asset
        self.open = stock.open
        self.close = stock.adj_close
        self.volume = stock.volume
        self.interday_volatility = stock.interday_volatility

class DataFrameCreator(object):
    def __init__(self, stocks):
        self.stocks = stocks

    def __iter__(self):
        for stock in self.stocks:
            date = stock.date
            total_length = Article.objects.filter(date_written=date).aggregate(total_length=Sum('length'))['total_length']
            smarter_sum = Article.objects.filter(date_written=date).aggregate(total_neg_words=Sum('smarter_negative_words'))['total_neg_words']
            negative_sentiment = smarter_sum/total_length
            yield {'date':date, 'sentiment':negative_sentiment, 'return': stock.logging()}


class AggregateRow(object):
    def __init__(self, date, neg_words, pos_sum, neg_sum, length, stock):
        self.date = date
        self.pos_sum = pos_sum
        self.neg_sum = neg_sum
        self.length = length
        self.neg_words = neg_words
        self.interday_volatility = stock.interday_volatility
        self.asset = stock.asset
        self.pos_sent = self.pos_sum/self.length if self.pos_sum else 0.0
        self.neg_sent = self.neg_sum/self.length if self.neg_sum else 0.0
        self.naive_sent = self.neg_words/self.length if self.neg_words else 0.0


class RelatedIterator(object):
    def __init__(self, articles):
        self.articles = [a[0] for a in articles]
    def __iter__(self):
        for row in self.get_rows():
            yield row

    def get_rows(self):
        for article in self.articles:
            date = article.date_written
            stocks = StockPrice.objects.filter(date=date)
            neg_words = article.negative_words
            pos_sum = article.smarter_positive_words
            neg_sum = article.smarter_negative_words
            total = article.length
            for stock in stocks:
                yield AggregateRow(date=date,
                            neg_words=neg_words,
                            pos_sum=pos_sum,
                            neg_sum=neg_sum,
                            length=total,
                            stock=stock
                            )

class AggregateRowIterator(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for row in self.get_rows():
            yield row

    def get_rows(self):
        for date in daterange(self.start, self.end):

            neg_sum = self.get_article_day_sum(date, 'negative_words')
            total_length = self.get_article_day_sum(date, 'length')
            smarter_pos_sum = self.get_article_day_sum(date, 'smarter_positive_words')
            smarter_neg_sum = self.get_article_day_sum(date, 'smarter_negative_words')

            stocks = StockPrice.objects.filter(date=date)

            for stock in stocks:
                yield AggregateRow(date = date,
                            neg_words=neg_sum,
                            pos_sum=smarter_pos_sum,
                            neg_sum=smarter_neg_sum,
                            length=total_length,
                            stock=stock
                            )
    def get_article_day_sum(self, date, column):
        articles = Article.objects.filter(date_written=date)
        sum = articles.aggregate(total=Sum(column))['total']
        return sum
# # TODO: turn this into a function where people just put in the queryset
class StockRowIterator(object):
    def __init__(self, stocks):
        self.stocks = stocks

    def __iter__(self):
        for stock in self.stocks:
            row = StockRow(stock)
            if row.interday_volatility == 0:
                continue
            yield row

class SentimentExtractor(object):
    def __init__(self, dates, category_words, filter, h_contents):
        print(h_contents)
        if len(h_contents) > 1:
            articles = []
            for h in h_contents:
                if h == "":
                    continue
                articles.extend(list(Article.objects.filter(date_written__range = [dates[0], dates[-1]]).filter(source__id__in=[filter]).filter(headline__icontains=h).values_list('date_written', 'tokens')))
        else:
            articles = list(Article.objects.filter(date_written__range = [dates[0], dates[-1]]).filter(source__id__in=[filter]).values_list('date_written', 'tokens'))
        print(len(articles))

        self.df = pd.DataFrame(articles, columns=['date_written', 'tokens'])
        self.dates = dates
        self.words = category_words
        self.words.sort()
    def __iter__(self):
        index = 0
        length = len(self.dates)
        for date in self.dates:
            index +=1
            day_count = 0
            day_articles = self.df.loc[self.df['date_written']==date]['tokens']
            if day_articles.empty:
                continue
            day_articles = " ".join(day_articles)
            tokens = word_tokenize(day_articles)
            tokens.sort()
            token_dict = {}
            for token in tokens:
                if token in token_dict:
                    token_dict[token] += 1
                else:
                    token_dict[token] = 1
            for word in token_dict:
                if word in self.words:
                    day_count += token_dict[word]
            progress(index, length, status="{}: {}".format(date, day_count/len(tokens)))

            yield {'date': date, 'count': day_count, 'length': len(tokens), 'sentiment': day_count/length}

class SentiWordNetExtractor(object):
    def __init__(self, dates, filter, h_contents):
        print(h_contents)
        if len(h_contents) > 1:

            articles = []
            for h in h_contents:
                if h == "":
                    continue
                print("Getting articles with {} in their headline".format(h))
                new_arts  = list(Article.objects.filter(date_written__range = [dates[0], dates[-1]]).filter(source__id__in=[filter]).filter(headline__icontains=h).values_list('date_written', 'tokens'))
                articles = list(set(articles) | set(new_arts))
        else:
            print("Getting articles without headline")
            articles = list(Article.objects.filter(date_written__range = [dates[0], dates[-1]]).filter(source__id__in=[filter]).values_list('date_written', 'tokens'))
        print(len(articles))
        print(filter)
        self.df = pd.DataFrame(articles, columns=['date_written', 'tokens']).sort_values('date_written')
        self.dates = dates
    def __iter__(self):
        index = 0
        length = len(self.dates)
        for date in self.dates:
            index +=1
            progress(index, length, status="{}".format(date))
            neg_day_count = 0
            pos_day_count = 0
            day_articles = self.df.loc[self.df['date_written']==date]['tokens']
            if day_articles.empty:
                continue
            day_articles = " ".join(day_articles)
            tokens = word_tokenize(day_articles)
            tokens.sort()
            token_dict = {}
            for word in tokens:
                if word in token_dict:
                    token_dict[word] += 1
                else:
                    token_dict[word] = 1
            for word in token_dict:
                try:
                    count = token_dict[word]
                    synset = list(swn.senti_synsets(word))[0]
                    n_score = synset.neg_score()
                    p_score = synset.pos_score()
                    neg_day_count += n_score*count
                    pos_day_count += p_score*count

                except:
                    continue

            yield {'date': date, 'length': len(tokens), 'n_sentiment': neg_day_count/length, 'p_sentiment': pos_day_count/length}



class SentimentIterator(object):
    def __init__(self, start=None, end=None):
        self.articles = Article.objects.all().order_by('date_written')
        if start:
            self.articles = self.articles.filter(date_written__gte=start)
        if end:
            self.articles = self.articles.filter(date_written__lte=end)

        self.dates = self.articles.values('date_written').distinct()

        self.neg_sums = self.get_column_sum('smarter_negative_words')
        self.pos_sums = self.get_column_sum('smarter_positive_words')
        self.total_lengths = self.get_column_sum('length')
        # # TODO: calculate the STD and the MEAN of the sentiments

    def get_column_sum(self, column):
        sum = self.articles.values('date_written').annotate(total = Sum(column))
        return sum

    def __iter__(self):
        for entry in self.dates:
            day = entry['date_written']
            neg = self.neg_sums.get(date_written=day)['total']
            pos = self.pos_sums.get(date_written=day)['total']
            total = self.total_lengths.get(date_written=day)['total']
            neg_sent = 0
            pos_sent = 0
            if neg and total:
                neg_sent = neg/total
            if pos and total:
                pos_sent = pos/total
            yield {'date': day, 'neg_sent': neg_sent, 'pos_sent': pos_sent}
