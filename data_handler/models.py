from django.db import models
from sentiment.models import Category
from datetime import datetime, date
from django.utils import timezone
from django.utils.translation import ugettext_lazy as _
from django.utils.timezone import now
import math

class Source(models.Model):
    name = models.CharField(max_length=255)
    country = models.IntegerField(
        choices = [('Ireland', 0), ('United Kingdom', 1), ('Unknown', None)],
        default = None
        )
    type = models.CharField(max_length=50)

    def __str__(self):
        return self.name

class Word(models.Model):
    word = models.CharField(max_length=50, unique=True) # longest word in enlish dictionary is 45 words
    date_added = models.DateTimeField(auto_now_add=True)
    categories = models.ManyToManyField(Category, related_name="words")

    def __str__(self):
        return self.word

    class Meta:
        ordering = ['word']

class Article(models.Model):
    headline = models.CharField(max_length=1024)
    date_added = models.DateTimeField(auto_now_add=True)
    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    length = models.IntegerField()
    date_written = models.DateField(auto_now_add=True,  editable=True)
    negative_words = models.IntegerField(null=True)
    smarter_negative_words = models.IntegerField(null=True)
    smarter_positive_words = models.IntegerField(null=True)
    contents = models.TextField(null=True)
    tokens = models.TextField(null=True)

    def __str__(self):
        return self.headline

    class Meta:
        ordering = ['date_written']

class WordsInArticleCount(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    word = models.ForeignKey(Word, on_delete=models.CASCADE)
    frequency = models.IntegerField()

    def __str___(self):
        return "Article: " + self.article + "\n" + "Word: " + self.word + "\n" + "Count: " + self.frequency

    def is_neg(self):
        if "Neg" in self.word.categories.all().values('name') or "Ngtv" in self.word.categories.all().values('name'):
            return True

class Asset(models.Model):
    name = models.CharField(max_length=30)
    ticker = models.CharField(max_length=10)

    def __str__(self):
        return self.name

class StockPrice(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.SET_NULL, null=True)
    date = models.DateField(default=date.today, editable=True)
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    adj_close = models.FloatField()
    volume = models.BigIntegerField()
    interday_volatility = models.FloatField()

    class Meta:
        ordering = ['date']

    def __str__(self):
        return self.asset.name + " on " + str(date) + ": " + str(self.close)

    def log_return(self, p1=None, p2=None):
        if not p1 or not p2:
            p1=self.close
            p2=self.open

        return math.log((p1/p2), 10)*100
