from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=30)
    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']

class Label(models.Model):
    model_name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField()

    def __str__(self):
        return self.model_name

    class Meta:
        ordering = ['-created_at']


class Column(models.Model):
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)

    class Meta:
        ordering = ['label']

class Value(models.Model):
    column = models.ForeignKey(Column, on_delete=models.CASCADE)
    date = models.DateField()
    value = models.FloatField()

    class Meta:
        ordering = ['column', 'date']

class Word2VecModel(models.Model):
    name = models.CharField(max_length=40)
    pathname = models.CharField(max_length=250)

    def __str__(self):
        return self.name


class DaySentiment(models.Model):
    date = models.DateField(auto_now_add=True, editable=True)
    total_neg_words = models.IntegerField()
    total_article_words = models.IntegerField()

    def __str__(self):
        return "{}/{} sentiment rating on {}".format(self.total_neg_words, self.total_article_words, self.article.date_written)

    class Meta:
        ordering=['date']
