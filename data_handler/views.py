from django.shortcuts import render
from django.views.generic import ListView, DetailView, FormView
from django.db.models import Count
from django.shortcuts import get_object_or_404
from django.db.models import Prefetch
from .preprocessing import produce_plots
from django.shortcuts import redirect

from django.views.generic.base import TemplateView
from .data_handler import get_related_articles, get_relevant_stocks, produce_article_freq_plots, produce_source_article_freq_plots, produce_stock_freq_plots
from rest_framework import generics
from data_handler.serializers import SourceSerializer, ArticleSerializer, WordCountSerializer, StockPriceSerializer, TimeSeriesSerializer, UniqueArticleSerializer
from data_handler.models import Source, Article, WordsInArticleCount, StockPrice, Asset
from sentiment.models import Category
from rest_pandas import PandasView
import pandas as pd
import os#
from .data_handler import frequency_count
from sentiment.model import DocModelGenerator
from datetime import datetime, timedelta, date
import datetime as dt
from .forms import FilterForm, StockFilterForm, FinancialDataForm, ArticleDataForm
from .data_handler import get_dates, produce_stock_plots
from .preprocessing import FinancialDataPreprocessor, ArticlePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_file(file, name):
    path = os.path.join(BASE_DIR, 'uploads/{}.csv'.format(name))
    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return path

class SourceViewSet(TemplateView):
    template_name='data/sources.html'
    queryset = Source.objects.all()
    paginate_by=10

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(columns=["Name", "Country", "Article Count (A)", "System % (B)", "Total (C)", "Total % (D)"])
        total_arts = len(Article.objects.all())
        df_1 = pd.DataFrame({'Country': ['Ireland', 'United Kingdom'],
                            'Article Count (A)':  [len(list(Article.objects.filter(source__country=0).values_list('date_written',flat=True))),
                                                len(list(Article.objects.filter(source__country=1).values_list('date_written',flat=True)))],
                            'Total (B)': [383244, 1503479],
                            'Total % (C)': ["{:.2f}%".format(len(list(Article.objects.filter(source__country=0).values_list('date_written',flat=True)))/383244*100),
                                        "{:.2f}%".format(len(list(Article.objects.filter(source__country=1).values_list('date_written',flat=True)))/1503479*100)]
                        },columns=['Country', 'Article Count (A)', 'Total (B)', 'Total % (C)'], index=[0,1])



        index = 0
        totals = {'Financial Times (London, England)': 385833, 'The Irish Times': 171288, 'Irish Independent': 211956,'The Times (London)': 799655,  'The Guardian(London)': 317991 }
        for source in self.queryset:
            count = len(list(Article.objects.filter(source=source.id).values_list('date_written',flat=True)))
            print(count)
            if source.country == 0:
                country = "Ireland"
            else:
                country = "United Kingdom"
            df_t = pd.DataFrame({"Name": source.name,
                                "Country": country,
                                "Article Count (A)": count,
                                "System % (B)": "{:.2f}%".format(count/total_arts*100),
                                "Total (C)" : int(totals[source.name]),
                                "Total % (D)": "{:.2f}%".format(count/totals[source.name]*100),
                                },
                                columns=["Name", "Country", "Article Count (A)", "System % (B)", "Total (C)", "Total % (D)"],
                                index=[index])
            df = df.append(df_t)
            index += 1
        html = df.to_html()
        html = html.replace('border=\"1\"', "")
        html = html.replace('dataframe','table table-bordered table-striped table-hover')
        count, cumsum, ewm = produce_source_article_freq_plots(self.queryset)
        html_1 = df_1.to_html()
        html_1 = html_1.replace('border=\"1\"', "")
        html_1 = html_1.replace('dataframe','table table-bordered table-striped table-hover')

        context['results'] = html
        context['results_1'] = html_1

        context['count'] = count
        context['cumsum'] = cumsum
        context['ewm'] = ewm

        return context

# Create your views here.
class ArticleView(ListView):
    template_name='data/articles.html'
    paginate_by = 1000
    model = Article

    def get_queryset(self):
        self.form = FilterForm(data=self.request.GET or None)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET

            self.date_start, self.date_end = get_dates(request)

            self.source_ids = request.getlist('source')
            self.queryset = Article.objects.filter(date_written__range=[self.date_start, self.date_end],
                                                source__id__in=self.source_ids)
        else:
            self.queryset = super().get_queryset()
        return self.queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['articles'] = self.queryset.count()
        context['plotdiv'] = produce_article_freq_plots(self.queryset)
        context['form'] = self.form
        return context

class ArticleDetailView(DetailView):
    template_name='data/articles_detail.html'
    model = Article

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        article = kwargs['object']
        dmg = DocModelGenerator()
        dmg.load_model("doc2vec.model")
        related = dmg.get_related_articles(article)
        related_stocks = get_relevant_stocks(related)
        plot_div = produce_plots(related)

        context['related_articles'] = related
        context['relevant_stocks'] = related_stocks
        context['plotdiv'] = plot_div
        return context

class WordCountView(ListView):
    paginate_by = 10
    serializer_class = WordCountSerializer
    template_name='data/wordcounts.html'
    def get_queryset(self):
        queryset = WordsInArticleCount.objects.filter(word__categories__name__in=["Negativ"]).values('word__word').annotate(total_articles=Count('word')).order_by('-total_articles')
        return queryset

class CategoryView(ListView):
    paginate_by = 10
    template_name = 'data/categories.html'
    queryset = Category.objects.all()


class FinancialDataView(ListView):
    template_name='data/stockprices.html'
    queryset = StockPrice.objects.filter(date__range=['2016-06-01', '2017-08-01']).order_by('date')
    model = StockPrice
    def get_queryset(self):
        # asset = Asset.objects.filter(name="FTSE 100").first()
        self.form = StockFilterForm(data=self.request.GET or None)
        print(self.request.GET)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET

            self.date_start, self.date_end = get_dates(request)

            self.asset_ids = request.getlist('assets')
            if not self.asset_ids:
                self.asset_ids = [asset.id for asset in (Asset.objects.all())]
            self.queryset = StockPrice.objects.filter(date__range=[self.date_start, self.date_end],
                                                asset__id__in=self.asset_ids)
        else:
            self.queryset = super().get_queryset()
            self.asset_ids = [asset.id for asset in (Asset.objects.all())]

        return self.queryset

    def get_context_data(self, **kwargs):
            context = super().get_context_data(**kwargs)
            retewm, volewm = produce_stock_plots(self.queryset)
            plt1, plt2, plt3 = produce_stock_freq_plots(self.asset_ids, self.queryset)
            context['price'] = plt2
            context['return'] = plt1
            context['ewm'] = plt3
            context['retewm'] = retewm
            context['volewm'] = volewm
            context['stocks'] = self.queryset.count()
            context['form'] = self.form
            return context

class SuccessfulUpload(TemplateView):
    template_name = 'data/success.html'

class FinancialDataPreprocessView(FormView):
    template_name = 'data/financeupload.html'
    form_class = FinancialDataForm
    success_url = 'success'

    def form_valid(self, form):
        path = save_file(self.request.FILES['file'], self.request.POST['title'])
        print("Success! File saved to {}".format(path))
        fdp = FinancialDataPreprocessor(path)
        asset = fdp.set_asset(self.request.POST['asset_name'], self.request.POST['ticker'])
        print(asset)
        fdp.create_objects()
        return super().form_valid(form)

class ArticlePreprocessorView(FormView):
    template_name = 'data/articleupload.html'
    form_class = ArticleDataForm
    success_url = 'success'

    def form_valid(self, form):
        path = save_file(self.request.FILES['file'], self.request.POST['title'])
        print("Success! File saved to {}".format(path))
        ap = ArticlePreprocessor(path)
        fdp.create_objects()
        return super().form_valid(form)

class ArticleWordCountView(TemplateView):
    template_name='data/wordfreqs.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = FilterForm(data=self.request.GET or None)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET

            self.date_start, self.date_end = get_dates(request)

            self.source_ids = request.getlist('source')
            articles = Article.objects.filter(date_written__range=[self.date_start, self.date_end],
                                                source__id__in=self.source_ids)
            html = frequency_count(articles).to_html()
            html = html.replace('border=\"1\"', "")
            html = html.replace('dataframe','table table-bordered table-striped table-hover')
            context['results'] = html

        else:
            context['results'] = ""
        context['form'] = self.form
        return context



class StockList(ListView):
    template_name = "data/stockslist.html"
    queryset = Asset.objects.all()
    def get_queryset(self):
        print(Asset.objects.all())
        return Asset.objects.all()

def delete_view(request, pk):
    w2v = Asset.objects.get(id=pk)
    scks = StockPrice.objects.filter(asset=w2v.id)
    for s in scks:
        s.delete()
    w2v.delete()
    response = redirect('stockslist')
    return response
