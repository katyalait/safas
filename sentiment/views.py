from django.shortcuts import render
from django.db.models import Count
from .aggregator import produce_table, get_article_sentiments, produce_plots
from . import models
from rest_framework import generics
from django.shortcuts import get_object_or_404
from django.views.generic import ListView, FormView
from django.views.generic.base import TemplateView
from data_handler.models import Asset, Article, Word, Source
from sentiment.models import Category, Word2VecModel, Label, Column, Value
from datetime import datetime, date
from .forms import DateModelForm, ModelLagForm, YearModelForm, YearModelLagForm, YearModelLagSigForm, DefinedModelForm, Word2VecSPModelForm, SentiWordNetForm, Word2VecForm, DictForm
from sentiment.model import SentimentPriceModel
from data_handler.data_handler import get_dates
from sentiment import model
import pandas as pd
import numpy as np
import os
from scipy.stats import t
from django.shortcuts import redirect
import gensim
from nltk.corpus import stopwords
from data_handler.helpers import progress
from rq import Queue
from worker import conn
import django_rq
from sentiment import model as md

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def html_convert(df, index=False):
    html = df.to_html(index=index)
    html = html.replace('dataframe','table table-sm table-bordered table-striped table-hover').replace('border=\"1\"', "style=\"text-align: center\"").replace("<th>", "<th style=\"text-align\": center>")
    return html
# Create your views here.
def save_file(file, name):
    path = os.path.join(BASE_DIR, 'uploads/{}.csv'.format(name))
    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return path

class Sentiment(TemplateView):
    template_name = 'sentiment/instructions.html'

class GraphSentiment(TemplateView):
    template_name = 'sentiment/overview.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = DateModelForm(data=self.request.GET or None)
        context['form'] = self.form
        spm = SentimentPriceModel()
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            date_start, date_end = get_dates(request)
            model_name = request.get('model_name')
            df = spm.load_db(label=model_name, set=True)

            df = df.loc[date_start:date_end]
            mp = spm.produce_plot_model(df=df)
            context['plotdiv'] = mp.get_html_plot()
        else:
            context['plotdiv'] = "<h5> Please choose date range and model. </h5>"
        return context

class CorrelationGraph(TemplateView):
    template_name = "sentiment/correlation_graph.html"
    model_name = 'word2vecmodel'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = ModelLagForm(data=self.request.GET or None)
        context['form'] = self.form
        model_name = self.model_name
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            model_name = request.get('model_name')
            indicator = request.get('variable')
            corr = model.CorrelationModel(model_name)

            indicator = "UK Negative" if not indicator in corr.columns else indicator
            plotdiv = corr.produce_plots(indicator)
            context['plotdiv'] = plotdiv
        else:
            context['plotdiv'] = " <h5> Please choose model and lag </h5>\n<p> \
                                    For information regarding the suggested \
                                    lag, please see the Lag Order Tables </p>"
        return context

class CorrelationTable(TemplateView):
    template_name = 'sentiment/correlation_table.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelForm(data=self.request.GET or None)
        context['form'] = self.form
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            corr = model.CorrelationModel(model_name)
            plotdiv = corr.produce_table(year)
            context['df_table'] = plotdiv
        else:
            context['df_table'] = " <h5> Please choose model and lag </h5>\n<p> \
                                    For information regarding the suggested \
                                    lag, please see the Lag Order Tables </p>"
        return context

class GrangerCausality(TemplateView):
    template_name = "sentiment/granger.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelLagForm(data=self.request.GET or None)
        context['form'] = self.form
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            variable = request.get('variable')

            order = int(request.get('lag_order'))
            spm = SentimentPriceModel()
            spm.load_db(model_name, set=True)
            df = spm.slice_year(year)
            var = spm.var(df=df)
            fit = var.fit_model(lag=order, freq="B")
            listdfs = []
            for col in df.columns:
                if col == variable:
                    continue
                c1 = fit.test_causality(col, variable)
                c2 = fit.test_causality(variable, col)
                conclusion  = ""
                if c1.pvalue <= 0.05 and c2.pvalue <= 0.05:
                    conclusion = "Bidirectional causation"
                elif c1.pvalue <= 0.05:
                    conclusion = "{} Granger-causes {}".format(col, variable)
                elif c2.pvalue <= 0.05:
                    conclusion = "{} Granger-causes {}".format(variable, col)
                else:
                    conclusion = "No causation"

                df = pd.DataFrame(np.array([["{} does not Granger-Cause {}".format(col, variable), c1.pvalue],
                                    ["{} does not Granger-Cause {}".format(variable, col), c2.pvalue],
                                    ["Conclusion", conclusion]]),
                                    columns=["Hypothesis", "P-value"])
                html = df.to_html(index=False)
                html = html.replace('dataframe','table table-sm table-bordered table-striped table-hover').replace('border=\"1\"', "style=\"text-align: center\"").replace("<th>", "<th style=\"text-align\": center>")
                listdfs.append(html)

            context['tables'] = listdfs
            context['set'] = True
        else:
            context['set'] = False
            context['tables'] = "<h5> Please select year, model and lag. </h5>\n\
                                    <p>By choosing a lag you can see what lag order best suits the variables</p>"
        return context

class StationaryView(TemplateView):
    template_name = 'sentiment/stationary.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelForm(data=self.request.GET or None)
        context['form'] = self.form
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            spm = SentimentPriceModel()
            spm.load_db(model_name, set=True)
            df = spm.slice_year(year)
            stationary_results = spm.test_stationary()

            cdf = pd.DataFrame(([col, trace, crit, trace > crit ] for col, trace, crit in spm.var().coint_results()),
                                columns = ["Variable", "Test Stat", "Critical Value", "Not-Cointegrated"])
            sdf = pd.DataFrame(([s.column, s.adf_stat, s.pval, s.H] for s in stationary_results),
                                columns=["Variable", "ADF", "P-value", "Hurst"])


            html1 = html_convert(sdf)
            html2 = html_convert(cdf)

            context['table'] = html1
            context['coint'] = html2

        else:
            context['table'] = "<h5> Please select year, model and lag. </h5>\n\
                                <p>By choosing a lag you can see what lag order best suits the variables</p>"
        return context

class LagOrderSelection(TemplateView):
    template_name = 'sentiment/lagorderselection.html'
    model_name = 'word2vecmodel'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelLagForm(data=self.request.GET or None)
        context['form'] = self.form
        model_name = self.model_name
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            order = int(request.get('lag_order'))
            spm = SentimentPriceModel()
            spm.load_db(model_name, set=True)
            df = spm.slice_year(year)
            context['table'] = spm.var(df).get_order_table(order)
        else:
            context['table'] = "<h5> Please select year, model and lag. </h5>\n\
                                <p>By choosing a lag you can see what lag order best suits the variables</p>"
        return context

class SignificanceTable(TemplateView):
    template_name = 'sentiment/significancetable.html'
    model_name = 'word2vecmodel'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelLagForm(data=self.request.GET or None)
        context['form'] = self.form
        model_name = self.model_name
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            lag_order = int(request.get('lag_order'))
            spm = SentimentPriceModel()
            spm.load_db(model_name, set=True)

            if not year == 0:
                start = "{}-01-01".format(year)
                end = "{}-01-01".format(year+1)
                startdate = pd.to_datetime(start).date()
                enddate = pd.to_datetime(end).date()
                spm.multivariate_df = spm.multivariate_df.loc[startdate:enddate].dropna()
            var = spm.var()
            html = var.pvalues(lag=lag_order, freq="B")
            context['table'] = html
        else:
            context['table'] = " <h5> Please choose model and lag </h5>\n<p> \
                                    For information regarding the suggested \
                                    lag, please see the Lag Order Tables </p>"
        return context

class SuccessfulUpload(TemplateView):
    template_name = 'data/success.html'

class ModelDefineView(FormView):
    template_name = "sentiment/definedmodel.html"
    form_class = DefinedModelForm
    success_url = 'success'

    def form_valid(self, form):
        request = self.request.POST
        wordset = set()
        cats = request.getlist('categories')
        assets = request.getlist('assets')
        source = int(request['source'])
        h_contents = request['headline_contents']
        weighted = int(request['weighted'])
        source_signals = int(request['source_signals'])
        included_countries = request.getlist('included_countries')
        include = request.getlist('include')
        h_contents = h_contents.split(',')
        if source==0 and (len(included_countries)==2 or len(included_countries)==0):
            irish_sources = list(Source.objects.filter(country=0).values_list('id', flat=True))
            uk_sources = list(Source.objects.filter(country=1).values_list('id', flat=True))
        elif source == 0 and len(included_countries)==1:
            if "Ireland" in included_countries:
                irish_sources = list(Source.objects.filter(country=0).values_list('id', flat=True))
                uk_sources = []
            elif "UK" in included_countries:
                irish_sources = []
                uk_sources = list(Source.objects.filter(country=1).values_list('id', flat=True))
        spm = model.SentimentPriceModel()
        q = django_rq.get_queue('default', default_timeout=36000)
        q.enqueue(spm.create_general_inquirer, request, cats, assets, source, h_contents, weighted, source_signals, included_countries, irish_sources, uk_sources, include)
        return super().form_valid(form)

        
class Word2VecSPModelView(FormView):
    template_name = "sentiment/w2vspmmodel.html"
    form_class = Word2VecSPModelForm
    success_url = 'success'

    def form_valid(self, form):
        cats = self.request.POST.getlist('l_categories')
        assets = self.request.POST.getlist('assets')
        include = self.request.POST.getlist('include')

        w_cats = self.request.POST.getlist('w_categories')
        topn = int(self.request.POST['topn'])
        source = int(self.request.POST['source'])
        weighted = int(self.request.POST['weighted'])
        source_signals = int(self.request.POST['source_signals'])

        q = django_rq.get_queue('default', default_timeout=36000)

        h_contents = self.request.POST['headline_contents']
        if h_contents:
            h_contents = h_contents.split(',')
            print("H contents found: {}".format(h_contents))
        if source==0:
            irish_sources = list(Source.objects.filter(country=0).values_list('id', flat=True))
            uk_sources = list(Source.objects.filter(country=1).values_list('id', flat=True))
        spm = model.SentimentPriceModel()
        word2vec = gensim.models.Word2Vec.load("word2vec.model")
        vectors = word2vec.wv
        include = self.request.POST.getlist('include')
        request = self.request.POST
        sw = stopwords.words('english')
        q.enqueue(spm.create_word2vec, request, cats, assets, include, w_cats, topn, source, weighted, source_signals, h_contents, irish_sources, uk_sources, vectors, sw)
        return super().form_valid(form)

class SentiWordNetView(FormView):
    template_name = "sentiment/sentiwordnetmodel.html"
    form_class = SentiWordNetForm
    success_url = 'success'

    def form_valid(self, form):
        assets = self.request.POST.getlist('assets')
        source = int(self.request.POST['source'])
        source_signals = int(self.request.POST['source_signals'])
        include_pos = int(self.request.POST['pos'])
        h_contents = self.request.POST['headline_contents']
        weighted = int(self.request.POST['weighted'])
        include = self.request.POST.getlist('include')
        if h_contents:
            h_contents = h_contents.split(',')
            print("H contents found: {}".format(h_contents))
        if include_pos == 0:
            include_pos = True
        else:
            include_pos = False

        if source==0:
            irish_sources = list(Source.objects.filter(country=0).values_list('id', flat=True))
            uk_sources = list(Source.objects.filter(country=1).values_list('id', flat=True))
        spm = model.SentimentPriceModel()
        request = self.request.POST
        q = django_rq.get_queue('default', default_timeout=36000)
        q.enqueue(spm.create_sentiwordnet, request, assets, source, source_signals, include_pos, h_contents, weighted, irish_sources, uk_sources, include)
        return super().form_valid(form)

class Word2VecMaker(FormView):
    template_name = 'sentiment/word2vecmaker.html'
    form_class = Word2VecForm
    success_url = 'success'

    def form_valid(self, form):
        path = save_file(self.request.FILES['file'], self.request.POST['title'])
        name = self.request.POST['name']
        headline = self.request.POST['headline']
        content = self.request.POST['content']
        q = django_rq.get_queue('default', default_timeout=36000)
        epochs = int(self.request.POST['epochs'])
        min_count = int(self.request.POST['min_count'])
        window = int(self.request.POST['window'])
        clean_lines = model.convert_to_lines_df(path, headline, content)
        clean_lines = model.preprocess(clean_lines)
        df = pd.DataFrame(([line for line in clean_lines]), columns=['clean_docs'])
        df = df.dropna()
        phrased = model.phrase_detector(df, min_count)
        model.create_model(phrased, epochs, min_count, window, name=name)
        wvm = Word2VecModel(name=name, pathname=os.path.join(BASE_DIR, name))
        wvm.save()

        return super().form_valid(form)

class DictMaker(FormView):
    template_name = 'sentiment/makedict.html'
    form_class = DictForm
    success_url = 'success'
    q = Queue(connection=conn)
    def form_valid(self, form):
        name = self.request.POST['name']
        categories = self.request.POST.getlist('categories')
        print(categories)
        gi_cats = Category.objects.filter(id__in=categories)
        print(gi_cats)

        new_cat = Category.objects.create(name=name)
        new_cat.save()
        for cat in gi_cats:
            words = Category.objects.get(name=cat).words.all()
            for word in words:
                word.categories.add(new_cat)
                word.save()

        return super().form_valid(form)

class VAROverview(TemplateView):
    template_name = "sentiment/varoverview.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = YearModelLagForm(data=self.request.GET or None)
        context['form'] = self.form
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            year = int(request.get('year'))
            model_name = request.get('model_name')
            variable = request.get('variable')

            order = int(request.get('lag_order'))
            spm = SentimentPriceModel()
            spm.load_db(model_name, set=True)
            df = spm.slice_year(year)
            var = spm.var(df=df)
            var_df = var.coefficient_df(order, variable)
            print(var_df)
            html = var_df.to_html()
            html = html.replace('dataframe','table table-sm table-bordered table-striped table-hover').replace('border=\"1\"', "")
            context['table'] = html
        else:
            context['table'] = "<h5> Please select year, model and lag. </h5>\n\
                                <p>By choosing a lag you can see what lag order best suits the variables</p>"
        return context

class ModelsView(ListView):
    template_name = "sentiment/modelslist.html"
    queryset = Label.objects.all()

def delete_view(request, pk):
    w2v = Label.objects.get(id=pk)
    cs = Column.objects.filter(label=pk)
    for c in cs:
        vs = Value.objects.filter(column=c.id)
        for v in vs:
            v.delete()
        c.delete()
    w2v.delete()
    response = redirect('models')
    return response
