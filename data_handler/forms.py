from django import forms
from .models import Article, Source, Asset
from datetime import date

class FilterForm(forms.Form):
    source_set = Source.objects.all()
    year_range = tuple([i for i in range(2016, 2021)])
    date_start = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    date_end = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    source = forms.MultipleChoiceField(choices=[(source.id, source.name) for source in source_set], widget=forms.CheckboxSelectMultiple())

    def __init__(self, *args, **kwargs):
        super(FilterForm, self).__init__(*args, **kwargs)

class StockFilterForm(forms.Form):
    year_range = tuple([i for i in range(2016, 2021)])
    assets = Asset.objects.all()
    # # TODO: Insert validator to ensure that start < end
    date_start = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    date_end = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    # # TODO: Change this ModelMultipleChoiceField
    assets = forms.MultipleChoiceField(choices=[(asset.id, asset.name) for asset in assets], widget=forms.CheckboxSelectMultiple(), required=False)

    def __init__(self, *args, **kwargs):
        super(StockFilterForm, self).__init__(*args, **kwargs)

class FinancialDataForm(forms.Form):
    asset_name = forms.CharField(max_length=100)
    ticker = forms.CharField(max_length=10)
    title = forms.CharField(max_length=100)
    file = forms.FileField()

class ArticleDataForm(forms.Form):
    title = forms.CharField(max_length=100)
    file = forms.FileField()
