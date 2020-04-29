from django import forms
from data_handler.models import Asset
from sentiment.models import Label, Category, Word2VecModel
from datetime import date

class DateModelForm(forms.Form):
    sentichoices = (('full_monty2', 'General Inquirer'),
                    ('sentiwordnet1', 'SentiWordNet'),
                    ('word2vecmodel', 'Word2Vec Expanded')
                    )
    year_range = tuple([i for i in range(2016, 2021)])
    assets = Asset.objects.all()
    date_start = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    date_end = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    model_name = forms.ModelChoiceField(queryset=Label.objects.all(), to_field_name='model_name')

    def __init__(self, *args, **kwargs):
        super(DateModelForm, self).__init__(*args, **kwargs)

class ModelLagForm(forms.Form):
    lags = [(i, i) for i in range(10)]
    lag_order = forms.IntegerField(label="Lag Order", widget = forms.Select(choices = lags), required=False)
    model_name = forms.ModelChoiceField(queryset=Label.objects.all(), to_field_name='model_name')
    variable = forms.CharField(label="Correlation Variable", help_text="Note: this field is case sensitive", max_length=100, required=False)

    def __init__(self, *args, **kwargs):
        super(ModelLagForm, self).__init__(*args, **kwargs)

class YearModelForm(forms.Form):
    year_range = ((2016, 2016),
                    (2017, 2017),
                    (2018, 2018),
                    (2019, 2019),
                    (0, 'Total'))
    year = forms.IntegerField(label="Year", widget = forms.Select(choices = year_range))
    model_name = forms.ModelChoiceField(queryset=Label.objects.all(), to_field_name='model_name', )


    def __init__(self, *args, **kwargs):
        super(YearModelForm, self).__init__(*args, **kwargs)

class YearModelLagForm(forms.Form):
    year_range = ((2016, 2016),
                    (2017, 2017),
                    (2018, 2018),
                    (2019, 2019),
                    (0, 'Total'))
    lags = [(i, i) for i in range(25)]
    lag_order = forms.IntegerField(label="Lag Order", widget = forms.Select(choices = lags))
    year = forms.IntegerField(label="Year", widget = forms.Select(choices = year_range))
    model_name = forms.ModelChoiceField(label="Model Name", queryset=Label.objects.all(), to_field_name='model_name')
    variable = forms.CharField(label="Key Variable", help_text="Note: this field is case sensitive", max_length=100, required=False)


    def __init__(self, *args, **kwargs):
        super(YearModelLagForm, self).__init__(*args, **kwargs)

class YearModelLagSigForm(forms.Form):
    year_range = ((2016, 2016),
                    (2017, 2017),
                    (2018, 2018),
                    (2019, 2019),
                    (0, 'Aggregate'))
    lags = [(i, i) for i in range(10)]
    sigs = [(i/100, i/100) for i in range(0, 25, 5)]
    lag_order = forms.IntegerField(label="Lag Order", widget = forms.Select(choices = lags))
    significance = forms.FloatField(label="Significance Label", widget = forms.Select(choices = sigs))
    year = forms.IntegerField(label="Year", widget = forms.Select(choices = year_range))
    model_name = forms.ModelChoiceField(label="Model Name", queryset=Label.objects.all(), to_field_name='model_name')


    def __init__(self, *args, **kwargs):
        super(YearModelLagSigForm, self).__init__(*args, **kwargs)

class DefinedModelForm(forms.Form):
    countries = ((0, 'Separate'),
                (1, 'Aggregate'))
    weighting = ((0, 'No'),
                (1, 'Yes'))
    signals = ((0, 'No'),
                (1, 'Yes'))

    name = forms.CharField(max_length=40)
    description = forms.CharField(max_length=200)
    headline_contents = forms.CharField(label="Headline Contains",
                                        max_length=300, required=False,
                                        help_text='Comma separated list e.g. brexit, deadline etc.')


    categories = forms.ModelMultipleChoiceField(
                        label="Categories",
                       queryset = Category.objects.all(),
                       to_field_name='name',
                       required=False
               )
    assets = forms.ModelMultipleChoiceField(
                        label="Assets",
                       queryset = Asset.objects.all(),
                       to_field_name='ticker',
                       required=False,
                       widget=forms.CheckboxSelectMultiple()
               )
    include = (("volume", "Volume"),
                ("assets", "Assets"))
    include = forms.MultipleChoiceField(
                        label="Asset Properties",
                       required=True,
                       widget=forms.CheckboxSelectMultiple(),
                       choices = include
               )
    included_countries = (("Ireland", "Ireland"), ("UK", "UK"))
    included_countries = forms.MultipleChoiceField(
                            label="Include countries",
                           required=False,
                           widget=forms.CheckboxSelectMultiple(),
                           choices = included_countries
                   )
    source = forms.ChoiceField(label="Separate sources by country?", widget = forms.RadioSelect, choices=countries)
    source_signals = forms.ChoiceField(label="Separate sources by publication?", widget = forms.RadioSelect, choices=signals)

    weighted = forms.ChoiceField(label="Weighted?", widget = forms.RadioSelect, choices=weighting)

    def __init__(self, *args, **kwargs):
        super(DefinedModelForm, self).__init__(*args, **kwargs)
    class Meta:
         help_texts = {
                'headline_contents': ('Comma separated list e.g. brexit, deadline etc.'),
        }

class Word2VecSPModelForm(forms.Form):
    countries = ((0, 'Separate'),
                (1, 'Aggregate'))
    weighting = ((0, 'No'),
                (1, 'Yes'))
    signals = ((0, 'No'),
                (1, 'Yes'))
    name = forms.CharField(max_length=40)
    description = forms.CharField(max_length=200)
    headline_contents = forms.CharField(label="Headline Contains",
                                        max_length=300, required=False,
                                        help_text='Comma separated list e.g. brexit, deadline etc.')

    L_categories = forms.ModelMultipleChoiceField(
                        label="Lexicon Categories",
                       queryset = Category.objects.all(),
                       to_field_name='name',
                       required=False
               )
    w_categories = forms.ModelMultipleChoiceField(
                        label="Word2Vec Categories",
                       queryset = Category.objects.all(),
                       to_field_name='name',
                       required=True
               )
    assets = forms.ModelMultipleChoiceField(
                        label="Assets",
                       queryset = Asset.objects.all(),
                       to_field_name='ticker',
                       required=True,
                       widget=forms.CheckboxSelectMultiple()
               )
    include = (("volume", "Volume"),
                ("assets", "Assets"))
    include = forms.MultipleChoiceField(
                        label="Asset Properties",
                       required=True,
                       widget=forms.CheckboxSelectMultiple(),
                       choices = include
               )
    included_countries = (("Ireland", "Ireland"), ("UK", "UK"))
    included_countries = forms.MultipleChoiceField(
                                label="Include countries",
                               required=False,
                               widget=forms.CheckboxSelectMultiple(),
                               choices = included_countries
                       )
    topn = forms.IntegerField(label="Top N")
    weighted = forms.ChoiceField(label="Weighted?", widget = forms.RadioSelect, choices=weighting)

    w2vm = forms.ModelChoiceField(label="Word2Vec Model", queryset=Word2VecModel.objects.all(), to_field_name='pathname', required=True)
    source = forms.ChoiceField(label="Separate sources by country?", widget = forms.RadioSelect, choices=countries)
    source_signals = forms.ChoiceField(label="Separate sources by publication?", widget = forms.RadioSelect, choices=signals)

    def __init__(self, *args, **kwargs):
        super(Word2VecSPModelForm, self).__init__(*args, **kwargs)

class SentiWordNetForm(forms.Form):
    countries = ((0, 'Separate'),
                (1, 'Aggregate'))
    weighting = ((0, 'No'),
                (1, 'Yes'))
    signals = ((0, 'No'),
                (1, 'Yes'))
    name = forms.CharField(max_length=40)
    description = forms.CharField(max_length=200)
    headline_contents = forms.CharField(label="Headline Contains",
                                        max_length=300, required=False,
                                        help_text='Comma separated list e.g. brexit, deadline etc.')

    sent = ((0, 'Yes'),
                (1, 'No'))
    assets = forms.ModelMultipleChoiceField(
                        label="Assets",
                       queryset = Asset.objects.all(),
                       to_field_name='ticker',
                       required=False,
                       widget=forms.CheckboxSelectMultiple()
               )
    source = forms.ChoiceField(label="Separate sources by country?", widget = forms.RadioSelect, choices=countries)
    source_signals = forms.ChoiceField(label="Separate sources by publication?", widget = forms.RadioSelect, choices=signals)
    include = (("volume", "Volume"),
                ("assets", "Assets"))
    include = forms.MultipleChoiceField(
                        label="Asset Properties",
                       required=True,
                       widget=forms.CheckboxSelectMultiple(),
                       choices = include
               )
    included_countries = (("Ireland", "Ireland"), ("UK", "UK"))
    included_countries = forms.MultipleChoiceField(
                            label="Include countries",
                           required=False,
                           widget=forms.CheckboxSelectMultiple(),
                           choices = included_countries
                   )
    weighted = forms.ChoiceField(label="Weighted?", widget = forms.RadioSelect, choices=weighting)

    pos = forms.ChoiceField(label="Include positive sentiment?", widget = forms.RadioSelect, choices=sent)

    def __init__(self, *args, **kwargs):
        super(SentiWordNetForm, self).__init__(*args, **kwargs)

class Word2VecForm(forms.Form):
    name = forms.CharField(label="Name of Model", max_length=40)
    file = forms.FileField(label="Corpus")
    title = forms.CharField(label="File name", max_length=40)
    headline = forms.CharField(label="Headline Column", max_length=40)
    content = forms.CharField(label="Content Column", max_length=40)
    epochs = forms.IntegerField(label="Epochs")
    min_count = forms.IntegerField(label="Minimum Count")
    window = forms.IntegerField(label="Window Size")


    def __init__(self, *args, **kwargs):
        super(Word2VecForm, self).__init__(*args, **kwargs)

class DictForm(forms.Form):
    name = forms.CharField(label="New Category name", max_length=40)
    categories = forms.ModelMultipleChoiceField(
                        label="GI Categories",
                       queryset = Category.objects.all(),
                       required=False
               )

    def __init__(self, *args, **kwargs):
        super(DictForm, self).__init__(*args, **kwargs)
