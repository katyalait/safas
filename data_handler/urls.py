from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    path('view_articles', views.ArticleView.as_view(), name="articles_view"),
    path('view_wordcount', views.WordCountView.as_view(), name="wordcount_view"),
    path('view_stockprice', views.FinancialDataView.as_view(), name="stockprice_view"),
    path('view_sources', views.SourceViewSet.as_view(), name="source_view"),
    path('view_categories', views.CategoryView.as_view(), name="category_view"),
    path('article_detail/<int:pk>', views.ArticleDetailView.as_view(), name="article_detail"),
    path('financeupload', views.FinancialDataPreprocessView.as_view(), name='financeupload'),
    path('articleupload', views.ArticlePreprocessorView.as_view(), name='articleupload'),
    path('success', views.SuccessfulUpload.as_view(), name='success'),
    path('wordfreqs', views.ArticleWordCountView.as_view(), name='freqs'),
    path('delete_stock/<int:pk>', views.delete_view, name='delete_stock'),
    path('stockslist', views.StockList.as_view(), name='stockslist'),


    ]
urlpatterns = format_suffix_patterns(urlpatterns)
