from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    path('', views.Sentiment.as_view(), name="sentiment"),
    path('view_graph', views.GraphSentiment.as_view(), name="graph_sent"),
    path('correlation_graph', views.CorrelationGraph.as_view(), name='correl_graph'),
    path('correlation_table', views.CorrelationTable.as_view(), name='correl_table'),
    path('lag_order_selection', views.LagOrderSelection.as_view(), name='lag_order'),
    path('varoverview', views.VAROverview.as_view(), name='varoverview'),
    path('granger', views.GrangerCausality.as_view(), name='granger'),
    path('stationary', views.StationaryView.as_view(), name='stationary'),



    path('tvalues', views.SignificanceTable.as_view(), name='tvals'),
    path('gi_model', views.ModelDefineView.as_view(), name='modelcreate'),
    path('w2vmodel', views.Word2VecSPModelView.as_view(), name='w2vmodel'),
    path('sentiwordnet', views.SentiWordNetView.as_view(), name='sentiwordnetmodel'),
    path('word2vecmaker', views.Word2VecMaker.as_view(), name='word2vecmaker'),
    path('makedict', views.DictMaker.as_view(), name='makedict'),

    path('models', views.ModelsView.as_view(), name='models'),
    path('delete_model/<int:pk>', views.delete_view, name='delete_model'),
    path('success', views.SuccessfulUpload.as_view(), name='success')


]
urlpatterns = format_suffix_patterns(urlpatterns)
