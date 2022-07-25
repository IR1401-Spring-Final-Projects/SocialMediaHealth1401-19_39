from django.urls import path
from . import views

urlpatterns = [
    path('', views.search, name='retriever-search'),
    path('classify/', views.classify, name='retriever-classify'),
    path('link-analysis/', views.link_analysis, name='retriever-link-analysis'),
]