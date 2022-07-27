from django.urls import path
from . import views

urlpatterns = [
    path('', views.search, name='retriever-search'),
    path('classify/', views.classify, name='retriever-classify'),
    path('link-analysis/', views.link_analysis, name='retriever-link-analysis'),
    path('link-analysis/social/', views.link_analysis_social, name='retriever-link-analysis-social'),
    path('link-analysis/health/', views.link_analysis_health, name='retriever-link-analysis-health'),
    path('classify/health/', views.classify_health, name='retriever-classify-health'),
    path('classify/social/', views.classify_social, name='retriever-classify-social'),
]