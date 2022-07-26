from django.shortcuts import render, redirect

from .bridge.assignment3 import *
from .bridge.assignment4 import *
from .bridge.assignment5 import *


SEARCH_TAB_KEY = 'search-tab'
CLASSIFY_TAB_KEY = 'classify-tab'
LINK_ANALYSIS_TAB_KEY = 'link-analysis-tab'

SUBJECT_TOGGLE_KEY = 'subject-toggle'

QUERY_EXPANSION_TOGGLE_KEY = 'query-expansion-toggle'
SEARCH_TERM_KEY = 'search-term'
BOOLEAN_SEARCH_BUTTON_KEY = 'boolean-search-button'
TFIDF_SEARCH_BUTTON_KEY = 'tfidf-search-button'
TRANSFORMER_SEARCH_BUTTON_KEY = 'transformer-search-button'
FASTTEXT_SEARCH_BUTTON_KEY = 'fasttext-search-button'
ELASTICSEARCH_SEARCH_BUTTON_KEY = 'elasticsearch-search-button'

CLASSIFY_TERM_KEY = 'classify-term'
CLASSIFY_CLUSTER_BUTTON_KEY = 'classify-cluster-button'

LINK_ANALYSIS_BUTTON_KEY = 'link-analysis-button'

def navbar(request):
    if SEARCH_TAB_KEY in request.POST:
        return redirect('retriever-search')
    elif CLASSIFY_TAB_KEY in request.POST:
        return redirect('retriever-classify')
    elif LINK_ANALYSIS_TAB_KEY in request.POST:
        return redirect('retriever-link-analysis')
    return None


def search(request):
    context = {
        "data": []
    }
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject =  request.POST.get(SUBJECT_TOGGLE_KEY, None)
        search_term = request.POST.get(SEARCH_TERM_KEY, None)
        expansion = request.POST.get(QUERY_EXPANSION_TOGGLE_KEY, None)
        if BOOLEAN_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = boolean_search(subject, search_term, expansion)
        elif TFIDF_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = tfidf_search(subject, search_term, expansion)
        elif TRANSFORMER_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = transformer_search(subject, search_term, expansion)
        elif FASTTEXT_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = fasttext_search(subject, search_term, expansion)
        elif ELASTICSEARCH_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = elasticsearch_search(subject, search_term, expansion)
    return render(request, 'retriever/search.html')


def classify(request):
    context = {
        "class": "",
        "cluster": ""
    }
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject =  request.POST.get(SUBJECT_TOGGLE_KEY, None)
        classify_term = request.POST.get(CLASSIFY_TERM_KEY, None)
        if CLASSIFY_CLUSTER_BUTTON_KEY in request.POST:
            context["class"], context["cluster"] = classify_cluster(subject, classify_term)
    return render(request, 'retriever/classify.html')


def link_analysis(request):
    context = {
        "data": []
    }
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject =  request.POST.get(SUBJECT_TOGGLE_KEY, None)
        if LINK_ANALYSIS_BUTTON_KEY in request.POST:
            context["data"] = link_analysis(subject)
    return render(request, 'retriever/link-analysis.html')