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
        "data": ['result is shown here ...']
    }
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject = True if request.POST.get(SUBJECT_TOGGLE_KEY, None) == 'on' else False
        search_term = request.POST.get(SEARCH_TERM_KEY, None)
        expansion = True if request.POST.get(QUERY_EXPANSION_TOGGLE_KEY, None) == 'on' else False
        if BOOLEAN_SEARCH_BUTTON_KEY in request.POST:
            results = boolean_search(subject, search_term, expansion)
            print(len(results))
            context["data"] = results
        elif TFIDF_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = tfidf_search(subject, search_term, expansion)
        elif TRANSFORMER_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = transformer_search(subject, search_term, expansion)
        elif FASTTEXT_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = fasttext_search(subject, search_term, expansion)
        elif ELASTICSEARCH_SEARCH_BUTTON_KEY in request.POST:
            context["data"] = elasticsearch_search(subject, search_term, expansion)
    return render(request, 'retriever/search-sara.html', context)


def classify(request):
    context = {
        "class": "",
        "cluster": ""
    }
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject = True if request.POST.get(SUBJECT_TOGGLE_KEY, None) == 'on' else False
        classify_term = request.POST.get(CLASSIFY_TERM_KEY, None)
        if CLASSIFY_CLUSTER_BUTTON_KEY in request.POST:
            context["class"], context["cluster"] = classify_cluster(subject, classify_term)
    return render(request, 'retriever/classify.html', context)


def link_analysis(request):
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
        subject = True if request.POST.get(SUBJECT_TOGGLE_KEY, None) == 'on' else False
        if LINK_ANALYSIS_BUTTON_KEY in request.POST:
            if subject:
                return redirect('retriever-link-analysis-health')
            else:
                return redirect('retriever-link-analysis-social')
    return render(request, 'retriever/link-analysis.html')

    
def link_analysis_social(request):
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
    return render(request, 'retriever/link-analysis-social.html')


def link_analysis_health(request):
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
    return render(request, 'retriever/link-analysis-health.html')

def classify_health(request):
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
    return render(request, 'retriever/classify-health.html')

    
def classify_social(request):
    if request.method == 'POST':
        nav = navbar(request)
        if nav:
            return nav
    return render(request, 'retriever/classify-social.html')