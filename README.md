# SocialMediaHealth1401-19_39
## Group 19 and 39 Final Project 
### Parsa Mohammadian - 98102284
### Sara Azarnoosh - 98170668
### Kahbod Aeini - 98101209
### Mohammadreza Daviran - 98101566
### Reza Erfan - 98106434 



# Overview
This is the final project of Modern Information Retrieval which is implemented by python, django and html. We designed an information retrieval system for social networks and health articles. In this system information can be retrieved with four different retrieval methods, boolean, tranfsormer-based, fasttext and TfIdf retrieval. There is also an elasticsearch module that is the engine we use to search through our datasets. These information retrievals and searchs are based on the user's query which can be expanded by query expansion module. Classification and Clustering is also done on the datasets.


Since this project is implemented on two different subjects we have two files for each module in the backend, one for the social network and another for health article retrieval. All the required modules are designed and developed. In the following we explain each modeule one by one.

# Retrievals
There are four retrieval methods in this project with which we can retrieve social and health information. Each retrieval system is a class that is trained with datasets csv files when the server is up and retrieve any query the user inputs. Boolean, Transformer-based, TfIdf and fasttext retrieval systems are developed in the python files in social and health folders separately.

# Elasticsearch
We can retrieve the information by elasticsearch engine. First we run an elasticsearch docker image and load the datasets on the docker and retrieve information by only calling the python function.

# Classification and Clustering
Classification and Clustering are almost the modules we used for the assignments which are improved by changing some learning parameters. Classification and Clustering evaluation scores are shown in their interface page. Also we cluster and classify a given query based on the clusters and classes we fetched from the dataset while training the models.

# Query Expansion
There are two query expansion models that expands the query and generate some query that can alternate the original one. Evaluation of the query expansion might not work with testing different queries since the datasets are not large enough to output different documents. So the evaluation is mainly shown by the MRR score we generated.

# UI/UX
UI pages are designed by html and css frameworks and are connected to the backend by django.
