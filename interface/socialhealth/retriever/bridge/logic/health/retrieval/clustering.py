import itertools

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import TruncatedSVD
from .requires import *
from scipy import sparse


def cluster(search_term):
    global kmeans, true_labels
    # PCA_func(true_labels, predicted_labels)
    # TSNE_func(true_labels, predicted_labels)

    RSS = calculate_RSS(doc_term_mat, kmeans.cluster_centers_, kmeans.labels_)
    # print("RSS is {}".format(RSS))

    silhouette_avg = silhouette_score(doc_term_mat, kmeans.labels_)
    # print("For n_clusters =", 7, "The average silhouette_score is :", silhouette_avg, )

    silhouette_visualizer(7)
    intercluster_distance_maps(7)

    score = purity_score(true_labels, predicted_labels)
    # print("purity score is {}".format(score))

    y = query(search_term)

    return RSS, silhouette_avg, score, y


def true_labels_func():
    categories = set()
    for i in bioset:
        categories.add(i['categories'][1])
    dic = {}
    counter = 0
    for category in categories:
        dic[category] = counter
        counter += 1
    true_labels = []
    for i in bioset:
        category = i['categories'][1]
        true_labels.append(dic[category])
    return np.array(true_labels)


def TSNE_func(true_labels, predicted_labels):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(doc_term_mat.toarray())

    # plot 1:
    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=true_labels, s=50, cmap='viridis')
    plt.title("True")
    plt.rcParams["figure.figsize"] = (15, 6)

    # plot 2:
    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=predicted_labels, s=50, cmap='viridis')
    plt.title("Predicted")
    plt.rcParams["figure.figsize"] = (15, 6)

    plt.show()


def PCA_func(true_labels, predicted_labels):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(doc_term_mat.toarray())

    plt.subplot(1, 2, 1)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=true_labels, s=50, cmap='viridis')
    plt.title("True")
    plt.rcParams["figure.figsize"] = (30, 6)

    # plot 2:
    plt.subplot(1, 2, 2)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=predicted_labels, s=50, cmap='viridis')
    plt.title("Predicted")
    plt.rcParams["figure.figsize"] = (30, 6)

    plt.show()


def calculate_RSS(docs, centers, labels):
    RSS = 0
    documents_length = 875
    for i in range(documents_length):
        RSS += np.sum(np.square(docs[i] - centers[labels[i]]))
    return RSS


def silhouette_visualizer(k=7):
    x = doc_term_mat.toarray()
    model = KMeans(k, random_state=0)
    visualizer = SilhouetteVisualizer(model)
    visualizer.fit(x)
    visualizer.show()


def intercluster_distance_maps(k=7):
    x = doc_term_mat.toarray()
    model = KMeans(k)
    visualizer = InterclusterDistance(model, random_state=0).fit(x)
    visualizer.show()


def purity_score(true_labels, predicted_labels):
    arr = np.zeros(true_labels.shape)

    labels = np.unique(true_labels)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        true_labels[true_labels == labels[k]] = ordered_labels[k]

    labels = np.unique(true_labels)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(predicted_labels):
        winner = np.argmax(np.histogram(true_labels[predicted_labels == cluster], bins=bins)[0])
        arr[predicted_labels == cluster] = winner

    return accuracy_score(true_labels, arr)


def process_query(query):
    splitted_input = query.split(" ")
    nsi = []
    for x in splitted_input:
        if x not in stopwords:
            nsi.append([normalizer.normalize(lemmatizer.lemmatize(x))])
    return nsi


def find_query_vector(tokens):
    vector = np.zeros(len(vocabulary))
    for token in itertools.chain(*tokens):
        try:
            index = vectorizer.vocabulary_[token]
            vector[index] = 1
        except ValueError:
            pass
    return vector


def query(search):
    global kmeans, svd
    query_vector = find_query_vector(process_query(search))
    query_vector = sparse.csr_matrix(query_vector)

    x_dr = svd.fit_transform(doc_term_mat)
    y = kmeans.fit_predict(x_dr)

    x_dr = svd.transform(query_vector)
    y = kmeans.predict(x_dr)

    return y[0]


print("training cluster retrieval system for health")
kmeans = KMeans(n_clusters=7, random_state=0).fit(doc_term_mat)
predicted_labels = kmeans.labels_
true_labels = true_labels_func()
svd = TruncatedSVD(n_components=7, n_iter=7, random_state=42)
print("training cluster retrieval system for health done")


def run(search_term):
    # return cluster(search_term)
    return query(search_term)
