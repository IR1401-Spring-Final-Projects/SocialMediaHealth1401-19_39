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
from .requires import *


def cluster():
    true_labels = true_labels_func()

    kmeans = KMeans(n_clusters=7, random_state=0).fit(doc_term_mat)
    predicted_labels = kmeans.labels_

    PCA_func(true_labels, predicted_labels)
    TSNE_func(true_labels, predicted_labels)

    RSS = calculate_RSS(doc_term_mat, kmeans.cluster_centers_, kmeans.labels_)
    print("RSS is {}".format(RSS))

    silhouette_avg = silhouette_score(doc_term_mat, kmeans.labels_)
    print("For n_clusters =", 7, "The average silhouette_score is :", silhouette_avg, )

    silhouette_visualizer(7)
    intercluster_distance_maps(7)

    score = purity_score(true_labels, predicted_labels)
    print("purity score is {}".format(score))


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
    # for i in range(2, 12):
    #     model = KMeans(i, random_state=0)
    #     visualizer = SilhouetteVisualizer(model)
    #     visualizer.fit(x)
    #     visualizer.show()
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


