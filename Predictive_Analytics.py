# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np
import pandas as pd
from copy import deepcopy

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """

def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss = 0
    for i, row in enumerate(clusters):
        if len(row) == 0:
            continue
        cluster_centroids = np.mean(row)
        wcss += np.sum(np.linalg.norm(row - cluster_centroids, axis=1))
    return wcss

def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
'''KNN Algorithm implementation'''

def calculate_distance_matrix(A,b,Y_train):
  d = np.linalg.norm(A - b, axis = 1)
  d = np.column_stack((d, Y_train))
  d = sorted(d, key =(lambda x: x[0]))
  d = np.asarray(d)
  return d

def KNN(X_train,X_test,Y_train):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
     predicted_labels = []
     for x_t in X_test:
         distance_matrix = calculate_distance_matrix(X_train, x_t, Y_train)
         distance_matrix = distance_matrix[0:k]
         test_labels = distance_matrix[::, -1]
         test_labels = test_labels.astype("int64")
         counts = np.bincount(test_labels)
         label = np.argmax(counts)
         predicted_labels.append(label)
     predicted_labels = np.asarray(predicted_labels)
     return predicted_labels

''' End of KNN Classifier Algorithm'''

def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """

''' K-Means Clustering Algorithm Implementation'''

def create_empty_clusters(k):
    clusters = []
    for i in range(k):
        clusters.append([])

    return clusters

def assigning_clusters(X,cluster_centers):
    clusters  = create_empty_clusters(cluster_centers.shape[0])
    for i in range(len(X)):
        dist = np.linalg.norm(X[i] - cluster_centers, axis = 1)
        dist = np.asarray(dist)
        min_dist_index = np.argmin(dist)
        clusters[min_dist_index].append(X[i])
    clusters = np.asarray(clusters)
    return clusters


def Kmeans(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    X = X_train
    k = N

    cluster_centers_old = np.zeros((k, X.shape[1]))
    cluster_centers = X[np.random.randint(X.shape[0], size=k)] #Randomly selecting data points to be initial cluster centroids
    error = np.sum(np.linalg.norm(cluster_centers - cluster_centers_old, axis=1))
    while error != 0:
        clusters = assigning_clusters(X, cluster_centers)
        cluster_centers_old = deepcopy(cluster_centers)
        for i, row in enumerate(clusters):
            if len(row) == 0:
                continue
            cluster_centers[i] = np.mean(row)
        error = np.sum(np.linalg.norm(cluster_centers - cluster_centers_old, axis=1))
    return clusters


''' End of K-means clustering algorithm'''


def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """


def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""



    
