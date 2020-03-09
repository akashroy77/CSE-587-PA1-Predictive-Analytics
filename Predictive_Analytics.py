# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

df = pd.read_csv("D:/UB Courses/cse587/Assignment1/data.csv", delimiter = ",")

y = df.iloc[:,-1]
y = y.to_numpy()
df = df.iloc[:,:-1]
df_normalized = (df - df.min())/ (df.max() - df.min())
X = df.to_numpy()
X = df_normalized.to_numpy()

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

no_of_classes = 11

knn = KNeighborsClassifier()
clf = SVC()
log_reg = LogisticRegression()
decision_tree = DecisionTreeClassifier()

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

def confusion_matrix(y_pred, y_test, no_of_classes):
    result=(y_test*(no_of_classes+1))+y_pred
    a = np.histogram(result, bins=range(0,((no_of_classes+1)**2)+1))
    final_conf=a[0].reshape((no_of_classes+1,no_of_classes+1))
    return final_conf


def plot_(y_pred, y_test, no_of_classes):
    data = confusion_matrix(y_pred, y_test,11)
    plt.imshow(data,cmap='Wistia')
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            plt.text(x,y,data[x, y],horizontalalignment='center',verticalalignment='center')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.grid(False)
    plt.colorbar()
    plt.show()


'''KNN Algorithm implementation'''

def calculate_distance_matrix(A,b,Y_train):
    d = np.linalg.norm(A - b, axis = 1)
    d = np.column_stack((d, Y_train))
    d = sorted(d, key =(lambda x: x[0]))
    d = np.asarray(d)
    return d

def KNN(X_train,X_test,Y_train):
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
    

def PCA(x_train, N):
    classes = []
    data = []
    rows = -1
    cols=0

    with open('D:/UB Courses/cse587/Assignment1/data.csv', 'r') as f:
        file = csv.reader(f, delimiter=' ')
        for i in file:
            rows += 1

    file1 = pd.read_csv('D:/UB Courses/cse587/Assignment1/data.csv')
    for i in file1:     
        cols += 1
    
    #calculate the mean of data
    mean_data = x_train.mean(0)    
    #normalization
    normalized_data = x_train - mean_data
    data_cov = np.cov(normalized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(data_cov)
    top_eigen_components = eigenvectors[:,0:N]
    sample_data = x_train.dot(top_eigen_components)
    return sample_data

    
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

def SklearnSupervisedLearning(x_train,y_train,x_test):
    knn_(x_train, x_test, 5)
    knn_grid(x_train, x_test, no_of_classes)
    SVM_(x_train, x_test, y_test)
    SVM_grid(x_train, x_test, y_test)
    logReg(x_train, x_test, y_test)
    logReg_grid(x_train, x_test, y_test)
    DT(x_train, x_test, y_test)
    DT_grid(x_train, x_test, y_test)
    

def SklearnVotingClassifier(x_train,y_train,x_test):
    estimators=[('knn', knn), ('svm', clf), ('DT',decision_tree), ('log_reg', log_reg)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(x_train, y_train)
    print(ensemble.score(x_test, y_test))
    y_pred = ensemble.predict(x_test)
    return y_pred

def knn_(x_train, x_test, N):
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred
    

def knn_grid(x_train, x_test, N):
    params_knn = {'n_neighbors': [N],
             'weights' : ['distance'],
             'metric' : ['euclidean']}

    knn_gs = GridSearchCV(KNeighborsClassifier(), params_knn, verbose=1, cv=5)
    knn_gs.fit(x_train, y_train)
    y_pred = knn_gs.predict(x_test)
    print('knn: {}'.format(knn_gs.score(x_test, y_test)))
    plot_(y_pred, y_test, N)
    return y_pred
    

def SVM_(x_train, x_test, y_test):
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    
    #clf = svm.NuSVC(gamma='scale')
    #clf = svm.SVC(kernel='linear', gamma = 'scale')
    #clf = svm.SVC(kernel='rbf',gamma = 'scale')
    clf = svm.SVC(kernel='poly',gamma = 'scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred
    

def SVM_grid(x_train, x_test, y_test):
    param_grid = {'C': [10],  
              'gamma': [ 0.1], 
              'kernel': ['linear']}  
  
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1) 
   
    grid.fit(x_train, y_train)
    y_pred = grid.predict(x_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred

    
def logReg(x_train, x_test, y_test):
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=5000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)

    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    
    print('log_reg: {}'.format(log_reg.score(x_test, y_test)))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred
    
    
def logReg_grid(x_train, x_test, y_test):
    hyperparameters = dict(C=np.logspace(0,4), penalty=['l2'])
    log = GridSearchCV(LogisticRegression(multi_class='auto', solver='lbfgs',max_iter=3000), hyperparameters)
    log.fit(x_train, y_train)
    y_pred = log.predict(x_test)
    print('log_reg_grid: {}'.format(log.score(x_test, y_test)))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred
    

def DT(x_train, x_test, y_test):
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=11)
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    print('DT: {}'.format(decision_tree.score(x_test, y_test)))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred

    
def DT_grid(x_train, x_test, y_test):
    parameters = dict(criterion = ['gini'],
                      max_depth = [6,10])
    dt_grid = GridSearchCV(DecisionTreeClassifier(), parameters )
    dt_grid.fit(x_train, y_train)
    y_pred = dt_grid.predict(x_test)
    print('DT_grid: {}'.format(dt_grid.score(x_test, y_test)))
    plot_(y_pred, y_test, no_of_classes)
    return y_pred
"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""
#https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
#https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a
#https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
#https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
#https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a
#https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/
#https://scikit-learn.org/stable/modules/tree.html
