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
'''
no_of_classes = 11

knn = KNeighborsClassifier()
clf = SVC()
log_reg = LogisticRegression()
decision_tree = DecisionTreeClassifier()
'''
def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    correct = 0
    for i in range(len(y_true)):
        if(y_true[i] == y_pred[i]):
            correct += 1
    return correct / float(len(y_true)) * 100.0
    
def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    return np.mean(recall)

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    return np.mean(precision)

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
<<<<<<< HEAD

=======
>>>>>>> 9d009247d42f5eb614eb92166c5203b0f6728290
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

def KNN(X_train,X_test,Y_train, k):
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

'''Starting Random Forest Algorithm'''

'''This Function will check the number of classes (between 1-11) for the particular'''
def number_of_class_in_a_set(input_set): 
    label = input_set[:, -1]
    unique_classes = np.unique(label)
    return unique_classes

'''This function will check which class has maximum number of counts in a given dataset'''
def max_class(input_val):
    label= input_val[:, -1]
    max_val=0
    index=[]
    unique_classes,counts_unique_classes = np.unique(label, return_counts=True)
    max_val=max(counts_unique_classes)
    index=[i for i, e in enumerate(counts_unique_classes) if e == max_val]
    max_class = unique_classes[index[0]]
    return max_class

'''Calculate all the possible splits'''
def all_splits(input_values, features):
    splits = {}
    column=np.random.choice(input_values.shape[1],features )
    for column_index in column:          
        values = input_values[:, column]
        unique_values = np.unique(values)
        splits[column_index] = unique_values
    return splits

''' Creating Entropy for each node'''
def class_entropy(data):
    label = data[:, -1]
    unique_classes,counts_unique_classes = np.unique(label, return_counts=True)
    p = counts_unique_classes / counts_unique_classes.sum()
    class_entropy = sum(p * -np.log2(p))
    return class_entropy

'''Creating the whole impurity using weighted average'''
def entropy_impurity(n1, n2):   
    n = len(n1) + len(n2)
    c1 = len(n1) / n
    c2 = len(n2) / n
    entropy =  (c1 * class_entropy(n1)  + c2 * class_entropy(n2))
    return entropy

'''Creating Node for a Particular Decision'''
def data_decision(input_feature, split_decision, threshold):
    divided_nodes = input_feature[:, split_decision]
    n1 = input_feature[divided_nodes <= threshold]
    n2 = input_feature[divided_nodes >  threshold]  
    return n1, n2

'''Creating the decision node'''
def decision_node_create(input_feature, input_feature_splits):
   max_entropy = 100000000
   split_column=0
   split_value=0
   for i in input_feature_splits:
     for j in input_feature_splits[i]:
       n1, n2 = data_decision(input_feature, i,j)
       entropy = entropy_impurity(n1, n2) 
       if entropy <= max_entropy:
                max_entropy = entropy 
                split_column = i
                split_value = j
   return split_column, split_value

'''Decision Tree Training'''
def decision_tree(df, count=0, random_features=7):
    if count == 0:
        global column,sample_size
        column = df.columns        
        sample_size=2 
        data=df.values
    else:
        data = df           
    classes=number_of_class_in_a_set(data)
    num_classes=len(classes)
    if ((num_classes==1)) or (len(data) < sample_size) or (count == 10):
        feature_class = max_class(data)
        return feature_class
    else:    
        count += 1 
        potential_splits = all_splits(data, random_features)
        feature, value = decision_node_create(data, potential_splits)
        n1, n2 = data_decision(data, feature, value)
        if len(n1) == 0 or len(n2) == 0:
            feature_class = max_class(data)
            return feature_class
        feature_name = column[feature]
        decision = "{} <= {}".format(feature_name, value)           
        tree = {decision: []}
        c1 = decision_tree(n1, count,random_features)
        c2 = decision_tree(n2, count,random_features)
        if c1 == c2:
            tree = c1
        else:
            tree[decision].append(c1)
            tree[decision].append(c2)
        return tree

''' Bagging the Data'''
def bagging(train_df):
    random_index = np.random.randint(low=0, high=len(train_df),size=400)
    data_with_replacement = train_df.iloc[random_index]
    return data_with_replacement

'''This function will create an example tree from the questions'''
def answer_prediction(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    if int(example[feature_name]) <= float(value):
      answer = tree[question][0]
    else:
      answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return answer_prediction(example, residual_tree)

''' Main Algorithm this will call the decision tree'''
def random_forest_classifier(train_df,features):
    random_forest = []
    number_of_trees=50
    for i in range(number_of_trees):
        data_with_replacement = bagging(train_df)
        tree = decision_tree(data_with_replacement, random_features=features)
        random_forest.append(tree)
    return random_forest

'''Decision Tree Prediction'''
def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(answer_prediction, args=(tree,), axis=1)
    return predictions

''' This will Predict the test data using decision tree'''    
def random_forest_predictions(test_df, forest):
    decisions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        decisions[column_name] = predictions

    decisions = pd.DataFrame(decisions)
    all_predictions = decisions.mode(axis=1)[0]
    
    return all_predictions
    
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    df1=pd.DataFrame(x_train)
    df2=pd.DataFrame(y_train)
    test_df=pd.DataFrame(x_test)
    train_df=pd.concat((df1,df2),axis=1)
    train_df,test_df=create_dataframe(x_train,y_train,x_test)
    train_df["label"] = train_df.iloc[:,-1]
    train_df.drop(train_df.columns[48], inplace=True)
    column_names = []
    for column in train_df.columns:
        column=str(column)
        name = column.replace(" ", "_")
        column_names.append(name)
    train_df.columns = column_names
    test_column_names = []
    for column in test_df.columns:
        column=str(column)
        name = column.replace(" ", "_")
        test_column_names.append(name)
    test_df.columns = test_column_names
    forest = random_forest_classifier(train_df,features=7)
    predictions = random_forest_predictions(test_df, forest)
    y_test=predictions.to_numpy()
    return y_test

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
    log = Gridsearchcv(LogisticRegression(), hyperparameters, cv=2)
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
#https://machinelearningmastery.com/implement-random-forest-scratch-python/
#https://www.python-course.eu/Random_Forests.php
#https://intellipaat.com/blog/what-is-random-forest-algorithm-in-python/