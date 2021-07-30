import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_wine
wine = load_wine()
# wine object is a sklearn.utils.bunch object, need to convert in pandas dataframe.

# REFERENCES
# https://www.superdatascience.com/pages/machine-learning
# This next 2 lines of code is taken from
# "https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset"
data = pd.DataFrame(wine.data,columns=wine.feature_names)
data['class'] = pd.Series(wine.target)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Using standardscalar to preprocess data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Support Vector Machine Classification
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("---------------------------------------------------------------------")
print("-Printing confusion matrix and accuracy score for SVM classification-")
print("---------------------------------------------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))

# KNN Classification method without cross validation
from sklearn.neighbors import KNeighborsClassifier
scores = []
index = range(1, 20)
for i in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))
plt.title("KNN Line graph")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.plot(index, scores, color="blue")
plt.ion()
plt.show()
plt.pause(2)
plt.close()

# From the line graph I decided to take the value of n_neighbors as 6, since after that its almost same level of accuracy.

classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("---------------------------------------------------------------------")
print("-Printing confusion matrix and accuracy score for KNN classification-")
print("---------------------------------------------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))

# KNN classification with Cross Validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

scores = []
index = range(1, 20)
for i in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
    accuracies.mean() * 100
    scores.append(accuracies.mean() * 100)
plt.title("KNN Line graph with cross validation")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.plot(index, scores, color="blue")
plt.ion()
plt.show()
plt.pause(2)
plt.close()
print("--------------------------------------------------------------------------------------------")
print("----------------------------Parameter tuning for n_neighbors--------------------------------")
print("-Using graph to find how many n_neighbors to chose for best accuracy using cross validation-")
print("--------------------------------------------------------------------------------------------")
print("* Maximum accuracy is found as: {0} at position: {1}, i.e for n_neighbors = {1}.".format(max(scores), scores.index(max(scores))))

classifier = KNeighborsClassifier(n_neighbors = 17)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("------------------------------------------------------------------------------------------------------------")
print("-Printing confusion matrix and accuracy score after parameter tuning for n_neighbors using cross validation-")
print("------------------------------------------------------------------------------------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))
print("* Clearly accuracy has been increased after n_neighbors parameter tuning from 98.15% to 100%!!")

# Applying PCA (Dimensionality reduction technique)
from sklearn.decomposition import PCA
pca = PCA(n_components = 7)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("\n")
print("--------------------------------------------------------------------------------------------------")
print("-----------Printing PCA variance preserved with each dimension and their cumulative sum-----------")
print("-We wanted to preserve 90% of the information hence took 7 components which represents nearly 90%-")
print("--------------------------------------------------------------------------------------------------")
print("Explained Variance :\n",pca.explained_variance_ratio_)
print("Explained Variance with cumulative sum :\n",pca.explained_variance_ratio_.cumsum())

from sklearn.neighbors import KNeighborsClassifier
scores = []
index = range(1, 20)
for i in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=3)
    accuracies.mean() * 100
    scores.append(accuracies.mean() * 100)
plt.title("KNN Line graph after Dimensionality Reduction(PCA)")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.plot(index, scores, color="blue")
plt.ion()
plt.show()
plt.pause(2)
plt.close()

classifier = KNeighborsClassifier(n_neighbors = 19)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("---------------------------------------------------------------------------------------------------------------")
print("-Printing confusion matrix and accuracy score after parameter tuning for n_neighbors after using PCA technique-")
print("---------------------------------------------------------------------------------------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))
print("* Clearly the same Accuracy has been achieved using KNN even after dimensionality reduction!!")

# Finding probabilities for roc_auc_score
knn_probabilities = classifier.predict_proba(X_test)

# Support Vector Machine Classification after dimensionality reduction
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0, probability=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("---------------------------------------------------------------------------------------------------------")
print("-Printing confusion matrix and accuracy score after Dimensionality reduction(PCA) for SVM classification-")
print("---------------------------------------------------------------------------------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))
print("* The Accuracy has reduced a bit using SVM after dimensionality reduction from 100% to 96.30%..")

# Finding probabilities for roc_auc_score
svm_probabilities = classifier.predict_proba(X_test)

# Random Forest classification
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("\n")
print("-------------------------------------------------------------------------------------------------------------")
print("-Printing parameter set for hyper_parameter tuning using RandomizedSearchCV for Random forest classification-")
print("-------------------------------------------------------------------------------------------------------------")
print(random_grid)
# Please wait,it may take some time for finding the best parameter set for the model using different combinations.
print("\n * Please wait,it may take some time for finding the best parameter set for the model using different "
      "combinations. \n\n")

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
# This will select the best parameter set
rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print("\n")
print("--------------------------------------------------------------")
print("-Printing best parameter set for Random forest classification-")
print("--------------------------------------------------------------")
print(rf_random.best_params_)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = rf_random.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("----------------------------------------------------------------------------------------------------------"
      "------------------------------")
print("-Printing confusion matrix and accuracy score after Dimensionality reduction(PCA) and parameter tuning for"
      " Random Forest classification-")
print("-----------------------------------------------------------------------------------------------------------"
      "------------------------------")
print(cm)
print(accuracy_score(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test,y_pred)*100))

# Finding probabilities for roc_auc_score
rf_probabilities = rf_random.predict_proba(X_test)

# Reference for finding roc_auc_score for multiclass classification
# https://github.com/manifoldailearning/Youtube/blob/master/ROC_AUC_on_Multiclass_classification.ipynb

from sklearn.metrics import roc_auc_score
knn_roc_auc_score = roc_auc_score(y_test, knn_probabilities, multi_class="ovr")
svm_roc_auc_score = roc_auc_score(y_test, svm_probabilities, multi_class="ovr")
rf_roc_auc_score = roc_auc_score(y_test, rf_probabilities, multi_class="ovr")
print("\n")
print("----------------------------------------------------------------")
print("-Printing the roc_auc_score for classification models after PCA-")
print("----------------------------------------------------------------")
print("KNN roc_auc_score: {:.5f} ".format(knn_roc_auc_score))
print("SVM roc_auc_score: {:.5f} ".format(svm_roc_auc_score))
print("Random Forest roc_auc_score: {:.5f} ".format(rf_roc_auc_score))