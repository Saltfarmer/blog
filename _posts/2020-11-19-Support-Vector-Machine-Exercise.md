---
title: "Support Vector Machine Exercise"
header :
  teaser: /assets/images/SVM.png
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Python
  - Machine Learning
  - Classification
  - EDA
  - Exercise
  - SVM
  - Grid Search
 

---

Exercise from [Jose Portilla Python for Data Science Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/).

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

Now Lets get started

# Support Vector Machines Project 

Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data

For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

Here's a picture of the three different Iris types:


```python
# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
```




![jpeg](http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg)




```python
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
```




![jpeg](http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg)




```python
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)
```




![jpeg](http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg)



The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)

The four features of the Iris dataset:

    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm

## Get the data

**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **


```python
import seaborn as sns 

iris = sns.load_dataset('iris')
```

Let's visualize the data and get you started!

## Exploratory Data Analysis

Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!

**Import some libraries you think you'll need.**


```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

%matplotlib inline 
```

** Create a pairplot of the data set. Which flower species seems to be the most separable?**


```python
sns.pairplot(iris, hue='species')
```




![png](https://i.ibb.co/P4wY92k/output-9-1.png)


**Create a kde plot of sepal_length versus sepal width for setosa species of flower.**


```python
sns.kdeplot(data=iris, x='sepal_width', y='sepal_length')
```




![png](https://i.ibb.co/XsCjD6b/output-11-1.png)


# Train Test Split

** Split your data into a training set and a testing set.**


```python
from sklearn.model_selection import train_test_split
```


```python
x = iris.drop(['species'], axis=1).copy()
y = iris['species'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y) 
```

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**


```python
from sklearn.svm import SVC
```


```python
svm = SVC()
```


```python
svm.fit(x_train, y_train)
```




    SVC()



## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**


```python
prediction = svm.predict(x_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(confusion_matrix(y_test, prediction))
```

    [[13  0  0]
     [ 0  9  1]
     [ 0  0 15]]



```python
print(classification_report(y_test, prediction))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.90      0.95        10
       virginica       0.94      1.00      0.97        15
    
        accuracy                           0.97        38
       macro avg       0.98      0.97      0.97        38
    weighted avg       0.98      0.97      0.97        38


​    

Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

## Gridsearch Practice

** Import GridsearchCV from SciKit Learn.**


```python
from sklearn.model_selection import GridSearchCV
```

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**


```python
param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.0001]}
```

** Create a GridSearchCV object and fit it to the training data.**


```python
grid = GridSearchCV(svm, param_grid, verbose=3)
grid.fit(x_train, y_train)
```

    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=1.000, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=0.913, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=1.000, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=1.000, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=0.864, total=   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] .................... C=0.1, gamma=0.1, score=0.870, total=   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] .................... C=0.1, gamma=0.1, score=0.913, total=   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] .................... C=0.1, gamma=0.1, score=0.955, total=   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] .................... C=0.1, gamma=0.1, score=0.909, total=   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] .................... C=0.1, gamma=0.1, score=0.818, total=   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ................... C=0.1, gamma=0.01, score=0.348, total=   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ................... C=0.1, gamma=0.01, score=0.348, total=   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ................... C=0.1, gamma=0.01, score=0.364, total=   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ................... C=0.1, gamma=0.01, score=0.364, total=   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ................... C=0.1, gamma=0.01, score=0.364, total=   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] ................. C=0.1, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] ................. C=0.1, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] ................. C=0.1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] ................. C=0.1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=0.1, gamma=0.0001 .............................................
    [CV] ................. C=0.1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=1, score=1.000, total=   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=1, score=0.957, total=   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=1, score=1.000, total=   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=1, score=0.955, total=   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................ C=1, gamma=1, score=0.909, total=   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................... C=1, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................... C=1, gamma=0.1, score=0.957, total=   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................... C=1, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................... C=1, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................... C=1, gamma=0.1, score=0.909, total=   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ..................... C=1, gamma=0.01, score=0.870, total=   0.0s
    [CV] C=1, gamma=0.01 .................................................


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s


    [CV] ..................... C=1, gamma=0.01, score=0.913, total=   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ..................... C=1, gamma=0.01, score=0.955, total=   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ..................... C=1, gamma=0.01, score=0.909, total=   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ..................... C=1, gamma=0.01, score=0.864, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ....................... C=10, gamma=1, score=1.000, total=   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ....................... C=10, gamma=1, score=0.913, total=   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ....................... C=10, gamma=1, score=1.000, total=   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ....................... C=10, gamma=1, score=0.955, total=   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] ....................... C=10, gamma=1, score=0.909, total=   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ..................... C=10, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ..................... C=10, gamma=0.1, score=0.957, total=   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ..................... C=10, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ..................... C=10, gamma=0.1, score=0.955, total=   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ..................... C=10, gamma=0.1, score=0.909, total=   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .................... C=10, gamma=0.01, score=1.000, total=   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .................... C=10, gamma=0.01, score=0.957, total=   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .................... C=10, gamma=0.01, score=1.000, total=   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .................... C=10, gamma=0.01, score=1.000, total=   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] .................... C=10, gamma=0.01, score=0.909, total=   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] .................. C=10, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] .................. C=10, gamma=0.0001, score=0.348, total=   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] .................. C=10, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] .................. C=10, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] .................. C=10, gamma=0.0001, score=0.364, total=   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ...................... C=100, gamma=1, score=1.000, total=   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ...................... C=100, gamma=1, score=0.913, total=   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ...................... C=100, gamma=1, score=1.000, total=   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ...................... C=100, gamma=1, score=0.955, total=   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ...................... C=100, gamma=1, score=1.000, total=   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] .................... C=100, gamma=0.1, score=0.957, total=   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] .................... C=100, gamma=0.1, score=0.957, total=   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] .................... C=100, gamma=0.1, score=1.000, total=   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] .................... C=100, gamma=0.1, score=0.955, total=   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] .................... C=100, gamma=0.1, score=0.955, total=   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ................... C=100, gamma=0.01, score=1.000, total=   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ................... C=100, gamma=0.01, score=0.957, total=   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ................... C=100, gamma=0.01, score=1.000, total=   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ................... C=100, gamma=0.01, score=0.955, total=   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ................... C=100, gamma=0.01, score=0.909, total=   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] ................. C=100, gamma=0.0001, score=0.870, total=   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] ................. C=100, gamma=0.0001, score=0.913, total=   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] ................. C=100, gamma=0.0001, score=0.955, total=   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] ................. C=100, gamma=0.0001, score=0.909, total=   0.0s
    [CV] C=100, gamma=0.0001 .............................................
    [CV] ................. C=100, gamma=0.0001, score=0.909, total=   0.0s


    [Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:    0.5s finished



    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.1, 1, 10, 100],
                             'gamma': [1, 0.1, 0.01, 0.0001]},
                 verbose=3)



** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**


```python
pred_grid = grid.predict(x_test)
```


```python
print(confusion_matrix(y_test, pred_grid))
```

    [[13  0  0]
     [ 0  9  1]
     [ 0  1 14]]



```python
print(classification_report(y_test, pred_grid))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        13
      versicolor       0.90      0.90      0.90        10
       virginica       0.93      0.93      0.93        15
    
        accuracy                           0.95        38
       macro avg       0.94      0.94      0.94        38
    weighted avg       0.95      0.95      0.95        38


​    

You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

## Great Job!