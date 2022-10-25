---
title: "K Nearest Neighbour Exercise"
header :
  image: /assets/images/python-head.jpg
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

# K Nearest Neighbors Project 

Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.

## Import Libraries

**Import pandas,seaborn, and the usual libraries.**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

## Get the Data

** Read the 'KNN_Project_Data csv file into a dataframe **


```python
data = pd.read_csv("KNN_Project_Data")
```

**Check the head of the dataframe.**


```python
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1636.670614</td>
      <td>817.988525</td>
      <td>2565.995189</td>
      <td>358.347163</td>
      <td>550.417491</td>
      <td>1618.870897</td>
      <td>2147.641254</td>
      <td>330.727893</td>
      <td>1494.878631</td>
      <td>845.136088</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1013.402760</td>
      <td>577.587332</td>
      <td>2644.141273</td>
      <td>280.428203</td>
      <td>1161.873391</td>
      <td>2084.107872</td>
      <td>853.404981</td>
      <td>447.157619</td>
      <td>1193.032521</td>
      <td>861.081809</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1300.035501</td>
      <td>820.518697</td>
      <td>2025.854469</td>
      <td>525.562292</td>
      <td>922.206261</td>
      <td>2552.355407</td>
      <td>818.676686</td>
      <td>845.491492</td>
      <td>1968.367513</td>
      <td>1647.186291</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1059.347542</td>
      <td>1066.866418</td>
      <td>612.000041</td>
      <td>480.827789</td>
      <td>419.467495</td>
      <td>685.666983</td>
      <td>852.867810</td>
      <td>341.664784</td>
      <td>1154.391368</td>
      <td>1450.935357</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1018.340526</td>
      <td>1313.679056</td>
      <td>950.622661</td>
      <td>724.742174</td>
      <td>843.065903</td>
      <td>1370.554164</td>
      <td>905.469453</td>
      <td>658.118202</td>
      <td>539.459350</td>
      <td>1899.850792</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



# EDA

Since this data is artificial, we'll just do a large pairplot with seaborn.

**Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**


```python
sns.pairplot(data)
```




![png](https://i.ibb.co/HCMpZMm/output-8-1.png)


# Standardize the Variables

Time to standardize the variables.

** Import StandardScaler from Scikit learn.**


```python
from sklearn.preprocessing import StandardScaler
```

** Create a StandardScaler() object called scaler.**


```python
scaler = StandardScaler()
```

** Fit scaler to the features.**


```python
scaler.fit(data.drop(['TARGET CLASS'], axis=1))
```




    StandardScaler()



**Use the .transform() method to transform the features to a scaled version.**


```python
scaler.transform(data.drop(['TARGET CLASS'], axis=1))
```




    array([[ 1.56852168, -0.44343461,  1.61980773, ..., -0.93279392,
             1.00831307, -1.06962723],
           [-0.11237594, -1.05657361,  1.7419175 , ..., -0.46186435,
             0.25832069, -1.04154625],
           [ 0.66064691, -0.43698145,  0.77579285, ...,  1.14929806,
             2.1847836 ,  0.34281129],
           ...,
           [-0.35889496, -0.97901454,  0.83771499, ..., -1.51472604,
            -0.27512225,  0.86428656],
           [ 0.27507999, -0.99239881,  0.0303711 , ..., -0.03623294,
             0.43668516, -0.21245586],
           [ 0.62589594,  0.79510909,  1.12180047, ..., -1.25156478,
            -0.60352946, -0.87985868]])



**Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**


```python
df_scaled = pd.DataFrame(scaler.transform(data.drop(['TARGET CLASS'], axis=1)), columns=data.drop(['TARGET CLASS'], axis=1).columns)
```

# Train Test Split

**Use train_test_split to split your data into a training set and a testing set.**


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(df_scaled, data["TARGET CLASS"], test_size =0.2, random_state=17)
```

# Using KNN

**Import KNeighborsClassifier from scikit learn.**


```python
from sklearn.neighbors import KNeighborsClassifier
```

**Create a KNN model instance with n_neighbors=1**


```python
knn = KNeighborsClassifier(n_neighbors=1)
```

**Fit this KNN model to the training data.**


```python
knn.fit(x_train, y_train)
```




    KNeighborsClassifier(n_neighbors=1)



# Predictions and Evaluations

Let's evaluate our KNN model!

**Use the predict method to predict values using your KNN model and X_test.**


```python
pred = knn.predict(x_test)
```

** Create a confusion matrix and classification report.**


```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```


```python
print(confusion_matrix(y_test, pred))
```

    [[59 26]
     [29 86]]



```python
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.67      0.69      0.68        85
               1       0.77      0.75      0.76       115
    
        accuracy                           0.73       200
       macro avg       0.72      0.72      0.72       200
    weighted avg       0.73      0.72      0.73       200


​    

# Choosing a K Value

Let's go ahead and use the elbow method to pick a good K Value!

** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**


```python
K = []
error_rate = []
for i in range(1,40):
    K.append(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
```

**Now create the following plot using the information from your for loop.**


```python
sns.lineplot(x=K, y=error_rate)
```




![png](https://i.ibb.co/VTf1LHf/output-38-1.png)


## Retrain with new K Value

**Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**


```python
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print("WITH K=31")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

    WITH K=31
    [[70 15]
     [21 94]]
                  precision    recall  f1-score   support
    
               0       0.77      0.82      0.80        85
               1       0.86      0.82      0.84       115
    
        accuracy                           0.82       200
       macro avg       0.82      0.82      0.82       200
    weighted avg       0.82      0.82      0.82       200


​    

# Great Job!