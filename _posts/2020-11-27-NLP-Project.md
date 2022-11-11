---
title: "NLP Project"
header :
  teaser: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - EDA
  - Classification
  - NLP
  - Data Visualization

---

Welcome to the NLP Project for this section of the course. In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews. This will be a simpler procedure than the lecture, since we will utilize the pipeline methods for more complex tasks.

We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users. 

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.

Let's get started! Just follow the directions below!

## Imports

 **Import the usual suspects. :) **


```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

%matplotlib inline
```

## The Data

**Read the yelp.csv file and set it as a dataframe called yelp.**


```python
yelp = pd.read_csv("yelp.csv")
```

** Check the head, info , and describe methods on yelp.**


```python
yelp.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




```python
yelp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   business_id  10000 non-null  object
     1   date         10000 non-null  object
     2   review_id    10000 non-null  object
     3   stars        10000 non-null  int64 
     4   text         10000 non-null  object
     5   type         10000 non-null  object
     6   user_id      10000 non-null  object
     7   cool         10000 non-null  int64 
     8   useful       10000 non-null  int64 
     9   funny        10000 non-null  int64 
    dtypes: int64(4), object(6)
    memory usage: 781.4+ KB



```python
yelp.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stars</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.777500</td>
      <td>0.876800</td>
      <td>1.409300</td>
      <td>0.701300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.214636</td>
      <td>2.067861</td>
      <td>2.336647</td>
      <td>1.907942</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>77.000000</td>
      <td>76.000000</td>
      <td>57.000000</td>
    </tr>
  </tbody>
</table>



**Create a new column called "text length" which is the number of words in the text column.**


```python
yelp['text length'] = yelp['text'].apply(lambda x:len(x))
```


```python
yelp['word length'] = yelp['text'].apply(lambda x: len(x.split(' ')))
```

# EDA

Let's explore the data

## Imports

**Import the data visualization libraries if you haven't done so already.**


```python
import seaborn as sns
```


```python
sns.jointplot(data=yelp, x='text length', y='word length')
```





![png](https://i.ibb.co/PMnywyj/output-15-1.png)


**Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**


```python
sns.displot(data=yelp, x="text length", col="stars")
```





![png](https://i.ibb.co/9tcSkKf/output-17-1.png)


**Create a boxplot of text length for each star category.**


```python
plt.figure(figsize=[15,6])
sns.boxplot(data=yelp, x="stars", y="text length")
```





![png](https://i.ibb.co/QMhmSH3/output-19-1.png)


**Create a countplot of the number of occurrences for each type of star rating.**


```python
plt.figure(figsize=[15,6])
sns.countplot(data=yelp, x="stars")
```





![png](https://i.ibb.co/mH7nkJY/output-21-1.png)


** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**


```python
yelp.groupby('stars').mean()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
      <th>word length</th>
    </tr>
    <tr>
      <th>stars</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.576769</td>
      <td>1.604806</td>
      <td>1.056075</td>
      <td>826.515354</td>
      <td>156.013351</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.719525</td>
      <td>1.563107</td>
      <td>0.875944</td>
      <td>842.256742</td>
      <td>158.508091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.788501</td>
      <td>1.306639</td>
      <td>0.694730</td>
      <td>758.498289</td>
      <td>143.043806</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.954623</td>
      <td>1.395916</td>
      <td>0.670448</td>
      <td>712.923142</td>
      <td>132.921441</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.944261</td>
      <td>1.381780</td>
      <td>0.608631</td>
      <td>624.999101</td>
      <td>116.054840</td>
    </tr>
  </tbody>
</table>






```python
star_mean = yelp.groupby('stars').mean().reset_index()
```

**Use the corr() method on that groupby dataframe to produce this dataframe:**


```python
star_mean.corr()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stars</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
      <th>word length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>stars</th>
      <td>1.000000</td>
      <td>0.964758</td>
      <td>-0.761288</td>
      <td>-0.950389</td>
      <td>-0.950171</td>
      <td>-0.953951</td>
    </tr>
    <tr>
      <th>cool</th>
      <td>0.964758</td>
      <td>1.000000</td>
      <td>-0.743329</td>
      <td>-0.944939</td>
      <td>-0.857664</td>
      <td>-0.865650</td>
    </tr>
    <tr>
      <th>useful</th>
      <td>-0.761288</td>
      <td>-0.743329</td>
      <td>1.000000</td>
      <td>0.894506</td>
      <td>0.699881</td>
      <td>0.690255</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>-0.950389</td>
      <td>-0.944939</td>
      <td>0.894506</td>
      <td>1.000000</td>
      <td>0.843461</td>
      <td>0.844066</td>
    </tr>
    <tr>
      <th>text length</th>
      <td>-0.950171</td>
      <td>-0.857664</td>
      <td>0.699881</td>
      <td>0.843461</td>
      <td>1.000000</td>
      <td>0.999648</td>
    </tr>
    <tr>
      <th>word length</th>
      <td>-0.953951</td>
      <td>-0.865650</td>
      <td>0.690255</td>
      <td>0.844066</td>
      <td>0.999648</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>





**Then use seaborn to create a heatmap based off that .corr() dataframe:**


```python
plt.figure(figsize=[14,10])
sns.heatmap(star_mean.corr(), annot=True)
```




![png](https://i.ibb.co/yQ6xyb8/output-28-1.png)


## NLP Classification Task

Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.

**Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**


```python
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
```

** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**


```python
X = yelp_class['text']
y = yelp_class['stars']
```

**Import CountVectorizer and create a CountVectorizer object.**


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```

** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**


```python
X = cv.fit_transform(X)
```

## Train Test Split

Let's split our data into training and testing data.

** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Training a Model

Time to train a model!

** Import MultinomialNB and create an instance of the estimator and call is nb **


```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
```

**Now fit nb using the training data.**


```python
nb.fit(X_train, y_train)
```




    MultinomialNB()



## Predictions and Evaluations

Time to see how our model did!

**Use the predict method off of nb to predict labels from X_test.**


```python
pred_nb = nb.predict(X_test)
```

** Create a confusion matrix and classification report using these predictions and y_test **


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
print(confusion_matrix(pred_nb, y_test))
print(classification_report(pred_nb, y_test))
```

    [[159  22]
     [ 69 976]]
                  precision    recall  f1-score   support
    
               1       0.70      0.88      0.78       181
               5       0.98      0.93      0.96      1045
    
        accuracy                           0.93      1226
       macro avg       0.84      0.91      0.87      1226
    weighted avg       0.94      0.93      0.93      1226


​    

**Great! Let's see what happens if we try to include TF-IDF to this process using a pipeline.**

# Using Text Processing

** Import TfidfTransformer from sklearn. **


```python
from sklearn.feature_extraction.text import TfidfTransformer
```

** Import Pipeline from sklearn. **


```python
from sklearn.pipeline import Pipeline
```

** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**


```python
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
```

## Using the Pipeline

**Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text**

### Train Test Split

**Redo the train test split on the yelp_class object.**


```python
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
```

**Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**


```python
pipeline.fit(X_train, y_train)
```




    Pipeline(steps=[('bow', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())])



### Predictions and Evaluation

** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**


```python
pred_pipe = pipeline.predict(X_test)
```


```python
print(confusion_matrix(pred_pipe, y_test))
print(classification_report(pred_pipe, y_test))
```

    [[  0   0]
     [228 998]]
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         0
               5       1.00      0.81      0.90      1226
    
        accuracy                           0.81      1226
       macro avg       0.50      0.41      0.45      1226
    weighted avg       1.00      0.81      0.90      1226


​    

    c:\users\saltfarmer\appdata\local\programs\python\python38\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


Looks like Tf-Idf actually made things worse! That is it for this project. But there is still a lot more you can play with:

**Some other things to try....**
Try going back and playing around with the pipeline steps and seeing if creating a custom analyzer like we did in the lecture helps (note: it probably won't). Or recreate the pipeline with just the CountVectorizer() and NaiveBayes. Does changing the ML model at the end to another classifier help at all?

# Great Job!