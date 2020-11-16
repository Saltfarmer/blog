---
title: "Logistic Regression Exercise"
header :
  image: /assets/images/python-head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Python
  - Regression
  - EDA
  - Exercise
 

---

Today, i will try Exploratory Data Analysis and regression with insurance data from Kaggle. Let's take a look


```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns 

%matplotlib inline
```


```python
data = pd.read_csv("insurance.csv")
```


```python
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>



Let's see the structure and the context of the data.

## Context

Machine Learning with R by Brett Lantz is a book that provides an introduction to machine learning using R. As far as I can tell, Packt Publishing does not make its datasets available online unless you buy the book and create a user account which can be a problem if you are checking the book out from the library or borrowing the book from a friend. All of these datasets are in the public domain but simply needed some cleaning up and recoding to match the format in the book.
Content

## Columns

    age: age of primary beneficiary
    
    sex: insurance contractor gender, female, male
    
    bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
    
    children: Number of children covered by health insurance / Number of dependents
    
    smoker: Smoking
    
    region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
    
    charges: Individual medical costs billed by health insurance



```python
data.isna().sum()
```


    age         0
    sex         0
    bmi         0
    children    0
    smoker      0
    region      0
    charges     0
    dtype: int64



Great! No missing values, let's check the info and description 


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB



```python
data.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>




```python
sns.pairplot(data)
```


![png](https://i.ibb.co/8KYBCLT/output-10-1.png)


Based on pairplot, it seems that age have 3 category of insurance. The BMI seems lack of correlation with charges. I am still not sure that the amount of children really correlated with insurance charges. Let's check out


```python
sns.heatmap(data.corr(), annot=True, cmap="cool")
```


![png](https://i.ibb.co/sq6F1Qg/output-12-1.png)



```python
sns.boxplot(data['charges'])
```


![png](https://i.ibb.co/3TKWynD/output-13-1.png)


We can see that there are many outliers in charges data. I expect that these outliers shows up in comparison to other categorical data

Now let's check categorical feature one by one


```python
sns.boxplot(data['sex'], data['charges'])
```


![png](https://i.ibb.co/r3TQdVx/output-16-1.png)


It seems the amount of charges is quite fair between male and female. But did you see that, it seems we have a lot of outliers here. Let's check the other


```python
sns.boxplot(data['smoker'], data['charges'])
```


![png](https://i.ibb.co/McWg3BR/output-18-1.png)


The smoker is a affecting the amount of insurance charges by it looks. Also there are no outliers towards smoker


```python
data['region'].value_counts()
```


    southeast    364
    northwest    325
    southwest    325
    northeast    324
    Name: region, dtype: int64



Seems balanced, let's check the boxplot


```python
sns.boxplot(data['region'], data['charges'])
```


![png](https://i.ibb.co/gmk8LD8/output-22-1.png)


East part of the region has a better Q3 value huh. It seem something big happens a lot in east part region


```python
sns.boxplot(data['children'], data['charges'])
```


![png](https://i.ibb.co/RCR2MWJ/output-24-1.png)


The amount of children probably does not raelly matter 


```python
from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)
```


```python
sns.heatmap(data.corr(), annot=True, cmap="cool")
```


![png](https://i.ibb.co/Hxj1MVd/output-27-1.png)


WOW. After the text encoding we can see that the correlation between smoker and insurance charges is really huge. Let's focus on smoker for a while


```python
sns.relplot(x="age", y="charges", data=data, hue='smoker');
```


![png](https://i.ibb.co/XbjrPLY/output-29-0.png)


With this visualization, we can understand that the smoker tends to have higher charges in insurances while the distribution of smoker spreads normally around all ages. The charges is also getting higher while you are getting older


```python
sns.boxplot(x="sex", y="charges", data=data, hue='smoker');
```


![png](https://i.ibb.co/N77vF1D/output-31-0.png)


Well we are sure that the insurance charges is not sexist afterall


```python
sns.relplot(x="bmi", y="charges", data=data, hue='smoker');
```


![png](https://i.ibb.co/S5cB6S5/output-33-0.png)


Surprisingly, BMI does not correlated that much towards smoker. I thought smoking have high influences on your weight


```python
sns.boxplot(x="children", y="charges", data=data, hue='smoker');
```


![png](https://i.ibb.co/XDP84nF/output-35-0.png)


Smoker hardly have more than 3 children according to that. Smoking is not good for your children after all


```python
sns.boxplot(x="region", y="charges", data=data, hue='smoker');
```


![png](https://i.ibb.co/f401FnL/output-37-0.png)


Nothing we can see here. Smoker spreads fairly in different region. I almost forgot the continuos values. Let's analyze them one by one


```python
sns.jointplot(x='age', y='charges', data=data[data['age'] <=20])
```


![png](https://i.ibb.co/WkHy7Dq/output-39-1.png)



```python
sns.jointplot(x='age', y='charges', data=data[data['age'] > 40])
```


![png](https://i.ibb.co/19s6FVQ/output-40-1.png)


The more old you are, the more charges on your insurance by it looks. The age distribution seems balanced to me


```python
sns.jointplot(x='bmi', y='charges', data=data[data['bmi'] <30])
```


![png](https://i.ibb.co/YhJGHVD/output-42-1.png)



```python
sns.jointplot(x='bmi', y='charges', data=data[data['bmi'] >=30])
```


![png](https://i.ibb.co/j39CmHk/output-43-1.png)


Eventhough the BMI value is normally distributed around 30 (The line between overweight or not) it seems that we cant rely that much on charge on how good your BMI are. The sure things is that there are more overweight people here

# Regression


```python
x = data.drop(['charges'], axis = 1)
y = data['charges']
```


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)
```


```python
from sklearn.linear_model import LinearRegression

lm1 = LinearRegression()
lm2 = LinearRegression(normalize=True)
```

I'm using 2 different linear regression. First without normalization and second with StandartScaler normalization


```python
from sklearn.model_selection import cross_val_score

print("Linear Regression score without normalization = ", cross_val_score(lm1, x_train, y_train, scoring='neg_mean_squared_error').mean())
print("Linear Regression score with normalization = ", cross_val_score(lm1, x_train, y_train, scoring='neg_mean_squared_error').mean())
```

    Linear Regression score without normalization =  -38555503.39899726
    Linear Regression score with normalization =  -38555503.39899726


Wow, pretty much the same. Now let's try to predict


```python
lm1.fit(x_train, y_train)
prediction = lm1.predict(x_test)
```


```python
from sklearn.metrics import mean_squared_error

print("The mean squared error is ", mean_squared_error(y_test, prediction))
```

    The mean squared error is  30084782.121418368



```python
sns.distplot((prediction**2)-(y_test**2))
```


![png](https://i.ibb.co/dp6J2YB/output-54-1.png)


Looks not really great with linear regression. We try to improve it later with different methods