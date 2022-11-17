---
title: "Categorical Encoding"
header :
  teaser: /assets/images/sklearn-head.png
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Preprocessing
  - Sklearn

---

Practically, in real dataset, the dataset contain categorical value. So what is the difference between casual string value and categorical value ? Well, sometimes categorical value contain numerical value. Also, categorical value is limited to some numbers, not continuous. Many machine learning algorithms can support categorical values without further manipulation but there are many more algorithms that do not. This means that categorical data must be encoded to numbers before we can use it to fit and evaluate a model. There are many ways to encode categorical variables for modeling, although the three most common are as follows:

1. **Label Encoder**: Where each unique label is mapped to an integer.
2. **One Hot Encoder**: Where each label is mapped to a binary vector.
3. **Ordinal Encoder**: Where a distributed representation of the categories is learned.

Before we start, let practice with exercise dataset from Seaborn

```python
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.DataFrame(sns.load_dataset("exercise"))
```


```python
data.dtypes
```


    Unnamed: 0       int64
    id               int64
    diet          category
    pulse            int64
    time          category
    kind          category
    dtype: object




```python
data['diet'].value_counts()
```


    low fat    45
    no fat     45
    Name: diet, dtype: int64




```python
data['time'].value_counts()
```


    30 min    30
    15 min    30
    1 min     30
    Name: time, dtype: int64




```python
data['kind'].value_counts()
```


    running    30
    walking    30
    rest       30
    Name: kind, dtype: int64



## Label Encoder

A Label encoding involves mapping each unique label to an integer value. Encode target labels with value between 0 and $n-1$. This transformer should be used to encode target values `y`, and not the input `X`.This type of encoding is really only appropriate if there is a <u>known continous relationship between the categories</u> (like continous scale grouping value).

```python
from sklearn.preprocessing import LabelEncoder

lencoder = LabelEncoder()
data['dietencode'] = lencoder.fit_transform(data['diet'])
data[['diet', 'dietencode']]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dietencode</th>
      <th>diet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>low fat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>low fat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>low fat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>low fat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>low fat</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1</td>
      <td>no fat</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1</td>
      <td>no fat</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1</td>
      <td>no fat</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1</td>
      <td>no fat</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1</td>
      <td>no fat</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 2 columns</p>



## One Hot Encoder

A one hot encoding is appropriate for categorical data where <u>no relationship exists</u> (like country, name types etc) between categories. It involves representing each categorical variable with a binary vector that has one element for each unique label and marking the class label with a 1 and all other elements 0. The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the `sparse` parameter). By default, the encoder derives the categories based on the unique values in each feature. Alternatively, you can also specify the `categories` manually.

```python
from sklearn.preprocessing import OneHotEncoder

hotencoder = OneHotEncoder()
data.join(pd.DataFrame(data=hotencoder.fit_transform(data[['diet', 'time']]).toarray(), columns=hotencoder.get_feature_names())) 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>diet</th>
      <th>pulse</th>
      <th>time</th>
      <th>kind</th>
      <th>dietencode</th>
      <th>x0_low fat</th>
      <th>x0_no fat</th>
      <th>x1_1 min</th>
      <th>x1_15 min</th>
      <th>x1_30 min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>low fat</td>
      <td>85</td>
      <td>1 min</td>
      <td>rest</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>low fat</td>
      <td>85</td>
      <td>15 min</td>
      <td>rest</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>low fat</td>
      <td>88</td>
      <td>30 min</td>
      <td>rest</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>low fat</td>
      <td>90</td>
      <td>1 min</td>
      <td>rest</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>low fat</td>
      <td>92</td>
      <td>15 min</td>
      <td>rest</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>85</td>
      <td>29</td>
      <td>no fat</td>
      <td>135</td>
      <td>15 min</td>
      <td>running</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>86</td>
      <td>29</td>
      <td>no fat</td>
      <td>130</td>
      <td>30 min</td>
      <td>running</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>87</td>
      <td>30</td>
      <td>no fat</td>
      <td>99</td>
      <td>1 min</td>
      <td>running</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>88</td>
      <td>30</td>
      <td>no fat</td>
      <td>111</td>
      <td>15 min</td>
      <td>running</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>89</td>
      <td>30</td>
      <td>no fat</td>
      <td>150</td>
      <td>30 min</td>
      <td>running</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 12 columns</p>



## Ordinal Encode

An ordinal encoding involves mapping each unique label to an integer value. As such, it is sometimes referred to simply as an integer encoding. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in the dataset, and ideally, this should be harnessed when preparing the data. In this case, we will ignore any possible existing ordinal relationship and assume all variables are categorical. It can still be helpful to use an ordinal encoding, at least as a point of reference with other encoding schemes.

```python
from sklearn.preprocessing import OrdinalEncoder

oencoder = OrdinalEncoder()
data.join(pd.DataFrame(data=oencoder.fit_transform(data[['diet', 'time']]))) 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>diet</th>
      <th>pulse</th>
      <th>time</th>
      <th>kind</th>
      <th>dietencode</th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>low fat</td>
      <td>85</td>
      <td>1 min</td>
      <td>rest</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>low fat</td>
      <td>85</td>
      <td>15 min</td>
      <td>rest</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>low fat</td>
      <td>88</td>
      <td>30 min</td>
      <td>rest</td>
      <td>0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>low fat</td>
      <td>90</td>
      <td>1 min</td>
      <td>rest</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>low fat</td>
      <td>92</td>
      <td>15 min</td>
      <td>rest</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>85</td>
      <td>29</td>
      <td>no fat</td>
      <td>135</td>
      <td>15 min</td>
      <td>running</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>86</td>
      <td>29</td>
      <td>no fat</td>
      <td>130</td>
      <td>30 min</td>
      <td>running</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>87</td>
      <td>30</td>
      <td>no fat</td>
      <td>99</td>
      <td>1 min</td>
      <td>running</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>88</td>
      <td>30</td>
      <td>no fat</td>
      <td>111</td>
      <td>15 min</td>
      <td>running</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>89</td>
      <td>30</td>
      <td>no fat</td>
      <td>150</td>
      <td>30 min</td>
      <td>running</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 9 columns</p>



## Common Questions

This section lists some common questions and answers when encoding categorical data.

#### **Q. What if I have a mixture of numeric and categorical data?**

Or, what if I have a mixture of categorical and ordinal data?

You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all of the prepared variables back together into a single array for fitting or evaluating the model.

#### **Q. What if I have hundreds of categories?**

Or, what if I concatenate many one hot encoded vectors to create a many thousand element input vector?

You can use a one hot encoding up to thousands and tens of thousands of categories. Also, having large vectors as input sounds intimidating, but the models can generally handle it.

Try an embedding; it offers the benefit of a smaller vector space (a projection) and the representation can have more meaning.

#### Q. What encoding technique is the best?

Unknown