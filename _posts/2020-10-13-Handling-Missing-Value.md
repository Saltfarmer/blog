---
title: "Handling Missing Values"
header :
  teaser: /assets/images/sklearn-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Preprocessing
  - Pandas
  - Sklearn

---

Missing value in your data is pretty common in real life. In fact, the chance that at least one data point is missing increases as the data set size increases. Missing data can occur any number of ways, some of which include the following.

- **Merging of source datasets**: A simple example commonly occurs when two data sets are merged by a sample identifier (ID). If an ID is present in only the first data set, then the merged data will contain missing values for that ID for all of the predictors in the second data set.
- **Equipment Errors**: Any measurement process is vulnerable to random events that prevent data collection. Consider the setting where data are collected in a medical diagnostic lab. Accidental misplacement or damage of a biological sample  would prevent measurements from being made on the sample, thus inducing missing values.
- **Human Errors**: Measurements from human tends to have one or two error. Especially in survey. For example, not all surveyor to perform 100 percent without making any mistake. Small unknown things will be happening. 

Moreover, missing values in the original predictors, regardless of any feature engineering, are intolerable in many kinds of predictive models. Therefore, to utilize predictors or feature engineering techniques, we must first address the missingness in the data. Also, the missingness itself may be an important predictor of the response.

## Types of missing data

> why are these values missing ?  

Sometimes the answer might already be known or could be easily inferred from studying the data. If the data stem from a scientific experiment or clinical study, information from laboratory notebooks or clinical study logs may provide a direct connection to the samples collected or to the patients studied that will reveal why measurements are missing. But for many other data sets, the cause of missing data may not be able to be determined. In cases like this, we need a framework for understanding missing data. This framework will, in turn, lead to appropriate techniques for handling the missing information.

One framework to view missing values is through the lens of the mechanisms of missing data. Three common mechanisms are:

- Structural deficiencies in the data
- Random occurrences, or
- Specific causes.

A structural deficiency can be defined as a missing component of a predictor that was omitted from the data. This type of missingness is often the easiest to resolve once the necessary component is identified. It may be tempting to simply remove this predictor because most of the values are missing. However doing this would throw away valuable predictive information. 

A second reason for missing values is due to random occurrences. Lets we split it to 3 categories

- Missing completely at random (MCAR): The fact that a certain value is missing has nothing to do with its hypothetical value and with the values of other variables. This is the best case situation.
- Missing at random (MAR): Missing at random means that the propensity for a data point to be missing is not related to the missing data, but it is related to some of the observed data. In this scenario, the probability of a missing result depends on the observed data but not on the unobserved data.

A third mechanism of missing data is missingness due to a specific cause (or missing not at random (MNAR)). Worst possible reasons are that the missing value depends on the hypothetical value  or missing value is dependent on some other variable’s value. Therefore, we must make a good effort to understanding the nature of the missing data prior to implementing any of the techniques. 

## Impute or Remove ?

In MCAR and MAR, it is safe to remove the data with missing values depending upon their occurrences, while in MNAR case removing observations with missing values can produce a bias in the model. So we have to be really careful before removing observations. Note that imputation does not necessarily give better results. 

## Removing

Listwise deletion (complete-case analysis) removes all data for an observation that has one or more missing values. Particularly if the missing data is limited to a small number of observations, you eliminate those cases from the analysis. However in most cases, it is often disadvantageous to use listwise deletion. This is because the assumptions of MCAR (Missing Completely at Random) are typically rare to support. As a result, listwise deletion methods produce biased parameters and estimates.

```python
df.dropna(inplace=True)
```

Sometimes you can drop variables if the data is missing for more than 60% observations but only if that variable is insignificant. Having said that, imputation is always a preferred choice over dropping variables

```python
del df.column_name
df.drop('column_name', axis=1, inplace=True)
```

## Imputing with Mean, Median, and Mode

Computing the overall mean, median or mode is a very basic imputation method, it is the only tested function that takes no advantage of the time series characteristics or relationship between the variables. It is very fast, but has clear disadvantages. One disadvantage is that mean imputation reduces variance in the dataset. It is pretty clear for categorical data we could impute with Mode, but what about continuous value ?

Overall, mean is much preferred to use. Median in the other hand had its own advantage. Outliers don’t have such an effect on the median. Therefore, here the median gives a more realistic picture of the data. Let see at the example

```python
num = np.random.randint(1, 100, 45)
null = []
for i in range(5):
    null.append(np.nan)

sample_num = pd.Series(sample)
sample_cat = pd.Series(["a", "b", np.nan, "c", "a", "b", np.nan, "a"])
```

 Then, we check if there is any missing value

```python
print(sample_num.isna().sum())
print(sample_cat.isna().sum())
```

```
5
2
```

Then lets try to fill the missing value

```python
# Fill it with mean
fill_mean = sample_num.fillna(sample_num.mean())
# Fill it with median
fill_median = sample_num.fillna(sample_num.median())
#  Fill it with mode
fill_mode = sample_cat.fillna(sample_cat.mode()[0])

print(fill_mean)
print(fill_median)
print(fill_mode)
```

```
0     55.911111
1     55.911111
2     55.911111
3     55.911111
4     55.911111
5     29.000000
6     88.000000
7     99.000000
8     75.000000
9     91.000000
10    10.000000
11    23.000000
12    23.000000
13    95.000000
14    74.000000
15    79.000000
16     3.000000
17     5.000000
18    64.000000
19    34.000000
20    44.000000
21    10.000000
22    82.000000
23    51.000000
24    80.000000
25    13.000000
26    73.000000
27    92.000000
28    69.000000
29    25.000000
30    93.000000
31    85.000000
32    47.000000
33    99.000000
34    97.000000
35    29.000000
36    17.000000
37    62.000000
38    56.000000
39    83.000000
40    57.000000
41    60.000000
42    45.000000
43    49.000000
44    92.000000
45     8.000000
46    97.000000
47    56.000000
48    13.000000
49    40.000000
dtype: float64
0     57.0
1     57.0
2     57.0
3     57.0
4     57.0
5     29.0
6     88.0
7     99.0
8     75.0
9     91.0
10    10.0
11    23.0
12    23.0
13    95.0
14    74.0
15    79.0
16     3.0
17     5.0
18    64.0
19    34.0
20    44.0
21    10.0
22    82.0
23    51.0
24    80.0
25    13.0
26    73.0
27    92.0
28    69.0
29    25.0
30    93.0
31    85.0
32    47.0
33    99.0
34    97.0
35    29.0
36    17.0
37    62.0
38    56.0
39    83.0
40    57.0
41    60.0
42    45.0
43    49.0
44    92.0
45     8.0
46    97.0
47    56.0
48    13.0
49    40.0
dtype: float64
0    a
1    b
2    a
3    c
4    a
5    b
6    a
7    a
dtype: object
```

You can also use imputer from Scikit-Learn. Sklearn imputer only can do it on DataFrames or 2-D array.

```python
data_sample = pd.DataFrame(data=sample_num)

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# strategy can be changed to "median" and “most_frequent”
imp.fit_transform(data_sample)
```

```
array([[55.91111111],
       [55.91111111],
       [55.91111111],
       [55.91111111],
       [55.91111111],
       [29.        ],
       [88.        ],
       [99.        ],
       [75.        ],
       [91.        ],
       [10.        ],
       [23.        ],
       [23.        ],
       [95.        ],
       [74.        ],
       [79.        ],
       [ 3.        ],
       [ 5.        ],
       [64.        ],
       [34.        ],
       [44.        ],
       [10.        ],
       [82.        ],
       [51.        ],
       [80.        ],
       [13.        ],
       [73.        ],
       [92.        ],
       [69.        ],
       [25.        ],
       [93.        ],
       [85.        ],
       [47.        ],
       [99.        ],
       [97.        ],
       [29.        ],
       [17.        ],
       [62.        ],
       [56.        ],
       [83.        ],
       [57.        ],
       [60.        ],
       [45.        ],
       [49.        ],
       [92.        ],
       [ 8.        ],
       [97.        ],
       [56.        ],
       [13.        ],
       [40.        ]])
```



# Bonus

## Fancy Imputer using KNN

In this method, k-nearest neighbors are chosen based on some distance measure and their average is used as an imputation estimate. The method requires the selection of the number of nearest neighbors, and a distance metric. KNN can predict both discrete attributes (the most frequent value among the k nearest neighbors) and continuous attributes (the mean among the k nearest neighbors)
The distance metric varies according to the type of data:

1. Continuous Data: The commonly used distance metrics for continuous data are Euclidean, Manhattan and Cosine
2. Categorical Data: Hamming distance is generally used in this case. It takes all the categorical attributes and for each, count one if the value is not the same between two points. The Hamming distance is then equal to the number of attributes for which the value was different.

One of the most attractive features of the KNN algorithm is that it is simple to understand and easy to implement. The non-parametric nature of KNN gives it an edge in certain settings where the data may be highly “unusual”.

One of the obvious drawbacks of the KNN algorithm is that it becomes time-consuming when analyzing large datasets because it searches for similar instances through the entire dataset. Furthermore, the accuracy of KNN can be severely degraded with high-dimensional data because there is little difference between the nearest and farthest neighbor.

```python
!pip install fancyimpute

from fancyimpute import KNN    
# Use 5 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).fit_transform(data_sample)
```

Also, `fancyimpute` have others method too

- `SimpleFill`: Replaces missing entries with the mean or median of each column.
- `SoftImpute`: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the [softImpute](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) package for R, which is based on [Spectral Regularization Algorithms for Learning Large Incomplete Matrices](http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf) by Mazumder et. al.
- `IterativeImputer`: A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion. A stub that links to `scikit-learn`'s [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html).
- `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.
- `MatrixFactorization`: Direct factorization of the incomplete matrix into low-rank `U` and `V`, with an L1 sparsity penalty on the elements of `U` and an L2 penalty on the elements of `V`. Solved by gradient descent.
- `NuclearNormMinimization`: Simple implementation of [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf) by Emmanuel Candes and Benjamin Recht using [cvxpy](http://www.cvxpy.org/). Too slow for large matrices.
- `BiScaler`: Iterative estimation of row/column means and standard deviations to get doubly normalized matrix. Not guaranteed to converge but works well in practice. Taken from [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](http://arxiv.org/abs/1410.2596).

## Random Forest Imputation using missingpy

MissForest imputes missing values using Random Forests in an iterative fashion [1]. By default, the imputer begins imputing missing values of the column (which is expected to be a variable) with the smallest number of missing values -- let's call this the candidate column. The first step involves filling any missing values of the remaining, non-candidate, columns with an initial guess, which is the column mean for columns representing numerical variables and the column mode for columns representing categorical variables. Note that the categorical variables need to be explicitly identified during the imputer's `fit()` method call (see API for more information). 

After that, the imputer fits a random forest model with the candidate column as the outcome variable and the remaining columns as the predictors over all rows where the candidate column values are not missing. After the fit, the missing rows of the candidate column are imputed using the prediction from the fitted Random Forest. The rows of the non-candidate columns act as the input data for the fitted model. Following this, the imputer moves on to the next candidate column with the second smallest number of missing values from among the non-candidate columns in the first round. The process repeats itself for each column with a missing value, possibly over multiple iterations or epochs for each column, until the stopping criterion is met. The stopping criterion is governed by the "difference" between the imputed arrays over successive iterations. 

```python
!pip install missingpy
from missingpy import MissForest
imputer = MissForest()
imputer.fit_transform(data_sample)
```

