---
title: "Feature Scaling"
header :
  teaser: /assets/images/sklearn_head.jpg
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

Numerical data is already digestible by machine learning or mathematical formula. But it doesn't mean that is no longer need feature engineering or preprocessing. If there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort. So these more significant number starts playing a more decisive role while training the model. The machine learning algorithm works on numbers and does not know what that number represents. A weight of 10 grams and a price of 10 dollars represents completely two different things, which doesn't make any sense for human. 

Another reason why feature scaling is applied is that few algorithms like Neural network gradient descent converge much faster with feature scaling than without it. In many algorithms, when we desire faster convergence, scaling is a "must do" in Neural Network. The machine learning algorithm which sensitive to the relative scales of features,usually uses the numeric values of the features rather than say their rank. Some examples of algorithms where feature scaling matters are :

- **K-nearest neighbors** (KNN) with a Euclidean distance measure is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.
- **K-Means** uses the Euclidean distance measure here feature scaling matters.
- Scaling is critical while performing **Principal Component Analysis(PCA)**. PCA tries to get the features with maximum variance, and the variance is high for high magnitude features and skews the PCA towards high magnitude features.
- We can speed up **gradient descent** in Neural Network by scaling because θ descends quickly on small ranges and slowly on large ranges, and oscillates inefficiently down to the optimum when the variables are very uneven.

Algorithms that do not require normalization/scaling are the ones that **rely on rules**. They would not be affected by any monotonic transformations of the variables. Scaling is a monotonic transformation. Examples of algorithms in this category are all the tree-based algorithms — **CART, Random Forests, Gradient Boosted Decision Trees**. These algorithms utilize rules (series of inequalities) and **do not require normalization**. Algorithms like **Linear Discriminant Analysis (LDA), Naive Bayes is** by design equipped to handle this and give weights to the features accordingly. Performing features scaling in these algorithms may not have much effect.

There are some ways we can do for feature scaling

## Min-Max Scaler

Transform features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set mostly between zero and one. This Scaler shrinks the data within the range of -1 to 1 if there are negative values. We can set the range like [0,1] or [0,5] or [-1,1]. The general formula for [0,1] min-max is
$$
x'={\frac  {x-{\text{min}}(x)}{{\text{max}}(x)-{\text{min}}(x)}}
$$


This Scaler responds well if the standard deviation is small and when a distribution is not Gaussian ( having the shape of a normal curve or a normal distribution). In the other hand, this Scaler is sensitive to outliers.

Let's try it using Boston House Pricing dataset from sklearn

```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_boston
bos = pd.DataFrame(load_boston().target)
sns.distplot(bos)
```

![](https://i.ibb.co/pWsTRyG/download.png)

Then we use min-max scaling on Boston House Pricing

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
minmax = scaler.fit_transform(bos)
sns.distplot(minmax)
```

![](https://i.ibb.co/p3XR9xc/download-1.png)

Before you take any conclusion, lets continue to other methods

## Standarization or Variance Scaling

The Standard Scaler assumes data is normally distributed within each feature and scales them such that the distribution centered around 0, with a standard deviation of 1. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. If data is not normally distributed, this is not the best Scaler to use. The general formula is
$$
x' = \frac{x - \bar{x}}{\sigma}
$$
where $\sigma$ is Standard Deviation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
stan = scaler.fit_transform(bos)
sns.distplot(stan)
```

![](https://i.ibb.co/G0PtWQC/download-2.png)

## Max Abs Scaler

Scale each feature by its maximum absolute value. This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set is 1.0. It does not shift/center the data and thus does not destroy any sparsity (the condition of not having enough of something). On positive-only data, this Scaler behaves similarly to Min Max Scaler and, therefore, also suffers from the presence of significant **outliers**. The general formula is
$$
x' = \frac{x}{|\max{x}|}
$$

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
maxabs = scaler.fit_transform(bos)
sns.distplot(maxabs)
```

![](https://i.ibb.co/xX1kQNC/download-3.png)

## Robust Scaler

As the name suggests, this Scaler is robust to outliers. If our data contains many outliers, scaling using the mean and standard deviation of the data won’t work well. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). The centering and scaling statistics of this Scaler are based on percentiles and are therefore not influenced by a few numbers of huge marginal outliers. Note that the outliers themselves are still present in the transformed data. If a separate outlier clipping is desirable, a non-linear transformation is required.

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
maxabs = scaler.fit_transform(bos)
sns.distplot(maxabs)
```

![](https://i.ibb.co/F7TgB90/download-4.png)

## Conclusion

Feature scaling wont change your data distribution. I think you need visualize your data before and after transformations. While we can understand generally what each scaler is *supposed* to do, each dataset is different; and things like outliers can make a difference in how these scalers perform. I think selecting a scaler and/or standardizer also depends on which algorithm you plan to use, since some algorithms have specific requirement