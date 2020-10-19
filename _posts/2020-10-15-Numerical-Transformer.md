---
title: "Numerical Transformer"
header :
  image: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Preprocessing
  - Numpy
  - Sklearn

---

After rescaling or normalize the data, there is another way to change the distribution of the data by transformation. There are 3 different ways to transform the data by using Power function, Quartile function, and custom function with Numpy.

## Power Transformer

The power transformer is a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to the variability of a variable that is unequal across the range (heteroscedasticity) or situations where normality is desired. The power transform finds the optimal scaling factor in stabilizing variance and minimizing skewness through maximum likelihood estimation. Currently, Sklearn implementation of PowerTransformer supports the **Box-Cox** transform and the **Yeo-Johnson** transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data. For Example 

```python
import pandas as pd
import numpy as np
import seaborn as sns
import quantumrandom

x = []
for i in range(100):
    x.append(quantumrandom.randint(0, 100))
    
df = pd.DataFrame(x)
sns.distplot(df, bins=50)
```

![](https://i.ibb.co/Bn228cX/download-5.png)

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')

sns.distplot(pt.fit_transform(df), bins=50)
```

![](https://i.ibb.co/jw3Fcvn/download-6.png)

## Quantile Transformer

This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is, therefore, a robust pre-processing scheme. The cumulative distribution function of a feature is used to project the original values. Note that this transform is non-linear and may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable. This is also sometimes called as Rank scaler.

```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=3)

sns.distplot(qt.fit_transform(df))
```

![](https://i.ibb.co/m42BrPf/download-7.png)

## Function Transformer

Constructs a transformer from an arbitrary callable. A FunctionTransformer forwards its arguments to a user-defined function or function object and returns the result of this function. This is useful for stateless transformations such as taking the log of frequencies, doing custom scaling, etc.

```python
from sklearn.preprocessing import FunctionTransformer
ft = FunctionTransformer(np.log1p)

sns.distplot(ft.fit_transform(df))
```

![](https://i.ibb.co/VWQVVz9/download-8.png)




