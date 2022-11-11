---
title: "Feature Selection"
header :
  teaser: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Preprocessing
  - Sklearn

---

Most data nowadays is huge and massive. Dataset often comes with many irrelevant features that do not contribute much to the accuracy of your predictive model. Imagine a music with too many instrument. Mostly, if the music is not planned properly, you wont enjoy listen to it right ? Just too noisy. Similarly, even the datasets encounter noise and its crucial to remove them for better model optimization. That’s where **feature selection** comes in action. Theoretically, **feature selection**

- ***reduces overfitting\*** ‘ The Curse of Dimensionality’ — If your dataset has more features/columns than samples, the model will be prone to overfitting. By removing irrelevant data/noise, the model gets to focus on essential features, leading to more generalization.
- ***simplifies models\*** — Dimensionality adds many layers to a model, making it needlessly complicated. Overengineering is fun but they may not be better than their simpler counterparts. Simpler models are easier to interpret and debug.
- ***reduces training time\*** — Lesser features/dimensions reduces the computation speed, speeding up model training.

Often, feature selection and dimensionality reduction are used interchangeably. However, there is an important difference between them. Feature selection <u>yields a subset of features</u> from the original set of features, which are the best representatives of the data. While dimensionality reduction is the <u>introduction of a new feature space</u> where the original features are represented. It basically transforms the feature space to a lower dimension, keeping the original features intact. This is done by either combining or excluding a few features. To sum up, you can consider feature selection as a part of dimensionality reduction.

There are various methods that could be used to perform Feature Selection, of which they fall into one of 3 categories. Each method has its own advantages and disadvantages. The categories are described in Guyon & Elisseeff (2003) as follows:

- **Filtering Methods**- Select subsets of variables as a pre-processing step, independently of the chosen predictor.
- **Wrapper Methods** - Utilize the learning machine of interest as a black box to score subsets of variables according to their predictive power.
- **Embedded Methods** - Perform variable selection in the process of training and are usually specific to given learning machines.

# Filtering Methods

Filter methods use univariate statistics to evaluate whether there is a statistically significant relationship from each input feature to the target feature. The features that provide the highest confidence are the features that we keep for our final model.  With filtering methods, we primarily apply a statistical measure that suits our data to assign each feature column a calculated score. Based on that score, it will be decided whether that feature will be kept or removed from our predictive model. These methods are computationally inexpensive and are best for eliminating redundant irrelevant features. However, one downside is that they don't take feature correlations into consideration since they work independently on each feature.

***Advantages***

- Robust against overfitting (that would introduce bias)
- Much faster than wrapper methods

***Disadvantages***

- Does not consider interactions between other features
- Does not consider the model being employed

Moreover, we have **Univariate filter** methods that work on ranking a single feature and **Multivariate filter** methods that evaluate the entire feature space. Let's explore the most notable filter methods of feature selection.

## **Missing Values Ratio**

Back on [Handling missing value](https://saltfarmer.github.io/blog/python/Handling-Missing-Value/), data columns with too many missing values won't be of much value. Theoretically, <u>25–30%</u> is the acceptable threshold of missing values, beyond which we should drop those features from the analysis. If you have the domain knowledge, it's always better to make an educated guess if the feature is crucial to the model. But sometimes you can try to impute missing values if the amount of missing value is tolerable.

## Variance Threshold

Features in which identical value occupies the majority of the samples are said to have zero variance. Such features carrying little information will not affect the target variable and can be dropped. You can adjust the threshold value, default is 0, i.e remove the features that have the same value in all samples. For quasi-constant features, that have the same value for a very large subset, use threshold as 0.01. In other words, drop the column where 99% of the values are similar.

```python
import pandas as pd
from sklearn.datasets import load_boston

bos = load_boston()
df = pd.DataFrame(data = bos.data,columns= bos.feature_names)

df["target"] = bos.target
df
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>391.99</td>
      <td>9.67</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.04527</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.120</td>
      <td>76.7</td>
      <td>2.2875</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>9.08</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>503</th>
      <td>0.06076</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.976</td>
      <td>91.0</td>
      <td>2.1675</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>5.64</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>504</th>
      <td>0.10959</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.794</td>
      <td>89.3</td>
      <td>2.3889</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>393.45</td>
      <td>6.48</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.04741</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.030</td>
      <td>80.8</td>
      <td>2.5050</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>7.88</td>
      <td>11.9</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 14 columns</p>



```python
df.var()
```


    CRIM          73.986578
    ZN           543.936814
    INDUS         47.064442
    CHAS           0.064513
    NOX            0.013428
    RM             0.493671
    AGE          792.358399
    DIS            4.434015
    RAD           75.816366
    TAX        28404.759488
    PTRATIO        4.686989
    B           8334.752263
    LSTAT         50.994760
    target        84.586724
    dtype: float64



```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=1.0)
df_vt = selector.fit_transform(df)
```



## Correlation Coefficient

Two independent features are highly correlated if they have a strong relationship with each other and move in a similar direction. In that case, you don't need two similar features to be fed to the model, if one can suffice. It centrally takes into consideration the fitted line, slope of the fitted line and the quality of the fit. There are various approaches for calculating correlation coefficients. 

```python
pearson = df.corr(method='pearson')['target']
# You can choose correlation method between Pearson, Spearman, Kendall
pearson
```


    CRIM      -0.388305
    ZN         0.360445
    INDUS     -0.483725
    CHAS       0.175260
    NOX       -0.427321
    RM         0.695360
    AGE       -0.376955
    DIS        0.249929
    RAD       -0.381626
    TAX       -0.468536
    PTRATIO   -0.507787
    B          0.333461
    LSTAT     -0.737663
    target     1.000000
    Name: target, dtype: float64



```python
abs_corr = abs(pearson)

# random threshold for features to keep
relevant_features = abs_corr[abs_corr>0.4]
relevant_features
```


    INDUS      0.483725
    NOX        0.427321
    RM         0.695360
    TAX        0.468536
    PTRATIO    0.507787
    LSTAT      0.737663
    target     1.000000
    Name: target, dtype: float64



```python
pearson_df = df[relevant_features.index]
pearson_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>5.33</td>
      <td>36.2</td>
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
    </tr>
    <tr>
      <th>501</th>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>9.67</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.04527</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.573</td>
      <td>6.120</td>
      <td>76.7</td>
      <td>2.2875</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>9.08</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>503</th>
      <td>0.06076</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.573</td>
      <td>6.976</td>
      <td>91.0</td>
      <td>2.1675</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>504</th>
      <td>0.10959</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.573</td>
      <td>6.794</td>
      <td>89.3</td>
      <td>2.3889</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>6.48</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.04741</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.573</td>
      <td>6.030</td>
      <td>80.8</td>
      <td>2.5050</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>7.88</td>
      <td>11.9</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 11 columns</p>



### Pearson Correlation (Continuous Data)

Pearson correlation is a parametric statistical test that measures the similarity between two variables. It means that this test assumes that the observed data follows some distribution pattern(e.g. normal, gaussian). Its coefficient values ranges between **-1(**negative correlation) to **1(**positive correlation) indicating how well the data fits the model. 

### Spearman Correlation (Continuous and Ordinal Data)

Spearman correlation is a non-parametric statistical test that works similar to Pearson, however, it does not make any assumptions about the data. Denoted by the symbol rho (-1<**ρ**<1**),** this test can be applied for both ordinal and continuous data that has failed the assumptions for conducting Pearson's correlation. For newbies, ordinal data is categorical data but with a slight nuance of ranking/ordering (e.g low, medium and high). An important assumption to be noted here is that there should be a monotonic relationship between the variables, i.e. variables increase in value together or if one increases, the other one decreases.

### Kendall Correlation (Ordinal/Discrete Data)

Kendall correlation coefficient is similar to Spearman correlation, this coefficient compares the number of concordant and discordant pairs of data.

> *Let's say we have a pair of observations (xᵢ, yᵢ), (xⱼ, yⱼ), with i < j, they are:*
> *** concordant if either (xᵢ > xⱼ and yᵢ > yⱼ) or (xᵢ < xⱼ and yᵢ < yⱼ)
> *** discordant if either (xᵢ < xⱼ and yᵢ > yⱼ) or (xᵢ > xⱼ and yᵢ < yⱼ)
> *** neither if there’s a tie in **x** (xᵢ = xⱼ) or a tie in **y** (yᵢ = yⱼ)

Denoted with the Greek letter tau (**τ**), this coefficient varies between -1 to 1 and is based on the difference in the counts of concordant and discordant pairs relative to the number of x-y pairs.

## Chi Square (Categorical Data)

In this method, we calculate the chi-square metric between the target and the numerical variable and only select the variable with the maximum chi-squared values. Chi-Square tests come in two variations - one that evaluates the **goodness-of-fit** and the other one where we will be focusing on is the **test of independence**. Primarily, it compares the observed data to a model that distributes the data according to the **expectation** that the variables are independent. Then, you basically need to check where the observed data doesn’t fit the model. If there are too many data points/outliers, there is a huge possibility that the variables are dependent, proving that the null hypothesis is incorrect!

## **Analysis of Variance (ANOVA)**

ANOVA is primarily an **extension of a t-test**. With a t-test, you can study only two groups but with ANOVA you need at least three groups to see if there’s a difference in means and determine if they came from the same population.

> It assumes Hypothesis as
> H0: Means of all groups are equal.
> H1: At least one mean of the groups are different.

Let’s say from our automobile dataset, we use a feature ‘fuel-type’ that has 2 groups/levels — ‘diesel’ and ‘gas’. So, our goal would be to determine if these two groups are statistically different by calculating whether the means of the groups are different from the overall mean of the independent variable i.e ‘fuel-type’. ANOVA uses F-Test for statistical significance, which is the ratio of the **variance between groups** to the **variance within groups** and the larger this number is, the more likely it is that the means of the groups really *are* different, and that you should reject the null hypothesis.

## Mutual Information (both regression & classification)

The mutual information measures the contribution of a variable towards another variable. In other words, how much will the target variable be impacted if we remove or add the feature? MI is 0 if both the variables are independent and ranges between 0 –1 if X is deterministic of Y. MI is primarily the entropy of X, which measures or quantifies the amount of information obtained about one random variable, through the other random variable.

## Using Chi2, Anova, and Mutual Information as Feature Selection in Scikit-Learn

There are different ways to selecting feature. Based on Scikit-Learn `sklearn.feature_selection`, you have 5 different mode.

1. ‘percentile’ : Percentage of how many feature you want to select.
2. ‘k_best’ : The $k$ amount of feature you want to select.
3. ‘fpr’ : Select features based on a false positive rate test ($FP/({FP+TN})$).
4. ‘fdr’ : Select features based on an estimated false discovery rate ($FP/({FP+TP})$).
5. ‘fwe’ : Select features based on family-wise error rate.

```python
df_target = df['target']
df_data = df.drop(columns=['target'])

from sklearn.feature_selection import GenericUnivariateSelect, chi2

transformer = GenericUnivariateSelect(chi2, mode='k_best', param=10)
df_new = transformer.fit_transform(df_data, df_target.astype('int'))
df_new.shape
```

```
(506, 10)
```



# Wrapping Methods

In Wrapper methods, we primarily choose a subset of features and train them using a machine learning algorithm. Based on the inferences from this model, we employ a search strategy to look through the space of possible feature subsets and decide which feature to add or remove for the next model development. This loop continues until the model performance no longer changes with the desired count of features*(k_features)*.

The downside is, it becomes computationally expensive as the features increase, but on the good side, it takes care of the interactions between the features, ultimately finding the optimal subset of features for your model with the lowest possible error.

***Advantages***

- Able to detect the interactions that take place between features
- Often results in better predictive accuracy than filter methods
- Finds the optimal feature subset

***Disadvantages***

- Computationally expensive
- Prone to overfitting

## Sequential Feature Selection

A greedy search algorithm, this comes in two variants- **Sequential Forward Selection** (SFS) and **Sequential Backward Selection** (SBS). It basically starts with a null set of features and then looks for a feature that **minimizes the cost function**. Once the feature is found, it gets added to the feature subset and in the same way one by one, it finds the right set of features to build an optimal model. That's how SFS works. With Sequential Backward Feature Selection, it takes a totally opposite route. It starts with all the features and iteratively removes one by one feature depending on the performance. Both algorithms have the same goal of attaining the lowest cost model.

> The main limitation of SFS is that it is *unable to remove features* that become non-useful after the addition of other features. The main limitation of SBS is its *inability to reevaluate* the usefulness of a feature after it has been discarded.

## Recursive Feature Selection

The goal of recursive feature elimination (RFE) is to select features by **recursively considering smaller and smaller sets of features.** First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a `*coef_*` attribute or through a `*feature_importances_*` attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

rfe = RFE(lin_reg, 10)
df_new1 = rfe.fit_transform(df_data, df_target.astype('int'))
df_new1.shape
```

```
(506, 10)
```



# Embedded Methods

These methods combine the functionalities of both Filter and Wrapper methods. The upside is that they perform feature selection during the process of training which is why they are called embedded! The computational speed is as good as of filter methods and of course better accuracy, making it a win-win model!

## Lasso Models

Before diving into L1, let's understand a bit about regularization. Primarily, it is a technique used to reduce overfitting to highly complex models. We add a penalty term to the cost function so that as the model complexity increases the cost function increases by a huge value. Coming back to LASSO (Least Absolute Shrinkage and Selection Operator) Regularization, what you need to understand here is that it comes with a parameter, **‘alpha’** and the higher the alpha is, the more feature coefficients of least important features are shrunk to zero. Eventually, we get a much simple model with the same or better accuracy!

However, in cases where a certain feature is important, you can try Ridge regularization (L2) or Elastic Net (a combination of L1 & L2), wherein instead of dropping it completely, it reduces the feature weightage.

## Tree Models

One of the most popular and accurate machine learning algorithms, random forests are an ensemble of randomized decision trees. An individual tree won't contain all the features and samples. The reason why we use these for feature selection is the way decision trees are constructed! What we mean by that is, during the process of tree building, it uses several feature selection methods that are built into it. Starting from the root, the function used to create the tree tries all possible splits by making conditional comparisons at each step and chooses the one that splits the data into the most homogenous groups (most pure). The importance of each feature is derived by how “pure” each of the sets is.

Using **Gini impurity** for classification and variance for regression, we can identify the features that would lead to an optimal model. The same concept can be applied to CART (Classification and Regression Trees) and boosting tree algorithms as well.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

selector2 = SelectFromModel(estimator=dt, max_features=10)
df_new2 = selector2.fit_transform(df_data, df_target.astype('int'))
df_new2.shape
```

```
(506, 6)
```

