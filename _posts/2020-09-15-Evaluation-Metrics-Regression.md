---
title: "Evaluation Metric Regression"
header :
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Scoring
  - Regression

---

Regression task is the prediction of the state of an outcome variable at a particular timepoint with the help of other correlated independent variables. The regression task, unlike the classification task, outputs **continuous** value within a given range. Regression is the task of predicting continuous values by learning from various independent features. The various metrics used to evaluate the results of the prediction are.

## Mean Square Error

MSE or Mean Squared Error simply average of the squared difference between the target value and the value predicted by the regression model. As it squares the differences, it penalizes even a small error which leads to over estimation of how bad the model is. It is preferred more than other metrics because it is differentiable and hence can be optimized better.


$$
{MSE} ={\frac {\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}}{n}}
$$


Where $Y_i - \hat Y_i$ is the difference between actual and predicted data and $n$ is the length of data

## Root Mean Square Error

RMSE or root mean square is preferred more in some cases because the errors are first squared before averaging which poses a **high penalty** on large errors. This implies that RMSE is useful when large errors are undesired.


$$
{RMSE} = {\sqrt{\frac {\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}}{n}}}
$$


Pretty much the same as before but with square root.

## Mean Absolute Error

MAE or Mean Absolute Error is the absolute difference between the target value and the value predicted by the model. MAE is more robust to outliers and does not penalize the errors as extremely as MSE. MAE is a linear score which means all the individual differences are weighted equally. It is not suitable for applications where you want to **pay more attention** to the outliers.


$$
{MAE} = {\frac {\sum _{i=1}^{n}|Y_{i}-{\hat {Y_{i}}}|{}}{n}}
$$


The difference from MSE is instead square the difference between actual and predicted data, you make the difference to be absolute number (always positive).

## R Square Error

Adjusted R² depicts the same meaning as R² but is an improvement of it. R² suffers from the problem that the scores improve on increasing terms even though the model is not improving which may misguide the researcher. Adjusted R² is always lower than R² as it adjusts for the increasing predictors and only shows improvement if there is a real improvement.


$$
R^2 = 1 - \frac{MSE(model)}{MSE(total)}
$$



or we can say


$$
R^2 = 1 - \frac{\sum_{i=1}^n Y_i - f_i}{\sum_{i=1}^n Y_i - \hat{Y_i}}
$$


where $f_i$ is the predicted value of $Y_i$. What is the different then ? $f_i$ domain only include $\sum_{i=1}^{n}$ where $n$ is the length of predicted value.

## Why is R²  can be negative?

There is a misconception among people that R² score ranges from 0 to 1 but actually it ranges from -∞ to 1. Due to this misconception, they are sometimes scared why the R² is negative which is not a possibility according to them.

The main reasons for R² to be negative are the following:

1. One of the main reason for R² to be negative is that the chosen model does not follow the trend of the data causing the R² to be negative. This causes the MSE of the chosen model(numerator) to be more than the MSE for constant baseline(denominator) resulting in negative R².
2. Maybe their area a large number of outliers in the data that causes the MSE of the model to be more than MSE of the baseline causing the R² to be negative(i.e the numerator is greater than the denominator).
3. Sometimes while coding the regression algorithm, the researcher might forget to add the intercept to the regressor which will also lead to R² being negative. This is because, without the benefit of an intercept, the regression could do worse than the sample mean(baseline) in terms of tracking the dependent variable (i.e., the numerator could be greater than the denominator). However, most of the standard machine learning library like scikit-learn include the intercept by default but if you are using stats-model library then you have to add the intercept manually.

``