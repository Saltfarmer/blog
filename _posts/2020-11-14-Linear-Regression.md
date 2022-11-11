---
title: "Linear Regression"
header :
  teaser: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Linear Regression
  - Regression
  - Sklearn

---

Linear regression is useful for finding relationship between two continuous variables. Linear regression is a **linear model**, a model that creates a linear relationship between the input variables $x$ and the single output variable $y$. More specifically, that y can be calculated from a linear combination of the input variables $x$. Different techniques can be used to prepare or train the linear regression equation from data, the most common of which is called **Ordinary Least Squares**. 

When there is a single input variable $x$, the method is referred to as **simple linear regression**. When there are multiple input variables, literature from statistics often refers to the method as **multiple linear regression**. For example, in a simple regression problem (a single input and output), the form of the model would be:
$$
y = \beta_0 + \beta_1*x
$$
The linear equation assigns one scale factor to each input value or column, called a coefficient and represented by $\beta$. One additional coefficient is also added, giving the line an additional degree of freedom (e.g. moving up and down on a two-dimensional plot) and is often called the intercept or the bias coefficient. In higher dimensions when we have more than one input, the line is called a plane or a **hyper-plane**. The representation therefore is the form of the equation and the specific values used for the coefficients.

In simple terms

- If $\beta_1$ > 0, then x(predictor) and y(target) have a positive relationship. That is increase in x will increase y.
- If $\beta_1$ < 0, then x(predictor) and y(target) have a negative relationship. That is increase in x will decrease y.
- If the model does not include x=0, then the prediction will become meaningless with only $\beta_0$. 
- If the model includes value 0, then $\beta_0$ will be the average of all predicted values when x=0. But, setting zero for all the predictor variables is often impossible.
- The value of $\beta_0$ guarantee that residual have mean zero. If there is no $\beta_0$ term, then regression will be forced to pass over the origin. Both the regression co-efficient and prediction will be biased.

![](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimage.slideserve.com%2F523137%2Flinear-regression36-l.jpg&f=1&nofb=1)

# Linear Regression Model

## 1. Simple Linear Regression

Simple linear regression is a single input model that we can use statistics to estimate the coefficients. This requires that you calculate statistical properties from the data such as means, standard deviations, correlations and covariance. All of the data must be available to traverse and calculate statistics.

## 2. Ordinary Least Square

When we have more than one input we can use Ordinary Least Squares to estimate the values of the coefficients. It procedure seeks to minimize the sum of the squared residuals. This means that given a regression line through the data we calculate the distance from each data point to the regression line, square it, and sum all of the squared errors together. This is the quantity that ordinary least squares seeks to minimize. This approach treats the data as a matrix and uses linear algebra operations to estimate the optimal values for the coefficients. It means that all of the data must be available and you must have enough memory to fit the data and perform matrix operations.

## 3. Gradient Descent

When there are one or more inputs you can use a process of optimizing the values of the coefficients by iteratively minimizing the error of the model on your training data. This operation is called Gradient Descent and works by starting with random values for each coefficient. The sum of the squared errors are calculated for each pair of input and output values. **A learning rate** $\alpha$ is used as a scale factor and the coefficients are updated in the direction towards minimizing the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible. When using this method, you must select a learning rate parameter that determines the size of the improvement step to take on each iteration of the procedure.

## 4. Regularization

There are extensions of the training of the linear model called regularization methods. These seek to both minimize the sum of the squared error of the model on the training data (using ordinary least squares) but also to reduce the complexity of the model (like the number or absolute size of the sum of all coefficients in the model). Two popular examples of regularization procedures for linear regression are:

- **Lasso Regression**: where Ordinary Least Squares is modified to also minimize the absolute sum of the coefficients (called L1 regularization).
- **Ridge Regression**: where Ordinary Least Squares is modified to also minimize the squared absolute sum of the coefficients (called L2 regularization).

# Preparing Data for Linear Regression

## Linear Assumption

Linear regression assumes that the relationship between your input and output is linear. It does not support anything else. This may be obvious, but it is good to remember when you have a lot of attributes. You may need to transform data to make the relationship linear (e.g. log transform for an exponential relationship).

## Remove Noise 

Linear regression assumes that your input and output variables are not noisy. Consider using data cleaning operations that let you better expose and clarify the signal in your data. This is most important for the output variable and you want to remove outliers in the output variable (y) if possible.

## Remove Collinearity

Linear regression will over-fit your data when you have highly correlated input variables. Consider calculating pairwise correlations for your input data and removing the most correlated.

## Gaussian Distributions

Linear regression will make more reliable predictions if your input and output variables have a Gaussian distribution. You may get some benefit using transforms (e.g. log or BoxCox) on you variables to make their distribution more Gaussian looking.

## Rescale Inputs 

Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.