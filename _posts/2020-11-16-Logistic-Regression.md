---
title: "Logistic Regression"
header :
  image: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Logistic Regression
  - Classification
  - Sklearn

---

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. It is a statistical machine learning algorithm that classifies the data by the outcome of variables on extreme ends and tries makes a logarithmic line that distinguishes between them. Logistic regression is named for the function used at the core of the method, **the logistic function** or **Sigmoid Function**.
$$
Sigmoid = \frac{1}{1-e^{-x}}
$$
where $x = \beta_{0} +\beta_{1} * x$ from Linear regression function

The hypothesis of logistic regression tends it to limit the cost function between 0 and 1. Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which is not possible as per the hypothesis of logistic regression. Logistic regression expected to give us a set of outputs or classes based on probability when we pass the inputs through a prediction function and returns a probability score between 0 and 1 with a line of threshold (mostly it is 0.5).

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.machinelearningplus.com%2Fwp-content%2Fuploads%2F2017%2F09%2Flinear_vs_logistic_regression.jpg&f=1&nofb=1)

Logistic regression is a linear method, but the predictions are transformed using the logistic function. The impact of this is that we can no longer understand the predictions as a linear combination of the inputs as we can with linear regression. The coefficients ($\beta$) of the logistic regression algorithm must be estimated from your training data. This is done using **maximum-likelihood estimation**. Maximum Likelihood Estimation (MLE) is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable. MLE is a common learning algorithm used by a variety of machine learning algorithms, although it does make assumptions about the distribution of your data (more on this when we talk about preparing your data).

The best coefficients would result in a model that would predict a value very close to 1 for the default class and a value very close to 0for the other class. The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data (e.g. probability of 1 if the data is the primary class).

# Cost Function

the cost function represents optimization objective for example we create a cost function and minimize it so that we can develop an accurate model with minimum error. If we try to use the cost function of the linear regression in Logistic Regression then it would be of no use as it would end up being a non-convex function with many local minimums, in which it would be very difficult to minimize the cost value and find the global minimum. 

how do we reduce the cost value ?. Well, this can be done by using **Gradient Descent**. The main goal of Gradient descent is to minimize the cost value. Now to minimize our cost function we need to run the gradient descent function on each parameter. Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do. Well, this action is analogous to calculating the gradient descent, and taking a step is analogous to one iteration of the update to the parameters.

# Prepare Data for Logistic Regression

## Binary Output Variable 

This might be obvious as we have already mentioned it, but logistic regression is intended for binary (two-class) classification problems. It will predict the probability of an instance belonging to the default class, which can be snapped into a 0 or 1 classification.

## Remove Noise

Logistic regression assumes no error in the output variable (y), consider removing outliers and possibly misclassified instances from your training data.

## Gaussian Distribution 

Logistic regression is a linear algorithm (with a non-linear transform on output). It does assume a linear relationship between the input variables with the output. Data transforms of your input variables that better expose this linear relationship can result in a more accurate model. For example, you can use log, root, Box-Cox and other univariate transforms to better expose this relationship.

## Remove Correlated Inputs

Like linear regression, the model can overfit if you have multiple highly-correlated inputs. Consider calculating the pairwise correlations between all inputs and removing highly correlated inputs.

## Fail to Converge

It is possible for the expected likelihood estimation process that learns the coefficients to fail to converge. This can happen if there are many highly correlated inputs in your data or the data is very sparse (e.g. lots of zeros in your input data).