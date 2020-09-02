---
title: "Types of Machine Learning"
header :
  image: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning

---

There are many different types of machine learning system. At a high-level, machine learning is simply the study of teaching a computer program or algorithm how to progressively improve upon a set task that it is given. On the research-side of things, machine learning can be viewed through the lens of theoretical and mathematical modeling of how this process works. However, more practically it is the study of how to build applications that exhibit this iterative improvement. It is useful to classify these broad categories based on 

- Wether or not they are trained with human supervision
- Wether or not they can learn continuously
- Whether they work by simply comparing new data points to known data points or instead detect patterns in the training data and build a predictive model

# Supervised or Unsupervised

Machine Learning systems can be classified according to the amount and type of supervision they get during training. There are 4 major categories.

## 1. Supervised Learning

The main task of supervised learning is to learn a model from labeled training data that allows us to make predictions about unseen or future data. Given data in the form of examples with labels, we can feed a learning algorithm these example-label pairs one by one, allowing the algorithm to predict the label for each example, and giving it feedback as to whether it predicted the right answer or not. Over time, the algorithm will learn to approximate the exact nature of the relationship between examples and their labels. When fully-trained, the supervised learning algorithm will be able to observe a new, never-before-seen example and predict a good label for it.

There are two main types of supervised learning problems: they are classification that involves predicting a class label and regression that involves predicting a numerical value.

### Classification

Supervised learning problem that involves predicting a **class** label. It can be either binary or multi class classification. In binary classification, model predicts either 0 or 1 ; yes or no but in case of multi class classification, model predicts more than one class. The goal here is to predict discrete values belonging to a particular class and evaluate on the basis of accuracy.

**Example** : email spam filtering, we can train a model using a supervised machine learning algorithm on a corpus of labeled email, email that are correctly marked as spam or not spam, to predict whether a new email belongs to either of the two categories.

### Regression

Supervised learning problem that involves predicting a **numerical** label. The goal here is to predict a value as much closer to actual output value as our model can and then evaluation is done by calculating error value. The smaller the error the greater the accuracy of our regression model.

**Example** : predicting the scores of students. If there is a relationship between the time spent studying  for the test and the final scores, we could use it as training data to learn a model that uses the study time to predict the test scores of future students who are planning to take this test.

## 2. Unsupervised Learning

Unsupervised learning is the training of machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. Here the task of machine is to group unsorted information according to similarities, patterns and differences without any prior training of data. 

What makes unsupervised learning such an interesting area is that an overwhelming majority of data in this world is unlabeled. Having intelligent algorithms that can take our terabytes and terabytes of unlabeled data and make sense of it is a huge source of potential profit for many industries. That alone could help boost productivity in a number of fields.

There are 5 main types of unsupervised learning problems: 

- **Clustering** : where you want to discover the inherent groupings in the data. For example, grouping customers by purchasing behavior.
- **Anomaly Detection** : Anomaly detection is where you want looking for unusual pattern in data or removing outliers from a dataset before feeding it to another learning algorithm. For example, Fraud Prevention in online shop.
- **Visualization** : to understand how data is organized and perhaps identify some patterns from complex and unlabelled data. For example, visualized geographic information of Covid-19 patient.
- **Dimensionality Reduction** : to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be very correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called *feature extraction*.
- **Association Rule** : where you want to discover rules that describe large portions of your data. For example, people that buy X also tend to buy Y.



## 3. Semi-Supervised Learning

Some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. This is called **semi-supervised learning**. Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms.

Conceptually, semi-supervised learning can be positioned halfway between unsupervised and supervised learning models. A semi-supervised learning problem starts with a series of labeled data points as well as some data point for which labels are not known. The goal of a semi-supervised model is to classify some of the unlabeled data using the labeled information set.

**Example** : Google Photos. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just one label per person, 4 and it is able to name everyone in every photo, which is useful for searching photos.

## 4. Reinforcement Learning

Reinforcement learning is fairly different when compared to supervised and unsupervised learning. Where we can easily see the relationship between supervised and unsupervised by the presence or absence of labels. The learning system, called an *agent* in this context, can observe the environment, select and perform actions, and get *rewards* in return or *penalties* in the form of negative rewards. It must then learn by itself what is the best strategy, called a *policy*, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

**Example** : OpenAI implement Reinforcement Learning algorithms to learn Dota 2 hero Shadow Fiend in 1vs1 match. The Reinforcement Learning algorithm works by making the AI play by itself in million of simulation. Then the AI will learn every iteration based on the rewards and penalties.

> To be continued tomorrow for next part

