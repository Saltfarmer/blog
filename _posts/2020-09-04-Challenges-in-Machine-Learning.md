---
title: "Challenges in Machine Learning"
header :
  teaser: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning

---

At a some level, running machine learning systems at scale is challenging for several reasons. The systems issues are often misunderstood. Although best practices are emerging quickly, they’re extremely decentralized. You need to research a lot of stuff from books, posts, conference talks, and github repositories. In this post, I would like to share into some challenges to running machine learning systems.

## Collecting Data

For beginners to experiment with machine learning, they can easily find data from Kaggle, UCI ML Repository, etc. But to implement real world scenarios, you need to collect the data through web-scraping, web-API or data from clients. Once the data is collected, we need to organize the data and store it in the database. This requires another knowledge in Big data or help from Big Data engineer.

If you plan to use personal data, you will probably face different challenges. People around the world are more aware of the importance of protecting their privacy. Not a lot of people may be unwilling to share them with you.

## The amount of training data

Machine Learning need to take a lot of data for most of the machine learning algorithms to work properly. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples.

## Non-representative training data

In order to machine learning algorithm works well, it is crucial that your training data be representative of the new cases you want to generalize to. This is correct whether you use instance-based learning or model-based learning.

By using a non representative training set, trained model is unlikely to make accurate predictions.

It is crucial to use a training set that is representative of the cases you want to train the data. If the sample is too small, you will have **sampling noise** (i.e., nonrepresentative data as a result of chance), but even very large samples can be non representative if the sampling method is flawed. This is called **sampling bias**.

## Poor Quality of Data

In reality, we don’t directly start training the model, analyzing data is the most important step. But the data we collected might not be ready for training, some samples are abnormal from others having outliers or missing values for instance. The truth is, most data scientists spend a significant part of their time doing just that.

If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually. Then, if some instances are missing a few features you must decide whether you want to ignore this attribute all of it, ignore these instances, fill in the missing values with the median or mean, or train one model with the feature and drop the feature.

## Organizing Experiment

Machine learning is an iterative process. You need to experiment with multiple combinations of data, learning algorithms and model parameters, and keep track of the impact these changes have on predictive performance. Over time this iterative experimentation can result in thousands of model training runs and model versions. This makes it hard to track the best performing models and their input configurations.

## Tuning Model Training

Tuning model training jobs is a simple matter when you’re training models in an interactive programming environment such as Jupyter notebooks. If you’re running the code manually, you’ll be met with exceptions and stack traces if training errors out. You’ll also be able to visualize learning curves and other metrics if training succeeds. These diagnostics can point out if issues like overfitting or vanishing gradients have occurred.

But tuning model training is really difficult and almost near impossible when model training are running as automated batch processes on a recurring schedule. While job schedulers will rerun jobs that explicitly fail, they can’t easily check for issues like overfitting and vanishing gradients unless you code up custom solutions. And since your goal as a data science team is to deploy more and more models, this problem is only going to get worse.

## Irrelevant Features

If the training data contains a large number of irrelevant features and enough relevant features, the machine learning system will not give the results as expected. A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called feature **engineering**, involves :

- **Feature selection**: selecting the most useful features to train on among existing features.
- **Feature extraction**: combining existing features to produce a more useful one
- Creating new features by gathering new data.

## Overfitting

Imagine you breakup with your boyfriend or girlfriend. Then you said all man or woman are jerk. Overgeneralizing is something that we humans do all too often, and unfortunately machines can fall into the same trap if we are not careful. In Machine Learning this is called **overfitting**. it means that the model performs well on the training data, but it does not generalize well.

There is a way to avoid overfitting. Constraining a model to make it simpler and reduce the risk of overfitting is called **regularization**. You want to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well. 

The amount of regularization to apply during learning can be controlled by a **hyperparameter**. A hyperparameter is a parameter of a learning algorithm. it must be set prior to training and remains constant during training. If you set the regularization hyperparameter to a very large value, you will get an almost flat model. The learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good result.

At the point when the model is excessively unpredictable when compared to the noisiness of the training dataset, overfitting occurs. We can avoid it by:

1. Gathering more training data.
2. Selecting a model with fewer features, a higher degree polynomial model is not preferred compared to the linear model.
3. Fix data errors, remove the outliers, and reduce the number of instances in the training set.

## Underfitting

Underfitting which is opposite to overfitting generally occurs when the model is too simple to understand the base structure of the data. It’s like trying to fit into undersized pants. It generally happens when we have less information to construct an exact model and when we attempt to build or develop a linear model with non-linear information.

Main options to reduce underfitting are:

1. Feature Engineering — feeding better features to the learning algorithm.
2. Removing noise from the data.
3. Increasing parameters and selecting a powerful model.

## Models deployment to production

A lot of machine learning practitioners can perform all steps but can lack the skills for deployment, bringing their cool applications into production has become one of the biggest challenges due to lack of practice and dependencies issues, low understanding of underlying models with business, understanding of business problems, unstable models.

There are multiple factors to consider when deciding how to deploy a machine learning model:

1. how frequently predictions should be generated
2. whether predictions should be generated for a single instance at a time or a batch of instances
3. the number of applications that will access the model
4. the latency requirements of these applications

Generally, many of the developers collect data from websites like Kaggle and start training the model. But in reality, we need to make a source for data collection, that varies dynamically. Offline learning or Batch learning may not be used for this type of variable data. The system is trained and then it is launched into production, runs without learning anymore.

It is always preferred to build a pipeline to collect, analyze, build/train, test and validate the dataset for any machine learning project and train the model in batches.

## Conclusion

So i think there are more challenge that i can cover or i can found. Especially challenge in production, i can't really explain properly because i never have a chance to participate in production. I will be happy if you can comment below what challenge for machine learning practitioner i have missed. 