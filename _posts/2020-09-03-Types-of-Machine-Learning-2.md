---
title: "Types of Machine Learning 2"
header :
  teaser: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning

---

..... Continue from [last post](https://saltfarmer.github.io/blog/machine%20learning/Types-of-Machine-Learning/)

# Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

## Batch Learning

In batch learning, the system is **not capable** of learning continuously. It means it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called **offline learning**.

![Batch Learning](https://i.ibb.co/0QG3wx0/Batch-Learning.png)

For example, If you want a batch learning system to know about new data you need to train a new version of the system from scratch on the full dataset
then and replace the old system with the new one. It works well for systems only have new data every week / day or more. But for systems such as Stock prices data is changing every minute, we need another solution.

Another example is imagine if you need to train huge chunk of dataset. Especially in real world case, the amount of dataset most likely will be very huge. Training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of resources. If the amount of data is huge, it may even be impossible or took forever to use a batch learning algorithm.

Luckily, there is a better option that capable to learn continuously.

## Online Learning

In online learning, your system **capable** to continuously feeding the data, either individually or by small groups called **mini-batches**. Each learning
step is fast and cheap, so the system can learn about new data on the fly, as it arrives. Online learning is great for systems that receive data as a continuous flow and need to adapt to change rapidly or autonomously. 

![Online Learning](https://i.ibb.co/j6hMw6L/Online-Learning.png)

Online learning is data efficient and adaptable. Online learning is data efficient because once data has been consumed it is no longer required. Technically, this means you don’t have to store your data. Online learning is adaptable because it makes no assumption about the distribution of your data. As your data distribution morphs or drifts, due to say changing customer behavior, the model can adapt on-the-fly to keep pace with trends in real-time. In order to do something similar with offline learning you’d have to create a sliding window of your data and retrain every time.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called **out-of-core learning**). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the **learning rate**. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data. So in summary, you need to find the balance in learning rate. If you set a low learning rate, the system will 
learn more slowly, but it will also be less sensitive to noise in the new data or to outliers.

But there is a risk of using online learning. The algorithm may receive new bad data, which will impact the system performance badly. To avoid that risk, you needwatch the model performance, and if you found the performance decreasing by new data, you can revert to the previous model. Also, you need to monitor the control the input data, controlling the abnormal data with anomaly detection algorithm.

# Instance and Model Based Learning

One more way to categorize Machine Learning systems is by how they generalize. Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before.

## Instance Based Learning

The system learns the examples by generalizing to new cases by comparing them to the learned examples using a **similarity measure**. The system will learn how identical between 2 or more data. This is called instance based learning.

For example, instead of just flagging emails that are identical to known spam emails, your spam filter could be programmed to also flag emails that are very similar to known spam emails. This requires a measure of similarity between two emails. A very basic similarity measure between two emails could be to count the number of words they have in common. The system would flag an email as spam if it has many words in common with a known spam email.

This method is not the worst solution, but surely not good enough. It simply just lazy.

## Model Based Learning

Another way to generalize from a set of examples is to build a model of these examples, then use that model to make *predictions*. This is called model based learning.

For example, you need to understand what is the sentiment analysis of movie, so you download IMDB movie review dataset. Then you created a model to make a prediction from IMDB dataset. If all went well, your model will make good predictions. If not, you may need to use more attributes, get more or better quality training data, or perhaps select a more powerful model.

In summary, the steps is 

1. You studied the Data
2. You choose a Model
3. You train your model based on training data
4. Then, you applied your model to make a prediction on new data or data test.

This is what most typical Machine Learning Project nowaday.