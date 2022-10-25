---
title: "Testing and Validation"
header :
  image: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Validation

---

To know how good your model is to actually try it on new cases or different cases. But if your model not doing good as it expected, surely you will hesitate to apply that model to production environment. A better option is to split your data into two sets: the **training set** and the **test set**. 

As these names imply, you train your model using the training set, and you test it using the test set. The error rate on new cases is called the **generalization error** and by evaluating your model on the test set, you get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before.

If the training error is low but the generalization error is high, it means that your model is **overfitting** the training data. Else if the trainin error is high while the generalization error is also high, it means that your model is **underfitting** the training data. 

The problem is that you measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model based on *that particular test set*. This would make the model is unlikely to perform as well on new
data.

A solution for this problem is using **holdout validation**. You simply hold out part of the training set to evaluate several candidate models and select the best one. The new holdout validation set is called **validation set**. Validation set is when the sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration. After this holdout validation process, you train the best model on the full training set and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error.

![Splits](https://i.ibb.co/D754xkZ/train-val-test-split.png)

But what if your validation set is too small ? your model will evaluation most likely will be not precise. Reversely, if validation set is too big, then the remaining training set will be much smaller than the full training set. One way to solve this problem to perform *k-fold* (the amount of validation) **cross-validation**, using more many (more than one) small validation sets.

![k-fold cross-validation](https://i.ibb.co/JxwTHDt/grid-search-cross-validation.jpg)

Each model is evaluated once per validation set, after it is trained on the rest of the data. By averaging out all the evaluations of a model, we get a much more accurate measure of its performance. However, there is a drawback, the training time is multiplied by the number of validation sets.  

But how many fold cross-validation is recommended ? In their book, Kuhn and Johnson have a section titled “Data Splitting Recommendations”, They go on to make a recommendation for small sample sizes of using 10-fold cross validation in general because of the desirable low bias and variance properties of the performance estimate. They recommend the holdout-validation in the case of comparing model performance because of the low variance in the performance estimate. For larger sample sizes, they again recommend a 10-fold cross-validation approach, in general.







