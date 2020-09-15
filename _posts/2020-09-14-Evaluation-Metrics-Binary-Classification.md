---
title: "Evaluation Metric Binary Classification"
header :
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Scoring
  - Classification

---

Classifier performance depends greatly on the characteristics of the data to be classified. There is no single classifier that works best on all given problems. After testing and validation from last post, we need to know what is the correct metrics for testing. The most common metric like accuracy is really well known for a lot of people. But did you know there are still a lot of evaluation metrics outside accuracy ?

To begin with, we need to understand **Confusion Matrix**. a confusion matrix, also known as an error matrix, is a table that allows us visualize of the performance of an algorithm. For example you have a data about COVID-19 disease. The actual data tells us where Covid-19 positive belong to class "1" and negative to class "0".

>  actual = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]

Then, you created a classifier that predicted COVID-19.

>  prediction = [0,0,0,1,1,1,1,1,0,0,0,1,1,1,0]

> condition positive (P) : the number of real positive cases in the data
>
> condition negative (N) : the number of real negative cases in the data

With these two labelled sets (actual and predictions) we can create a confusion matrix that will summarize the results of testing the classifier:

|               |          |  Actual  |  Class   |
| :-----------: | :------: | :------: | :------: |
|               |          | Positive | Negative |
| **Predicted** | Positive |    5     |    3     |
|   **Class**   | Negative |    3     |    4     |

In abstract terms, the confusion matrix is as follows:

|               |          |  Actual  |  Class   |
| :-----------: | :------: | :------: | :------: |
|               |          | Positive | Negative |
| **Predicted** | Positive |    TP    |    FP    |
|   **Class**   | Negative |    FN    |    TN    |

> true positive (TP) : Correct Positive Prediction
>
> true negative (TN) : Correct Negative Prediction
>
> false positive (FP) : Wrong Positive Prediction (Type 1 Error)
>
> false negative (FN) : Wrong Negative Prediction (Type 2 Error)

In statistical hypothesis testing, a **type I error** is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion; example: "an innocent person is convicted"), while a **type II error** is the non-rejection of a false null hypothesis (also known as a "false negative" finding or conclusion; example: "a guilty person is not convicted").

# Classification Metrics

In a classification task, the terms ‘’positive’’ and ‘’negative’’ refer to the classifier’s prediction, and the terms ‘’true’’ and ‘’false’’ refer to whether that prediction corresponds to the external judgment (sometimes known as the ‘’observation’’).

## <u>Based on Actual Data</u> (Subtracted by Actual Data)

### True Positive Rate, Recall, or Sensitivity

$$
{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }
$$

### True Negative rate or Specificity

$$
{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }
$$

### False Positive Rate or False Alarm

$$
{\displaystyle \mathrm {FPR} ={\frac {\mathrm {FP} }{\mathrm {N} }}={\frac {\mathrm {FP} }{\mathrm {FP} +\mathrm {TN} }}=1-\mathrm {TNR} }
$$

### False Negative Rate or Miss Rate

$$
{\displaystyle \mathrm {FNR} ={\frac {\mathrm {FN} }{\mathrm {P} }}={\frac {\mathrm {FN} }{\mathrm {FN} +\mathrm {TP} }}=1-\mathrm {TPR} }
$$



## <u>Based on Prediction Data</u> (Subtracted by Prediction Result)

### Positive Predictive Value or Precision

$$
{\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }
$$

### Negative Predictive Value

$$
{\displaystyle \mathrm {NPV} ={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FN} }}=1-\mathrm {FOR} }
$$

### False Discovery Rate

$$
{\displaystyle \mathrm {FDR} ={\frac {\mathrm {FP} }{\mathrm {FP} +\mathrm {TP} }}=1-\mathrm {PPV} }
$$

### False Omission Rate

$$
{\displaystyle \mathrm {FOR} ={\frac {\mathrm {FN} }{\mathrm {FN} +\mathrm {TN} }}=1-\mathrm {NPV} }
$$

## Accuracy

According to ISO 5725-1,[[1\]](https://en.wikipedia.org/wiki/Accuracy_and_precision#cite_note-iso5725-1) the general term "accuracy" is used to describe the closeness of a measurement to the true value (Correct Values). Accuracy is the most common evaluation metrics used in classification. 
$$
{\displaystyle \mathrm {ACC} ={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {P} +\mathrm {N} }}={\frac {\mathrm {TP} +\mathrm {TN} }{\mathrm {TP} +\mathrm {TN} +\mathrm {FP} +\mathrm {FN} }}}
$$
Then what's wrong with accuracy ? Well, nothing is wrong with  accuracy, if your dataset is already balanced, What is balanced dataset ? balanced dataset is when the proportion rate of positive and negative is equal or almost equal. The closer amount of positive and negative data, the more balance the data.

If you have imbalance dataset and still want to use accuracy as evaluation metrics, you should use **Balanced Accuracy**.
$$
{\displaystyle \mathrm {BA} ={\frac {TPR+TNR}{2}}}
$$
Let's try from the example of this data

|               |          |  Actual  |  Class   |
| :-----------: | :------: | :------: | :------: |
|               |          | Positive | Negative |
| **Predicted** | Positive |    5     |    3     |
|   **Class**   | Negative |    3     |    4     |

Accuracy = 9/15 = 0.6

Balanced Accuracy = (5/8 + 4/7) / 2 =  0.59821428571

let's round up Balanced Accuracy to 2 digits then the result basically the same. Let's try another example with Imbalanced data.

|               |          |  Actual  |  Class   |
| :-----------: | :------: | :------: | :------: |
|               |          | Positive | Negative |
| **Predicted** | Positive |    8     |    3     |
|   **Class**   | Negative |    3     |    1     |

Accuracy = 9/15 = 0.6

Balanced Accuracy = (8/11 + 1/3) / 2 = 0.53 

Now we see the metric is different. You see any difference between those 2 data ? Correct, the ratio of positive value is larger than the negative value. This metric will be important for imbalance dataset let's say Spam Detection or Fraud Detection where both of those happen not happening a lot.

## F1 Score

F1 score (also F-score or F-measure) is another measure evaluation metrics. It is calculated from the precision and recall of the test, where the precision is the number of correctly identified positive results divided by the number of all positive results, including those not identified correctly, and the recall is the number of correctly identified positive results divided by the number of all samples that should have been identified as positive. 
$$
{\displaystyle \mathrm {F} _{1}=2\cdot {\frac {\mathrm {PPV} \cdot \mathrm {TPR} }{\mathrm {PPV} +\mathrm {TPR} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}
$$
F1 Score is needed when you want to seek a balance between Precision and Recall. So what is the difference between F1 Score and Accuracy then? We have previously seen that accuracy can be largely contributed by a large number of True Negatives which in most circumstances, we do not focus on much whereas False Negative and False Positive usually has business costs (tangible & intangible) thus F1 Score might be a better measure to use if we need to seek a balance between Precision and Recall AND there is an imbalance data.

Let's try with Imbalanced data before.



|               |          |  Actual  |  Class   |
| :-----------: | :------: | :------: | :------: |
|               |          | Positive | Negative |
| **Predicted** | Positive |    8     |    3     |
|   **Class**   | Negative |    3     |    1     |

Accuracy = 9/15 = 0.6

F1-Score = 2 * 8 / (2*8 + 3 + 3) = 0.73

Then which one is the better evaluation metrics for imbalance dataset ? F1-Score is much better evaluation metrics because they look both the amount of positive and negative value whereas Balanced Accuracy works more better in positive > negative condition.

That's all for today. In the next post, I will explain the other evaluation metric especially for Regression problem and Clustering problem.  



