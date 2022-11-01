---
title: "Automated EDA Library for Python"
header : 
  image : /assets/images/python-head.jpg
categories:
  - Python
tags:
  - Python
  - EDA
---

After I reviewed my knowledge of exploratory data analysis (EDA) [here](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python), I am wondering if there is some way or a new way to understand your dataset more easily. Luckily, I stumbled on this [site](https://github.com/mstaniak/autoEDA-resources). 

During data exploration, it is necessary to understand the data before jumping into pre-processing, machine learning model, and conclusion you can get from data analysis.

It used to be that you had to write basic functions for EDA using Pandas and Numpy (even you are too lazy to do visualization and just visualize it using pandas built-in visualization). But with the advent of advanced techniques such as Automatic Machine Learning, it has become obvious that EDA could benefit from automation as well. 

So the main point of doing EDA are :

- EDA aims to examine the data for distributions, outliers and anomalies to direct specific testing of our hypothesis.
- EDA display and summarize data that are obtained from a sample, by means of visualization and statistical techniques.
- EDA helps us to find out the natural patterns and the features which are of utmost significance.
- EDA allows us to prepare the report containing the details of the data insights which can be shared with the business owners and stakeholders.

In this post, I try to explore automated EDA libraries with Python. I know there is some interesting library that I don't put in this post because of some problem when installing the library with my PIP.

## Pandas Profiling

![](https://camo.githubusercontent.com/8a45c0936d6113b12b7b32942f448270eda8f714665ba8629f36c291f0ccd5fd/68747470733a2f2f70616e6461732d70726f66696c696e672e6769746875622e696f2f70616e6461732d70726f66696c696e672f646f63732f6173736574732f6c6f676f5f6865616465722e706e67)

The most commonly used library for getting quick data summaries and correlation analysis. It generates profile reports for the mentioned dataframe. Usually, we use df.describe() function for this but it is not sufficient for in-depth exploratory data analysis. pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.

Depending upon the relevant data type of the column, the following details are present in an interactive HTML report:-

- **Type inference:** detect the [types](https://github.com/pandas-profiling/pandas-profiling#types) of columns in a dataframe.
- **Essentials**: type, unique values, missing values
- **Quantile statistics** like minimum value, Q1, median, Q3, maximum, range, interquartile range
- **Descriptive statistics** like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
- **Most frequent values**
- **Histogram**
- **Correlations** highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
- **Missing values** matrix, count, heatmap and dendrogram of missing values
- **Text analysis** learns about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data.
- **File and Image analysis** extract file sizes, creation dates and dimensions and scan for truncated images or those containing EXIF information.

**Installation**

```python
pip install pandas-profiling
```

**Profile Reports**

```python
import pandas as pd

# Read the Titanic Dataset
file_name = cache_file(
	"titanic.csv",
	"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
)

df = pd.read_csv(file_name)
# Generate the Profiling Report
profile = ProfileReport(
	df, title="Titanic Dataset", html={"style": {"full_width": True}}, sort="None"
)

profile
```

You can see the result of profile report [here](https://pandas-profiling.github.io/pandas-profiling/examples/master/titanic/titanic_report.html).



