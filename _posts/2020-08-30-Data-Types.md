---
title: "Data Types"

comments : true
share : true
categories:
  - Data Science
tags:
  - Data Types
  - Data Science
  - Pandas

---

There are many different kind of data types. In this blog, i will explain these data types based on most common understanding in Data Science. Specifically in Python Pandas. When doing data analysis, it is important to make sure you are using the correct data types; otherwise you may get unexpected results or errors. 

Datatypes are an important concept because statistical methods can only be used with certain data types. You have to analyze continuous data differently than categorical data otherwise it would result in a wrong analysis. Therefore knowing the types of data you are dealing with, enables you to choose the correct method of analysis.

To start with, here are the most common data types in Pandas

| Pandas dtype  | Python type  | NumPy type                                                   | Usage                                        |
| :------------ | :----------- | :----------------------------------------------------------- | :------------------------------------------- |
| object        | str or mixed | string_, unicode_, mixed types                               | Text or mixed numeric and non-numeric values |
| int64         | int          | int_, int8, int16, int32, int64, uint8, uint16, uint32, uint64 | Integer numbers                              |
| float64       | float        | float_, float16, float32, float64                            | Floating point numbers                       |
| bool          | bool         | bool_                                                        | True/False values                            |
| datetime64    | NA           | datetime64[ns]                                               | Date and time values                         |
| timedelta[ns] | NA           | NA                                                           | Differences between two datetimes            |
| category      | NA           | NA                                                           | Finite list of text values                   |

For the most part, there is no need to worry about determining if you should try to explicitly force the pandas type to a corresponding to NumPy type. Most of the time, using pandas default `int64` and `float64` types will work. 

One other item I want to highlight is that the `object` data type can actually contain multiple different types. For instance, the a column could include integers, floats and strings which collectively are labeled as an `object`. Therefore, you may need some additional techniques to handle mixed data types in `object` columns.

Then, Let's check the types of data based on their characteristic

# Numerical Data

## Discrete Data

We speak of discrete data if its values are distinct and separate. In other words: We speak of discrete data if the data can only take on certain values. This type of data **can’t be measured but it can be counted**. It basically represents information that can be categorized into a classification. An example is how many time did you run.

## Continous Data

Continuous Data represents measurements and therefore their values **can’t be counted but they can be measured**. An example would be the height of a person, which you can describe by using intervals on the real number line.

### Interval Data

Interval values represent **ordered units that have the same difference**. Therefore we speak of interval data when we have a variable that contains numeric values that are ordered and where we know the exact differences between the values. An example is what is the temperature from 0 to 100 in Celcius. The problem with interval values data is that they **"don’t have a true zero“**. That means there is no such thing as no temperature. With interval data, we can add and subtract, but we cannot multiply, divide or calculate ratios. Because there is no true zero, a lot of descriptive and inferential statistics can’t be applied.

### Ratio Data

Ratio values are also ordered units that have the same difference. Ratio values are **the same as interval values, with the difference that they do have an absolute zero**. Good examples are height, weight, length etc.

# Categorical data

Categorical data represents characteristics. Therefore it can represent things like a person’s gender, language etc. Categorical data can also take on numerical values (Example: 1 for female and 0 for male). Note that those numbers don’t have mathematical meaning. 

## Nominal Data

Nominal values represent discrete units and are used to label variables, that have no quantitative value. Just think of them as labels. Note that **nominal data that has no order**. Therefore if you would change the order of its values, the meaning would not change. An example is which group these people grouped into.

## Ordinal Data

Ordinal values represent discrete and ordered units. It is therefore nearly the same as nominal data, except that it’s **ordering matters**. For example, what is your level in video games.

