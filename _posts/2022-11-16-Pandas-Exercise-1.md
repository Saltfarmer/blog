---
title: "Pandas Exercise 1 : Knowing your data"
header :
  teaser: /assets/images/pandas-head.jpg

categories:
  - Python
tags:
  - Pandas
  - Python
  - Exercise

---

So in this exercise is we are going to use dataset from the internet to make it easier. You could download the exercise from [here](https://github.com/guipsamora/pandas_exercises/archive/refs/heads/master.zip). I just bored and keep trying to grind myself at least the basic. My suggestion is that you learn a topic in a tutorial, video or documentation and then do the first exercises. Learn one more topic and do more exercises. Never ever you try to check the solutions at all. Suggestions and collaborations are more than welcome. In this case I only show the answer and the result. For the last part I will only show the question because it might be more challenging.

## Chipotle dataset
First exercise, we are going to practice the very basic of Pandas. In this case I will use [Chipotle Dataset](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv). Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

### Step 1. Import the necessary libraries
```python
import pandas as pd
```
### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv). 

### Step 3. Assign it to a variable called chipo.
```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')
```
### Step 4. See the first 10 entries
```python
chipo.head(10)
```

|      | order_id | quantity | item_name                             | choice_description                                                                                                 | item_price |
| ---: | -------: | -------: | :------------------------------------ | :----------------------------------------------------------------------------------------------------------------- | :--------- |
|    0 |        1 |        1 | Chips and Fresh Tomato Salsa          | nan                                                                                                                | $2.39      |
|    1 |        1 |        1 | Izze                                  | [Clementine]                                                                                                       | $3.39      |
|    2 |        1 |        1 | Nantucket Nectar                      | [Apple]                                                                                                            | $3.39      |
|    3 |        1 |        1 | Chips and Tomatillo-Green Chili Salsa | nan                                                                                                                | $2.39      |
|    4 |        2 |        2 | Chicken Bowl                          | [Tomatillo-Red Chili Salsa (Hot), [Black Beans, Rice, Cheese, Sour Cream]]                                         | $16.98     |
|    5 |        3 |        1 | Chicken Bowl                          | [Fresh Tomato Salsa (Mild), [Rice, Cheese, Sour Cream, Guacamole, Lettuce]]                                        | $10.98     |
|    6 |        3 |        1 | Side of Chips                         | nan                                                                                                                | $1.69      |
|    7 |        4 |        1 | Steak Burrito                         | [Tomatillo Red Chili Salsa, [Fajita Vegetables, Black Beans, Pinto Beans, Cheese, Sour Cream, Guacamole, Lettuce]] | $11.75     |
|    8 |        4 |        1 | Steak Soft Tacos                      | [Tomatillo Green Chili Salsa, [Pinto Beans, Cheese, Sour Cream, Lettuce]]                                          | $9.25      |
|    9 |        5 |        1 | Steak Burrito                         | [Fresh Tomato Salsa, [Rice, Black Beans, Pinto Beans, Cheese, Sour Cream, Lettuce]]                                | $9.25      |

### Step 5. What is the number of observations in the dataset?
```python
chipo.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4622 entries, 0 to 4621
Data columns (total 5 columns):
order_id              4622 non-null int64
quantity              4622 non-null int64
item_name             4622 non-null object
choice_description    3376 non-null object
item_price            4622 non-null object
dtypes: int64(2), object(3)
memory usage: 180.6+ KB
```

### Step 6. What is the number of columns in the dataset?
```python
chipo.shape[1]
```

```
5
```

### Step 7. Print the name of all the columns.
```python
chipo.columns
```
```
Index([u'order_id', u'quantity', u'item_name', u'choice_description',
       u'item_price'],
      dtype='object')
```
### Step 8. How is the dataset indexed?
```python
chipo.index
```
```
RangeIndex(start=0, stop=4622, step=1)
```
### Step 9. Which was the most-ordered item? 
```python
chipo.groupby(['item_name']).count().sort_values(['quantity'], ascending = False).head(1).reset_index()['item_name']
```
```
0    Chicken Bowl
Name: item_name, dtype: object
```
### Step 10. For the most-ordered item, how many items were ordered?
```python
chipo.groupby(['item_name']).sum().sort_values(['quantity'], ascending = False).head(1)['quantity']
```
### Step 11. What was the most ordered item in the choice_description column?
```python
chipo.groupby(['choice_description']).sum().sort_values(['quantity'], ascending = False).head(1)
```
| choice_description | order_id | quantity |
| :----------------- | -------: | -------: |
| [Diet Coke]        |   123455 |      159 |

### Step 12. How many items were orderd in total?
```python
chipo['quantity'].sum()
```
```
4972
```
### Step 13. Turn the item price into a float
#### Step 13.a. Check the item price type
```python
chipo['item_price'].dtype
```
```
dtype('O')
```
#### Step 13.b. Create a lambda function and change the type of item price
```python
chipo['item_price'] = chipo['item_price'].apply(lambda x : float(x[1:-1])) 
```
#### Step 13.c. Check the item price type
```python
chipo['item_price'].dtype
```
```
dtype('float64')
```
### Step 14. How much was the revenue for the period in the dataset?
```python
chipo['item_price'].sum()
```
```
34500.16
```
### Step 15. How many orders were made in the period?
```python
len(chipo['order_id'].unique())
```
```
1834
```
### Step 16. What is the average revenue amount per order?
```python
chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']
```
```
21.394231188658654
```
### Step 17. How many different items are sold?
```python
chipo['item_name'].value_counts().count()
```
```
50
```

## Occupation dataset
This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

### Step 1. Import the necessary libraries
```python
import pandas as pd
```
### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 
### Step 3. Assign it to a variable called users and use the 'user_id' as index
```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
user_id = pd.read_csv(url, sep='|')
```
### Step 4. See the first 25 entries
```python
user_id.head(25)
```
|      | user_id |  age | gender | occupation    | zip_code |
| ---: | ------: | ---: | :----- | :------------ | -------: |
|    0 |       1 |   24 | M      | technician    |    85711 |
|    1 |       2 |   53 | F      | other         |    94043 |
|    2 |       3 |   23 | M      | writer        |    32067 |
|    3 |       4 |   24 | M      | technician    |    43537 |
|    4 |       5 |   33 | F      | other         |    15213 |
|    5 |       6 |   42 | M      | executive     |    98101 |
|    6 |       7 |   57 | M      | administrator |    91344 |
|    7 |       8 |   36 | M      | administrator |    05201 |
|    8 |       9 |   29 | M      | student       |    01002 |
|    9 |      10 |   53 | M      | lawyer        |    90703 |
|   10 |      11 |   39 | F      | other         |    30329 |
|   11 |      12 |   28 | F      | other         |    06405 |
|   12 |      13 |   47 | M      | educator      |    29206 |
|   13 |      14 |   45 | M      | scientist     |    55106 |
|   14 |      15 |   49 | F      | educator      |    97301 |
|   15 |      16 |   21 | M      | entertainment |    10309 |
|   16 |      17 |   30 | M      | programmer    |    06355 |
|   17 |      18 |   35 | F      | other         |    37212 |
|   18 |      19 |   40 | M      | librarian     |    02138 |
|   19 |      20 |   42 | F      | homemaker     |    95660 |
|   20 |      21 |   26 | M      | writer        |    30068 |
|   21 |      22 |   25 | M      | writer        |    40206 |
|   22 |      23 |   30 | F      | artist        |    48197 |
|   23 |      24 |   21 | F      | artist        |    94533 |
|   24 |      25 |   39 | M      | engineer      |    55107 |
### Step 5. See the last 10 entries
```python
user_id.tail(10)
```
|      | user_id |  age | gender | occupation    | zip_code |
| ---: | ------: | ---: | :----- | :------------ | -------: |
|  933 |     934 |   61 | M      | engineer      |    22902 |
|  934 |     935 |   42 | M      | doctor        |    66221 |
|  935 |     936 |   24 | M      | other         |    32789 |
|  936 |     937 |   48 | M      | educator      |    98072 |
|  937 |     938 |   38 | F      | technician    |    55038 |
|  938 |     939 |   26 | F      | student       |    33319 |
|  939 |     940 |   32 | M      | administrator |    02215 |
|  940 |     941 |   20 | M      | student       |    97229 |
|  941 |     942 |   48 | F      | librarian     |    78209 |
|  942 |     943 |   22 | M      | student       |    77841 |
### Step 6. What is the number of observations in the dataset?
```python
user_id.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 943 entries, 0 to 942
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   user_id     943 non-null    int64 
 1   age         943 non-null    int64 
 2   gender      943 non-null    object
 3   occupation  943 non-null    object
 4   zip_code    943 non-null    object
dtypes: int64(2), object(3)
memory usage: 37.0+ KB
```
### Step 7. What is the number of columns in the dataset?
```python
user_id.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 943 entries, 0 to 942
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   user_id     943 non-null    int64 
 1   age         943 non-null    int64 
 2   gender      943 non-null    object
 3   occupation  943 non-null    object
 4   zip_code    943 non-null    object
dtypes: int64(2), object(3)
memory usage: 37.0+ KB
```
### Step 8. Print the name of all the columns.
```python
user_id.columns
```
```
Index(['user_id', 'age', 'gender', 'occupation', 'zip_code'], dtype='object')
```
### Step 9. How is the dataset indexed?
```python
user_id.index
```
```
RangeIndex(start=0, stop=943, step=1)
```
### Step 10. What is the data type of each column?
```python
user_id.dtypes
```
```
user_id        int64
age            int64
gender        object
occupation    object
zip_code      object
dtype: object
```
### Step 11. Print only the occupation column
```python
user_id['occupation']
```
```
0         technician
1              other
2             writer
3         technician
4              other
           ...      
938          student
939    administrator
940          student
941        librarian
942          student
Name: occupation, Length: 943, dtype: object
```
### Step 12. How many different occupations are in this dataset?
```python
user_id['occupation'].value_counts().count()
```
```
21
```
### Step 13. What is the most frequent occupation?
```python
user_id.groupby(['occupation']).count().sort_values(['user_id'], ascending = False).head(1)
```
| occupation | user_id |  age | gender | zip_code |
| :--------- | ------: | ---: | -----: | -------: |
| student    |     196 |  196 |    196 |      196 |
### Step 14. Summarize the DataFrame.
```python
user_id.describe()
```
|       | user_id |     age |
| :---- | ------: | ------: |
| count |     943 |     943 |
| mean  |     472 |  34.052 |
| std   | 272.365 | 12.1927 |
| min   |       1 |       7 |
| 25%   |   236.5 |      25 |
| 50%   |     472 |      31 |
| 75%   |   707.5 |      43 |
| max   |     943 |      73 |
### Step 15. Summarize all the columns
```python
user_id.describe(include= 'all')
```
|        | user_id |     age | gender | occupation | zip_code |
| :----- | ------: | ------: | :----- | :--------- | -------: |
| count  |     943 |     943 | 943    | 943        |      943 |
| unique |     nan |     nan | 2      | 21         |      795 |
| top    |     nan |     nan | M      | student    |    55414 |
| freq   |     nan |     nan | 670    | 196        |        9 |
| mean   |     472 |  34.052 | nan    | nan        |      nan |
| std    | 272.365 | 12.1927 | nan    | nan        |      nan |
| min    |       1 |       7 | nan    | nan        |      nan |
| 25%    |   236.5 |      25 | nan    | nan        |      nan |
| 50%    |     472 |      31 | nan    | nan        |      nan |
| 75%    |   707.5 |      43 | nan    | nan        |      nan |
| max    |     943 |      73 | nan    | nan        |      nan |
### Step 16. Summarize only the occupation column
```python
user_id['occupation'].describe()
```
```
count         943
unique         21
top       student
freq          196
Name: occupation, dtype: object
```
### Step 17. What is the mean age of users?
```python
user_id['age'].mean()
```
```
34.05196182396607
```
### Step 18. What is the age with least occurrence?
```python
user_id['age'].value_counts().tail()
```
```
7     1
66    1
11    1
10    1
73    1
Name: age, dtype: int64
```

## World Food Facts Dataset
For this exercise, you could try it by yourself. Have fun !!!