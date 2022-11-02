---
title: "Python Pandas"
header :
  image: /assets/images/pandas-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Machine Learning
  - Python
  - Pandas

---

Next, let's discussing Pandas. Preparing the data and munging the same was the initial outcomes of  python before the introduction of Panda libraries. after the  introduction of panda libraries python began to flourish a lot in the  analytics sector. The major outcomes of panda are analysis of data, preparation of data, data manipulation, data modeling, and data analysis. 

## Series

First of all, Lets discuss about `pd.series`. Series is a one-dimensional ndarray with axis labels (including time series).

Labels need not be unique but must be a hashable type. The object supports both integer- and label-based indexing and provides a host of methods for performing operations involving the index. Statistical methods from ndarray have been overridden to automatically exclude missing data (currently represented as NaN).

lets check this example

```python
import numpy as np
import pandas as pd

arr = np.random.randint(1,10,5)
d = {'a': 2, 'b': 3, 'c' : 7, 'd' : 9, 'e' :1} 

list1 = pd.Series(data=arr)
list2 = pd.Series(data=d)

print(list1)
print(list2)
```

```
0    3
1    6
2    1
3    2
4    8
dtype: int64
a    2
b    3
c    7
d    9
e    1
dtype: int64
```

So we see the difference between series from list and dictionary is that series from list automatically indexing from 0-n while index series from dictionary follow from dictionary. You can also put a label from unlabelled data like this example

```python
import numpy as np
import pandas as pd

arr = np.random.randint(1,10,5)
label = ['ran1', 'ran2', 'ran3', 'ran4', 'ran5']

list1 = pd.Series(data=arr,index=label)
print(list1)
```

```
ran1    4
ran2    1
ran3    3
ran4    4
ran5    6
dtype: int64
```

You can also combine between two or more series. For Example

```python
import numpy as np
import pandas as pd

arr = np.random.randint(1,10,5)
d = {'a': 2, 'b': 3, 'c' : 7, 'd' : 9, 'e' :1} 

list1 = pd.Series(data=arr, index=['a', 'b', 'c', 'd', 'e'])
list2 = pd.Series(data=d)
list3 = list1 + list2

print(list3)
```

```
a     3
b     5
c     8
d    15
e     2
dtype: int64
```

Why I put index on my randomized series ? Because if i dont it would ends up as index [0,1,2,3,4,a,b,c,d,e] and NaN values.

## Dataframes

`pd.DataFrame` is Two-dimensional, size-mutable, potentially heterogeneous tabular data. Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dictionary-like container for Series objects. The primary pandas data structure.

Let Start creating dataframe with random numbers

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(data=np.random.randn(5,4), index=[1,2,3,4,5], columns=['a', 'b', 'c', 'd'])
df
```

```
		a			b			c		d
1	-2.826205	-0.020080	-1.188318	-1.239230
2	-1.503837	0.028629	-1.210348	-0.132313
3	1.368030	-1.905128	-1.719268	-0.090431
4	1.263813	-0.870573	-0.685124	2.135516
5	0.061481	0.884779	-0.836704	-0.574821
```

You can show the column(s) like this

```python
df[['a', 'b']]
```

```
	a			b
1	0.086870	-0.639393
2	-0.955657	0.258113
3	-0.669046	0.417024
4	-1.212063	0.090159
5	-1.022556	2.135724
```

And selecting row(s) like this

```python
df.loc[1] 
# Based on the index name
df.iloc[0]
# Based on the index location
```

```
a    0.086870
b   -0.639393
c   -0.762429
d   -0.306316
Name: 1, dtype: float64
a    0.086870
b   -0.639393
c   -0.762429
d   -0.306316
Name: 1, dtype: float64
```

It is the same because index "1" located first (python numbering start with zero). And then you can call the Dataframes based on row and column like this

```python
df.loc[[1,3], ['a','b','d']]
```

```
	a	b	d
1	0.086870	-0.639393	-0.306316
3	-0.669046	0.417024	0.870600
```

And finally, you can create a new column and delete column with this

```python
df['new'] = df['a'] + df['c']
print(df)
df_drop = df.drop(columns=['new'])
print(df_drop)
```

```
          a         b         c         d       new
1  0.086870 -0.639393 -0.762429 -0.306316 -0.675559
2 -0.955657  0.258113 -0.027353  1.256810 -0.983010
3 -0.669046  0.417024  1.412859  0.870600  0.743814
4 -1.212063  0.090159  1.125037 -0.265073 -0.087026
5 -1.022556  2.135724  1.208373  0.219263  0.185817
          a         b         c         d
1  0.086870 -0.639393 -0.762429 -0.306316
2 -0.955657  0.258113 -0.027353  1.256810
3 -0.669046  0.417024  1.412859  0.870600
4 -1.212063  0.090159  1.125037 -0.265073
5 -1.022556  2.135724  1.208373  0.219263
```

You can use conditional operator inside Dataframes. It will help you to select  particular number or data. Here is the example

```python
print(df)
print(df>1)
print(df[df['a']>0])
```

```
          a         b         c         d       new
1  0.086870 -0.639393 -0.762429 -0.306316 -0.675559
2 -0.955657  0.258113 -0.027353  1.256810 -0.983010
3 -0.669046  0.417024  1.412859  0.870600  0.743814
4 -1.212063  0.090159  1.125037 -0.265073 -0.087026
5 -1.022556  2.135724  1.208373  0.219263  0.185817
       a      b      c      d    new
1  False  False  False  False  False
2  False  False  False   True  False
3  False  False   True  False  False
4  False  False   True  False  False
5  False   True   True  False  False
         a         b         c         d       new
1  0.08687 -0.639393 -0.762429 -0.306316 -0.675559
```

For two conditions you can use `|` and `&` with parenthesis like this

```python
df[(df['a']>0) | (df['c'] > 1)]
```

```
a	b	c	d
1	0.426039	-0.310240	-0.842375	-0.376677
3	-0.696941	-0.368143	2.026986	-0.486401
```

 lets take a look about more details in indexing dataframes. You can reset your index by usinf `df.reset_index()`. Letes take a look at the example

```python
print(df)
print(df.reset_index())
```

```
          a         b         c         d
1  0.426039 -0.310240 -0.842375 -0.376677
2 -2.555266  1.060464  0.697077 -0.611722
3 -0.696941 -0.368143  2.026986 -0.486401
4 -0.410270  0.059790 -0.804746 -0.293890
5 -0.554131 -0.375219 -0.831811 -0.059533

   index         a         b         c         d
0      1  0.426039 -0.310240 -0.842375 -0.376677
1      2 -2.555266  1.060464  0.697077 -0.611722
2      3 -0.696941 -0.368143  2.026986 -0.486401
3      4 -0.410270  0.059790 -0.804746 -0.293890
4      5 -0.554131 -0.375219 -0.831811 -0.059533
```

You can see, pandas resetting the index from 0-n and make the old index names to columns. Then you can set the index with `df.set_index`.

```python
df = df.reset_index()
newind = 'CA NY WY OR CO'.split()
df['States'] = newind
df.set_index('States')
```

```
	index	a	b	c	d
States					
CA	1	0.486785	0.907007	-0.176515	0.136101
NY	2	0.071172	1.313467	0.507755	-1.628941
WY	3	-1.492063	-0.929157	-0.394949	0.706727
OR	4	-1.512844	-0.058844	0.029634	0.887493
CO	5	-1.445499	0.715998	0.997913	-1.257716
```

## Handling missing data with pandas

Basically, there are 2 option dealing with missing value in dataset. Obviously, not all dataset is ready. Sometimes the dataset have a lot or some missing values. Let's take a look how to handle missing value by dropping with `df.dropna()`

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})

print(df)
print(df.dropna(axis=0))
print(df.dropna(axis=1))
```

```
    A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
2  NaN  NaN  3
     A    B  C
0  1.0  5.0  1
   C
0  1
1  2
2  3
```

You see the difference between dropping values from axis=0 and axis=1. Dropping with axis=0 means that you delete the rows based on missing value. In the other hand you use axis=1 to drop the columns. You can also put a threshold of the minimal count of missing values.

```python
print(df.dropna(thresh=2))
```

```
     A    B  C
0  1.0  5.0  1
1  2.0  NaN  2
```

Now, you can try filling missing values with either mode, mean, and median. You can use `df.fillna()` to do that

```python
print(df['A'].fillna(value=df['A'].mean()))
```

```
0    1.0
1    2.0
2    1.5
Name: A, dtype: float64
```