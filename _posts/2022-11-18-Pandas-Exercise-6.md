---
title: "Pandas Exercise 5 : Stats"
header :
  teaser: /assets/images/pandas-head.jpg

categories:
  - Python
tags:
  - Pandas
  - Python
  - Exercise

---

The continuity of my practice on Pandas exercise from [guisapmora](https://github.com/guipsamora/pandas_exercises/archive/refs/heads/master.zip).

# US - Baby Names Dataset

### Introduction:

We are going to use a subset of [US Baby Names](https://www.kaggle.com/kaggle/us-baby-names) from Kaggle.  
In the file it will be names from 2004 until 2014


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv). 

### Step 3. Assign it to a variable called baby_names.


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv'
baby_name = pd.read_csv(url)
```

### Step 4. See the first 10 entries


```python
baby_name.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Id</th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11349</td>
      <td>11350</td>
      <td>Emma</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11350</td>
      <td>11351</td>
      <td>Madison</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11351</td>
      <td>11352</td>
      <td>Hannah</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11352</td>
      <td>11353</td>
      <td>Grace</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11353</td>
      <td>11354</td>
      <td>Emily</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11354</td>
      <td>11355</td>
      <td>Abigail</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>37</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11355</td>
      <td>11356</td>
      <td>Olivia</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11356</td>
      <td>11357</td>
      <td>Isabella</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11357</td>
      <td>11358</td>
      <td>Alyssa</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>29</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11358</td>
      <td>11359</td>
      <td>Sophia</td>
      <td>2004</td>
      <td>F</td>
      <td>AK</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. Delete the column 'Unnamed: 0' and 'Id'


```python
baby_name = baby_name.drop(columns=['Unnamed: 0', 'Id'], axis=1)
```

### Step 6. Is there more male or female names in the dataset?


```python
baby_name['Gender'].value_counts()
```




    F    558846
    M    457549
    Name: Gender, dtype: int64



### Step 7. Group the dataset by name and assign to names


```python
names = baby_name.groupby(['Name']).count()
```

### Step 8. How many different names exist in the dataset?


```python
names.shape[0]
```




    17632



### Step 9. What is the name with most occurrences?


```python
names['Count'].sort_values(ascending=False).head(1)
```




    Name
    Riley    1112
    Name: Count, dtype: int64



### Step 10. How many different names have the least occurrences?


```python
names[names['Count'] == names['Count'].min()].shape[0]
```




    3682



### Step 11. What is the median name occurrence?


```python
names['Count'].median()
```




    8.0



### Step 12. What is the standard deviation of names?


```python
names['Count'].std()
```




    122.02996350814125



### Step 13. Get a summary with the mean, min, max, std and quartiles.


```python
names['Count'].describe()
```




    count    17632.000000
    mean        57.644907
    std        122.029964
    min          1.000000
    25%          2.000000
    50%          8.000000
    75%         39.000000
    max       1112.000000
    Name: Count, dtype: float64


# Wind Statistics Dateset

### Introduction:

The data have been modified to contain some missing values, identified by NaN.  
Using pandas should make this exercise
easier, in particular for the bonus question.

You should be able to perform all of these operations without using
a for loop or other looping construct.


1. The data in 'wind.data' has the following format:


```python
"""
Yr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL
61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04
61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71
"""
```




    '\nYr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL\n61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04\n61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83\n61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71\n'



   The first three columns are year, month and day.  The
   remaining 12 columns are average windspeeds in knots at 12
   locations in Ireland on that day.   

   More information about the dataset go [here](wind.desc).

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data)

### Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data'
data = pd.read_csv(url, sep='\s+', parse_dates=[0,1,2])
```

### Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.


```python
data['Yr'] = data['Yr'].apply(lambda x : '19' + x)
```

### Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].


```python
data.set_index(pd.to_datetime(data['Yr']+data['Mo']+data['Dy'], format='%Y%m%d'), inplace=True)
data.drop(columns=['Yr', 'Mo', 'Dy'], axis=1, inplace=True)
```

### Step 6. Compute how many values are missing for each location over the entire record.  
#### They should be ignored in all calculations below. 


```python
data.isna().sum()
```




    RPT    6
    VAL    3
    ROS    2
    KIL    5
    SHA    2
    BIR    0
    DUB    3
    CLA    2
    MUL    3
    CLO    1
    BEL    0
    MAL    4
    dtype: int64



### Step 7. Compute how many non-missing values there are in total.


```python
(data.count() - data.isna().sum()).sum()
```




    78826



### Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
#### A single number for the entire dataset.


```python
data.mean().mean()
```




    10.227982360836924



### Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days 

#### A different set of numbers for each location.


```python
loc_stats = pd.DataFrame(data.describe()).T
loc_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RPT</th>
      <td>6568.0</td>
      <td>12.362987</td>
      <td>5.618413</td>
      <td>0.67</td>
      <td>8.12</td>
      <td>11.71</td>
      <td>15.92</td>
      <td>35.80</td>
    </tr>
    <tr>
      <th>VAL</th>
      <td>6571.0</td>
      <td>10.644314</td>
      <td>5.267356</td>
      <td>0.21</td>
      <td>6.67</td>
      <td>10.17</td>
      <td>14.04</td>
      <td>33.37</td>
    </tr>
    <tr>
      <th>ROS</th>
      <td>6572.0</td>
      <td>11.660526</td>
      <td>5.008450</td>
      <td>1.50</td>
      <td>8.00</td>
      <td>10.92</td>
      <td>14.67</td>
      <td>33.84</td>
    </tr>
    <tr>
      <th>KIL</th>
      <td>6569.0</td>
      <td>6.306468</td>
      <td>3.605811</td>
      <td>0.00</td>
      <td>3.58</td>
      <td>5.75</td>
      <td>8.42</td>
      <td>28.46</td>
    </tr>
    <tr>
      <th>SHA</th>
      <td>6572.0</td>
      <td>10.455834</td>
      <td>4.936125</td>
      <td>0.13</td>
      <td>6.75</td>
      <td>9.96</td>
      <td>13.54</td>
      <td>37.54</td>
    </tr>
    <tr>
      <th>BIR</th>
      <td>6574.0</td>
      <td>7.092254</td>
      <td>3.968683</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>6.83</td>
      <td>9.67</td>
      <td>26.16</td>
    </tr>
    <tr>
      <th>DUB</th>
      <td>6571.0</td>
      <td>9.797343</td>
      <td>4.977555</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>9.21</td>
      <td>12.96</td>
      <td>30.37</td>
    </tr>
    <tr>
      <th>CLA</th>
      <td>6572.0</td>
      <td>8.495053</td>
      <td>4.499449</td>
      <td>0.00</td>
      <td>5.09</td>
      <td>8.08</td>
      <td>11.42</td>
      <td>31.08</td>
    </tr>
    <tr>
      <th>MUL</th>
      <td>6571.0</td>
      <td>8.493590</td>
      <td>4.166872</td>
      <td>0.00</td>
      <td>5.37</td>
      <td>8.17</td>
      <td>11.19</td>
      <td>25.88</td>
    </tr>
    <tr>
      <th>CLO</th>
      <td>6573.0</td>
      <td>8.707332</td>
      <td>4.503954</td>
      <td>0.04</td>
      <td>5.33</td>
      <td>8.29</td>
      <td>11.63</td>
      <td>28.21</td>
    </tr>
    <tr>
      <th>BEL</th>
      <td>6574.0</td>
      <td>13.121007</td>
      <td>5.835037</td>
      <td>0.13</td>
      <td>8.71</td>
      <td>12.50</td>
      <td>16.88</td>
      <td>42.38</td>
    </tr>
    <tr>
      <th>MAL</th>
      <td>6570.0</td>
      <td>15.599079</td>
      <td>6.699794</td>
      <td>0.67</td>
      <td>10.71</td>
      <td>15.00</td>
      <td>19.83</td>
      <td>42.54</td>
    </tr>
  </tbody>
</table>
</div>



### Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.

#### A different set of numbers for each day.


```python
day_stats = pd.DataFrame(data.T.describe().T)
day_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1961-01-01</th>
      <td>11.0</td>
      <td>13.018182</td>
      <td>2.808875</td>
      <td>9.29</td>
      <td>10.5400</td>
      <td>13.170</td>
      <td>15.0000</td>
      <td>18.50</td>
    </tr>
    <tr>
      <th>1961-01-02</th>
      <td>11.0</td>
      <td>11.336364</td>
      <td>3.188994</td>
      <td>6.50</td>
      <td>9.7300</td>
      <td>10.830</td>
      <td>13.2250</td>
      <td>17.54</td>
    </tr>
    <tr>
      <th>1961-01-03</th>
      <td>11.0</td>
      <td>11.641818</td>
      <td>3.681912</td>
      <td>6.17</td>
      <td>9.3150</td>
      <td>11.250</td>
      <td>12.7300</td>
      <td>18.50</td>
    </tr>
    <tr>
      <th>1961-01-04</th>
      <td>12.0</td>
      <td>6.619167</td>
      <td>3.198126</td>
      <td>1.79</td>
      <td>4.5700</td>
      <td>5.855</td>
      <td>9.1175</td>
      <td>11.75</td>
    </tr>
    <tr>
      <th>1961-01-05</th>
      <td>12.0</td>
      <td>10.630000</td>
      <td>2.445356</td>
      <td>6.17</td>
      <td>9.8075</td>
      <td>11.170</td>
      <td>12.1700</td>
      <td>13.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1978-12-27</th>
      <td>12.0</td>
      <td>16.708333</td>
      <td>7.868076</td>
      <td>8.08</td>
      <td>13.8025</td>
      <td>15.025</td>
      <td>17.3025</td>
      <td>40.08</td>
    </tr>
    <tr>
      <th>1978-12-28</th>
      <td>12.0</td>
      <td>15.150000</td>
      <td>9.687857</td>
      <td>5.00</td>
      <td>9.0950</td>
      <td>13.895</td>
      <td>16.7000</td>
      <td>41.46</td>
    </tr>
    <tr>
      <th>1978-12-29</th>
      <td>12.0</td>
      <td>14.890000</td>
      <td>5.756836</td>
      <td>8.71</td>
      <td>10.4775</td>
      <td>14.210</td>
      <td>17.0350</td>
      <td>29.58</td>
    </tr>
    <tr>
      <th>1978-12-30</th>
      <td>12.0</td>
      <td>15.367500</td>
      <td>5.540437</td>
      <td>9.13</td>
      <td>12.3750</td>
      <td>13.455</td>
      <td>18.1850</td>
      <td>28.79</td>
    </tr>
    <tr>
      <th>1978-12-31</th>
      <td>12.0</td>
      <td>15.402500</td>
      <td>5.702483</td>
      <td>9.59</td>
      <td>11.5300</td>
      <td>12.080</td>
      <td>19.5200</td>
      <td>27.29</td>
    </tr>
  </tbody>
</table>
<p>6574 rows × 8 columns</p>
</div>



### Step 11. Find the average windspeed in January for each location.  
#### Treat January 1961 and January 1962 both as January.


```python
data[data.index.month == 1].mean()
```




    RPT    14.407735
    VAL    12.362146
    ROS    13.290470
    KIL     6.926239
    SHA    11.205064
    BIR     7.827393
    DUB    11.953120
    CLA     9.377511
    MUL     9.469915
    CLO     9.880812
    BEL    14.582350
    MAL    18.332051
    dtype: float64



### Step 12. Downsample the record to a yearly frequency for each location.


```python
data.groupby(data.index.year).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RPT</th>
      <th>VAL</th>
      <th>ROS</th>
      <th>KIL</th>
      <th>SHA</th>
      <th>BIR</th>
      <th>DUB</th>
      <th>CLA</th>
      <th>MUL</th>
      <th>CLO</th>
      <th>BEL</th>
      <th>MAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1961</th>
      <td>12.299583</td>
      <td>10.351796</td>
      <td>11.362369</td>
      <td>6.958227</td>
      <td>10.881763</td>
      <td>7.729726</td>
      <td>9.733923</td>
      <td>8.858788</td>
      <td>8.647652</td>
      <td>9.835577</td>
      <td>13.502795</td>
      <td>13.680773</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>12.246923</td>
      <td>10.110438</td>
      <td>11.732712</td>
      <td>6.960440</td>
      <td>10.657918</td>
      <td>7.393068</td>
      <td>11.020712</td>
      <td>8.793753</td>
      <td>8.316822</td>
      <td>9.676247</td>
      <td>12.930685</td>
      <td>14.323956</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>12.813452</td>
      <td>10.836986</td>
      <td>12.541151</td>
      <td>7.330055</td>
      <td>11.724110</td>
      <td>8.434712</td>
      <td>11.075699</td>
      <td>10.336548</td>
      <td>8.903589</td>
      <td>10.224438</td>
      <td>13.638877</td>
      <td>14.999014</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>12.363661</td>
      <td>10.920164</td>
      <td>12.104372</td>
      <td>6.787787</td>
      <td>11.454481</td>
      <td>7.570874</td>
      <td>10.259153</td>
      <td>9.467350</td>
      <td>7.789016</td>
      <td>10.207951</td>
      <td>13.740546</td>
      <td>14.910301</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>12.451370</td>
      <td>11.075534</td>
      <td>11.848767</td>
      <td>6.858466</td>
      <td>11.024795</td>
      <td>7.478110</td>
      <td>10.618712</td>
      <td>8.879918</td>
      <td>7.907425</td>
      <td>9.918082</td>
      <td>12.964247</td>
      <td>15.591644</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>13.461973</td>
      <td>11.557205</td>
      <td>12.020630</td>
      <td>7.345726</td>
      <td>11.805041</td>
      <td>7.793671</td>
      <td>10.579808</td>
      <td>8.835096</td>
      <td>8.514438</td>
      <td>9.768959</td>
      <td>14.265836</td>
      <td>16.307260</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>12.737151</td>
      <td>10.990986</td>
      <td>11.739397</td>
      <td>7.143425</td>
      <td>11.630740</td>
      <td>7.368164</td>
      <td>10.652027</td>
      <td>9.325616</td>
      <td>8.645014</td>
      <td>9.547425</td>
      <td>14.774548</td>
      <td>17.135945</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>11.835628</td>
      <td>10.468197</td>
      <td>11.409754</td>
      <td>6.477678</td>
      <td>10.760765</td>
      <td>6.067322</td>
      <td>8.859180</td>
      <td>8.255519</td>
      <td>7.224945</td>
      <td>7.832978</td>
      <td>12.808634</td>
      <td>15.017486</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>11.166356</td>
      <td>9.723699</td>
      <td>10.902000</td>
      <td>5.767973</td>
      <td>9.873918</td>
      <td>6.189973</td>
      <td>8.564493</td>
      <td>7.711397</td>
      <td>7.924521</td>
      <td>7.754384</td>
      <td>12.621233</td>
      <td>15.762904</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>12.600329</td>
      <td>10.726932</td>
      <td>11.730247</td>
      <td>6.217178</td>
      <td>10.567370</td>
      <td>7.609452</td>
      <td>9.609890</td>
      <td>8.334630</td>
      <td>9.297616</td>
      <td>8.289808</td>
      <td>13.183644</td>
      <td>16.456027</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>11.273123</td>
      <td>9.095178</td>
      <td>11.088329</td>
      <td>5.241507</td>
      <td>9.440329</td>
      <td>6.097151</td>
      <td>8.385890</td>
      <td>6.757315</td>
      <td>7.915370</td>
      <td>7.229753</td>
      <td>12.208932</td>
      <td>15.025233</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>12.463962</td>
      <td>10.561311</td>
      <td>12.058333</td>
      <td>5.929699</td>
      <td>9.430410</td>
      <td>6.358825</td>
      <td>9.704508</td>
      <td>7.680792</td>
      <td>8.357295</td>
      <td>7.515273</td>
      <td>12.727377</td>
      <td>15.028716</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>11.828466</td>
      <td>10.680493</td>
      <td>10.680493</td>
      <td>5.547863</td>
      <td>9.640877</td>
      <td>6.548740</td>
      <td>8.482110</td>
      <td>7.614274</td>
      <td>8.245534</td>
      <td>7.812411</td>
      <td>12.169699</td>
      <td>15.441096</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>13.643096</td>
      <td>11.811781</td>
      <td>12.336356</td>
      <td>6.427041</td>
      <td>11.110986</td>
      <td>6.809781</td>
      <td>10.084603</td>
      <td>9.896986</td>
      <td>9.331753</td>
      <td>8.736356</td>
      <td>13.252959</td>
      <td>16.947671</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>12.008575</td>
      <td>10.293836</td>
      <td>11.564712</td>
      <td>5.269096</td>
      <td>9.190082</td>
      <td>5.668521</td>
      <td>8.562603</td>
      <td>7.843836</td>
      <td>8.797945</td>
      <td>7.382822</td>
      <td>12.631671</td>
      <td>15.307863</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>11.737842</td>
      <td>10.203115</td>
      <td>10.761230</td>
      <td>5.109426</td>
      <td>8.846339</td>
      <td>6.311038</td>
      <td>9.149126</td>
      <td>7.146202</td>
      <td>8.883716</td>
      <td>7.883087</td>
      <td>12.332377</td>
      <td>15.471448</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>13.099616</td>
      <td>11.144493</td>
      <td>12.627836</td>
      <td>6.073945</td>
      <td>10.003836</td>
      <td>8.586438</td>
      <td>11.523205</td>
      <td>8.378384</td>
      <td>9.098192</td>
      <td>8.821616</td>
      <td>13.459068</td>
      <td>16.590849</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>12.504356</td>
      <td>11.044274</td>
      <td>11.380000</td>
      <td>6.082356</td>
      <td>10.167233</td>
      <td>7.650658</td>
      <td>9.489342</td>
      <td>8.800466</td>
      <td>9.089753</td>
      <td>8.301699</td>
      <td>12.967397</td>
      <td>16.771370</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13. Downsample the record to a monthly frequency for each location.


```python
data.groupby(data.index.month).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RPT</th>
      <th>VAL</th>
      <th>ROS</th>
      <th>KIL</th>
      <th>SHA</th>
      <th>BIR</th>
      <th>DUB</th>
      <th>CLA</th>
      <th>MUL</th>
      <th>CLO</th>
      <th>BEL</th>
      <th>MAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.407735</td>
      <td>12.362146</td>
      <td>13.290470</td>
      <td>6.926239</td>
      <td>11.205064</td>
      <td>7.827393</td>
      <td>11.953120</td>
      <td>9.377511</td>
      <td>9.469915</td>
      <td>9.880812</td>
      <td>14.582350</td>
      <td>18.332051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.710906</td>
      <td>12.111122</td>
      <td>12.879132</td>
      <td>6.942411</td>
      <td>11.551772</td>
      <td>7.633858</td>
      <td>11.206024</td>
      <td>9.341437</td>
      <td>9.313169</td>
      <td>9.518051</td>
      <td>13.728898</td>
      <td>17.156142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.158687</td>
      <td>11.505842</td>
      <td>12.648118</td>
      <td>7.265907</td>
      <td>11.554516</td>
      <td>7.959409</td>
      <td>11.310179</td>
      <td>9.635896</td>
      <td>9.700324</td>
      <td>10.096953</td>
      <td>13.810609</td>
      <td>16.909317</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.555648</td>
      <td>10.429759</td>
      <td>12.204815</td>
      <td>6.898037</td>
      <td>10.677667</td>
      <td>7.441389</td>
      <td>10.221315</td>
      <td>8.909056</td>
      <td>8.930870</td>
      <td>9.158019</td>
      <td>12.664759</td>
      <td>14.937611</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11.724032</td>
      <td>10.145619</td>
      <td>11.550394</td>
      <td>6.307487</td>
      <td>10.224301</td>
      <td>6.942061</td>
      <td>8.797738</td>
      <td>8.452903</td>
      <td>8.040806</td>
      <td>8.524857</td>
      <td>12.767258</td>
      <td>13.736039</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.451317</td>
      <td>8.949704</td>
      <td>10.361315</td>
      <td>5.652278</td>
      <td>9.529926</td>
      <td>6.410093</td>
      <td>8.009556</td>
      <td>7.920796</td>
      <td>7.639796</td>
      <td>7.729185</td>
      <td>12.246407</td>
      <td>12.861818</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.992007</td>
      <td>8.357778</td>
      <td>9.349642</td>
      <td>5.416935</td>
      <td>9.302634</td>
      <td>5.972348</td>
      <td>7.843501</td>
      <td>7.262760</td>
      <td>7.544480</td>
      <td>7.321416</td>
      <td>11.676505</td>
      <td>12.800789</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10.213411</td>
      <td>8.415143</td>
      <td>9.993441</td>
      <td>5.270681</td>
      <td>8.901559</td>
      <td>5.891057</td>
      <td>7.772312</td>
      <td>6.842025</td>
      <td>7.240573</td>
      <td>7.002783</td>
      <td>11.110090</td>
      <td>12.565943</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11.458519</td>
      <td>9.981002</td>
      <td>10.756883</td>
      <td>5.615176</td>
      <td>9.766315</td>
      <td>6.566222</td>
      <td>8.609722</td>
      <td>7.745677</td>
      <td>7.610556</td>
      <td>7.689278</td>
      <td>12.686389</td>
      <td>14.761963</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12.660610</td>
      <td>11.010681</td>
      <td>11.453943</td>
      <td>6.065215</td>
      <td>10.550251</td>
      <td>7.159910</td>
      <td>9.387778</td>
      <td>8.726308</td>
      <td>8.347181</td>
      <td>8.850376</td>
      <td>14.155323</td>
      <td>16.697151</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.778291</td>
      <td>12.140912</td>
      <td>12.663775</td>
      <td>6.567949</td>
      <td>10.952411</td>
      <td>7.447849</td>
      <td>11.137247</td>
      <td>8.753590</td>
      <td>8.874886</td>
      <td>9.258917</td>
      <td>14.044117</td>
      <td>18.066268</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14.486328</td>
      <td>12.456597</td>
      <td>13.100194</td>
      <td>6.903722</td>
      <td>11.354931</td>
      <td>7.959569</td>
      <td>11.711028</td>
      <td>9.246833</td>
      <td>9.439875</td>
      <td>9.721625</td>
      <td>14.257306</td>
      <td>18.476042</td>
    </tr>
  </tbody>
</table>
</div>



### Step 14. Downsample the record to a weekly frequency for each location.


```python
data.groupby(data.index.isocalendar().week).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RPT</th>
      <th>VAL</th>
      <th>ROS</th>
      <th>KIL</th>
      <th>SHA</th>
      <th>BIR</th>
      <th>DUB</th>
      <th>CLA</th>
      <th>MUL</th>
      <th>CLO</th>
      <th>BEL</th>
      <th>MAL</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13.920000</td>
      <td>11.710880</td>
      <td>12.853016</td>
      <td>6.617302</td>
      <td>10.473175</td>
      <td>7.578492</td>
      <td>11.623651</td>
      <td>9.123600</td>
      <td>9.272222</td>
      <td>9.870635</td>
      <td>14.241746</td>
      <td>18.249841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.444717</td>
      <td>13.504528</td>
      <td>14.104717</td>
      <td>7.094906</td>
      <td>12.230566</td>
      <td>8.184717</td>
      <td>12.113585</td>
      <td>9.939245</td>
      <td>9.692264</td>
      <td>9.559434</td>
      <td>14.948302</td>
      <td>18.242830</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.676154</td>
      <td>11.921538</td>
      <td>13.295385</td>
      <td>6.330769</td>
      <td>11.052308</td>
      <td>8.214615</td>
      <td>11.654615</td>
      <td>9.586154</td>
      <td>9.337692</td>
      <td>9.860769</td>
      <td>14.151538</td>
      <td>19.094615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.208333</td>
      <td>14.823333</td>
      <td>16.127500</td>
      <td>7.858333</td>
      <td>13.040833</td>
      <td>8.395000</td>
      <td>13.055833</td>
      <td>10.645833</td>
      <td>10.765000</td>
      <td>11.393333</td>
      <td>16.794167</td>
      <td>20.674167</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.640098</td>
      <td>12.991569</td>
      <td>13.281667</td>
      <td>7.487647</td>
      <td>11.969118</td>
      <td>8.234902</td>
      <td>12.148333</td>
      <td>9.689216</td>
      <td>9.834118</td>
      <td>10.153431</td>
      <td>14.735392</td>
      <td>17.584608</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13.203730</td>
      <td>11.508492</td>
      <td>12.142619</td>
      <td>6.469127</td>
      <td>11.025476</td>
      <td>7.166746</td>
      <td>11.115317</td>
      <td>8.894206</td>
      <td>9.323413</td>
      <td>9.298175</td>
      <td>13.903254</td>
      <td>17.921587</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14.517460</td>
      <td>12.681349</td>
      <td>13.219524</td>
      <td>7.148160</td>
      <td>12.141032</td>
      <td>7.906190</td>
      <td>11.390238</td>
      <td>9.749524</td>
      <td>9.417778</td>
      <td>9.605476</td>
      <td>13.776667</td>
      <td>17.130159</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.422857</td>
      <td>11.822937</td>
      <td>13.367381</td>
      <td>7.142400</td>
      <td>11.480635</td>
      <td>7.729683</td>
      <td>11.329048</td>
      <td>9.329762</td>
      <td>9.247063</td>
      <td>9.541984</td>
      <td>13.132302</td>
      <td>16.882540</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12.761360</td>
      <td>11.837460</td>
      <td>12.627440</td>
      <td>6.620000</td>
      <td>10.937143</td>
      <td>7.457778</td>
      <td>10.622778</td>
      <td>9.387778</td>
      <td>8.864160</td>
      <td>9.611508</td>
      <td>13.724048</td>
      <td>16.602080</td>
    </tr>
    <tr>
      <th>10</th>
      <td>13.526720</td>
      <td>11.758095</td>
      <td>13.151667</td>
      <td>7.317063</td>
      <td>11.477143</td>
      <td>7.965397</td>
      <td>10.751111</td>
      <td>9.151825</td>
      <td>9.367302</td>
      <td>9.761429</td>
      <td>13.055000</td>
      <td>16.047381</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.597143</td>
      <td>11.495079</td>
      <td>13.032698</td>
      <td>7.446270</td>
      <td>11.689206</td>
      <td>7.990238</td>
      <td>11.444365</td>
      <td>10.038968</td>
      <td>9.783730</td>
      <td>10.398730</td>
      <td>13.823968</td>
      <td>17.331280</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.006825</td>
      <td>11.621032</td>
      <td>12.579048</td>
      <td>7.390476</td>
      <td>11.850476</td>
      <td>8.288095</td>
      <td>12.008413</td>
      <td>10.136746</td>
      <td>10.176270</td>
      <td>10.259048</td>
      <td>14.389524</td>
      <td>17.859365</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12.772540</td>
      <td>11.076349</td>
      <td>12.104444</td>
      <td>7.386640</td>
      <td>11.664286</td>
      <td>8.042063</td>
      <td>11.306587</td>
      <td>9.494365</td>
      <td>9.977760</td>
      <td>10.038571</td>
      <td>14.241667</td>
      <td>16.486508</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13.903968</td>
      <td>11.025238</td>
      <td>14.149286</td>
      <td>7.798968</td>
      <td>11.482857</td>
      <td>8.006905</td>
      <td>11.525238</td>
      <td>9.579921</td>
      <td>9.894762</td>
      <td>9.948175</td>
      <td>13.386270</td>
      <td>16.691349</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12.599444</td>
      <td>9.966667</td>
      <td>11.654286</td>
      <td>6.758175</td>
      <td>10.642540</td>
      <td>7.377778</td>
      <td>10.677143</td>
      <td>9.328413</td>
      <td>9.248333</td>
      <td>9.338413</td>
      <td>12.842460</td>
      <td>15.900476</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12.753968</td>
      <td>11.111587</td>
      <td>11.966587</td>
      <td>7.110000</td>
      <td>10.983730</td>
      <td>7.660794</td>
      <td>9.981905</td>
      <td>8.763571</td>
      <td>8.802063</td>
      <td>9.326905</td>
      <td>12.472857</td>
      <td>14.229524</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10.966349</td>
      <td>9.647698</td>
      <td>11.272619</td>
      <td>6.019286</td>
      <td>9.627857</td>
      <td>6.687063</td>
      <td>8.870714</td>
      <td>8.091825</td>
      <td>7.836032</td>
      <td>8.238333</td>
      <td>11.971667</td>
      <td>13.097460</td>
    </tr>
    <tr>
      <th>18</th>
      <td>12.897222</td>
      <td>11.095079</td>
      <td>11.256825</td>
      <td>6.745794</td>
      <td>11.035952</td>
      <td>7.405238</td>
      <td>9.318730</td>
      <td>8.674206</td>
      <td>8.340952</td>
      <td>8.952857</td>
      <td>12.928968</td>
      <td>14.523810</td>
    </tr>
    <tr>
      <th>19</th>
      <td>13.061429</td>
      <td>11.275159</td>
      <td>12.630397</td>
      <td>7.005238</td>
      <td>10.944683</td>
      <td>7.559603</td>
      <td>9.624320</td>
      <td>9.111349</td>
      <td>8.604683</td>
      <td>9.243254</td>
      <td>13.721984</td>
      <td>14.288016</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11.338571</td>
      <td>9.769200</td>
      <td>11.566746</td>
      <td>6.195238</td>
      <td>9.967937</td>
      <td>6.811429</td>
      <td>8.732857</td>
      <td>8.495317</td>
      <td>7.984365</td>
      <td>8.507540</td>
      <td>12.740952</td>
      <td>13.859286</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11.282698</td>
      <td>9.673889</td>
      <td>11.223968</td>
      <td>5.975680</td>
      <td>10.051905</td>
      <td>6.730476</td>
      <td>8.435397</td>
      <td>8.150556</td>
      <td>7.871587</td>
      <td>8.278413</td>
      <td>12.536270</td>
      <td>13.626825</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9.673333</td>
      <td>8.264683</td>
      <td>10.382302</td>
      <td>5.195397</td>
      <td>8.696825</td>
      <td>5.802063</td>
      <td>7.275873</td>
      <td>7.533016</td>
      <td>7.120873</td>
      <td>7.247222</td>
      <td>11.238810</td>
      <td>11.770317</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10.092143</td>
      <td>8.980556</td>
      <td>10.030476</td>
      <td>5.304048</td>
      <td>8.882143</td>
      <td>5.816825</td>
      <td>7.420952</td>
      <td>7.387857</td>
      <td>7.031587</td>
      <td>7.173810</td>
      <td>11.605635</td>
      <td>12.395476</td>
    </tr>
    <tr>
      <th>24</th>
      <td>10.131905</td>
      <td>8.741984</td>
      <td>10.518254</td>
      <td>5.498492</td>
      <td>8.997381</td>
      <td>6.189762</td>
      <td>7.320317</td>
      <td>7.597143</td>
      <td>7.172857</td>
      <td>7.259524</td>
      <td>11.835873</td>
      <td>12.041270</td>
    </tr>
    <tr>
      <th>25</th>
      <td>11.680238</td>
      <td>9.898016</td>
      <td>10.730397</td>
      <td>6.294444</td>
      <td>11.030397</td>
      <td>7.410952</td>
      <td>9.410238</td>
      <td>8.822698</td>
      <td>8.728016</td>
      <td>8.677063</td>
      <td>13.494762</td>
      <td>14.329841</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10.036000</td>
      <td>8.770635</td>
      <td>10.113333</td>
      <td>5.798492</td>
      <td>9.680159</td>
      <td>6.474603</td>
      <td>8.226032</td>
      <td>7.920556</td>
      <td>7.906270</td>
      <td>7.937778</td>
      <td>12.462222</td>
      <td>12.993840</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9.991984</td>
      <td>7.916984</td>
      <td>8.741032</td>
      <td>5.276825</td>
      <td>8.998175</td>
      <td>5.852302</td>
      <td>7.817680</td>
      <td>7.182143</td>
      <td>7.662143</td>
      <td>7.451349</td>
      <td>11.582857</td>
      <td>13.096984</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9.962143</td>
      <td>8.272619</td>
      <td>9.651349</td>
      <td>5.461270</td>
      <td>9.344206</td>
      <td>5.951111</td>
      <td>7.937857</td>
      <td>7.341746</td>
      <td>7.671905</td>
      <td>7.267063</td>
      <td>11.450159</td>
      <td>12.932460</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10.281746</td>
      <td>8.422063</td>
      <td>9.755079</td>
      <td>5.672778</td>
      <td>9.766667</td>
      <td>6.300238</td>
      <td>8.277619</td>
      <td>7.568175</td>
      <td>7.725952</td>
      <td>7.616349</td>
      <td>12.021667</td>
      <td>12.880476</td>
    </tr>
    <tr>
      <th>30</th>
      <td>9.760397</td>
      <td>8.211984</td>
      <td>9.181587</td>
      <td>5.195476</td>
      <td>8.920873</td>
      <td>5.657222</td>
      <td>7.566905</td>
      <td>6.904603</td>
      <td>7.078571</td>
      <td>7.010317</td>
      <td>11.410794</td>
      <td>12.279683</td>
    </tr>
    <tr>
      <th>31</th>
      <td>9.721984</td>
      <td>8.218571</td>
      <td>9.585556</td>
      <td>5.078651</td>
      <td>8.940397</td>
      <td>5.692857</td>
      <td>7.470397</td>
      <td>6.646746</td>
      <td>7.039206</td>
      <td>6.615556</td>
      <td>11.020317</td>
      <td>12.348492</td>
    </tr>
    <tr>
      <th>32</th>
      <td>10.202640</td>
      <td>8.669683</td>
      <td>9.357619</td>
      <td>5.524206</td>
      <td>9.143492</td>
      <td>6.091984</td>
      <td>7.518413</td>
      <td>6.896190</td>
      <td>7.409365</td>
      <td>7.057840</td>
      <td>11.154365</td>
      <td>12.746587</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10.213254</td>
      <td>8.000476</td>
      <td>10.086667</td>
      <td>5.169127</td>
      <td>8.638413</td>
      <td>5.652619</td>
      <td>7.635635</td>
      <td>6.591746</td>
      <td>7.033413</td>
      <td>6.502460</td>
      <td>10.647222</td>
      <td>11.315680</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10.675556</td>
      <td>9.065317</td>
      <td>10.466190</td>
      <td>5.380635</td>
      <td>9.343651</td>
      <td>6.275317</td>
      <td>8.164603</td>
      <td>7.503889</td>
      <td>7.753730</td>
      <td>7.779206</td>
      <td>11.881667</td>
      <td>13.764762</td>
    </tr>
    <tr>
      <th>35</th>
      <td>10.378571</td>
      <td>8.525476</td>
      <td>10.496746</td>
      <td>5.398571</td>
      <td>8.739444</td>
      <td>5.938968</td>
      <td>8.088889</td>
      <td>6.907857</td>
      <td>7.063016</td>
      <td>7.213175</td>
      <td>11.606825</td>
      <td>13.390873</td>
    </tr>
    <tr>
      <th>36</th>
      <td>10.764365</td>
      <td>9.296190</td>
      <td>9.931270</td>
      <td>5.387381</td>
      <td>9.245873</td>
      <td>6.353810</td>
      <td>8.314603</td>
      <td>7.012857</td>
      <td>7.449762</td>
      <td>7.221905</td>
      <td>11.597619</td>
      <td>13.754921</td>
    </tr>
    <tr>
      <th>37</th>
      <td>11.371190</td>
      <td>9.722857</td>
      <td>10.963889</td>
      <td>5.638810</td>
      <td>9.603968</td>
      <td>6.646587</td>
      <td>8.698492</td>
      <td>7.704841</td>
      <td>7.532143</td>
      <td>7.645952</td>
      <td>12.658968</td>
      <td>14.827540</td>
    </tr>
    <tr>
      <th>38</th>
      <td>9.929524</td>
      <td>8.863095</td>
      <td>10.245280</td>
      <td>4.462080</td>
      <td>8.346190</td>
      <td>5.408175</td>
      <td>7.285079</td>
      <td>6.724640</td>
      <td>6.387381</td>
      <td>6.423095</td>
      <td>11.394127</td>
      <td>13.094921</td>
    </tr>
    <tr>
      <th>39</th>
      <td>13.960635</td>
      <td>12.361920</td>
      <td>11.787302</td>
      <td>6.898095</td>
      <td>11.971032</td>
      <td>7.882540</td>
      <td>10.012619</td>
      <td>9.489206</td>
      <td>8.933413</td>
      <td>9.402540</td>
      <td>14.980000</td>
      <td>16.928095</td>
    </tr>
    <tr>
      <th>40</th>
      <td>12.199841</td>
      <td>10.619127</td>
      <td>11.392698</td>
      <td>5.854048</td>
      <td>10.308175</td>
      <td>6.841508</td>
      <td>9.301905</td>
      <td>8.581111</td>
      <td>8.561667</td>
      <td>8.519524</td>
      <td>13.326429</td>
      <td>15.758095</td>
    </tr>
    <tr>
      <th>41</th>
      <td>12.260800</td>
      <td>10.581508</td>
      <td>11.316190</td>
      <td>5.468889</td>
      <td>9.998254</td>
      <td>6.667460</td>
      <td>8.667778</td>
      <td>8.279444</td>
      <td>7.782540</td>
      <td>8.176825</td>
      <td>13.550952</td>
      <td>15.663571</td>
    </tr>
    <tr>
      <th>42</th>
      <td>12.777460</td>
      <td>10.966587</td>
      <td>11.381667</td>
      <td>6.349048</td>
      <td>10.716032</td>
      <td>7.526032</td>
      <td>9.941349</td>
      <td>8.802222</td>
      <td>8.460873</td>
      <td>9.039286</td>
      <td>14.088175</td>
      <td>17.116111</td>
    </tr>
    <tr>
      <th>43</th>
      <td>13.358651</td>
      <td>11.915238</td>
      <td>11.888095</td>
      <td>6.554286</td>
      <td>11.363571</td>
      <td>7.769524</td>
      <td>10.170873</td>
      <td>9.458889</td>
      <td>8.925840</td>
      <td>9.800873</td>
      <td>16.055873</td>
      <td>18.580952</td>
    </tr>
    <tr>
      <th>44</th>
      <td>14.077179</td>
      <td>12.215077</td>
      <td>12.850000</td>
      <td>6.805128</td>
      <td>11.049692</td>
      <td>7.597077</td>
      <td>10.420667</td>
      <td>9.005949</td>
      <td>8.835436</td>
      <td>9.559487</td>
      <td>14.284103</td>
      <td>17.692462</td>
    </tr>
    <tr>
      <th>45</th>
      <td>14.400698</td>
      <td>12.762605</td>
      <td>13.244791</td>
      <td>6.862791</td>
      <td>11.556355</td>
      <td>7.894000</td>
      <td>11.490186</td>
      <td>9.447116</td>
      <td>9.364977</td>
      <td>9.642093</td>
      <td>14.449116</td>
      <td>18.232791</td>
    </tr>
    <tr>
      <th>46</th>
      <td>14.096846</td>
      <td>12.565000</td>
      <td>13.101462</td>
      <td>6.983154</td>
      <td>11.603462</td>
      <td>7.844615</td>
      <td>11.637769</td>
      <td>9.014615</td>
      <td>9.155154</td>
      <td>9.582923</td>
      <td>14.284692</td>
      <td>19.428923</td>
    </tr>
    <tr>
      <th>47</th>
      <td>11.731508</td>
      <td>10.361667</td>
      <td>11.107698</td>
      <td>5.413968</td>
      <td>9.200952</td>
      <td>6.172540</td>
      <td>10.071920</td>
      <td>7.322222</td>
      <td>7.690397</td>
      <td>7.933095</td>
      <td>12.399762</td>
      <td>16.473254</td>
    </tr>
    <tr>
      <th>48</th>
      <td>13.817239</td>
      <td>12.136074</td>
      <td>12.050061</td>
      <td>6.297791</td>
      <td>11.072699</td>
      <td>7.641472</td>
      <td>11.273804</td>
      <td>8.655706</td>
      <td>8.907791</td>
      <td>9.249939</td>
      <td>14.161288</td>
      <td>17.849571</td>
    </tr>
    <tr>
      <th>49</th>
      <td>14.776094</td>
      <td>13.048205</td>
      <td>12.752650</td>
      <td>7.306880</td>
      <td>11.743291</td>
      <td>8.209444</td>
      <td>12.114231</td>
      <td>9.496624</td>
      <td>9.844060</td>
      <td>10.012222</td>
      <td>14.860000</td>
      <td>18.847436</td>
    </tr>
    <tr>
      <th>50</th>
      <td>14.520350</td>
      <td>12.493846</td>
      <td>12.796224</td>
      <td>7.096643</td>
      <td>11.650769</td>
      <td>8.277483</td>
      <td>11.867343</td>
      <td>9.456434</td>
      <td>9.483706</td>
      <td>9.968462</td>
      <td>14.752168</td>
      <td>18.312517</td>
    </tr>
    <tr>
      <th>51</th>
      <td>13.639921</td>
      <td>11.697619</td>
      <td>12.501667</td>
      <td>5.991270</td>
      <td>10.541667</td>
      <td>7.275159</td>
      <td>10.301667</td>
      <td>8.462619</td>
      <td>8.438730</td>
      <td>8.687937</td>
      <td>13.298889</td>
      <td>17.182063</td>
    </tr>
    <tr>
      <th>52</th>
      <td>14.469134</td>
      <td>11.692283</td>
      <td>14.210000</td>
      <td>7.018189</td>
      <td>10.856905</td>
      <td>7.379213</td>
      <td>12.011969</td>
      <td>9.121811</td>
      <td>9.665197</td>
      <td>9.677953</td>
      <td>13.549055</td>
      <td>19.183228</td>
    </tr>
    <tr>
      <th>53</th>
      <td>12.694286</td>
      <td>9.451905</td>
      <td>13.526190</td>
      <td>5.343810</td>
      <td>8.040476</td>
      <td>5.884286</td>
      <td>10.366190</td>
      <td>5.556190</td>
      <td>6.954286</td>
      <td>7.820952</td>
      <td>11.028095</td>
      <td>17.103810</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.


```python
data.groupby(data.index.isocalendar().week).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">RPT</th>
      <th colspan="2" halign="left">VAL</th>
      <th>...</th>
      <th colspan="2" halign="left">BEL</th>
      <th colspan="8" halign="left">MAL</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>126.0</td>
      <td>13.920000</td>
      <td>6.341312</td>
      <td>2.62</td>
      <td>9.2600</td>
      <td>13.020</td>
      <td>17.3850</td>
      <td>33.34</td>
      <td>125.0</td>
      <td>11.710880</td>
      <td>...</td>
      <td>18.8700</td>
      <td>38.20</td>
      <td>126.0</td>
      <td>18.249841</td>
      <td>7.047385</td>
      <td>4.17</td>
      <td>13.3225</td>
      <td>17.435</td>
      <td>23.2900</td>
      <td>37.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53.0</td>
      <td>15.444717</td>
      <td>5.462035</td>
      <td>5.29</td>
      <td>11.3400</td>
      <td>15.090</td>
      <td>19.7500</td>
      <td>28.75</td>
      <td>53.0</td>
      <td>13.504528</td>
      <td>...</td>
      <td>18.8400</td>
      <td>31.08</td>
      <td>53.0</td>
      <td>18.242830</td>
      <td>6.881317</td>
      <td>3.63</td>
      <td>12.5000</td>
      <td>19.040</td>
      <td>23.3300</td>
      <td>31.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.0</td>
      <td>13.676154</td>
      <td>4.875431</td>
      <td>7.54</td>
      <td>9.7100</td>
      <td>11.460</td>
      <td>18.0800</td>
      <td>22.37</td>
      <td>13.0</td>
      <td>11.921538</td>
      <td>...</td>
      <td>15.5000</td>
      <td>23.04</td>
      <td>13.0</td>
      <td>19.094615</td>
      <td>6.579779</td>
      <td>6.79</td>
      <td>15.7500</td>
      <td>19.550</td>
      <td>20.8300</td>
      <td>32.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.0</td>
      <td>16.208333</td>
      <td>6.112314</td>
      <td>2.08</td>
      <td>11.7600</td>
      <td>17.870</td>
      <td>19.5925</td>
      <td>24.25</td>
      <td>12.0</td>
      <td>14.823333</td>
      <td>...</td>
      <td>20.8150</td>
      <td>29.88</td>
      <td>12.0</td>
      <td>20.674167</td>
      <td>8.356038</td>
      <td>12.29</td>
      <td>14.1975</td>
      <td>17.795</td>
      <td>25.5275</td>
      <td>40.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>102.0</td>
      <td>14.640098</td>
      <td>5.648340</td>
      <td>2.71</td>
      <td>10.8425</td>
      <td>14.310</td>
      <td>18.5375</td>
      <td>33.84</td>
      <td>102.0</td>
      <td>12.991569</td>
      <td>...</td>
      <td>19.1350</td>
      <td>27.71</td>
      <td>102.0</td>
      <td>17.584608</td>
      <td>6.482263</td>
      <td>4.83</td>
      <td>12.9350</td>
      <td>17.625</td>
      <td>22.1700</td>
      <td>34.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>126.0</td>
      <td>13.203730</td>
      <td>5.620867</td>
      <td>2.79</td>
      <td>9.1800</td>
      <td>12.105</td>
      <td>17.3600</td>
      <td>30.13</td>
      <td>126.0</td>
      <td>11.508492</td>
      <td>...</td>
      <td>17.8200</td>
      <td>28.12</td>
      <td>126.0</td>
      <td>17.921587</td>
      <td>6.092634</td>
      <td>5.29</td>
      <td>12.9600</td>
      <td>17.330</td>
      <td>21.6925</td>
      <td>33.04</td>
    </tr>
    <tr>
      <th>7</th>
      <td>126.0</td>
      <td>14.517460</td>
      <td>5.916627</td>
      <td>2.92</td>
      <td>10.2725</td>
      <td>14.190</td>
      <td>18.2925</td>
      <td>29.17</td>
      <td>126.0</td>
      <td>12.681349</td>
      <td>...</td>
      <td>18.2700</td>
      <td>32.08</td>
      <td>126.0</td>
      <td>17.130159</td>
      <td>6.819665</td>
      <td>5.09</td>
      <td>11.8225</td>
      <td>16.315</td>
      <td>21.7900</td>
      <td>37.04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>126.0</td>
      <td>13.422857</td>
      <td>5.923096</td>
      <td>1.54</td>
      <td>9.0625</td>
      <td>12.310</td>
      <td>17.4900</td>
      <td>32.38</td>
      <td>126.0</td>
      <td>11.822937</td>
      <td>...</td>
      <td>16.7275</td>
      <td>35.08</td>
      <td>126.0</td>
      <td>16.882540</td>
      <td>7.251216</td>
      <td>3.04</td>
      <td>11.3025</td>
      <td>16.130</td>
      <td>22.2200</td>
      <td>38.20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>125.0</td>
      <td>12.761360</td>
      <td>5.240324</td>
      <td>0.67</td>
      <td>9.0800</td>
      <td>12.580</td>
      <td>16.3800</td>
      <td>28.91</td>
      <td>126.0</td>
      <td>11.837460</td>
      <td>...</td>
      <td>18.7950</td>
      <td>31.13</td>
      <td>125.0</td>
      <td>16.602080</td>
      <td>6.191920</td>
      <td>3.17</td>
      <td>12.5000</td>
      <td>15.750</td>
      <td>20.5400</td>
      <td>31.66</td>
    </tr>
    <tr>
      <th>10</th>
      <td>125.0</td>
      <td>13.526720</td>
      <td>6.165086</td>
      <td>2.29</td>
      <td>8.7100</td>
      <td>12.620</td>
      <td>17.0400</td>
      <td>35.80</td>
      <td>126.0</td>
      <td>11.758095</td>
      <td>...</td>
      <td>17.2275</td>
      <td>32.63</td>
      <td>126.0</td>
      <td>16.047381</td>
      <td>6.572057</td>
      <td>2.46</td>
      <td>11.3900</td>
      <td>15.670</td>
      <td>20.4700</td>
      <td>37.59</td>
    </tr>
    <tr>
      <th>11</th>
      <td>126.0</td>
      <td>13.597143</td>
      <td>6.412912</td>
      <td>1.46</td>
      <td>8.8725</td>
      <td>12.665</td>
      <td>18.1450</td>
      <td>33.04</td>
      <td>126.0</td>
      <td>11.495079</td>
      <td>...</td>
      <td>19.0175</td>
      <td>34.92</td>
      <td>125.0</td>
      <td>17.331280</td>
      <td>7.428510</td>
      <td>4.54</td>
      <td>11.5400</td>
      <td>16.880</td>
      <td>21.4600</td>
      <td>40.37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>126.0</td>
      <td>13.006825</td>
      <td>5.509073</td>
      <td>2.92</td>
      <td>9.2400</td>
      <td>12.645</td>
      <td>16.8475</td>
      <td>32.13</td>
      <td>126.0</td>
      <td>11.621032</td>
      <td>...</td>
      <td>18.6875</td>
      <td>31.71</td>
      <td>126.0</td>
      <td>17.859365</td>
      <td>6.902719</td>
      <td>3.21</td>
      <td>13.2650</td>
      <td>18.185</td>
      <td>22.3075</td>
      <td>37.99</td>
    </tr>
    <tr>
      <th>13</th>
      <td>126.0</td>
      <td>12.772540</td>
      <td>5.278774</td>
      <td>2.79</td>
      <td>8.6125</td>
      <td>12.895</td>
      <td>16.0500</td>
      <td>27.12</td>
      <td>126.0</td>
      <td>11.076349</td>
      <td>...</td>
      <td>17.5800</td>
      <td>32.21</td>
      <td>126.0</td>
      <td>16.486508</td>
      <td>6.644671</td>
      <td>2.04</td>
      <td>12.6525</td>
      <td>15.955</td>
      <td>20.3800</td>
      <td>31.20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>126.0</td>
      <td>13.903968</td>
      <td>5.698092</td>
      <td>3.88</td>
      <td>9.8225</td>
      <td>13.790</td>
      <td>17.4900</td>
      <td>32.58</td>
      <td>126.0</td>
      <td>11.025238</td>
      <td>...</td>
      <td>17.4100</td>
      <td>25.50</td>
      <td>126.0</td>
      <td>16.691349</td>
      <td>6.979889</td>
      <td>3.71</td>
      <td>11.8225</td>
      <td>16.125</td>
      <td>20.9775</td>
      <td>34.08</td>
    </tr>
    <tr>
      <th>15</th>
      <td>126.0</td>
      <td>12.599444</td>
      <td>5.335705</td>
      <td>4.42</td>
      <td>8.4300</td>
      <td>11.650</td>
      <td>15.6900</td>
      <td>29.79</td>
      <td>126.0</td>
      <td>9.966667</td>
      <td>...</td>
      <td>15.9975</td>
      <td>27.00</td>
      <td>126.0</td>
      <td>15.900476</td>
      <td>6.795427</td>
      <td>4.33</td>
      <td>11.3225</td>
      <td>14.335</td>
      <td>20.7775</td>
      <td>33.95</td>
    </tr>
    <tr>
      <th>16</th>
      <td>126.0</td>
      <td>12.753968</td>
      <td>5.248461</td>
      <td>3.67</td>
      <td>8.6325</td>
      <td>12.145</td>
      <td>16.2375</td>
      <td>27.84</td>
      <td>126.0</td>
      <td>11.111587</td>
      <td>...</td>
      <td>15.5675</td>
      <td>23.96</td>
      <td>126.0</td>
      <td>14.229524</td>
      <td>5.328738</td>
      <td>4.17</td>
      <td>10.4800</td>
      <td>13.750</td>
      <td>17.6975</td>
      <td>28.67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>126.0</td>
      <td>10.966349</td>
      <td>5.378706</td>
      <td>3.46</td>
      <td>6.3225</td>
      <td>9.960</td>
      <td>14.0900</td>
      <td>26.83</td>
      <td>126.0</td>
      <td>9.647698</td>
      <td>...</td>
      <td>15.7300</td>
      <td>26.12</td>
      <td>126.0</td>
      <td>13.097460</td>
      <td>6.018248</td>
      <td>2.21</td>
      <td>8.3800</td>
      <td>12.580</td>
      <td>16.8250</td>
      <td>32.05</td>
    </tr>
    <tr>
      <th>18</th>
      <td>126.0</td>
      <td>12.897222</td>
      <td>4.367054</td>
      <td>1.63</td>
      <td>9.8825</td>
      <td>12.830</td>
      <td>16.0650</td>
      <td>27.25</td>
      <td>126.0</td>
      <td>11.095079</td>
      <td>...</td>
      <td>16.3800</td>
      <td>28.33</td>
      <td>126.0</td>
      <td>14.523810</td>
      <td>6.216011</td>
      <td>2.58</td>
      <td>10.2000</td>
      <td>13.415</td>
      <td>18.6475</td>
      <td>28.75</td>
    </tr>
    <tr>
      <th>19</th>
      <td>126.0</td>
      <td>13.061429</td>
      <td>5.517038</td>
      <td>3.54</td>
      <td>8.5925</td>
      <td>13.025</td>
      <td>16.4900</td>
      <td>30.91</td>
      <td>126.0</td>
      <td>11.275159</td>
      <td>...</td>
      <td>17.2875</td>
      <td>32.91</td>
      <td>126.0</td>
      <td>14.288016</td>
      <td>5.699596</td>
      <td>3.33</td>
      <td>10.0625</td>
      <td>14.500</td>
      <td>18.3400</td>
      <td>26.83</td>
    </tr>
    <tr>
      <th>20</th>
      <td>126.0</td>
      <td>11.338571</td>
      <td>5.404169</td>
      <td>2.42</td>
      <td>6.8400</td>
      <td>10.125</td>
      <td>15.0775</td>
      <td>25.17</td>
      <td>125.0</td>
      <td>9.769200</td>
      <td>...</td>
      <td>16.1475</td>
      <td>26.42</td>
      <td>126.0</td>
      <td>13.859286</td>
      <td>5.865042</td>
      <td>3.58</td>
      <td>9.5225</td>
      <td>13.480</td>
      <td>17.3850</td>
      <td>32.17</td>
    </tr>
    <tr>
      <th>21</th>
      <td>126.0</td>
      <td>11.282698</td>
      <td>5.200316</td>
      <td>2.62</td>
      <td>7.5600</td>
      <td>10.815</td>
      <td>13.9725</td>
      <td>28.79</td>
      <td>126.0</td>
      <td>9.673889</td>
      <td>...</td>
      <td>15.6975</td>
      <td>28.12</td>
      <td>126.0</td>
      <td>13.626825</td>
      <td>5.258812</td>
      <td>4.04</td>
      <td>9.4550</td>
      <td>13.480</td>
      <td>16.6875</td>
      <td>29.50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>126.0</td>
      <td>9.673333</td>
      <td>4.053188</td>
      <td>2.04</td>
      <td>6.7300</td>
      <td>9.250</td>
      <td>11.9875</td>
      <td>21.84</td>
      <td>126.0</td>
      <td>8.264683</td>
      <td>...</td>
      <td>13.6575</td>
      <td>26.38</td>
      <td>126.0</td>
      <td>11.770317</td>
      <td>5.001233</td>
      <td>1.75</td>
      <td>7.8525</td>
      <td>11.650</td>
      <td>15.1000</td>
      <td>28.04</td>
    </tr>
    <tr>
      <th>23</th>
      <td>126.0</td>
      <td>10.092143</td>
      <td>4.516349</td>
      <td>2.17</td>
      <td>6.7600</td>
      <td>9.665</td>
      <td>12.5700</td>
      <td>23.21</td>
      <td>126.0</td>
      <td>8.980556</td>
      <td>...</td>
      <td>14.5400</td>
      <td>21.34</td>
      <td>126.0</td>
      <td>12.395476</td>
      <td>5.383357</td>
      <td>2.21</td>
      <td>8.3825</td>
      <td>11.895</td>
      <td>16.0700</td>
      <td>32.79</td>
    </tr>
    <tr>
      <th>24</th>
      <td>126.0</td>
      <td>10.131905</td>
      <td>4.410570</td>
      <td>1.00</td>
      <td>6.5400</td>
      <td>9.960</td>
      <td>12.8300</td>
      <td>21.59</td>
      <td>126.0</td>
      <td>8.741984</td>
      <td>...</td>
      <td>14.6475</td>
      <td>25.25</td>
      <td>126.0</td>
      <td>12.041270</td>
      <td>5.419266</td>
      <td>2.79</td>
      <td>7.8525</td>
      <td>11.685</td>
      <td>15.3400</td>
      <td>27.16</td>
    </tr>
    <tr>
      <th>25</th>
      <td>126.0</td>
      <td>11.680238</td>
      <td>4.415008</td>
      <td>2.92</td>
      <td>7.9700</td>
      <td>11.145</td>
      <td>14.5300</td>
      <td>22.71</td>
      <td>126.0</td>
      <td>9.898016</td>
      <td>...</td>
      <td>16.2400</td>
      <td>29.79</td>
      <td>126.0</td>
      <td>14.329841</td>
      <td>5.873487</td>
      <td>3.08</td>
      <td>10.1400</td>
      <td>14.020</td>
      <td>17.4475</td>
      <td>28.38</td>
    </tr>
    <tr>
      <th>26</th>
      <td>125.0</td>
      <td>10.036000</td>
      <td>3.868826</td>
      <td>3.46</td>
      <td>6.9200</td>
      <td>9.670</td>
      <td>12.7900</td>
      <td>20.50</td>
      <td>126.0</td>
      <td>8.770635</td>
      <td>...</td>
      <td>15.6050</td>
      <td>24.71</td>
      <td>125.0</td>
      <td>12.993840</td>
      <td>4.516285</td>
      <td>4.54</td>
      <td>9.3300</td>
      <td>12.710</td>
      <td>15.9200</td>
      <td>25.17</td>
    </tr>
    <tr>
      <th>27</th>
      <td>126.0</td>
      <td>9.991984</td>
      <td>4.742847</td>
      <td>2.00</td>
      <td>6.7900</td>
      <td>8.960</td>
      <td>12.4100</td>
      <td>25.84</td>
      <td>126.0</td>
      <td>7.916984</td>
      <td>...</td>
      <td>14.5000</td>
      <td>23.00</td>
      <td>126.0</td>
      <td>13.096984</td>
      <td>5.292558</td>
      <td>4.50</td>
      <td>9.0800</td>
      <td>12.560</td>
      <td>16.2800</td>
      <td>29.63</td>
    </tr>
    <tr>
      <th>28</th>
      <td>126.0</td>
      <td>9.962143</td>
      <td>4.661313</td>
      <td>2.75</td>
      <td>6.2500</td>
      <td>9.105</td>
      <td>13.3500</td>
      <td>22.50</td>
      <td>126.0</td>
      <td>8.272619</td>
      <td>...</td>
      <td>14.4175</td>
      <td>23.83</td>
      <td>126.0</td>
      <td>12.932460</td>
      <td>5.565273</td>
      <td>2.13</td>
      <td>8.7400</td>
      <td>12.420</td>
      <td>16.3200</td>
      <td>28.46</td>
    </tr>
    <tr>
      <th>29</th>
      <td>126.0</td>
      <td>10.281746</td>
      <td>4.578130</td>
      <td>1.71</td>
      <td>6.8325</td>
      <td>10.330</td>
      <td>13.4575</td>
      <td>22.34</td>
      <td>126.0</td>
      <td>8.422063</td>
      <td>...</td>
      <td>15.2175</td>
      <td>24.41</td>
      <td>126.0</td>
      <td>12.880476</td>
      <td>5.472898</td>
      <td>2.29</td>
      <td>8.5000</td>
      <td>12.810</td>
      <td>17.3800</td>
      <td>25.37</td>
    </tr>
    <tr>
      <th>30</th>
      <td>126.0</td>
      <td>9.760397</td>
      <td>4.056483</td>
      <td>1.25</td>
      <td>6.4800</td>
      <td>9.335</td>
      <td>12.5500</td>
      <td>20.33</td>
      <td>126.0</td>
      <td>8.211984</td>
      <td>...</td>
      <td>14.2700</td>
      <td>22.37</td>
      <td>126.0</td>
      <td>12.279683</td>
      <td>5.055287</td>
      <td>2.88</td>
      <td>8.2100</td>
      <td>12.310</td>
      <td>15.6900</td>
      <td>25.37</td>
    </tr>
    <tr>
      <th>31</th>
      <td>126.0</td>
      <td>9.721984</td>
      <td>4.143365</td>
      <td>2.79</td>
      <td>6.5825</td>
      <td>9.420</td>
      <td>12.1075</td>
      <td>22.42</td>
      <td>126.0</td>
      <td>8.218571</td>
      <td>...</td>
      <td>13.7375</td>
      <td>21.67</td>
      <td>126.0</td>
      <td>12.348492</td>
      <td>5.053851</td>
      <td>2.25</td>
      <td>9.0925</td>
      <td>12.625</td>
      <td>16.3875</td>
      <td>24.83</td>
    </tr>
    <tr>
      <th>32</th>
      <td>125.0</td>
      <td>10.202640</td>
      <td>3.982815</td>
      <td>2.67</td>
      <td>6.8700</td>
      <td>10.250</td>
      <td>13.1300</td>
      <td>20.08</td>
      <td>126.0</td>
      <td>8.669683</td>
      <td>...</td>
      <td>13.6925</td>
      <td>24.08</td>
      <td>126.0</td>
      <td>12.746587</td>
      <td>5.353812</td>
      <td>3.04</td>
      <td>8.9650</td>
      <td>12.500</td>
      <td>16.3125</td>
      <td>29.95</td>
    </tr>
    <tr>
      <th>33</th>
      <td>126.0</td>
      <td>10.213254</td>
      <td>4.914926</td>
      <td>3.17</td>
      <td>6.6400</td>
      <td>9.040</td>
      <td>13.4075</td>
      <td>26.38</td>
      <td>126.0</td>
      <td>8.000476</td>
      <td>...</td>
      <td>13.5650</td>
      <td>23.25</td>
      <td>125.0</td>
      <td>11.315680</td>
      <td>5.587084</td>
      <td>2.17</td>
      <td>7.1700</td>
      <td>10.750</td>
      <td>14.6200</td>
      <td>34.33</td>
    </tr>
    <tr>
      <th>34</th>
      <td>126.0</td>
      <td>10.675556</td>
      <td>4.874965</td>
      <td>2.42</td>
      <td>7.1700</td>
      <td>10.460</td>
      <td>14.2800</td>
      <td>24.67</td>
      <td>126.0</td>
      <td>9.065317</td>
      <td>...</td>
      <td>16.0700</td>
      <td>24.83</td>
      <td>126.0</td>
      <td>13.764762</td>
      <td>5.642352</td>
      <td>3.37</td>
      <td>9.1250</td>
      <td>13.290</td>
      <td>18.0000</td>
      <td>30.46</td>
    </tr>
    <tr>
      <th>35</th>
      <td>126.0</td>
      <td>10.378571</td>
      <td>4.976603</td>
      <td>0.96</td>
      <td>6.0800</td>
      <td>10.040</td>
      <td>13.5900</td>
      <td>23.38</td>
      <td>126.0</td>
      <td>8.525476</td>
      <td>...</td>
      <td>14.7900</td>
      <td>26.46</td>
      <td>126.0</td>
      <td>13.390873</td>
      <td>5.962103</td>
      <td>3.83</td>
      <td>8.7500</td>
      <td>12.395</td>
      <td>17.3900</td>
      <td>28.84</td>
    </tr>
    <tr>
      <th>36</th>
      <td>126.0</td>
      <td>10.764365</td>
      <td>5.086312</td>
      <td>1.50</td>
      <td>6.6500</td>
      <td>10.400</td>
      <td>14.3675</td>
      <td>29.54</td>
      <td>126.0</td>
      <td>9.296190</td>
      <td>...</td>
      <td>14.9150</td>
      <td>24.17</td>
      <td>126.0</td>
      <td>13.754921</td>
      <td>5.692819</td>
      <td>3.37</td>
      <td>9.4700</td>
      <td>13.940</td>
      <td>17.0300</td>
      <td>30.34</td>
    </tr>
    <tr>
      <th>37</th>
      <td>126.0</td>
      <td>11.371190</td>
      <td>5.662910</td>
      <td>1.79</td>
      <td>7.2600</td>
      <td>10.775</td>
      <td>14.9800</td>
      <td>31.42</td>
      <td>126.0</td>
      <td>9.722857</td>
      <td>...</td>
      <td>16.2500</td>
      <td>25.75</td>
      <td>126.0</td>
      <td>14.827540</td>
      <td>6.350553</td>
      <td>2.83</td>
      <td>10.3800</td>
      <td>14.830</td>
      <td>18.0500</td>
      <td>35.13</td>
    </tr>
    <tr>
      <th>38</th>
      <td>126.0</td>
      <td>9.929524</td>
      <td>4.585719</td>
      <td>0.79</td>
      <td>6.5325</td>
      <td>9.165</td>
      <td>12.7800</td>
      <td>24.79</td>
      <td>126.0</td>
      <td>8.863095</td>
      <td>...</td>
      <td>14.7400</td>
      <td>23.04</td>
      <td>126.0</td>
      <td>13.094921</td>
      <td>5.420672</td>
      <td>0.67</td>
      <td>9.0825</td>
      <td>12.670</td>
      <td>17.2975</td>
      <td>29.63</td>
    </tr>
    <tr>
      <th>39</th>
      <td>126.0</td>
      <td>13.960635</td>
      <td>5.313510</td>
      <td>3.37</td>
      <td>10.0100</td>
      <td>13.545</td>
      <td>17.1300</td>
      <td>27.00</td>
      <td>125.0</td>
      <td>12.361920</td>
      <td>...</td>
      <td>19.4950</td>
      <td>28.79</td>
      <td>126.0</td>
      <td>16.928095</td>
      <td>6.943457</td>
      <td>4.04</td>
      <td>11.9125</td>
      <td>15.895</td>
      <td>20.8100</td>
      <td>33.63</td>
    </tr>
    <tr>
      <th>40</th>
      <td>126.0</td>
      <td>12.199841</td>
      <td>4.458238</td>
      <td>2.37</td>
      <td>8.9200</td>
      <td>12.380</td>
      <td>15.0900</td>
      <td>25.29</td>
      <td>126.0</td>
      <td>10.619127</td>
      <td>...</td>
      <td>17.4275</td>
      <td>26.71</td>
      <td>126.0</td>
      <td>15.758095</td>
      <td>6.364950</td>
      <td>2.04</td>
      <td>11.2900</td>
      <td>15.565</td>
      <td>20.4100</td>
      <td>32.96</td>
    </tr>
    <tr>
      <th>41</th>
      <td>125.0</td>
      <td>12.260800</td>
      <td>5.400736</td>
      <td>3.04</td>
      <td>8.2900</td>
      <td>11.790</td>
      <td>15.1600</td>
      <td>25.92</td>
      <td>126.0</td>
      <td>10.581508</td>
      <td>...</td>
      <td>17.4375</td>
      <td>34.83</td>
      <td>126.0</td>
      <td>15.663571</td>
      <td>6.537501</td>
      <td>3.33</td>
      <td>10.7200</td>
      <td>15.120</td>
      <td>19.9875</td>
      <td>36.51</td>
    </tr>
    <tr>
      <th>42</th>
      <td>126.0</td>
      <td>12.777460</td>
      <td>5.718162</td>
      <td>2.04</td>
      <td>8.6600</td>
      <td>12.145</td>
      <td>16.2700</td>
      <td>28.62</td>
      <td>126.0</td>
      <td>10.966587</td>
      <td>...</td>
      <td>17.0825</td>
      <td>32.63</td>
      <td>126.0</td>
      <td>17.116111</td>
      <td>6.401907</td>
      <td>5.04</td>
      <td>12.1125</td>
      <td>16.395</td>
      <td>21.5900</td>
      <td>33.45</td>
    </tr>
    <tr>
      <th>43</th>
      <td>126.0</td>
      <td>13.358651</td>
      <td>6.116732</td>
      <td>1.50</td>
      <td>9.0200</td>
      <td>12.670</td>
      <td>17.0100</td>
      <td>29.08</td>
      <td>126.0</td>
      <td>11.915238</td>
      <td>...</td>
      <td>20.7000</td>
      <td>38.96</td>
      <td>126.0</td>
      <td>18.580952</td>
      <td>6.799425</td>
      <td>6.17</td>
      <td>12.9900</td>
      <td>17.965</td>
      <td>23.4100</td>
      <td>36.63</td>
    </tr>
    <tr>
      <th>44</th>
      <td>195.0</td>
      <td>14.077179</td>
      <td>5.833335</td>
      <td>2.00</td>
      <td>9.9600</td>
      <td>13.620</td>
      <td>17.5400</td>
      <td>32.17</td>
      <td>195.0</td>
      <td>12.215077</td>
      <td>...</td>
      <td>18.1450</td>
      <td>39.04</td>
      <td>195.0</td>
      <td>17.692462</td>
      <td>7.021542</td>
      <td>4.54</td>
      <td>12.3550</td>
      <td>17.250</td>
      <td>21.8400</td>
      <td>37.59</td>
    </tr>
    <tr>
      <th>45</th>
      <td>215.0</td>
      <td>14.400698</td>
      <td>6.254240</td>
      <td>1.63</td>
      <td>9.7700</td>
      <td>13.790</td>
      <td>18.1200</td>
      <td>33.12</td>
      <td>215.0</td>
      <td>12.762605</td>
      <td>...</td>
      <td>18.7300</td>
      <td>33.34</td>
      <td>215.0</td>
      <td>18.232791</td>
      <td>7.140148</td>
      <td>2.96</td>
      <td>13.2900</td>
      <td>17.960</td>
      <td>23.1300</td>
      <td>38.04</td>
    </tr>
    <tr>
      <th>46</th>
      <td>130.0</td>
      <td>14.096846</td>
      <td>6.405892</td>
      <td>2.92</td>
      <td>9.0800</td>
      <td>13.000</td>
      <td>18.7800</td>
      <td>30.21</td>
      <td>130.0</td>
      <td>12.565000</td>
      <td>...</td>
      <td>18.8800</td>
      <td>31.96</td>
      <td>130.0</td>
      <td>19.428923</td>
      <td>8.585392</td>
      <td>2.00</td>
      <td>13.1800</td>
      <td>18.935</td>
      <td>24.7150</td>
      <td>41.25</td>
    </tr>
    <tr>
      <th>47</th>
      <td>126.0</td>
      <td>11.731508</td>
      <td>5.234852</td>
      <td>2.67</td>
      <td>8.0950</td>
      <td>10.750</td>
      <td>14.8300</td>
      <td>26.50</td>
      <td>126.0</td>
      <td>10.361667</td>
      <td>...</td>
      <td>16.4100</td>
      <td>30.00</td>
      <td>126.0</td>
      <td>16.473254</td>
      <td>6.677397</td>
      <td>4.25</td>
      <td>10.8625</td>
      <td>15.710</td>
      <td>20.8900</td>
      <td>34.83</td>
    </tr>
    <tr>
      <th>48</th>
      <td>163.0</td>
      <td>13.817239</td>
      <td>6.125372</td>
      <td>3.08</td>
      <td>9.0800</td>
      <td>13.170</td>
      <td>17.8100</td>
      <td>34.37</td>
      <td>163.0</td>
      <td>12.136074</td>
      <td>...</td>
      <td>18.7900</td>
      <td>42.38</td>
      <td>163.0</td>
      <td>17.849571</td>
      <td>6.642826</td>
      <td>4.79</td>
      <td>12.7300</td>
      <td>17.290</td>
      <td>22.5850</td>
      <td>42.54</td>
    </tr>
    <tr>
      <th>49</th>
      <td>233.0</td>
      <td>14.776094</td>
      <td>6.192947</td>
      <td>0.67</td>
      <td>10.3400</td>
      <td>13.920</td>
      <td>19.0800</td>
      <td>35.38</td>
      <td>234.0</td>
      <td>13.048205</td>
      <td>...</td>
      <td>18.8700</td>
      <td>34.42</td>
      <td>234.0</td>
      <td>18.847436</td>
      <td>7.020126</td>
      <td>3.25</td>
      <td>14.2300</td>
      <td>19.165</td>
      <td>23.0975</td>
      <td>38.79</td>
    </tr>
    <tr>
      <th>50</th>
      <td>143.0</td>
      <td>14.520350</td>
      <td>6.803921</td>
      <td>1.96</td>
      <td>9.7300</td>
      <td>14.420</td>
      <td>18.2900</td>
      <td>30.91</td>
      <td>143.0</td>
      <td>12.493846</td>
      <td>...</td>
      <td>18.7900</td>
      <td>29.25</td>
      <td>143.0</td>
      <td>18.312517</td>
      <td>6.500093</td>
      <td>6.50</td>
      <td>13.6700</td>
      <td>18.710</td>
      <td>21.8950</td>
      <td>37.12</td>
    </tr>
    <tr>
      <th>51</th>
      <td>126.0</td>
      <td>13.639921</td>
      <td>5.313674</td>
      <td>1.79</td>
      <td>9.5225</td>
      <td>13.270</td>
      <td>16.9950</td>
      <td>28.71</td>
      <td>126.0</td>
      <td>11.697619</td>
      <td>...</td>
      <td>17.2675</td>
      <td>30.84</td>
      <td>126.0</td>
      <td>17.182063</td>
      <td>6.303056</td>
      <td>2.62</td>
      <td>13.0100</td>
      <td>16.685</td>
      <td>20.7175</td>
      <td>38.25</td>
    </tr>
    <tr>
      <th>52</th>
      <td>127.0</td>
      <td>14.469134</td>
      <td>5.701056</td>
      <td>3.00</td>
      <td>10.2450</td>
      <td>13.620</td>
      <td>18.1250</td>
      <td>32.50</td>
      <td>127.0</td>
      <td>11.692283</td>
      <td>...</td>
      <td>17.2300</td>
      <td>31.25</td>
      <td>127.0</td>
      <td>19.183228</td>
      <td>6.834899</td>
      <td>4.58</td>
      <td>14.6450</td>
      <td>18.710</td>
      <td>22.7700</td>
      <td>41.46</td>
    </tr>
    <tr>
      <th>53</th>
      <td>21.0</td>
      <td>12.694286</td>
      <td>5.804934</td>
      <td>3.71</td>
      <td>8.6700</td>
      <td>11.750</td>
      <td>16.7500</td>
      <td>23.83</td>
      <td>21.0</td>
      <td>9.451905</td>
      <td>...</td>
      <td>13.3700</td>
      <td>30.46</td>
      <td>21.0</td>
      <td>17.103810</td>
      <td>5.909188</td>
      <td>7.12</td>
      <td>13.7000</td>
      <td>16.580</td>
      <td>23.1300</td>
      <td>26.83</td>
    </tr>
  </tbody>
</table>
<p>53 rows × 96 columns</p>
</div>


