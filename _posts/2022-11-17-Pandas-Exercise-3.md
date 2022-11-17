---
title: "Pandas Exercise 3 : Grouping"
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

# Alcohol Consumption Dataset


### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  
### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 

### Step 3. Assign it to a variable called drinks.


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
drinks = pd.read_csv(url)
```

### Step 4. Which continent drinks more beer on average?


```python
drinks[drinks['beer_servings'] > drinks['beer_servings'].mean()].groupby(['continent']).count()
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
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. For each continent print the statistics for wine consumption.


```python
drinks.groupby(['continent']).sum()['wine_servings']
```




    continent
    AF     862
    AS     399
    EU    6400
    OC     570
    SA     749
    Name: wine_servings, dtype: int64



### Step 6. Print the mean alcohol consumption per continent for every column


```python
drinks.groupby(['continent']).mean()
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
      <td>16.339623</td>
      <td>16.264151</td>
      <td>3.007547</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
      <td>60.840909</td>
      <td>9.068182</td>
      <td>2.170455</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
      <td>58.437500</td>
      <td>35.625000</td>
      <td>3.381250</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
      <td>114.750000</td>
      <td>62.416667</td>
      <td>6.308333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
drinks.groupby(['continent']).median()
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>32.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>17.5</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>219.0</td>
      <td>122.0</td>
      <td>128.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>52.5</td>
      <td>37.0</td>
      <td>8.5</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>162.5</td>
      <td>108.5</td>
      <td>12.0</td>
      <td>6.85</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
pd.DataFrame(data = {'mean' : drinks.groupby(['continent']).mean()['spirit_servings'], 
                    'min' : drinks.groupby(['continent']).min()['spirit_servings'],
                    'max' : drinks.groupby(['continent']).max()['spirit_servings']})
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
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>16.339623</td>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>60.840909</td>
      <td>0</td>
      <td>326</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>132.555556</td>
      <td>0</td>
      <td>373</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>58.437500</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>114.750000</td>
      <td>25</td>
      <td>302</td>
    </tr>
  </tbody>
</table>
</div>

# Occupation Dataset

### Introduction:

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 

### Step 3. Assign it to a variable called users.


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
users = pd.read_csv(url, sep='|')
```

### Step 4. Discover what is the mean age per occupation


```python
users.groupby(['occupation']).mean()['age']
```




    occupation
    administrator    38.746835
    artist           31.392857
    doctor           43.571429
    educator         42.010526
    engineer         36.388060
    entertainment    29.222222
    executive        38.718750
    healthcare       41.562500
    homemaker        32.571429
    lawyer           36.750000
    librarian        40.000000
    marketing        37.615385
    none             26.555556
    other            34.523810
    programmer       33.121212
    retired          63.071429
    salesman         35.666667
    scientist        35.548387
    student          22.081633
    technician       33.148148
    writer           36.311111
    Name: age, dtype: float64



### Step 5. Discover the Male ratio per occupation and sort it from the most to the least


```python
(users[users['gender'] == 'M'].groupby(['occupation']).count()['gender'] / \
 users.groupby(['occupation']).count()['gender']).sort_values(ascending=False)
```




    occupation
    doctor           1.000000
    engineer         0.970149
    technician       0.962963
    retired          0.928571
    programmer       0.909091
    executive        0.906250
    scientist        0.903226
    entertainment    0.888889
    lawyer           0.833333
    salesman         0.750000
    educator         0.726316
    student          0.693878
    other            0.657143
    marketing        0.615385
    writer           0.577778
    none             0.555556
    administrator    0.544304
    artist           0.535714
    librarian        0.431373
    healthcare       0.312500
    homemaker        0.142857
    Name: gender, dtype: float64



### Step 6. For each occupation, calculate the minimum and maximum ages


```python
pd.DataFrame(data = {'min' : users.groupby(['occupation']).min()['age'],
                     'max' : users.groupby(['occupation']).max()['age']})
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
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>administrator</th>
      <td>21</td>
      <td>70</td>
    </tr>
    <tr>
      <th>artist</th>
      <td>19</td>
      <td>48</td>
    </tr>
    <tr>
      <th>doctor</th>
      <td>28</td>
      <td>64</td>
    </tr>
    <tr>
      <th>educator</th>
      <td>23</td>
      <td>63</td>
    </tr>
    <tr>
      <th>engineer</th>
      <td>22</td>
      <td>70</td>
    </tr>
    <tr>
      <th>entertainment</th>
      <td>15</td>
      <td>50</td>
    </tr>
    <tr>
      <th>executive</th>
      <td>22</td>
      <td>69</td>
    </tr>
    <tr>
      <th>healthcare</th>
      <td>22</td>
      <td>62</td>
    </tr>
    <tr>
      <th>homemaker</th>
      <td>20</td>
      <td>50</td>
    </tr>
    <tr>
      <th>lawyer</th>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>librarian</th>
      <td>23</td>
      <td>69</td>
    </tr>
    <tr>
      <th>marketing</th>
      <td>24</td>
      <td>55</td>
    </tr>
    <tr>
      <th>none</th>
      <td>11</td>
      <td>55</td>
    </tr>
    <tr>
      <th>other</th>
      <td>13</td>
      <td>64</td>
    </tr>
    <tr>
      <th>programmer</th>
      <td>20</td>
      <td>63</td>
    </tr>
    <tr>
      <th>retired</th>
      <td>51</td>
      <td>73</td>
    </tr>
    <tr>
      <th>salesman</th>
      <td>18</td>
      <td>66</td>
    </tr>
    <tr>
      <th>scientist</th>
      <td>23</td>
      <td>55</td>
    </tr>
    <tr>
      <th>student</th>
      <td>7</td>
      <td>42</td>
    </tr>
    <tr>
      <th>technician</th>
      <td>21</td>
      <td>55</td>
    </tr>
    <tr>
      <th>writer</th>
      <td>18</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. For each combination of occupation and gender, calculate the mean age


```python
users.groupby(['occupation', 'gender']).mean()['age']
```




    occupation     gender
    administrator  F         40.638889
                   M         37.162791
    artist         F         30.307692
                   M         32.333333
    doctor         M         43.571429
    educator       F         39.115385
                   M         43.101449
    engineer       F         29.500000
                   M         36.600000
    entertainment  F         31.000000
                   M         29.000000
    executive      F         44.000000
                   M         38.172414
    healthcare     F         39.818182
                   M         45.400000
    homemaker      F         34.166667
                   M         23.000000
    lawyer         F         39.500000
                   M         36.200000
    librarian      F         40.000000
                   M         40.000000
    marketing      F         37.200000
                   M         37.875000
    none           F         36.500000
                   M         18.600000
    other          F         35.472222
                   M         34.028986
    programmer     F         32.166667
                   M         33.216667
    retired        F         70.000000
                   M         62.538462
    salesman       F         27.000000
                   M         38.555556
    scientist      F         28.333333
                   M         36.321429
    student        F         20.750000
                   M         22.669118
    technician     F         38.000000
                   M         32.961538
    writer         F         37.631579
                   M         35.346154
    Name: age, dtype: float64



### Step 8.  For each occupation present the percentage of women and men


```python
pd.DataFrame(data = {'male percentage' : users[users['gender'] == 'M'].groupby(['occupation']).count()['gender'] / \
                     users.groupby(['occupation']).count()['gender'] * 100,
                    'female percentage' : users[users['gender'] == 'F'].groupby(['occupation']).count()['gender'] / \
                    users.groupby(['occupation']).count()['gender'] * 100}) 
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
      <th>male percentage</th>
      <th>female percentage</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>administrator</th>
      <td>54.430380</td>
      <td>45.569620</td>
    </tr>
    <tr>
      <th>artist</th>
      <td>53.571429</td>
      <td>46.428571</td>
    </tr>
    <tr>
      <th>doctor</th>
      <td>100.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>educator</th>
      <td>72.631579</td>
      <td>27.368421</td>
    </tr>
    <tr>
      <th>engineer</th>
      <td>97.014925</td>
      <td>2.985075</td>
    </tr>
    <tr>
      <th>entertainment</th>
      <td>88.888889</td>
      <td>11.111111</td>
    </tr>
    <tr>
      <th>executive</th>
      <td>90.625000</td>
      <td>9.375000</td>
    </tr>
    <tr>
      <th>healthcare</th>
      <td>31.250000</td>
      <td>68.750000</td>
    </tr>
    <tr>
      <th>homemaker</th>
      <td>14.285714</td>
      <td>85.714286</td>
    </tr>
    <tr>
      <th>lawyer</th>
      <td>83.333333</td>
      <td>16.666667</td>
    </tr>
    <tr>
      <th>librarian</th>
      <td>43.137255</td>
      <td>56.862745</td>
    </tr>
    <tr>
      <th>marketing</th>
      <td>61.538462</td>
      <td>38.461538</td>
    </tr>
    <tr>
      <th>none</th>
      <td>55.555556</td>
      <td>44.444444</td>
    </tr>
    <tr>
      <th>other</th>
      <td>65.714286</td>
      <td>34.285714</td>
    </tr>
    <tr>
      <th>programmer</th>
      <td>90.909091</td>
      <td>9.090909</td>
    </tr>
    <tr>
      <th>retired</th>
      <td>92.857143</td>
      <td>7.142857</td>
    </tr>
    <tr>
      <th>salesman</th>
      <td>75.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>scientist</th>
      <td>90.322581</td>
      <td>9.677419</td>
    </tr>
    <tr>
      <th>student</th>
      <td>69.387755</td>
      <td>30.612245</td>
    </tr>
    <tr>
      <th>technician</th>
      <td>96.296296</td>
      <td>3.703704</td>
    </tr>
    <tr>
      <th>writer</th>
      <td>57.777778</td>
      <td>42.222222</td>
    </tr>
  </tbody>
</table>
</div>

# Regiment Dataset

### Introduction:

Special thanks to: http://chrisalbon.com/ for sharing the dataset and materials.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Create the DataFrame with the following values:


```python
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
```

### Step 3. Assign it to a variable called regiment.
#### Don't forget to name each column


```python
regiment = pd.DataFrame(raw_data)
```

### Step 4. What is the mean preTestScore from the regiment Nighthawks?  


```python
regiment.groupby(['regiment']).mean().filter(['Nighthawks'], axis=0)['preTestScore']
```




    regiment
    Nighthawks    15.25
    Name: preTestScore, dtype: float64



### Step 5. Present general statistics by company


```python
regiment.groupby(['company']).describe()
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
      <th colspan="8" halign="left">preTestScore</th>
      <th colspan="8" halign="left">postTestScore</th>
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
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>company</th>
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
      <th>1st</th>
      <td>6.0</td>
      <td>6.666667</td>
      <td>8.524475</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>3.5</td>
      <td>4.00</td>
      <td>24.0</td>
      <td>6.0</td>
      <td>57.666667</td>
      <td>27.485754</td>
      <td>25.0</td>
      <td>34.25</td>
      <td>66.0</td>
      <td>70.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>6.0</td>
      <td>15.500000</td>
      <td>14.652645</td>
      <td>2.0</td>
      <td>2.25</td>
      <td>13.5</td>
      <td>29.25</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>67.000000</td>
      <td>14.057027</td>
      <td>57.0</td>
      <td>58.25</td>
      <td>62.0</td>
      <td>68.0</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. What is the mean of each company's preTestScore?


```python
regiment.groupby(['company']).mean()['preTestScore']
```




    company
    1st     6.666667
    2nd    15.500000
    Name: preTestScore, dtype: float64



### Step 7. Present the mean preTestScores grouped by regiment and company


```python
regiment.groupby(['regiment', 'company']).mean()['preTestScore']
```




    regiment    company
    Dragoons    1st         3.5
                2nd        27.5
    Nighthawks  1st        14.0
                2nd        16.5
    Scouts      1st         2.5
                2nd         2.5
    Name: preTestScore, dtype: float64



### Step 8. Present the mean preTestScores grouped by regiment and company without heirarchical indexing


```python
regiment.groupby(['regiment', 'company']).mean()['preTestScore'].unstacktack()
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
      <th>company</th>
      <th>1st</th>
      <th>2nd</th>
    </tr>
    <tr>
      <th>regiment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dragoons</th>
      <td>3.5</td>
      <td>27.5</td>
    </tr>
    <tr>
      <th>Nighthawks</th>
      <td>14.0</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>Scouts</th>
      <td>2.5</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Group the entire dataframe by regiment and company


```python
regiment.groupby(['regiment', 'company']).describe()
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
      <th></th>
      <th colspan="8" halign="left">preTestScore</th>
      <th colspan="8" halign="left">postTestScore</th>
    </tr>
    <tr>
      <th></th>
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
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>regiment</th>
      <th>company</th>
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
      <th rowspan="2" valign="top">Dragoons</th>
      <th>1st</th>
      <td>2.0</td>
      <td>3.5</td>
      <td>0.707107</td>
      <td>3.0</td>
      <td>3.25</td>
      <td>3.5</td>
      <td>3.75</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>47.5</td>
      <td>31.819805</td>
      <td>25.0</td>
      <td>36.25</td>
      <td>47.5</td>
      <td>58.75</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2.0</td>
      <td>27.5</td>
      <td>4.949747</td>
      <td>24.0</td>
      <td>25.75</td>
      <td>27.5</td>
      <td>29.25</td>
      <td>31.0</td>
      <td>2.0</td>
      <td>75.5</td>
      <td>26.162951</td>
      <td>57.0</td>
      <td>66.25</td>
      <td>75.5</td>
      <td>84.75</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Nighthawks</th>
      <th>1st</th>
      <td>2.0</td>
      <td>14.0</td>
      <td>14.142136</td>
      <td>4.0</td>
      <td>9.00</td>
      <td>14.0</td>
      <td>19.00</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>59.5</td>
      <td>48.790368</td>
      <td>25.0</td>
      <td>42.25</td>
      <td>59.5</td>
      <td>76.75</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2.0</td>
      <td>16.5</td>
      <td>20.506097</td>
      <td>2.0</td>
      <td>9.25</td>
      <td>16.5</td>
      <td>23.75</td>
      <td>31.0</td>
      <td>2.0</td>
      <td>59.5</td>
      <td>3.535534</td>
      <td>57.0</td>
      <td>58.25</td>
      <td>59.5</td>
      <td>60.75</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Scouts</th>
      <th>1st</th>
      <td>2.0</td>
      <td>2.5</td>
      <td>0.707107</td>
      <td>2.0</td>
      <td>2.25</td>
      <td>2.5</td>
      <td>2.75</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>66.0</td>
      <td>5.656854</td>
      <td>62.0</td>
      <td>64.00</td>
      <td>66.0</td>
      <td>68.00</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2.0</td>
      <td>2.5</td>
      <td>0.707107</td>
      <td>2.0</td>
      <td>2.25</td>
      <td>2.5</td>
      <td>2.75</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>66.0</td>
      <td>5.656854</td>
      <td>62.0</td>
      <td>64.00</td>
      <td>66.0</td>
      <td>68.00</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 10. What is the number of observations in each regiment and company


```python
regiment.groupby(['regiment', 'company']).count()
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
      <th></th>
      <th>name</th>
      <th>preTestScore</th>
      <th>postTestScore</th>
    </tr>
    <tr>
      <th>regiment</th>
      <th>company</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Dragoons</th>
      <th>1st</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Nighthawks</th>
      <th>1st</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Scouts</th>
      <th>1st</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Step 11. Iterate over a group and print the name and the whole data from the regiment


```python
for group in regiment.groupby(['regiment']):
    print(group)
```

    ('Dragoons',    regiment company    name  preTestScore  postTestScore
    4  Dragoons     1st   Cooze             3             70
    5  Dragoons     1st   Jacon             4             25
    6  Dragoons     2nd  Ryaner            24             94
    7  Dragoons     2nd    Sone            31             57)
    ('Nighthawks',      regiment company      name  preTestScore  postTestScore
    0  Nighthawks     1st    Miller             4             25
    1  Nighthawks     1st  Jacobson            24             94
    2  Nighthawks     2nd       Ali            31             57
    3  Nighthawks     2nd    Milner             2             62)
    ('Scouts',    regiment company   name  preTestScore  postTestScore
    8    Scouts     1st  Sloan             2             62
    9    Scouts     1st  Piger             3             70
    10   Scouts     2nd  Riani             2             62
    11   Scouts     2nd    Ali             3             70)
    
