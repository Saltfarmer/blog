---
title: "Pandas Exercise 5 : Merge"
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

# Fictitious Names Dataset

### Introduction:

This time you will create a data again 

Special thanks to [Chris Albon](http://chrisalbon.com/) for sharing the dataset and materials.
All the credits to this exercise belongs to him.  

In order to understand about it go [here](https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/).

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Create the 3 DataFrames based on the following raw data


```python
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
```

### Step 3. Assign each to a variable called data1, data2, data3


```python
data1 = pd.DataFrame(raw_data_1)
data2 = pd.DataFrame(raw_data_2)
data3 = pd.DataFrame(raw_data_3)
```

### Step 4. Join the two dataframes along rows and assign all_data


```python
all_data = pd.concat((data1, data2), axis=0)
```

### Step 5. Join the two dataframes along columns and assing to all_data_col


```python
all_data_col = pd.concat((data1, data2), axis=1)
```

### Step 6. Print data3


```python
data3
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
      <th>subject_id</th>
      <th>test_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>61</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Merge all_data and data3 along the subject_id value


```python
all_data.merge(data3, on=['subject_id'])
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
      <th>subject_id</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>test_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alex</td>
      <td>Anderson</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Amy</td>
      <td>Ackerman</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Allen</td>
      <td>Ali</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>Aoni</td>
      <td>61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Billy</td>
      <td>Bonder</td>
      <td>61</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>Atiches</td>
      <td>16</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>Brian</td>
      <td>Black</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Bryce</td>
      <td>Brice</td>
      <td>14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Betty</td>
      <td>Btisan</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Merge only the data that has the same 'subject_id' on both data1 and data2


```python
data1.merge(data2, on=['subject_id'], how='inner')
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
      <th>subject_id</th>
      <th>first_name_x</th>
      <th>last_name_x</th>
      <th>first_name_y</th>
      <th>last_name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>Alice</td>
      <td>Aoni</td>
      <td>Billy</td>
      <td>Bonder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>Atiches</td>
      <td>Brian</td>
      <td>Black</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Merge all values in data1 and data2, with matching records from both sides where available.


```python
data1.merge(data2, on=['subject_id'], how='outer')
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
      <th>subject_id</th>
      <th>first_name_x</th>
      <th>last_name_x</th>
      <th>first_name_y</th>
      <th>last_name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alex</td>
      <td>Anderson</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Amy</td>
      <td>Ackerman</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Allen</td>
      <td>Ali</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>Aoni</td>
      <td>Billy</td>
      <td>Bonder</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>Atiches</td>
      <td>Brian</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bran</td>
      <td>Balwner</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bryce</td>
      <td>Brice</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Betty</td>
      <td>Btisan</td>
    </tr>
  </tbody>
</table>
</div>

# MPG  Dataset

### Introduction:

The following exercise utilizes data from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the first dataset [cars1](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv) and [cars2](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv).  

   ### Step 3. Assign each to a variable called cars1 and cars2


```python
url1 = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv'
url2 = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv'

cars1 = pd.read_csv(url1)
cars2 = pd.read_csv(url2)
```

### Step 4. Oops, it seems our first dataset has some unnamed blank columns, fix cars1


```python
cars1 = cars1.dropna(how ='all', axis=1)
```

### Step 5. What is the number of observations in each dataset?


```python
print(cars1.shape[0])
print(cars2.shape[0])
```

    198
    200
    

### Step 6. Join cars1 and cars2 into a single DataFrame called cars


```python
cars = pd.concat((cars1, cars1), axis=0)
cars.columns
```




    Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model', 'origin', 'car'],
          dtype='object')



### Step 7. Oops, there is a column missing, called owners. Create a random number Series from 15,000 to 73,000.


```python
import numpy as np

owners = np.random.randint(low=15000, high=73000, size= cars.shape[0])
```

### Step 8. Add the column owners to cars


```python
cars['owner'] = owners
```
