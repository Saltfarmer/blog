---
title: "Pandas Exercise 7 : Visualization"
header :
  teaser: /assets/images/pandas-head.jpg

categories:
  - Python
tags:
  - Pandas
  - Python
  - Exercise
  - Visualization

---

The continuity of my practice on Pandas exercise from [guisapmora](https://github.com/guipsamora/pandas_exercises/archive/refs/heads/master.zip). This one is interesting because it covers the basic exercise of visualization in Matplotlib.

# Visualizing Chipotle's Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

### Step 1. Import the necessary libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# set this so the graphs open internally
%matplotlib inline
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv). 

### Step 3. Assign it to a variable called chipo.


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep ='\t')
```

### Step 4. See the first 10 entries


```python
chipo.head()
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
      <th>order_id</th>
      <th>quantity</th>
      <th>item_name</th>
      <th>choice_description</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Chips and Fresh Tomato Salsa</td>
      <td>NaN</td>
      <td>$2.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Izze</td>
      <td>[Clementine]</td>
      <td>$3.39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Nantucket Nectar</td>
      <td>[Apple]</td>
      <td>$3.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Chips and Tomatillo-Green Chili Salsa</td>
      <td>NaN</td>
      <td>$2.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>Chicken Bowl</td>
      <td>[Tomatillo-Red Chili Salsa (Hot), [Black Beans...</td>
      <td>$16.98</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. Create a histogram of the top 5 items bought


```python
chipo.groupby(['item_name']).sum()[['quantity']].sort_values(['quantity'],ascending=False).head(5).plot(kind='bar')
```




    <AxesSubplot:xlabel='item_name'>




    
![png](https://i.ibb.co/FhHF9fH/output-9-1.png)
    


### Step 6. Create a scatterplot with the number of items orderered per order price
#### Hint: Price should be in the X-axis and Items ordered in the Y-axis


```python
chipo['item_price'] = chipo['item_price'].apply(lambda x : x[1:]).astype('float')
```


```python
chipo.groupby(['item_name']).sum()[['quantity', 'item_price']].plot(x='item_price', y='quantity',kind='scatter')
```




    <AxesSubplot:xlabel='item_price', ylabel='quantity'>




    
![png](https://i.ibb.co/Nj5Q6B7/output-12-1.png)
    


### Step 7. BONUS: Create a question and a graph to answer your own question.


```python
chipo.groupby(['item_name']).sum()[['quantity', 'item_price']].plot(x='item_price',bins = 15, kind='hist')
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](https://i.ibb.co/tz5rYy1/output-14-1.png)
    

