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
    

# Scores Dataset

### Introduction:

This time you will create the data.

***Exercise based on [Chris Albon](http://chrisalbon.com/) work, the credits belong to him.***

### Step 1. Import the necessary libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 2. Create the DataFrame that should look like the one below.


```python
dict = {'first name' : ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],\
        'last_name' : ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],\
        'age' : [42, 52, 36, 24, 73],\
        'female' : [0, 1, 1, 0, 1],\
        'preTestScore' : [4, 24, 31, 2, 3],\
        'postTestScore' : [25, 94, 57, 62, 70]}

df = pd.DataFrame(dict)
df
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
      <th>first name</th>
      <th>last_name</th>
      <th>age</th>
      <th>female</th>
      <th>preTestScore</th>
      <th>postTestScore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>0</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>1</td>
      <td>24</td>
      <td>94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>1</td>
      <td>31</td>
      <td>57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>0</td>
      <td>2</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>1</td>
      <td>3</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>



### Step 3. Create a Scatterplot of preTestScore and postTestScore, with the size of each point determined by age
#### Hint: Don't forget to place the labels


```python
sns.scatterplot(data=df, x='preTestScore', y='postTestScore', size='age')
```




    <AxesSubplot:xlabel='preTestScore', ylabel='postTestScore'>




    
![png](https://i.ibb.co/1bjkNwY/output-6-1.png)
    


### Step 4. Create a Scatterplot of preTestScore and postTestScore.
### This time the size should be 4.5 times the postTestScore and the color determined by sex


```python
sns.scatterplot(x=df['preTestScore'], y=df['postTestScore'], s=df['age']*4.5, hue=df['female'])
```




    <AxesSubplot:xlabel='preTestScore', ylabel='postTestScore'>




    
![png](https://i.ibb.co/VWpQyy9/output-8-1.png)
    


### BONUS: Create your own question and answer it.


```python

```

# Visualizing the Titanic Disaster

### Introduction:

This exercise is based on the titanic Disaster dataset avaiable at [Kaggle](https://www.kaggle.com/c/titanic).  
To know more about the variables check [here](https://www.kaggle.com/c/titanic/data)


### Step 1. Import the necessary libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Titanic_Desaster/train.csv)

### Step 3. Assign it to a variable titanic 


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Titanic_Desaster/train.csv'
titanic = pd.read_csv(url)
```

### Step 4. Set PassengerId as the index 


```python
titanic.set_index(['PassengerId'], inplace=True)
```

### Step 5. Create a pie chart presenting the male/female proportion


```python
titanic['Sex'].value_counts().plot(kind='pie')
```




    <AxesSubplot:ylabel='Sex'>




    
![png](https://i.ibb.co/SRfCJKs/output-9-1.png)
    


### Step 6. Create a scatterplplotwith the Fare payed and the Age, differ the plot color by gender


```python
sns.scatterplot(x=titanic['Fare'], y=titanic['Age'], hue=titanic['Sex'])
```




    <AxesSubplot:xlabel='Fare', ylabel='Age'>




    
![png](https://i.ibb.co/2KT67Th/output-11-1.png)
    


### Step 7. How many people survived?


```python
titanic['Survived'].sum()
```




    342



### Step 8. Create a histogram with the Fare payed


```python
sns.histplot(x=titanic['Fare'])
```




    <AxesSubplot:xlabel='Fare', ylabel='Count'>




    
![png](https://i.ibb.co/mGhJN32/output-15-1.png)
    


### BONUS: Create your own question and answer it.


```python

```
  