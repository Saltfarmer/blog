---
title: "Pandas Exercise 2 : Filtering and Sorting"
header :
  teaser: /assets/images/pandas-head.

categories:
  - Python
tags:
  - Pandas
  - Python
  - Exercise

---

The continuity of my practice on Pandas exercise from [guisapmora](https://github.com/guipsamora/pandas_exercises/archive/refs/heads/master.zip).

# Chipotle dataset

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
chipo = pd.read_csv(url, sep='\t')
```

### Step 4. How many products cost more than $10.00?


```python
chipo['item_price'] = chipo['item_price'].apply(lambda x : x[1:]).astype('float').copy()
```


```python
chipo[chipo['item_price'] > 10]['item_name'].value_counts().count()
```




    31



### Step 5. What is the price of each item? 
###### print a data frame with only two columns item_name and item_price


```python
chipo.groupby(['item_name']).mean().reset_index()[['item_name', 'item_price']]
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
      <th>item_name</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6 Pack Soft Drink</td>
      <td>6.610185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barbacoa Bowl</td>
      <td>10.187273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barbacoa Burrito</td>
      <td>9.832418</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barbacoa Crispy Tacos</td>
      <td>10.928182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barbacoa Salad Bowl</td>
      <td>10.640000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Barbacoa Soft Tacos</td>
      <td>10.018400</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bottled Water</td>
      <td>1.867654</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bowl</td>
      <td>14.800000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Burrito</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Canned Soda</td>
      <td>1.320577</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Canned Soft Drink</td>
      <td>1.457641</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Carnitas Bowl</td>
      <td>10.833971</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Carnitas Burrito</td>
      <td>10.132712</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Carnitas Crispy Tacos</td>
      <td>11.137143</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Carnitas Salad</td>
      <td>8.990000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Carnitas Salad Bowl</td>
      <td>11.056667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Carnitas Soft Tacos</td>
      <td>9.398500</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Chicken Bowl</td>
      <td>10.113953</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Chicken Burrito</td>
      <td>10.082857</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Chicken Crispy Tacos</td>
      <td>10.045319</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chicken Salad</td>
      <td>9.010000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Chicken Salad Bowl</td>
      <td>11.170455</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Chicken Soft Tacos</td>
      <td>9.635565</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Chips</td>
      <td>2.342844</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Chips and Fresh Tomato Salsa</td>
      <td>3.285091</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Chips and Guacamole</td>
      <td>4.595073</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Chips and Mild Fresh Tomato Salsa</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Chips and Roasted Chili Corn Salsa</td>
      <td>3.084091</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Chips and Roasted Chili-Corn Salsa</td>
      <td>2.390000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Chips and Tomatillo Green Chili Salsa</td>
      <td>3.087209</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Chips and Tomatillo Red Chili Salsa</td>
      <td>3.072917</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Chips and Tomatillo-Green Chili Salsa</td>
      <td>2.544194</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Chips and Tomatillo-Red Chili Salsa</td>
      <td>2.987500</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Crispy Tacos</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Izze</td>
      <td>3.390000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Nantucket Nectar</td>
      <td>3.641111</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Salad</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Side of Chips</td>
      <td>1.840594</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Steak Bowl</td>
      <td>10.711801</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Steak Burrito</td>
      <td>10.465842</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Steak Crispy Tacos</td>
      <td>10.209714</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Steak Salad</td>
      <td>8.915000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Steak Salad Bowl</td>
      <td>11.847931</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Steak Soft Tacos</td>
      <td>9.746364</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Veggie Bowl</td>
      <td>10.211647</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Veggie Burrito</td>
      <td>9.839684</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Veggie Crispy Tacos</td>
      <td>8.490000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Veggie Salad</td>
      <td>8.490000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Veggie Salad Bowl</td>
      <td>10.138889</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Veggie Soft Tacos</td>
      <td>10.565714</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. Sort by the name of the item


```python
chipo.groupby(['item_name']).mean().sort_index().reset_index()[['item_name', 'item_price']]
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
      <th>item_name</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6 Pack Soft Drink</td>
      <td>6.610185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barbacoa Bowl</td>
      <td>10.187273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barbacoa Burrito</td>
      <td>9.832418</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barbacoa Crispy Tacos</td>
      <td>10.928182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barbacoa Salad Bowl</td>
      <td>10.640000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Barbacoa Soft Tacos</td>
      <td>10.018400</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bottled Water</td>
      <td>1.867654</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bowl</td>
      <td>14.800000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Burrito</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Canned Soda</td>
      <td>1.320577</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Canned Soft Drink</td>
      <td>1.457641</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Carnitas Bowl</td>
      <td>10.833971</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Carnitas Burrito</td>
      <td>10.132712</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Carnitas Crispy Tacos</td>
      <td>11.137143</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Carnitas Salad</td>
      <td>8.990000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Carnitas Salad Bowl</td>
      <td>11.056667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Carnitas Soft Tacos</td>
      <td>9.398500</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Chicken Bowl</td>
      <td>10.113953</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Chicken Burrito</td>
      <td>10.082857</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Chicken Crispy Tacos</td>
      <td>10.045319</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chicken Salad</td>
      <td>9.010000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Chicken Salad Bowl</td>
      <td>11.170455</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Chicken Soft Tacos</td>
      <td>9.635565</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Chips</td>
      <td>2.342844</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Chips and Fresh Tomato Salsa</td>
      <td>3.285091</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Chips and Guacamole</td>
      <td>4.595073</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Chips and Mild Fresh Tomato Salsa</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Chips and Roasted Chili Corn Salsa</td>
      <td>3.084091</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Chips and Roasted Chili-Corn Salsa</td>
      <td>2.390000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Chips and Tomatillo Green Chili Salsa</td>
      <td>3.087209</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Chips and Tomatillo Red Chili Salsa</td>
      <td>3.072917</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Chips and Tomatillo-Green Chili Salsa</td>
      <td>2.544194</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Chips and Tomatillo-Red Chili Salsa</td>
      <td>2.987500</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Crispy Tacos</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Izze</td>
      <td>3.390000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Nantucket Nectar</td>
      <td>3.641111</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Salad</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Side of Chips</td>
      <td>1.840594</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Steak Bowl</td>
      <td>10.711801</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Steak Burrito</td>
      <td>10.465842</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Steak Crispy Tacos</td>
      <td>10.209714</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Steak Salad</td>
      <td>8.915000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Steak Salad Bowl</td>
      <td>11.847931</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Steak Soft Tacos</td>
      <td>9.746364</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Veggie Bowl</td>
      <td>10.211647</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Veggie Burrito</td>
      <td>9.839684</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Veggie Crispy Tacos</td>
      <td>8.490000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Veggie Salad</td>
      <td>8.490000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Veggie Salad Bowl</td>
      <td>10.138889</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Veggie Soft Tacos</td>
      <td>10.565714</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. What was the quantity of the most expensive item ordered?


```python
chipo[chipo['item_price'] >= chipo['item_price'].max()]['quantity']
```




    3598    15
    Name: quantity, dtype: int64



### Step 8. How many times was a Veggie Salad Bowl ordered?


```python
chipo.groupby(['item_name']).sum().filter(['Veggie Salad Bowl'], axis=0)['quantity']
```




    item_name
    Veggie Salad Bowl    18
    Name: quantity, dtype: int64



### Step 9. How many times did someone order more than one Canned Soda?


```python
chipo[chipo['quantity'] > 1].groupby(['item_name']).count().filter(['Canned Soda'], axis=0)['quantity']
```




    item_name
    Canned Soda    20
    Name: quantity, dtype: int64

# Euro12 Dataset

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv). 

### Step 3. Assign it to a variable called euro12.


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
euro12 = pd.read_csv(url)
```

### Step 4. Select only the Goal column.


```python
euro12['Goals']
```




    0      4
    1      4
    2      4
    3      5
    4      3
    5     10
    6      5
    7      6
    8      2
    9      2
    10     6
    11     1
    12     5
    13    12
    14     5
    15     2
    Name: Goals, dtype: int64



### Step 5. How many team participated in the Euro2012?


```python
euro12['Team'].value_counts().count()
```




    16



### Step 6. What is the number of columns in the dataset?


```python
len(euro12.columns)
```




    35



### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline


```python
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline
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
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Sort the teams by Red Cards, then to Yellow Cards


```python
discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False)
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
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Calculate the mean Yellow Cards given per Team


```python
discipline['Yellow Cards'].mean()
```




    7.4375



### Step 10. Filter teams that scored more than 6 goals


```python
euro12[euro12['Goals'] > 6]['Team']
```




    5     Germany
    13      Spain
    Name: Team, dtype: object



### Step 11. Select the teams that start with G


```python
euro12[euro12['Team'].str[0] == 'G']['Team']
```




    5    Germany
    6     Greece
    Name: Team, dtype: object



### Step 12. Select the first 7 columns


```python
euro12.iloc[:, :7]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>6</td>
      <td>34</td>
      <td>45</td>
      <td>43.0%</td>
      <td>7.5%</td>
      <td>110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>2</td>
      <td>12</td>
      <td>36</td>
      <td>25.0%</td>
      <td>4.1%</td>
      <td>60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>2</td>
      <td>15</td>
      <td>23</td>
      <td>39.4%</td>
      <td>5.2%</td>
      <td>48</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>6</td>
      <td>22</td>
      <td>42</td>
      <td>34.3%</td>
      <td>9.3%</td>
      <td>82</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>36.8%</td>
      <td>5.2%</td>
      <td>28</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>5</td>
      <td>9</td>
      <td>31</td>
      <td>22.5%</td>
      <td>12.5%</td>
      <td>59</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>5</td>
      <td>17</td>
      <td>19</td>
      <td>47.2%</td>
      <td>13.8%</td>
      <td>39</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>2</td>
      <td>7</td>
      <td>26</td>
      <td>21.2%</td>
      <td>6.0%</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13. Select all columns except the last 3.


```python
euro12.iloc[:, :-3]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Clean Sheets</th>
      <th>Blocks</th>
      <th>Goals conceded</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>5</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>29</td>
      <td>3</td>
      <td>22</td>
      <td>88.1%</td>
      <td>43</td>
      <td>45</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
      <td>54.6%</td>
      <td>36</td>
      <td>51</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>23</td>
      <td>7</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>6</td>
      <td>34</td>
      <td>45</td>
      <td>43.0%</td>
      <td>7.5%</td>
      <td>110</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>18</td>
      <td>7</td>
      <td>20</td>
      <td>74.1%</td>
      <td>101</td>
      <td>89</td>
      <td>16</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>2</td>
      <td>12</td>
      <td>36</td>
      <td>25.0%</td>
      <td>4.1%</td>
      <td>60</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>9</td>
      <td>5</td>
      <td>12</td>
      <td>70.6%</td>
      <td>35</td>
      <td>30</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>2</td>
      <td>15</td>
      <td>23</td>
      <td>39.4%</td>
      <td>5.2%</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>6</td>
      <td>66.7%</td>
      <td>48</td>
      <td>56</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>6</td>
      <td>22</td>
      <td>42</td>
      <td>34.3%</td>
      <td>9.3%</td>
      <td>82</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>11</td>
      <td>4</td>
      <td>10</td>
      <td>71.5%</td>
      <td>73</td>
      <td>90</td>
      <td>10</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>36.8%</td>
      <td>5.2%</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>23</td>
      <td>9</td>
      <td>17</td>
      <td>65.4%</td>
      <td>43</td>
      <td>51</td>
      <td>11</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>5</td>
      <td>9</td>
      <td>31</td>
      <td>22.5%</td>
      <td>12.5%</td>
      <td>59</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>10</td>
      <td>77.0%</td>
      <td>34</td>
      <td>43</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>15</td>
      <td>93.8%</td>
      <td>102</td>
      <td>83</td>
      <td>19</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>5</td>
      <td>17</td>
      <td>19</td>
      <td>47.2%</td>
      <td>13.8%</td>
      <td>39</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>8</td>
      <td>61.6%</td>
      <td>35</td>
      <td>51</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>2</td>
      <td>7</td>
      <td>26</td>
      <td>21.2%</td>
      <td>6.0%</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>13</td>
      <td>76.5%</td>
      <td>48</td>
      <td>31</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 32 columns</p>
</div>



### Step 14. Present only the Shooting Accuracy from England, Italy and Russia


```python
euro12[euro12['Team'].isin(['England', 'Italy', 'Russia'])][['Team', 'Shooting Accuracy']]
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
      <th>Team</th>
      <th>Shooting Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>50.0%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>43.0%</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>22.5%</td>
    </tr>
  </tbody>
</table>
</div>

# Fictional Army dataset

### Introduction:

This exercise was inspired by this [page](http://chrisalbon.com/python/)

Special thanks to: https://github.com/chrisalbon for sharing the dataset and materials.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. This is the data given as a dictionary


```python
# Create an example dataframe about a fictional army
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}
```

### Step 3. Create a dataframe and assign it to a variable called army. 

#### Don't forget to include the columns names in the order presented in the dictionary ('regiment', 'company', 'deaths'...) so that the column index order is consistent with the solutions. If omitted, pandas will order the columns alphabetically.


```python
army = pd.DataFrame(raw_data)
```

### Step 4. Set the 'origin' colum as the index of the dataframe


```python
army = army.set_index(['origin'])
```

### Step 5. Print only the column veterans


```python
army['veterans']
```




    origin
    Arizona         1
    California      5
    Texas          62
    Florida        26
    Maine          73
    Iowa           37
    Alaska        949
    Washington     48
    Oregon         48
    Wyoming       435
    Louisana       63
    Georgia       345
    Name: veterans, dtype: int64



### Step 6. Print the columns 'veterans' and 'deaths'


```python
army[['veterans', 'deaths']]
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
      <th>veterans</th>
      <th>deaths</th>
    </tr>
    <tr>
      <th>origin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arizona</th>
      <td>1</td>
      <td>523</td>
    </tr>
    <tr>
      <th>California</th>
      <td>5</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>62</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>26</td>
      <td>616</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>73</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>37</td>
      <td>234</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>949</td>
      <td>523</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>48</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>48</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>435</td>
      <td>73</td>
    </tr>
    <tr>
      <th>Louisana</th>
      <td>63</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>345</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the name of all the columns.


```python
army.columns
```




    Index(['regiment', 'company', 'deaths', 'battles', 'size', 'veterans',
           'readiness', 'armored', 'deserters'],
          dtype='object')



### Step 8. Select the 'deaths', 'size' and 'deserters' columns from Maine and Alaska


```python
army[['deaths', 'size', 'deserters']].loc[['Maine', 'Alaska']]
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
      <th>deaths</th>
      <th>size</th>
      <th>deserters</th>
    </tr>
    <tr>
      <th>origin</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Maine</th>
      <td>43</td>
      <td>1592</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>523</td>
      <td>987</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Select the rows 3 to 7 and the columns 3 to 6


```python
army.iloc[2:7,2:6]
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
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
    </tr>
    <tr>
      <th>origin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Texas</th>
      <td>25</td>
      <td>2</td>
      <td>1099</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>616</td>
      <td>2</td>
      <td>1400</td>
      <td>26</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>43</td>
      <td>4</td>
      <td>1592</td>
      <td>73</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>234</td>
      <td>7</td>
      <td>1006</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>523</td>
      <td>8</td>
      <td>987</td>
      <td>949</td>
    </tr>
  </tbody>
</table>
</div>



### Step 10. Select every row after the fourth row and all columns


```python
army.iloc[4:,::]
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
      <th>regiment</th>
      <th>company</th>
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
      <th>readiness</th>
      <th>armored</th>
      <th>deserters</th>
    </tr>
    <tr>
      <th>origin</th>
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
      <th>Maine</th>
      <td>Dragoons</td>
      <td>1st</td>
      <td>43</td>
      <td>4</td>
      <td>1592</td>
      <td>73</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>Dragoons</td>
      <td>1st</td>
      <td>234</td>
      <td>7</td>
      <td>1006</td>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>Dragoons</td>
      <td>2nd</td>
      <td>523</td>
      <td>8</td>
      <td>987</td>
      <td>949</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>Dragoons</td>
      <td>2nd</td>
      <td>62</td>
      <td>3</td>
      <td>849</td>
      <td>48</td>
      <td>3</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>Scouts</td>
      <td>1st</td>
      <td>62</td>
      <td>4</td>
      <td>973</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>Scouts</td>
      <td>1st</td>
      <td>73</td>
      <td>7</td>
      <td>1005</td>
      <td>435</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Louisana</th>
      <td>Scouts</td>
      <td>2nd</td>
      <td>37</td>
      <td>8</td>
      <td>1099</td>
      <td>63</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>Scouts</td>
      <td>2nd</td>
      <td>35</td>
      <td>9</td>
      <td>1523</td>
      <td>345</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Step 11. Select every row up to the 4th row and all columns


```python
army.iloc[:4,::]
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
      <th>regiment</th>
      <th>company</th>
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
      <th>readiness</th>
      <th>armored</th>
      <th>deserters</th>
    </tr>
    <tr>
      <th>origin</th>
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
      <th>Arizona</th>
      <td>Nighthawks</td>
      <td>1st</td>
      <td>523</td>
      <td>5</td>
      <td>1045</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>California</th>
      <td>Nighthawks</td>
      <td>1st</td>
      <td>52</td>
      <td>42</td>
      <td>957</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>Nighthawks</td>
      <td>2nd</td>
      <td>25</td>
      <td>2</td>
      <td>1099</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>Nighthawks</td>
      <td>2nd</td>
      <td>616</td>
      <td>2</td>
      <td>1400</td>
      <td>26</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Step 12. Select the 3rd column up to the 7th column


```python
army.iloc[:,2:7]
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
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
      <th>readiness</th>
    </tr>
    <tr>
      <th>origin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arizona</th>
      <td>523</td>
      <td>5</td>
      <td>1045</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>California</th>
      <td>52</td>
      <td>42</td>
      <td>957</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>25</td>
      <td>2</td>
      <td>1099</td>
      <td>62</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>616</td>
      <td>2</td>
      <td>1400</td>
      <td>26</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>43</td>
      <td>4</td>
      <td>1592</td>
      <td>73</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>234</td>
      <td>7</td>
      <td>1006</td>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>523</td>
      <td>8</td>
      <td>987</td>
      <td>949</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>62</td>
      <td>3</td>
      <td>849</td>
      <td>48</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>62</td>
      <td>4</td>
      <td>973</td>
      <td>48</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>73</td>
      <td>7</td>
      <td>1005</td>
      <td>435</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Louisana</th>
      <td>37</td>
      <td>8</td>
      <td>1099</td>
      <td>63</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>35</td>
      <td>9</td>
      <td>1523</td>
      <td>345</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13. Select rows where df.deaths is greater than 50


```python
army[army['deaths'] >50]['deaths']
```




    origin
    Arizona       523
    California     52
    Florida       616
    Iowa          234
    Alaska        523
    Washington     62
    Oregon         62
    Wyoming        73
    Name: deaths, dtype: int64



### Step 14. Select rows where df.deaths is greater than 500 or less than 50


```python
army[(army['deaths'] < 50) | (army['deaths'] > 500)]['deaths']
```




    origin
    Arizona     523
    Texas        25
    Florida     616
    Maine        43
    Alaska      523
    Louisana     37
    Georgia      35
    Name: deaths, dtype: int64



### Step 15. Select all the regiments not named "Dragoons"


```python
army[~army['regiment'].isin(['Dragoons'])]
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
      <th>regiment</th>
      <th>company</th>
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
      <th>readiness</th>
      <th>armored</th>
      <th>deserters</th>
    </tr>
    <tr>
      <th>origin</th>
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
      <th>Arizona</th>
      <td>Nighthawks</td>
      <td>1st</td>
      <td>523</td>
      <td>5</td>
      <td>1045</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>California</th>
      <td>Nighthawks</td>
      <td>1st</td>
      <td>52</td>
      <td>42</td>
      <td>957</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>Nighthawks</td>
      <td>2nd</td>
      <td>25</td>
      <td>2</td>
      <td>1099</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>Nighthawks</td>
      <td>2nd</td>
      <td>616</td>
      <td>2</td>
      <td>1400</td>
      <td>26</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>Scouts</td>
      <td>1st</td>
      <td>62</td>
      <td>4</td>
      <td>973</td>
      <td>48</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>Scouts</td>
      <td>1st</td>
      <td>73</td>
      <td>7</td>
      <td>1005</td>
      <td>435</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Louisana</th>
      <td>Scouts</td>
      <td>2nd</td>
      <td>37</td>
      <td>8</td>
      <td>1099</td>
      <td>63</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>Scouts</td>
      <td>2nd</td>
      <td>35</td>
      <td>9</td>
      <td>1523</td>
      <td>345</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Step 16. Select the rows called Texas and Arizona


```python
army.loc[['Texas', 'Arizona']]
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
      <th>regiment</th>
      <th>company</th>
      <th>deaths</th>
      <th>battles</th>
      <th>size</th>
      <th>veterans</th>
      <th>readiness</th>
      <th>armored</th>
      <th>deserters</th>
    </tr>
    <tr>
      <th>origin</th>
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
      <th>Texas</th>
      <td>Nighthawks</td>
      <td>2nd</td>
      <td>25</td>
      <td>2</td>
      <td>1099</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>Nighthawks</td>
      <td>1st</td>
      <td>523</td>
      <td>5</td>
      <td>1045</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Step 17. Select the third cell in the row named Arizona


```python
army.loc[['Arizona']].iloc[:, 2]
```




    origin
    Arizona    523
    Name: deaths, dtype: int64



### Step 18. Select the third cell down in the column named deaths


```python
army[['deaths']].iloc[2:, :]
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
      <th>deaths</th>
    </tr>
    <tr>
      <th>origin</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Texas</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>616</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>43</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>234</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>523</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>62</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>62</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>73</td>
    </tr>
    <tr>
      <th>Louisana</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>