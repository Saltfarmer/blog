---
title: "Pandas Exercise 7 : Visualization Bonus"
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

# Tips Dataset

### Introduction:

This exercise was created based on the tutorial and documentation from [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/index.html)  
The dataset being used is tips from Seaborn.

### Step 1. Import the necessary libraries:


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv). 

### Step 3. Assign it to a variable called tips


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv'
tips = pd.read_csv(url)
```

### Step 4. Delete the Unnamed 0 column


```python
tips.drop(['Unnamed: 0'], axis=1, inplace=True)
```

### Step 5. Plot the total_bill column histogram


```python
sns.histplot(tips['total_bill'])
```




    <AxesSubplot:xlabel='total_bill', ylabel='Count'>




    
![png](https://i.ibb.co/BsYbdyC/output-9-1.png)
    


### Step 6. Create a scatter plot presenting the relationship between total_bill and tip


```python
sns.scatterplot(x=tips['total_bill'], y=tips['tip'])
```




    <AxesSubplot:xlabel='total_bill', ylabel='tip'>




    
![png](https://i.ibb.co/nBCnVcX/output-11-1.png)
    


### Step 7.  Create one image with the relationship of total_bill, tip and size.
#### Hint: It is just one function.


```python
sns.pairplot(tips)
```




    <seaborn.axisgrid.PairGrid at 0x2b3d89aa070>




    
![png](https://i.ibb.co/mGKVGk8/output-13-1.png)
    


### Step 8. Present the relationship between days and total_bill value


```python
sns.barplot(x=tips['day'], y=tips['total_bill'])
```




    <AxesSubplot:xlabel='day', ylabel='total_bill'>




    
![png](https://i.ibb.co/N6dXxg2/output-15-1.png)
    


### Step 9. Create a scatter plot with the day as the y-axis and tip as the x-axis, differ the dots by sex


```python
sns.stripplot(x=tips['tip'], y=tips['day'], hue=tips['sex'], alpha=0.8)
```




    <AxesSubplot:xlabel='tip', ylabel='day'>




    
![png](https://i.ibb.co/fNJ7XNH/output-17-1.png)
    


### Step 10.  Create a box plot presenting the total_bill per day differetiation the time (Dinner or Lunch)


```python
sns.boxplot(x=tips['tip'], y=tips['day'], hue=tips['time'])
```




    <AxesSubplot:xlabel='tip', ylabel='day'>




    
![png](https://i.ibb.co/mSQw1L3/output-19-1.png)
    


### Step 11. Create two histograms of the tip value based for Dinner and Lunch. They must be side by side.


```python
sns.FacetGrid(tips, col='time').map(sns.histplot, x=tips['tip'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b3dcce9910>




    
![png](https://i.ibb.co/PrmZCMX/output-21-1.png)
    


### Step 12. Create two scatterplots graphs, one for Male and another for Female, presenting the total_bill value and tip relationship, differing by smoker or no smoker
### They must be side by side.


```python
sns.FacetGrid(tips, col='sex').map(sns.scatterplot, x=tips['total_bill'], y=tips['tip'], hue=tips['smoker'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b3dcd03d00>




    
![png](https://i.ibb.co/DK4ghDk/output-23-1.png)
    


### BONUS: Create your own question and answer it using a graph.


```python
sns.FacetGrid(tips, col='day', row='time').map(sns.scatterplot, x=tips['total_bill'], y=tips['tip'],\
                                               hue=tips['sex'], style=tips['smoker'], s=tips['size']*25)
```




    <seaborn.axisgrid.FacetGrid at 0x2b3e099dac0>




    
![png](https://i.ibb.co/nf6gzsW/output-25-1.png)
    

