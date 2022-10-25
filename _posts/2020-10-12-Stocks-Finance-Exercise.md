---
title: "Stocks Finance Exercise"
header :
  image: /assets/images/stonks.png
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Exercise
  - Pandas
  - Seaborn
  - EDA
 

---

In this data project we will focus on exploratory data analysis of stock prices. Keep in mind, this project is just meant to practice your visualization and pandas skills, it is not meant to be a robust financial analysis or be taken as financial advice.

____

> NOTE: This project is extremely challenging because it will introduce a lot of new concepts and have you looking things up on your own (we'll point you in the right direction) to try to solve the tasks issued. Feel free to just go through the solutions lecture notebook and video as a "walkthrough" project if you don't want to have to look things up yourself. You'll still learn a lot that way! **

____

We'll focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016. You can download the notebooks [here](https://drive.google.com/file/d/1_xUpC4AjzEjODZ9oKRY6vkd0g8UF6zqO/view?usp=sharing) if you want to try it all by yourself.

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

Now Lets get started

# Exercises

## Get the Data

In this section we will learn how to use pandas to directly read data from Google finance using pandas!

First we need to start with the proper imports, which we've already laid out for you here.

*Note: [You'll need to install pandas-datareader for this to work!](https://github.com/pydata/pandas-datareader) Pandas datareader allows you to [read stock information directly from the internet](http://pandas.pydata.org/pandas-docs/stable/remote_data.html) Use these links for install guidance (**pip install pandas-datareader**), or just follow along with the video lecture.*

### The Imports

Already filled out for you.


```python
from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
plt.rcParams['figure.figsize'] = (20.0, 10.0)
```

## Data

We need to get data using pandas datareader. We will get stock information for the following banks:

*  Bank of America
*  CitiGroup
*  Goldman Sachs
*  JPMorgan Chase
*  Morgan Stanley
*  Wells Fargo

** Figure out how to get the stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks. Set each bank to be a separate dataframe, with the variable name for that bank being its ticker symbol. This will involve a few steps:**

1. Use datetime to set start and end datetime objects.
2. Figure out the ticker symbol for each bank.
3. Figure out how to use datareader to grab info on the stock.

** Use [this documentation page](http://pandas.pydata.org/pandas-docs/stable/remote_data.html) for hints and instructions (it should just be a matter of replacing certain values. Use google finance as a source, for example:**
    

```python
# Bank of America
BAC = data.DataReader("BAC", 'google', start, end)
```



```python
start = pd.to_datetime("2006-1-1")
end = pd.to_datetime("2016-1-1")
```


```python
BAC = data.DataReader("BAC", "yahoo", start, end)
C = data.DataReader("C", "yahoo", start, end)
GS = data.DataReader("GS", "yahoo", start, end)
JPM = data.DataReader("JPM", "yahoo", start, end)
MS = data.DataReader("MS", "yahoo", start, end)
WFC = data.DataReader("WFC", "yahoo", start, end)
# https://github.com/pydata/pandas-datareader/issues/768
```

** Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers**


```python
tickers = ("BAC C GS JPM MS WFC").split()
```

** Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.**


```python
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
```

** Set the column name levels (this is filled out for you):**


```python
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
```

** Check the head of the bank_stocks dataframe.**


```python
bank_stocks.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Bank Ticker</th>
      <th colspan="6" halign="left">BAC</th>
      <th colspan="4" halign="left">C</th>
      <th>...</th>
      <th colspan="4" halign="left">MS</th>
      <th colspan="6" halign="left">WFC</th>
    </tr>
    <tr>
      <th>Stock Info</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>...</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2006-01-03</th>
      <td>47.180000</td>
      <td>46.150002</td>
      <td>46.919998</td>
      <td>47.080002</td>
      <td>16296700.0</td>
      <td>34.811729</td>
      <td>493.799988</td>
      <td>481.100006</td>
      <td>490.000000</td>
      <td>492.899994</td>
      <td>...</td>
      <td>57.169998</td>
      <td>58.310001</td>
      <td>5377000.0</td>
      <td>37.170383</td>
      <td>31.975000</td>
      <td>31.195000</td>
      <td>31.600000</td>
      <td>31.900000</td>
      <td>11016400.0</td>
      <td>20.324865</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>47.240002</td>
      <td>46.450001</td>
      <td>47.000000</td>
      <td>46.580002</td>
      <td>17757900.0</td>
      <td>34.442013</td>
      <td>491.000000</td>
      <td>483.500000</td>
      <td>488.600006</td>
      <td>483.799988</td>
      <td>...</td>
      <td>58.700001</td>
      <td>58.349998</td>
      <td>7977800.0</td>
      <td>37.195881</td>
      <td>31.820000</td>
      <td>31.365000</td>
      <td>31.799999</td>
      <td>31.530001</td>
      <td>10870000.0</td>
      <td>20.089125</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>46.830002</td>
      <td>46.320000</td>
      <td>46.580002</td>
      <td>46.639999</td>
      <td>14970700.0</td>
      <td>34.486385</td>
      <td>487.799988</td>
      <td>484.000000</td>
      <td>484.399994</td>
      <td>486.200012</td>
      <td>...</td>
      <td>58.549999</td>
      <td>58.509998</td>
      <td>5778000.0</td>
      <td>37.297882</td>
      <td>31.555000</td>
      <td>31.309999</td>
      <td>31.500000</td>
      <td>31.495001</td>
      <td>10158000.0</td>
      <td>20.066822</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>46.910000</td>
      <td>46.349998</td>
      <td>46.799999</td>
      <td>46.570000</td>
      <td>12599800.0</td>
      <td>34.434616</td>
      <td>489.000000</td>
      <td>482.000000</td>
      <td>488.799988</td>
      <td>486.200012</td>
      <td>...</td>
      <td>58.770000</td>
      <td>58.570000</td>
      <td>6889800.0</td>
      <td>37.336124</td>
      <td>31.775000</td>
      <td>31.385000</td>
      <td>31.580000</td>
      <td>31.680000</td>
      <td>8403800.0</td>
      <td>20.184694</td>
    </tr>
    <tr>
      <th>2006-01-09</th>
      <td>46.970001</td>
      <td>46.360001</td>
      <td>46.720001</td>
      <td>46.599998</td>
      <td>15619400.0</td>
      <td>34.456806</td>
      <td>487.399994</td>
      <td>483.000000</td>
      <td>486.000000</td>
      <td>483.899994</td>
      <td>...</td>
      <td>58.630001</td>
      <td>59.189999</td>
      <td>4144500.0</td>
      <td>37.731342</td>
      <td>31.825001</td>
      <td>31.555000</td>
      <td>31.674999</td>
      <td>31.674999</td>
      <td>5619600.0</td>
      <td>20.181515</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>

# EDA

Let's explore the data a bit! Before continuing, I encourage you to check out the documentation on [Multi-Level Indexing](http://pandas.pydata.org/pandas-docs/stable/advanced.html) and [Using .xs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html).
Reference the solutions if you can not figure out how to use .xs(), since that will be a major part of this project.

** What is the max Close price for each bank's stock throughout the time period?**


```python
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
```


    Bank Ticker
    BAC     54.900002
    C      564.099976
    GS     247.919998
    JPM     70.080002
    MS      89.300003
    WFC     58.520000
    dtype: float64



** Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**

$$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$


```python
returns = pd.DataFrame()
```

** We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.**


```python
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BAC Return</th>
      <th>C Return</th>
      <th>GS Return</th>
      <th>JPM Return</th>
      <th>MS Return</th>
      <th>WFC Return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-03</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>-0.010620</td>
      <td>-0.018462</td>
      <td>-0.013812</td>
      <td>-0.014183</td>
      <td>0.000686</td>
      <td>-0.011599</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>0.001288</td>
      <td>0.004961</td>
      <td>-0.000393</td>
      <td>0.003029</td>
      <td>0.002742</td>
      <td>-0.001110</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>-0.001501</td>
      <td>0.000000</td>
      <td>0.014169</td>
      <td>0.007046</td>
      <td>0.001025</td>
      <td>0.005874</td>
    </tr>
    <tr>
      <th>2006-01-09</th>
      <td>0.000644</td>
      <td>-0.004731</td>
      <td>0.012030</td>
      <td>0.016242</td>
      <td>0.010586</td>
      <td>-0.000158</td>
    </tr>
  </tbody>
</table>



** Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?**


```python
import seaborn as sns
sns.pairplot(returns)
```


    <seaborn.axisgrid.PairGrid at 0x2b15117fe50>




![png](https://i.ibb.co/r6ZNy4k/output-22-1.png)


** Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?**


```python
returns.idxmax()
```


    BAC Return   2009-04-09
    C Return     2008-11-24
    GS Return    2008-11-24
    JPM Return   2009-01-21
    MS Return    2008-10-13
    WFC Return   2008-07-16
    dtype: datetime64[ns]




```python
returns.idxmin()
```


    BAC Return   2009-01-20
    C Return     2009-02-27
    GS Return    2009-01-20
    JPM Return   2009-01-20
    MS Return    2008-10-09
    WFC Return   2009-01-20
    dtype: datetime64[ns]

There is something happen in 2009-01-20, Let's check out. According to https://www.onthisday.com/date/2009/january/20. Barack Obama, inaugurated as the 44th President of the United States of America, becomes the United States' first African-American president

** You should have noticed that Citigroup's largest drop and biggest gain were very close to one another, did anythign significant happen in that time frame? **

** Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?**


```python
returns.std()
```


    BAC Return    0.036647
    C Return      0.038672
    GS Return     0.025390
    JPM Return    0.027667
    MS Return     0.037819
    WFC Return    0.030238
    dtype: float64




```python
returns.loc["2015-01-01" : "2015-12-31"].std()
```


    BAC Return    0.016163
    C Return      0.015289
    GS Return     0.014046
    JPM Return    0.014017
    MS Return     0.016249
    WFC Return    0.012591
    dtype: float64



According to Standart Deviation, Morgan Stanley is the most riskiest to get invested in 2015

** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **


```python
sns.distplot(returns["MS Return"].loc["2015-01-01" : "2015-12-31"])
```


    <AxesSubplot:xlabel='MS Return'>




![png](https://i.ibb.co/PMRx9T1/output-33-1.png)


** Create a distplot using seaborn of the 2008 returns for CitiGroup **


```python
sns.distplot(returns["C Return"].loc["2008-01-01" : "2008-12-31"])
```


    <AxesSubplot:xlabel='C Return'>




![png](https://i.ibb.co/wNQKqvb/output-35-1.png)


____

# More Visualization

A lot of this project will focus on visualizations. Feel free to use any of your preferred visualization libraries to try to recreate the described plots below, seaborn, matplotlib, plotly and cufflinks, or just pandas.

### Imports

** Create a line plot showing Close price for each bank for the entire index of time. (Hint: Try using a for loop, or use [.xs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html) to get a cross section of the data.)**


```python
sns.lineplot(data=returns, palette="tab10")
```


    <AxesSubplot:>




![png](https://i.ibb.co/Hg6pZyZ/output-38-1.png)



```python
sns.lineplot(data=returns.rolling(window=90).mean(), palette="tab10")
```


    <AxesSubplot:>




![png](https://i.ibb.co/cD0bjhr/output-39-1.png)



```python
sns.lineplot(data=returns.rolling(window=365).mean(), palette="tab10")
```


    <AxesSubplot:>




![png](https://i.ibb.co/xFTN16y/output-40-1.png)


## Moving Averages

Let's analyze the moving averages for these stocks in the year 2008. 

** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**


```python
sns.lineplot(data=BAC["Close"].loc['2008-01-01':'2009-01-01'])
sns.lineplot(data=BAC["Close"].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean())
```


    <AxesSubplot:>




![png](https://i.ibb.co/X7FtQms/output-42-1.png)


** Create a heatmap of the correlation between the stocks Close Price.**


```python
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr())
```


    <AxesSubplot:xlabel='Bank Ticker', ylabel='Bank Ticker'>




![png](https://i.ibb.co/vJfgd4T/output-44-1.png)


** Optional: Use seaborn's clustermap to cluster the correlations together:**


```python
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr())
```


    <seaborn.matrix.ClusterGrid at 0x2b166301c70>




![png](https://i.ibb.co/5scD0RQ/output-46-1.png)


# Great Job!

Definitely a lot of more specific finance topics here, so don't worry if you didn't understand them all! The only thing you should be concerned with understanding are the basic pandas and visualization operations.








