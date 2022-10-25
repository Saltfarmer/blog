---
title: "Python Crash Course Exercise 3"
header :
  image: /assets/images/pandas-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Exercise
  - Pandas
 

---

Today i will completing Pandas Exercise using Ecommerce Purchase. If you want to solve it all by yourself, you can download notebooks file [here](https://drive.google.com/file/d/1ehpuK3y8FTyCmqwgJ7jIv7zsT5vMCOj7/view?usp=sharing) and dataset [here](https://drive.google.com/file/d/1tBDm1wSiVvuS9H_rjtK4HpI5CWerCiaD/view?usp=sharing)

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

# Ecommerce Purchases Exercise

In this Exercise you will be given some Fake Data about some purchases done through Amazon! Just go ahead and follow the directions and try your best to answer the questions and complete the tasks. Feel free to reference the solutions. Most of the tasks can be solved in different ways. For the most part, the questions get progressively harder.

Please excuse anything that doesn't make "Real-World" sense in the dataframe, all the data is fake and made-up.

Also note that all of these questions can be answered with one line of code.

____

** Import pandas and read in the Ecommerce Purchases csv file and set it to a DataFrame called ecom. **


```python
import pandas as pd
```


```python
ecom = pd.read_csv("Ecommerce_Purchases.csv")
```

**Check the head of the DataFrame.**


```python
ecom.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Address</th>
      <th>Lot</th>
      <th>AM or PM</th>
      <th>Browser Info</th>
      <th>Company</th>
      <th>Credit Card</th>
      <th>CC Exp Date</th>
      <th>CC Security Code</th>
      <th>CC Provider</th>
      <th>Email</th>
      <th>Job</th>
      <th>IP Address</th>
      <th>Language</th>
      <th>Purchase Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16629 Pace Camp Apt. 448\nAlexisborough, NE 77...</td>
      <td>46 in</td>
      <td>PM</td>
      <td>Opera/9.56.(X11; Linux x86_64; sl-SI) Presto/2...</td>
      <td>Martinez-Herman</td>
      <td>6011929061123406</td>
      <td>02/20</td>
      <td>900</td>
      <td>JCB 16 digit</td>
      <td>pdunlap@yahoo.com</td>
      <td>Scientist, product/process development</td>
      <td>149.146.147.205</td>
      <td>el</td>
      <td>98.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9374 Jasmine Spurs Suite 508\nSouth John, TN 8...</td>
      <td>28 rn</td>
      <td>PM</td>
      <td>Opera/8.93.(Windows 98; Win 9x 4.90; en-US) Pr...</td>
      <td>Fletcher, Richards and Whitaker</td>
      <td>3337758169645356</td>
      <td>11/18</td>
      <td>561</td>
      <td>Mastercard</td>
      <td>anthony41@reed.com</td>
      <td>Drilling engineer</td>
      <td>15.160.41.51</td>
      <td>fr</td>
      <td>70.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Unit 0065 Box 5052\nDPO AP 27450</td>
      <td>94 vE</td>
      <td>PM</td>
      <td>Mozilla/5.0 (compatible; MSIE 9.0; Windows NT ...</td>
      <td>Simpson, Williams and Pham</td>
      <td>675957666125</td>
      <td>08/19</td>
      <td>699</td>
      <td>JCB 16 digit</td>
      <td>amymiller@morales-harrison.com</td>
      <td>Customer service manager</td>
      <td>132.207.160.22</td>
      <td>de</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7780 Julia Fords\nNew Stacy, WA 45798</td>
      <td>36 vm</td>
      <td>PM</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0 ...</td>
      <td>Williams, Marshall and Buchanan</td>
      <td>6011578504430710</td>
      <td>02/24</td>
      <td>384</td>
      <td>Discover</td>
      <td>brent16@olson-robinson.info</td>
      <td>Drilling engineer</td>
      <td>30.250.74.19</td>
      <td>es</td>
      <td>78.04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23012 Munoz Drive Suite 337\nNew Cynthia, TX 5...</td>
      <td>20 IE</td>
      <td>AM</td>
      <td>Opera/9.58.(X11; Linux x86_64; it-IT) Presto/2...</td>
      <td>Brown, Watson and Andrews</td>
      <td>6011456623207998</td>
      <td>10/25</td>
      <td>678</td>
      <td>Diners Club / Carte Blanche</td>
      <td>christopherwright@gmail.com</td>
      <td>Fine artist</td>
      <td>24.140.33.94</td>
      <td>es</td>
      <td>77.82</td>
    </tr>
  </tbody>
</table>

** How many rows and columns are there? **


```python
ecom.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Address           10000 non-null  object 
     1   Lot               10000 non-null  object 
     2   AM or PM          10000 non-null  object 
     3   Browser Info      10000 non-null  object 
     4   Company           10000 non-null  object 
     5   Credit Card       10000 non-null  int64  
     6   CC Exp Date       10000 non-null  object 
     7   CC Security Code  10000 non-null  int64  
     8   CC Provider       10000 non-null  object 
     9   Email             10000 non-null  object 
     10  Job               10000 non-null  object 
     11  IP Address        10000 non-null  object 
     12  Language          10000 non-null  object 
     13  Purchase Price    10000 non-null  float64
    dtypes: float64(1), int64(2), object(11)
    memory usage: 1.1+ MB


** What is the average Purchase Price? **


```python
ecom['Purchase Price'].mean()
```


    50.347302



** What were the highest and lowest purchase prices? **


```python
ecom["Purchase Price"].max()
```


    99.99




```python
ecom["Purchase Price"].min()
```


    0.0



** How many people have English 'en' as their Language of choice on the website? **


```python
ecom[ecom["Language"] == "en"].count()
```


    Address             1098
    Lot                 1098
    AM or PM            1098
    Browser Info        1098
    Company             1098
    Credit Card         1098
    CC Exp Date         1098
    CC Security Code    1098
    CC Provider         1098
    Email               1098
    Job                 1098
    IP Address          1098
    Language            1098
    Purchase Price      1098
    dtype: int64



** How many people have the job title of "Lawyer" ? **



```python
ecom[ecom["Job"] == "Lawyer"].count()
```


    Address             30
    Lot                 30
    AM or PM            30
    Browser Info        30
    Company             30
    Credit Card         30
    CC Exp Date         30
    CC Security Code    30
    CC Provider         30
    Email               30
    Job                 30
    IP Address          30
    Language            30
    Purchase Price      30
    dtype: int64



** How many people made the purchase during the AM and how many people made the purchase during PM ? **

**(Hint: Check out [value_counts()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) ) **


```python
ecom["AM or PM"].value_counts()
```


    PM    5068
    AM    4932
    Name: AM or PM, dtype: int64



** What are the 5 most common Job Titles? **


```python
ecom["Job"].value_counts().head(5)
```


    Interior and spatial designer        31
    Lawyer                               30
    Social researcher                    28
    Designer, jewellery                  27
    Research officer, political party    27
    Name: Job, dtype: int64



** Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction? **


```python
ecom[ecom["Lot"] == "90 WT"]["Purchase Price"]
```


    513    75.1
    Name: Purchase Price, dtype: float64



** What is the email of the person with the following Credit Card Number: 4926535242672853 **


```python
ecom[ecom["Credit Card"] == 4926535242672853]["Email"]
```


    1234    bondellen@williams-garza.com
    Name: Email, dtype: object



** How many people have American Express as their Credit Card Provider *and* made a purchase above $95 ?**


```python
ecom[(ecom["CC Provider"] == "American Express") & (ecom["Purchase Price"] > 95)].count()
```


    Address             39
    Lot                 39
    AM or PM            39
    Browser Info        39
    Company             39
    Credit Card         39
    CC Exp Date         39
    CC Security Code    39
    CC Provider         39
    Email               39
    Job                 39
    IP Address          39
    Language            39
    Purchase Price      39
    dtype: int64



** Hard: How many people have a credit card that expires in 2025? **


```python
ecom[ecom["CC Exp Date"].apply(lambda x : x.split("/")[1]) == "25"].count()
```


    Address             1033
    Lot                 1033
    AM or PM            1033
    Browser Info        1033
    Company             1033
    Credit Card         1033
    CC Exp Date         1033
    CC Security Code    1033
    CC Provider         1033
    Email               1033
    Job                 1033
    IP Address          1033
    Language            1033
    Purchase Price      1033
    dtype: int64



** Hard: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...) **


```python
ecom["Email"].apply(lambda x : x.split("@")[1]).value_counts().head(5)
```


    hotmail.com     1638
    yahoo.com       1616
    gmail.com       1605
    smith.com         42
    williams.com      37
    Name: Email, dtype: int64



# Great Job!