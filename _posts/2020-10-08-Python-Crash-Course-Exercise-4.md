---
title: "Python Crash Course Exercise 4"
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

Today i will completing Pandas Exercise using SF Salaries. If you want to solve it all by yourself, you can download notebooks file [here](https://drive.google.com/file/d/1NYKbH7Lo6sV21jbXn2Qs5ZQlqy2v3fha/view?usp=sharing) and dataset [here](https://drive.google.com/file/d/1LtynC1iXncAfuPLUCLDoZXoc_Js3rlqq/view?usp=sharing)

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

# SF Salaries Exercise 

Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

** Import pandas as pd.**


```python
import pandas as pd
```

** Read Salaries.csv as a dataframe called sal.**


```python
sal = pd.read_csv("Salaries.csv")
```

** Check the head of the DataFrame. **


```python
sal.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.00</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966.02</td>
      <td>245131.88</td>
      <td>137811.38</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739.13</td>
      <td>106088.18</td>
      <td>16452.60</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916.00</td>
      <td>56120.71</td>
      <td>198306.90</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134401.60</td>
      <td>9737.00</td>
      <td>182234.59</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

** Use the .info() method to find out how many entries there are.**


```python
sal.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 148654 entries, 0 to 148653
    Data columns (total 13 columns):
     #   Column            Non-Null Count   Dtype  
    ---  ------            --------------   -----  
     0   Id                148654 non-null  int64  
     1   EmployeeName      148654 non-null  object 
     2   JobTitle          148654 non-null  object 
     3   BasePay           148045 non-null  float64
     4   OvertimePay       148650 non-null  float64
     5   OtherPay          148650 non-null  float64
     6   Benefits          112491 non-null  float64
     7   TotalPay          148654 non-null  float64
     8   TotalPayBenefits  148654 non-null  float64
     9   Year              148654 non-null  int64  
     10  Notes             0 non-null       float64
     11  Agency            148654 non-null  object 
     12  Status            0 non-null       float64
    dtypes: float64(8), int64(2), object(3)
    memory usage: 14.7+ MB


**What is the average BasePay ?**


```python
sal["BasePay"].mean()
```


    66325.4488404877



** What is the highest amount of OvertimePay in the dataset ? **


```python
sal["OvertimePay"].max()
```


    245131.88



** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **


```python
sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["JobTitle"]
```


    24    CAPTAIN, FIRE SUPPRESSION
    Name: JobTitle, dtype: object



** How much does JOSEPH DRISCOLL make (including benefits)? **


```python
sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["TotalPayBenefits"]
```


    24    270324.91
    Name: TotalPayBenefits, dtype: float64



** What is the name of highest paid person (including benefits)?**


```python
sal[sal["TotalPayBenefits"] == sal["TotalPayBenefits"].max()]["EmployeeName"]
```


    0    NATHANIEL FORD
    Name: EmployeeName, dtype: object



** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**


```python
sal[sal["TotalPayBenefits"] == sal["TotalPayBenefits"].min()]["EmployeeName"]
# Negative result of Total Pay with Benefits
```


    148653    Joe Lopez
    Name: EmployeeName, dtype: object



** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **


```python
sal.groupby("Year").mean()["BasePay"]
```


    Year
    2011    63595.956517
    2012    65436.406857
    2013    69630.030216
    2014    66564.421924
    Name: BasePay, dtype: float64



** How many unique job titles are there? **


```python
sal["JobTitle"].nunique()
```


    2159



** What are the top 5 most common jobs? **


```python
sal["JobTitle"].value_counts().head(5)
```


    Transit Operator                7036
    Special Nurse                   4389
    Registered Nurse                3736
    Public Svc Aide-Public Works    2518
    Police Officer 3                2421
    Name: JobTitle, dtype: int64



** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **


```python
sum(sal[sal["Year"] == 2013]["JobTitle"].value_counts() == 1)
```


    202



** How many people have the word Chief in their job title? (This is pretty tricky) **


```python
sal[sal["JobTitle"].apply(lambda x : "chief" in x.lower())]["JobTitle"].count()
```


    627



** Bonus: Is there a correlation between length of the Job Title string and Salary? **


```python
sal["title_len"] = sal["JobTitle"].apply(lambda x : len(x))
sal[["title_len", "TotalPayBenefits"]].corr()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_len</th>
      <th>TotalPayBenefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>title_len</th>
      <td>1.000000</td>
      <td>-0.036878</td>
    </tr>
    <tr>
      <th>TotalPayBenefits</th>
      <td>-0.036878</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

# Great Job!