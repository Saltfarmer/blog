---
title: "Day 5 Algorit.ma : Classification Model"
header :
  image : /assets/images/AlgoritmaBanner.jpg
  teaser: /assets/images/Algoritma.png
comments : true
share : true
categories:
  - Data Science
  - Sklearn
  - Statsmodel
tags:
  - Python
  - Algoritma
  - Pandas
  - Classification

---

Day 5, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Inclass: Classification Model**
- Durasi: 7 hours
- _Last Updated_: Desember 2023

___

- Disusun dan dikurasi oleh tim produk dan instruktur [Algoritma Data Science School](https://algorit.ma).

# Classification in Machine Learning


```python
import pandas as pd # preprocessing data
import math # math operations
%matplotlib inline 

pd.set_option('display.max_columns', 100) # set max columns display
from matplotlib.pyplot import figure
figure(figsize=(9, 15), dpi=80)
```




    <Figure size 720x1200 with 0 Axes>




    <Figure size 720x1200 with 0 Axes>


# Introduction

* Klasifikasi bertujuan untuk memprediksi **target variable kategorik** seperti label/kelas
* Label/kelas yang dapat diprediksi antara lain berjenis 2 kelas (**binary**) atau >2 kelas (**multiclass**).

# Logistic Regression Concept

Logistic Regression merupakan salah satu metode klasifikasi yang konsepnya hampir mirip dengan regresi linear. Hanya saja, dalam logistic regression tidak menghitung secara spesifik nilai prediksi target variable, namun menghitung **kemungkinan/peluang** pada masing-masing kelas target.

- Linier regresion: y (numerik) -> **-inf, + inf**
- Logistic regression: y (peluang) -> **0, 1**

![](assets/data-science-programming-contrast-linear-logistic-regression.jpg)

📝 Hasil dari regresi logistik dapat digunakan untuk:
- keperluan interpretasi
- keperluan prediksi

❓ Bagaimana regresi logistik bekerja? 

Suatu regresi yang dapat menghasilkan nilai (-inf sd. +inf), lalu dikonversikan ke bentuk peluang (0 - 1)
  - nilai yg dihasilkan oleh algoritma logistic regression: log of odds
  - nilai dapat dikonversikan antara **log of odds** - **odds** - **peluang**:

<div>
<img src="assets/linear_vs_logistic_regression.png" width="700"/>
</div>

## Basic Intuition: Probability

* Pada dasarnya, ketika kita melakukan klasifikasi, kita mempertimbangkan **peluang**.

**Probability** : kemungkinan terjadi suatu kejadian dari seluruh kejadian yang ada.

$$P(A) = \frac{n}{S} $$ 

* $P(A)$ : peluang kejadian A
* $n$ : banyak kejadian A
* $S$ : total seluruh kejadian

💭❓ **Analytical Question**

Terdapat 100 data transaksi dari sebuah Bank, 10 diantaranya merupakan transaksi `fraud` (palsu), sedangkan sisanya sebanyak 90 adalah transaksi `not fraud`. Berapakah peluang kejadian transaksi `fraud`?


```python
# probability fraud
p_fraud =  10/100
p_fraud 
```




    0.1



📝 **Note:**  Range dari probability : **0 - 1**

## Odds 

Ketika kita menebak suatu nilai dalam regresi, range nilai yang kita tebak adalah **$-\infty - \infty$**. Sedangkan dalam klasifikasi, range nilai yang kita tebak adalah **0 - 1**. Oleh karena itu, kita memerlukan suatu jembatan untuk bisa menghubungkan antara nilai numerik menjadi suatu nilai peluang. Jembatan tersebut disebut **Odds**.

**Odds** : perbandingan probability kejadian sukses (yang diamati) dibandingkan dengan probability kejadian tidak sukses (tidak diamati)

$$Odds = \frac{p}{1-p}$$

$p$ : merupakan probability kejadian

Jika ingin mengetahui odds dari kejadian 'yes', maka:

$$Odds(yes) = \frac{p(yes)}{1-p(yes)}$$

Jika ingin mengetahui odds dari kejadian 'no' maka:

$$Odds(no) = \frac{p(no)}{1-p(no)}$$


```python
# odds fraud
odds_fraud = 0.1 / 0.9
odds_fraud
```




    0.11111111111111112



> 📈 Interpretasi: Kemungkinan/probability transaksi sebagai fraud adalah **XX KALI** lebih mungkin dibandingkan diketahui sebagai not fraud


```python
# odds not fraud
odds_not_fraud = 0.9 / 0.1
odds_not_fraud
```




    9.0



> 📈 Interpretasi: Kemungkinan jenis tanah diketahui sebagai not fraud adalah **XX KALI** lebih mungkin dibandingkan diketahui sebagai fraud

📝 **Note:**  Range dari odds : **0 - inf**

## Log of Odds

**Log of Odds** : suatu nilai odds yang di logaritmakan.

$$logit(p) = log(\frac{p}{1-p})$$

💭❓ Berapakah log of odds transaksi fraud?


```python
# log of odds fraud
log_odds_fraud = math.log(0.11111111111111112)
log_odds_fraud
```




    -2.197224577336219



 **💡Highlight Point:💡**
 
 - Untuk menginterpretasikan log of odds kedalam nilai odds -> `math.exp()`
 
 - Untuk menginterpretasikan log of odds kedalam probability -> $\frac{odds}{odds+1}$ atau 
 
```python
from scipy.special import expit
expit()
```


```python
# example menginterpretasikan dari log of odds --> probability
from scipy.special import expit
expit(log_odds_fraud)
```




    0.10000000000000002



![](assets/prob_to_logofodds_sigmoid.png)

## Logistic Regression Modeling Workflow

Berikut adalah urutan *workflow* model Logistic Regression :

1. Mempersiapkan data
2. *Exploratory Data Analysis*
3. Data *Pre-Processing*
4. Membuat model logistic regression & interpretasi
5. Melakukan prediksi
6. Model evaluasi

## Study Case : Fraud Bank Account 

Berbagai penipuan yang marak terjadi melibatkan penggunaan rekening bank. Tentunya hal ini meresahkan dan menyebabkan adanya kerugian baik untuk nasabah maupun bank ini sendiri. Kerugian ini bisa berupa kerugian material sampai menurunnya kepercayaan masyarakat terhadap suatu bank.

Data yang akan kita gunakan saat ini merupakan data akun bank yang sudah disesuaikan untuk pembelajaran di workshop ini.

### Import data

Dalam pembelajaran kali ini kita akan menggunakan data `fraud_dataset.csv` yang tersimpan pada folder `data_input`. 

Data ini dapat dieksplorasi di luar kelas, tetapi untuk kepentingan pembelajaran kita hanya akan mengambil 11 variabel yang nantinya akan digunakan pada model. Silakan jalankan kode berikut ini:


```python
import pandas as pd

fraud = pd.read_csv('data_input/fraud_dataset.csv')
col_used = ['income', 'name_email_similarity', 'intended_balcon_amount', 'zip_count_4w', 
            'credit_risk_score', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 
            'proposed_credit_limit', 'source', 'fraud_bool']
fraud = fraud[col_used]

fraud.sample(3)
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>source</th>
      <th>fraud_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8890</th>
      <td>0.2</td>
      <td>0.812254</td>
      <td>-1.326747</td>
      <td>823</td>
      <td>169.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1500.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8083</th>
      <td>0.9</td>
      <td>0.143784</td>
      <td>51.139202</td>
      <td>1333</td>
      <td>135.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>758</th>
      <td>0.8</td>
      <td>0.737511</td>
      <td>22.804448</td>
      <td>1573</td>
      <td>28.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Data Description:**

- `income` (numeric): _Annual income of the applicant (in decile form). Ranges between [0.1, 0.9]._
- `name_email_similarity` (numeric): _Metric of similarity between email and applicant’s name. Higher values represent higher similarity. Ranges between [0, 1]._
- `intended_balcon_amount` (numeric): _Initial transferred amount for application. Ranges between [−16, 114] (negatives are missing values)._
- `zip_count_4w` (numeric): _Number of applications within same zip code in last 4 weeks. Ranges between [1, 6830]._
- `credit_risk_score` (numeric): _Internal score of application risk. Ranges between [−191, 389]._
- `phone_home_valid` (binary): _Validity of provided home phone._
- `phone_mobile_valid` (binary): _Validity of provided mobile phone._
- `has_other_cards` (binary): _If applicant has other cards from the same banking company. _
- `proposed_credit_limit` (numeric): _Applicant’s proposed credit limit. Ranges between [200, 2000]._
- `source` (categorical): _Online source of application. Either browser (INTERNET) or app (TELEAPP)._
- `fraud_bool` (binary): _If the application is fraudulent or not._


Sebelum masuk pada tahap pembuatan model, kita akan melakukan EDA untuk mengetahui variabel prediktor yang perlu dimasukkan dalam model dan yang tidak.

### Wrangling Data

#### Mengubah tipe data

Sebelum melakukan perubahan tipe data, silakan cek terlebih dahulu jenis tipe datanya dengan menggunakan method `dtypes`/`info()`


```python
# code here
fraud.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14905 entries, 0 to 14904
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   income                  14905 non-null  float64
     1   name_email_similarity   14905 non-null  float64
     2   intended_balcon_amount  14905 non-null  float64
     3   zip_count_4w            14905 non-null  int64  
     4   credit_risk_score       14905 non-null  float64
     5   phone_home_valid        14905 non-null  int64  
     6   phone_mobile_valid      14905 non-null  int64  
     7   has_other_cards         14905 non-null  int64  
     8   proposed_credit_limit   14905 non-null  float64
     9   source                  14905 non-null  object 
     10  fraud_bool              14905 non-null  int64  
    dtypes: float64(5), int64(5), object(1)
    memory usage: 1.3+ MB
    

❓ Kolom apa saja yang belum memliki tipe data yang tepat?

- `source`


```python
# list berisi nama kolom yang ingin diubah dalam format sama
listkolom = ['source']

# Mengubah tipe data beberapa kolom
fraud['source'] = fraud['source'].astype('category')

# cek kembali tipe data
fraud.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14905 entries, 0 to 14904
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype   
    ---  ------                  --------------  -----   
     0   income                  14905 non-null  float64 
     1   name_email_similarity   14905 non-null  float64 
     2   intended_balcon_amount  14905 non-null  float64 
     3   zip_count_4w            14905 non-null  int64   
     4   credit_risk_score       14905 non-null  float64 
     5   phone_home_valid        14905 non-null  int64   
     6   phone_mobile_valid      14905 non-null  int64   
     7   has_other_cards         14905 non-null  int64   
     8   proposed_credit_limit   14905 non-null  float64 
     9   source                  14905 non-null  category
     10  fraud_bool              14905 non-null  int64   
    dtypes: category(1), float64(5), int64(5)
    memory usage: 1.2 MB
    

#### Cek Missing Value & Duplicate Data

Dalam pengecekan *missing values* disediakan fungsi `isna()` yang dapat mengecek ke setiap baris data dan menunjukan *logical value*. Untuk mempermudah pengecekannya, fungsi tersebut dapat digabungkan dengan fungsi `.sum()`.

Dalam pengecekan *nilai duplikat* disediakan sebuah fungsi `duplicated()` yang dapat mengecek ke setiap baris data dan menunjukan *logical value*. Untuk mempermudah pengecekannya, fungsi tersebut dapat digabungkan dengan fungsi `.any()`.


```python
# cek missing value
fraud.isna().sum()
```




    income                    0
    name_email_similarity     0
    intended_balcon_amount    0
    zip_count_4w              0
    credit_risk_score         0
    phone_home_valid          0
    phone_mobile_valid        0
    has_other_cards           0
    proposed_credit_limit     0
    source                    0
    fraud_bool                0
    dtype: int64




```python
# cek duplicate
fraud.duplicated().any()
```




    False



### Exploratory Data Analysis (EDA)

**Analisis `describe()`**

Pada tahapan ini kita akan mencoba untuk melakkan analisis apakah terdapat sebuah hal yang menarik dari hasil fungsi `describe()` untuk masing-masing kelas target


```python
fraud.describe()
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>fraud_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.571110</td>
      <td>0.481305</td>
      <td>7.986892</td>
      <td>1571.105736</td>
      <td>136.478363</td>
      <td>0.400671</td>
      <td>0.883126</td>
      <td>0.213485</td>
      <td>551.910768</td>
      <td>0.113116</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.291264</td>
      <td>0.292755</td>
      <td>19.702913</td>
      <td>998.577819</td>
      <td>73.059616</td>
      <td>0.490051</td>
      <td>0.321280</td>
      <td>0.409781</td>
      <td>516.560244</td>
      <td>0.316746</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>0.000093</td>
      <td>-12.537085</td>
      <td>36.000000</td>
      <td>-154.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>190.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.300000</td>
      <td>0.206239</td>
      <td>-1.173150</td>
      <td>893.000000</td>
      <td>85.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>200.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.600000</td>
      <td>0.472416</td>
      <td>-0.834826</td>
      <td>1267.000000</td>
      <td>127.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>200.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.800000</td>
      <td>0.748003</td>
      <td>-0.204896</td>
      <td>1941.000000</td>
      <td>186.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.900000</td>
      <td>0.999997</td>
      <td>111.697355</td>
      <td>6349.000000</td>
      <td>378.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2100.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fraud[fraud['intended_balcon_amount'] < 0]
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>source</th>
      <th>fraud_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>-1.074674</td>
      <td>3483</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>-1.043444</td>
      <td>2849</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9</td>
      <td>0.401137</td>
      <td>-0.394588</td>
      <td>780</td>
      <td>74.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>0.720006</td>
      <td>-0.487785</td>
      <td>4527</td>
      <td>136.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.4</td>
      <td>0.241164</td>
      <td>-1.459099</td>
      <td>1434</td>
      <td>144.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>500.0</td>
      <td>INTERNET</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14896</th>
      <td>0.3</td>
      <td>0.886668</td>
      <td>-0.941696</td>
      <td>922</td>
      <td>52.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14899</th>
      <td>0.9</td>
      <td>0.142189</td>
      <td>-0.780839</td>
      <td>1728</td>
      <td>54.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14900</th>
      <td>0.9</td>
      <td>0.225052</td>
      <td>-0.521791</td>
      <td>507</td>
      <td>106.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14902</th>
      <td>0.1</td>
      <td>0.494256</td>
      <td>-0.973377</td>
      <td>1177</td>
      <td>121.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14903</th>
      <td>0.7</td>
      <td>0.051507</td>
      <td>-1.162706</td>
      <td>1176</td>
      <td>103.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>INTERNET</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>11344 rows × 11 columns</p>
</div>




```python
fraud_clean = fraud.drop(columns='intended_balcon_amount')
```


```python
fraud[(fraud['proposed_credit_limit'] < 200) |(fraud['proposed_credit_limit'] > 2000)]
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>source</th>
      <th>fraud_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3452</th>
      <td>0.9</td>
      <td>0.217841</td>
      <td>-1.341128</td>
      <td>1079</td>
      <td>305.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2100.0</td>
      <td>INTERNET</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4132</th>
      <td>0.8</td>
      <td>0.217669</td>
      <td>-0.440791</td>
      <td>2315</td>
      <td>159.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2100.0</td>
      <td>INTERNET</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6750</th>
      <td>0.1</td>
      <td>0.455694</td>
      <td>20.775652</td>
      <td>427</td>
      <td>103.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>190.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7963</th>
      <td>0.9</td>
      <td>0.120395</td>
      <td>-0.656258</td>
      <td>57</td>
      <td>93.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>190.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11876</th>
      <td>0.6</td>
      <td>0.881736</td>
      <td>-0.444774</td>
      <td>3065</td>
      <td>109.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>190.0</td>
      <td>INTERNET</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fraud_clean = fraud_clean[(fraud_clean['proposed_credit_limit'] >= 200)  & (fraud_clean['proposed_credit_limit'] <= 2000)]
```


```python
fraud_clean.shape
```




    (14900, 10)



💭 **Insight**: Terdapat ketidaksesuaian data dengan deskripsi data terutama pada kolom `intended_balcon_amount` dan `proposed_credit_limit`

**Analisis Korelasi**


```python
import seaborn as sns
import matplotlib
matplotlib.rc('figure', figsize=(10, 5)) # Buat melebarkan gambar

sns.heatmap(fraud_clean.select_dtypes(include=['int64', 'float64']).corr(), # nilai korelasi
            annot=True,   # anotasi angka di dalam kotak heatmap
            fmt=".3f",    # format 3 angka dibelakang koma 
            cmap='Blues'); # warna heatmap
```


    
![png](output_41_0.png)
    


###  Data Pre-Processing

Terdapat 2 hal yang biasanya dilakukan pada tahapan data pre-processing yaitu **Dummy Variable Encoding** dan juga **Cross Validation**


```python

```

#### Dummy Variable Encoding 

Variabel yang kita miliki terdapat variabel dengan tipe data category, oleh karena itu kita perlu membuat dummy variabel terlebih dahulu. Untuk algoritma Logistic Regression, karena masih terdapat asumsi multicolinearity, maka yang akan dipakai adalah dummy variable. 
    
Mari lakukan metode tersebut dengan memanfaatkan fungsi berikut ini `pd.get_dummies()` dan mengisinya dengan beberapa parameter antara lain:

- `data`: data yang ingin diubah menjadi numerikal
- `columns`: list kolom yang akan dilakukan dummy variable encoding
- `drop_first`: apakah ingin drop kolom pertama. Default False. Namun akan kita atur sebagai True agar kolom hasil dummies tidak redundan
- `dtype` = memasukan tipe data yang ingin di-isi


```python
# code here
fraud_enc = pd.get_dummies(data=fraud_clean,columns=['source'],drop_first=True, dtype= 'int64')
fraud_enc
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>fraud_bool</th>
      <th>source_TELEAPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>3483</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>2849</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>0.370933</td>
      <td>406</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9</td>
      <td>0.401137</td>
      <td>780</td>
      <td>74.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>0.720006</td>
      <td>4527</td>
      <td>136.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14900</th>
      <td>0.9</td>
      <td>0.225052</td>
      <td>507</td>
      <td>106.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14901</th>
      <td>0.4</td>
      <td>0.766147</td>
      <td>1822</td>
      <td>130.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>500.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14902</th>
      <td>0.1</td>
      <td>0.494256</td>
      <td>1177</td>
      <td>121.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14903</th>
      <td>0.7</td>
      <td>0.051507</td>
      <td>1176</td>
      <td>103.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14904</th>
      <td>0.1</td>
      <td>0.780972</td>
      <td>1513</td>
      <td>97.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>14900 rows × 10 columns</p>
</div>



#### Cross Validation

*Cross Validation* adalah metode yang kita gunakan untuk mengetahui seberapa baik performa model kita memprediksi terhadap data baru.

Lantas, bagaimana cara mengetahui apakah model yang kita buat telah baik dalam memprediksi data baru? Di sinilah mengapa kita melakukan Train-test splitting. Kita membagi data kita menjadi 2 kelompok, yaitu data `train` dan `test`.

<img src="assets/test-train.png" width="600"/>

- Data `train`: Data yang model gunakan untuk training.

- Data `test`: Data untuk evaluasi model (Untuk melihat seberapa baik model memprediksi terhadap data yang tidak digunakan untuk training)

📌 **Analogi sederhana**

- Seorang siswa dapat dikatakan pintar ketika dapat menjawab benar soal-soal ujian yang tidak pernah dikerjakannya pada soal-soal latihan untuk persiapan ujian.
- Data `train` diibaratkan soal latihan, dan data `test` diibaratkan soal ujian. Adapun `model` kita diibaratkan sebagai siswa.


Kita dapat menggunakan fungsi `train_test_split` dengan beberapa parameter sebagai berikut.
- `arrays`: dataframe yang kita gunakan (dipisah , untuk yang prediktor dan target variable)
- `test_size`: jumlah persentase dari data yang akan digunakan sebagai data test
- `train_size`: jumlah persentase dari data yang akan digunakan sebagai data test (akan otomatis terisi jika `test_size` diberi nilai)
- `random_state`: nilai random number generator (RNG). Jika kita memasukkan suatu nilai integer untuk parameter ini maka akan menghasilkan hasil yang sama untuk nilai yang sama. Jika kita mengubah nilainya, maka hasilnya akan berbeda.
- `stratify`: memastikan pembagian di data train dan test memiliki proporsi target yang sama dengan data awal

> **💡 NOTES**: Biasanya data dibagi menjadi 80:20 atau 70:30 (train size:test size). Porsi yang besar selalu digunakan untuk training


```python
# Total dimensi awal sebelum split
fraud_enc.shape
```




    (14905, 11)




```python
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
```


```python
# Tahapan 1 - Memisahkan prediktor dengan target
## prediktor
X = sm.add_constant(fraud_enc.drop(columns='fraud_bool'))

## target
Y = fraud_enc['fraud_bool']
```


```python
# Tahapan 2 - Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, # kolom prediktor
                                                   Y, # kolom target
                                                   test_size = 0.2, # 80% training and 20% test
                                                   random_state = 10,
                                                   stratify=Y)
```


```python
X_train.shape
```




    (11920, 10)




```python
y_train.value_counts(normalize=True)
```




    fraud_bool
    0    0.886997
    1    0.113003
    Name: proportion, dtype: float64



❓ **Mengapa kita perlu mengunci sifat random yang ada?**

- Agar kita mendapatkan hasil antara data train dan data test yang sama 
- Ketika kita ingin melakukan adjustment/tunning pada model yang sudah ada, data yang akan dimasukan kembali ke model tersebut sama dengan model yang sebelumnya. Sehingga kita bisa melakukan komparasi yang apple to apple terhadap kedua model tersebut.

**Cek Proporsi Kelas Target**

Setelah melakukan cross validation, kita perlu memastikan bahwa proporsi kelas target kita sudah seimbang atau belum.

❓ **Mengapa kita harus mencari tau proporsi targetnya seimbang/tidak?**

- Proporsi yang seimbang penting untuk agar model dapat mempelajari karakteristik kelas positif maupun negatif secara seimbang
- Dalam kata lain, tidak hanya belajar dari satu kelas saja. Hal ini mencegah model dari *hanya baik memprediksi 1 kelas saja*

Dalam melakukan pengecekan, pandas sudah menyediakan sebuah fungsi `crosstab()`. Pada fungsi tersebut akan di-isi dengan 3 parameter yaitu

- `index`: parameter ini akan di-isi dengan target data train kita
- `columns`: parameter ini akan di-isi dengan target variable
- `normalize`: dapat di-isi dengan True untuk menunjukan hasil dalam bentuk persentase.


```python
# Code here
pd.crosstab(index = y_train, 
            columns = 'count', 
            normalize = True).round(2)
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
      <th>col_0</th>
      <th>count</th>
    </tr>
    <tr>
      <th>fraud_bool</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.11</td>
    </tr>
  </tbody>
</table>
</div>



Proporsi yang imbalance sebenarnya cukup subjektif dan tidak ada aturan bakunya. Akan tetapi ketika proporsinya targetnya *90%:10%* atau *95%:5%*, target variable tersebut akan dianggap tidak seimbang.

**Action Plan ketika datanya imbalance:**

- Tambah data real $\rightarrow$ memerlukan waktu
- Metode *downSampling* $\rightarrow$ Membuang observasi dari kelas mayoritas, sehingga seimbang.
- Metode *upSampling* $\rightarrow$ Duplikasi observasi dari kelas minoritas, sehingga seimbang.

Metode pada poin kedua dan ketiga di atas tidak akan kita pelajari di kelas, tetapi Anda bisa membaca dokumentasinya pada link berikut: [downSampling](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) dan [upSampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html).

### Model Fitting

Untuk membuat model logistic regression, kita bisa menggunakan fungsi `Logit()` dari package `statsmodels` atau `sm`.


```python
# membuat model
model_logit = sm.Logit(y_train, X_train)
model_logit.fit().summary()
```

    Optimization terminated successfully.
             Current function value: 0.298085
             Iterations 7
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>fraud_bool</td>    <th>  No. Observations:  </th>   <td> 11920</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td> 11910</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     9</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 09 Jan 2024</td> <th>  Pseudo R-squ.:     </th>   <td>0.1550</td>  
</tr>
<tr>
  <th>Time:</th>                <td>13:54:54</td>     <th>  Log-Likelihood:    </th>  <td> -3553.2</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -4204.8</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>6.344e-275</td>
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                 <td>   -2.8681</td> <td>    0.156</td> <td>  -18.417</td> <td> 0.000</td> <td>   -3.173</td> <td>   -2.563</td>
</tr>
<tr>
  <th>income</th>                <td>    1.4400</td> <td>    0.121</td> <td>   11.898</td> <td> 0.000</td> <td>    1.203</td> <td>    1.677</td>
</tr>
<tr>
  <th>name_email_similarity</th> <td>   -1.3942</td> <td>    0.111</td> <td>  -12.603</td> <td> 0.000</td> <td>   -1.611</td> <td>   -1.177</td>
</tr>
<tr>
  <th>zip_count_4w</th>          <td> 9.678e-05</td> <td>  3.1e-05</td> <td>    3.126</td> <td> 0.002</td> <td> 3.61e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>credit_risk_score</th>     <td>    0.0064</td> <td>    0.001</td> <td>   11.733</td> <td> 0.000</td> <td>    0.005</td> <td>    0.008</td>
</tr>
<tr>
  <th>phone_home_valid</th>      <td>   -0.7981</td> <td>    0.074</td> <td>  -10.764</td> <td> 0.000</td> <td>   -0.943</td> <td>   -0.653</td>
</tr>
<tr>
  <th>phone_mobile_valid</th>    <td>   -0.5900</td> <td>    0.095</td> <td>   -6.192</td> <td> 0.000</td> <td>   -0.777</td> <td>   -0.403</td>
</tr>
<tr>
  <th>has_other_cards</th>       <td>   -1.3055</td> <td>    0.103</td> <td>  -12.713</td> <td> 0.000</td> <td>   -1.507</td> <td>   -1.104</td>
</tr>
<tr>
  <th>proposed_credit_limit</th> <td>    0.0005</td> <td> 6.96e-05</td> <td>    7.734</td> <td> 0.000</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>source_TELEAPP</th>        <td>   -0.1356</td> <td>    0.491</td> <td>   -0.276</td> <td> 0.782</td> <td>   -1.097</td> <td>    0.826</td>
</tr>
</table>



**Interpretasi Model**

Nilai intercept dan slope tidak bisa diinterpretasikan secara langsung karena nilainya masih berupa log of odds. Oleh karena itu, perlu dilakukan interpretasi menggunakan nilai odds. Untuk mengubah nilai log of odds menjadi odds bisa menggunakan fungsi `exp()` dari package `math`.    


```python
model_logit.fit().params.values
```

    Optimization terminated successfully.
             Current function value: 0.298085
             Iterations 7
    




    array([-2.86808695e+00,  1.43996937e+00, -1.39418862e+00,  9.67818085e-05,
            6.43433541e-03, -7.98147401e-01, -5.89977432e-01, -1.30554459e+00,
            5.38173698e-04, -1.35643304e-01])




```python
import numpy as np
np.exp(model_logit.fit().params)
```

    Optimization terminated successfully.
             Current function value: 0.298085
             Iterations 7
    




    const                    0.056807
    income                   4.220567
    name_email_similarity    0.248034
    zip_count_4w             1.000097
    credit_risk_score        1.006455
    phone_home_valid         0.450162
    phone_mobile_valid       0.554340
    has_other_cards          0.271025
    proposed_credit_limit    1.000538
    source_TELEAPP           0.873154
    dtype: float64



Hasil formula model yang diperoleh adalah sebagai berikut :

$$logit(y)= \beta_0 +\beta_1 \times x_1 + ... +\beta_n \times x_n$$

- **Interpretasi intercept/`const`**

- **Interpretasi variabel numerik**:

    -
    -

- **Interpretasi variabel kategorik**:

    -
    -

Interpretasi: ...

### Model Prediction

Ketika kita sudah berhasil membuat model, kita akan mencoba melakukan prediksi terhadap data *test* yang sudah kita persiapkan pada tahap *cross validation*

Dalam melakukan prediksi, kita bisa memanfaaatkan fungsi `predict()`. Dengan syntax sebagai berikut:

`<nama_model>.predict(<var_prediktor>)`


```python
# code of predict value from model
logit_pred = model_logit.fit().predict(X_test)
logit_pred
```

    Optimization terminated successfully.
             Current function value: 0.298085
             Iterations 7
    




    1355     0.027437
    5510     0.046101
    11107    0.310836
    14625    0.023955
    6431     0.118358
               ...   
    582      0.052350
    12280    0.020636
    12577    0.244421
    12116    0.247063
    9421     0.030503
    Length: 2980, dtype: float64



Hasil prediksi yang dikeluarkan masih berupa probability dengan range 0-1. Untuk dapat mengubah nilai probability tersebut, kita bisa menetapkan threshold pada probability untuk masuk ke kelas 1 atau 0. Umumnya threshold yang digunakan yaitu 0.5. 


```python
# change probability to predict class
pred_label = logit_pred.apply(lambda x: 1 if x > 0.5 else 0)
pred_label.sample(5)
```




    11461    0
    11986    0
    13478    0
    14160    0
    10966    0
    dtype: int64



### Model Evaluation

Setelah dilakukan prediksi menggunakan model, masih ada saja prediksi yang salah. Pada klasifikasi, kita mengevaluasi model berdasarkan **confusion matrix**:

- Penentuan kelas:
  + kelas positif: kelas yang lebih difokuskan 
  + kelas negatif: kelas yang tidak difokuskan
 
- Contoh kasus: 
  + Machine learning untuk deteksi pasien covid:
    * kelas positif: terdeteksi covid $\rightarrow$ Jangan sampai orang yang terkena covid dibiarkan bebas karena dapat menularkan ke orang banyak
    * kelas negatif: terdeteksi sehat
    
  + Machine learning untuk deteksi apakah seseorang bisa bayar pinjaman atau tidak
    * kelas positf: yang tidak bisa bayar $\rightarrow$ karna kita perlu berhati2 apakah nasabah tersebut bisa tidak bayar, kalo tidak bayar perusahaan bisa rugi. 
    * kelas negatif: yang bisa bayar

- Isi dari confusion matrix
    * TP (True Positive) = Ketika kita memprediksi kelas `positive`, dan benar bahwa data aktualnya `positive`
    * TN (True Negative) = Ketika kita memprediksi kelas `negative`, dan benar bahwa data aktualnya `negative`
    * FP (False Positive) = Ketika kita memprediksi kelas `positive`, namun data aktualnya `negative`
    * FN (False Negative) = Ketika kita memprediksi kelas `negative`, namun data aktualnya `positive`
    
![](assets/tnfp.PNG)


```python
# confusion matrix sederhana (perbandingan antara pred label dengan data test)
pd.crosstab(y_test, pred_label)
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
      <th>col_0</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>fraud_bool</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2625</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>306</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>



* TP = 37
* TN = 2615
* FN = 29
* FP = 300

4 metrics performa model: **Accuracy, Sensitivity/Recall, Precision, Specificity**

- **Accuracy**: seberapa tepat model kita memprediksi kelas target (secara global)   
- **Sensitivity**/ **Recall**: ukuran kebaikan model terhadap kelas `positif`   
- **Specificity**: ukuran kebaikan model terhadap kelas `negatif`   
- Pos Pred Value/**Precision**: seberapa presisi model memprediksi kelas positif  

### Accuracy

Seberapa baik model kita menjelaskan kelas target (baik positif maupun negatif). Dipakai ketika kelas positif dan negatif sama pentingnya atau ketika proporsi kelas seimbang.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$


```python
# nilai akurasi
from sklearn import metrics

metrics.accuracy_score(y_test, pred_label)
```




    0.8912751677852349



Dalam bisnis/real-case, tak selamanya kita hanya mementingkan metric accuracy. Sering kali harus memilih antara meninggikan **recall/precision**. Hal ini tergantung pada kasus bisnis/efek yang ditimbulkan dari hasil prediksi tersebut.

### Recall / Sensitivity

Seberapa banyak yang **benar diprediksi positif** dari yang **re**alitynya (aktualnya) positif.

![](assets/recall.png)

$$
Recall = \frac{TP}{TP + FN}
$$


```python
# nilai recall
metrics.recall_score(y_test, pred_label)
```




    0.09198813056379822



### Precision

Seberapa banyak yang **benar diprediksi positif** dari yang di**pre**diksi positif.

![](assets/precision.png)

$$
Precision = \frac{TP}{TP + FP}
$$


```python
# nilai precision
metrics.precision_score(y_test, pred_label)
```




    0.6326530612244898



**Cara Cepat**

Selain melakukan perhitungan manual, kita juga dapat memanfaatkan fungsi yang sudah disediakan oleh library sklearn dengan syntax 

`*_score(y_true, y_pred)`


```python
from sklearn.metrics import recall_score, precision_score, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, pred_label)}')
print(f'Recall score: {recall_score(y_test, pred_label)}')
print(f'Precision score: {precision_score(y_test, pred_label)}')
```

    Accuracy score: 0.8912751677852349
    Recall score: 0.09198813056379822
    Precision score: 0.6326530612244898
    

## Buatlah model dengal menghilangkan salah satu variable yang berkorelasi kuat tersebut


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

vif = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
pd.Series(data=vif, index = X_train.columns).sort_values(ascending=False)
```




    const                    23.641677
    credit_risk_score         1.737997
    proposed_credit_limit     1.683440
    phone_home_valid          1.109834
    phone_mobile_valid        1.087907
    income                    1.048076
    has_other_cards           1.039245
    zip_count_4w              1.019992
    name_email_similarity     1.006883
    source_TELEAPP            1.001757
    dtype: float64




```python
X_train_drop = X_train.drop(columns='proposed_credit_limit')
X_test_drop = X_test.drop(columns='proposed_credit_limit')
```


```python
X_train_drop.columns
```




    Index(['const', 'income', 'name_email_similarity', 'zip_count_4w',
           'credit_risk_score', 'phone_home_valid', 'phone_mobile_valid',
           'has_other_cards', 'source_TELEAPP'],
          dtype='object')




```python
model_logit2 = sm.Logit(y_train, X_train_drop).fit()
label_pred2 = model_logit2.predict(X_test_drop)
label_pred2 = label_pred2.apply(lambda x: 1 if x > 0.5 else 0)
```

    Optimization terminated successfully.
             Current function value: 0.300542
             Iterations 7
    


```python
from sklearn.metrics import recall_score, precision_score, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, label_pred2)}')
print(f'Recall score: {recall_score(y_test, label_pred2)}')
print(f'Precision score: {precision_score(y_test, label_pred2)}')
```

    Accuracy score: 0.8919463087248322
    Recall score: 0.0830860534124629
    Precision score: 0.6829268292682927
    


```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(max_depth=2)
gbc.fit(X_train, y_train)
pred3 = gbc.predict(X_test)
```


```python
from sklearn.metrics import recall_score, precision_score, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, pred3)}')
print(f'Recall score: {recall_score(y_test, pred3)}')
print(f'Precision score: {precision_score(y_test, pred3)}')
```

    Accuracy score: 0.8889261744966444
    Recall score: 0.0771513353115727
    Precision score: 0.5652173913043478
    


```python
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
pred4 = xgb.predict(X_test)
```


```python
from sklearn.metrics import recall_score, precision_score, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, pred4)}')
print(f'Recall score: {recall_score(y_test, pred4)}')
print(f'Precision score: {precision_score(y_test, pred4)}')
```

    Accuracy score: 0.886241610738255
    Recall score: 0.1543026706231454
    Precision score: 0.49056603773584906
    


```python
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define grid
weights = [0.1, 0.25, 0.5, 0.66, 1, 10, 25, 50, 99]
param_grid = dict(scale_pos_weight=weights)

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
```


```python
%%time
grid_result = grid.fit(X_train, y_train)
pred5 = grid.predict(X_test)
```

    CPU times: total: 1.39 s
    Wall time: 3.2 s
    


```python
from sklearn.metrics import recall_score, precision_score, accuracy_score

print(f'Accuracy score: {accuracy_score(y_test, pred5)}')
print(f'Recall score: {recall_score(y_test, pred5)}')
print(f'Precision score: {precision_score(y_test, pred5)}')
```

    Accuracy score: 0.6758389261744966
    Recall score: 0.543026706231454
    Precision score: 0.18391959798994975
    


Ketika tidak puas dengan hasil model performance diatas, yang bisa dilakukan adalah:

1. Tunning model (membuat model baru dengan kombinasi prediktor yang lain)
2. Menambahkan data
3. Melakukan penggeseran threshold
4. Menggunakan metode yang lain

### Asumsi Logistic Regression

Asumsi Logistic Regression :

* **No Multicollinearity**: antar prediktor tidak saling berkorelasi. Untuk melakukan pengecekannya sama seperti dalam linear regression yaitu menggunakan nilai VIF. 

  + apabila ada prediktor yang terindikasi multikolinearity, kita bisa menggunakan salah satu variabel saja atau membuat variabel baru yang men-summary dari kedua variabel tersebut (mean)
  +  dari VIF kita ingin variabel kita memiliki VIF < 10
  
* **Independence of Observations**: antar observasi saling independen & tidak berasal dari pengukuran berulang (repeated measurement).

* **Linearity of Predictor & Log of Odds**: cara interpretasi mengacu pada asumsi ini. untuk variabel numerik, peningkatan 1 nilai akan menaikan log of odds (peluang).

# K-Nearest Neighbour Algorithm

Metode k-NN akan mengkasifikasi data baru dengan membandingkan karakteristik data baru (data test) dengan data yang ada (data train). Kedekatan karakteristik tersebut diukur dengan Euclidean Distance yaitu pengukuran jarak. Kemudian akan dipilih k tetangga terdekat dari data baru tersebut, kemudian ditentukan kelasnya menggunakan majority voting.

### Karakteristik k-NN

- tidak ada asumsi
- dapat memprediksi multiclass
- baik untuk prediktor numerik (karena mengklasifikasikan berdasarkan jarak), tidak baik untuk prediktor kategorik
- robust: performa nya bagus -> error nya kecil
- tidak interpretable

## Data Cleansing

Kita akan menggunakan data yang sama dengan metode sebelumnya, tetapi kali ini kita akan menggunakan data tanpa kategorikal sama sekali.


```python
fraud_enc.head(3)
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>fraud_bool</th>
      <th>source_TELEAPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>3483</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>2849</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>0.370933</td>
      <td>406</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fraud_enc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 14900 entries, 0 to 14904
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   income                 14900 non-null  float64
     1   name_email_similarity  14900 non-null  float64
     2   zip_count_4w           14900 non-null  int64  
     3   credit_risk_score      14900 non-null  float64
     4   phone_home_valid       14900 non-null  int64  
     5   phone_mobile_valid     14900 non-null  int64  
     6   has_other_cards        14900 non-null  int64  
     7   proposed_credit_limit  14900 non-null  float64
     8   fraud_bool             14900 non-null  int64  
     9   source_TELEAPP         14900 non-null  int64  
    dtypes: float64(4), int64(6)
    memory usage: 1.3 MB
    


```python
catbool = ['phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'source_TELEAPP']
```


```python
fraud_knn = fraud_enc.drop(columns=catbool)

fraud_knn.head()
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
      <th>income</th>
      <th>name_email_similarity</th>
      <th>zip_count_4w</th>
      <th>credit_risk_score</th>
      <th>proposed_credit_limit</th>
      <th>fraud_bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>3483</td>
      <td>20.0</td>
      <td>200.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>2849</td>
      <td>3.0</td>
      <td>200.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>0.370933</td>
      <td>406</td>
      <td>50.0</td>
      <td>200.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9</td>
      <td>0.401137</td>
      <td>780</td>
      <td>74.0</td>
      <td>200.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>0.720006</td>
      <td>4527</td>
      <td>136.0</td>
      <td>200.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Cross Validation

Gunakan metode train-test splitting dengan proporsi dan random_state yang sudah kita gunakan pada kasus sebelumnya.


```python
# prediktor
X = fraud_knn.drop(columns='fraud_bool')
# target
y = fraud_knn['fraud_bool']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, # kolom prediktor
                                                   y, # kolom target
                                                   test_size = 0.2, # 80% training and 20% test
                                                   random_state = 10,
                                                   stratify = y)
```


```python
vif = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
pd.Series(data=vif, index = X_train.columns).sort_values(ascending=False)
```




    credit_risk_score        6.586484
    income                   3.779879
    proposed_credit_limit    3.572963
    name_email_similarity    2.967258
    zip_count_4w             2.653914
    dtype: float64



## Data Preprocessing

**Feature Scalling**

🔎 Scaling: menyamaratakan range variable prediktor

Scaling bisa menggunakan **min-max normalization atau z-score standarization**

1.  **Min-max normalization** --> bekerja dengan mentransformasi fitur sehingga nilainya berada dalam rentang 0 hingga 1.

> Formula: $x_{new}=\frac{(x-min(x))}{(max(x)-min(x))}$

- Nilai fitur yang dinormalisasi secara efektif mengomunikasikan seberapa jauh, dalam persentase, nilai asli berada di sepanjang rentang semua nilai fitur *x*.
- digunakan ketika tau angka pasti min dan max nya. misalnya nilai ujian matematika pasti nilai min-max nya 0 - 100.

2. **z-score standardization** mengurangi fitur x dengan rata-rata dan dibagi dengan standar deviasi dari fitur.

> Formula: $x_{new}=\frac{(x-\bar x)}{std(x)}$ 

- digunakan ketika tidak diketahui angka min dan max pastinya. misalnya temperature bisa dari kisaran -inf s.d +inf



🔻 Menormalisasi menjadi z-score:


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```


```python
# subset kolom numerik
cols = X_train.columns
```


```python
# transform
scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)
```

Data prediktor discaling menggunakan z-score standarization. Data test juga harus discaling menggunakan parameter dari data train (karena menganggap data test adalah unseen data).

Untuk data test:
- diperlakukan sebagai data unseen
- ketika ingin discaling prediktornya harus menggunakan informasi mean dan sd dari data train

## Training Model

Untuk membuat model K-NN, kita akan memanfaatkan library `KNeighborsClassifier` yang berada pada `sklearn.neighbors`. Tetapi sebelumnya kita harus menentukan dulu jumlah tetangga yang harus kita perhitungkan.

### Choosing an appropriate *k*

Berikut adalah intuisi dasar pemilihan nilai K optimal:

- Jangan terlalu besar: pemilihan kelas hanya berdasarkan kelas yang dominan dan mengabaikan data kecil yang ternyata penting.
- Jangan terlalu kecil: rentan mengklasifikasikan data baru ke kelas outlier.
- Penentuan k optimum biasanya menggunakan akar dari jumlah data train kita: `sqrt(nrow(data))`


```python
math.sqrt(fraud_clean.shape[0])
```




    122.06555615733703



k-NN akan menghitung jumlah kelas pada tetangga terdekat suatu data dan kelas terbanyak inilah akan menjadi hasil klasifikasi data kita. Bila hasil majority voting seri, maka kelas akan dipilih secara random. Maka dari itu, untuk meminimalisir seri ketika majority voting:

+ k harus ganjil bila jumlah kelas target genap
+ k harus genap bila jumlah kelas target ganjil
+ k tidak boleh angka kelipatan jumlah kelas target

Nilai hasil perhitungan di atas perlu dibulatkan berdasarkan arahan ini. Mari kita gunakan nilainya pada pembuatan model k-NN.


```python
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=123, weights='distance')
model_knn.fit(X_train_scale, y_train)
```




<style>#sk-container-id-7 {color: black;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=123, weights=&#x27;distance&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=123, weights=&#x27;distance&#x27;)</pre></div></div></div></div></div>



## Model prediction
Sama seperti model sebelumnya, model yang sudah dipersiapkan untuk melakukan prediksi pada data test yang sudah dipersipakan dengan menggunakan fungsi `predict()`.


```python
knn_pred = model_knn.predict(X_test_scale)
```

### Model Evaluation


```python
# Hasil evaluasi KNN
print(f'Accuracy score: {accuracy_score(y_test, knn_pred)}')
print(f'Recall score: {recall_score(y_test, knn_pred)}')
print(f'Precision score: {precision_score(y_test, knn_pred)}')
```

    Accuracy score: 0.8895973154362417
    Recall score: 0.04154302670623145
    Precision score: 0.7
    

# Logistic Regression & k-NN Comparation

![](assets/Karakter.png)

# Glossary & Additional Information

<details>
    <summary>Click once on <font color="red"><b>this text</b></font> to hide/unhide additional information in Summary</summary>
    
**[Optional] Other Information in Summary**

1. Tabel 1, sisi kiri menyimpan informasi dasar dari model

| Variable | Description | 
| :--- | :--- | 
| Dep. Variable  | Dependent variable atau target variabel (Y) | 
| Model | Model logistic regression| 
| Method | Metode yang digunakan untuk membuat model logistic regression: Maximum Likelihood Estimator | 
| No. Observations| Jumlah observasi (baris) yang digunakan ketika membuat model regresi linier | 
| DF Residuals | Degrees of freedom error/residual (jumlah observasi/baris - parameter) |
| DF Model  | Degrees of freedom model (jumlah prediktor) | 
| Covariance type | tipe nonrobust berarti tidak ada penghapusan data untuk menghitung kovarian antar fitur. Kovarian menunjukkan bagaimana dua variabel bergerak terhadap satu sama lain (+ atau -, tidak menghitung kekuatannya) | 
    
    
<br> 
<br>  

2. Tabel 1, sisi kanan menyimpan informasi kebaikan model

| Variable | Description | 
| :--- | :--- | 
| Pseudo R-squared  | Goodness of fit. Rasio dari log-likelihood null model dibandingkan dengan full model. | 
| Log-likelihood  | [Conditional probability](https://en.wikipedia.org/wiki/Conditional_probability) bahwa data yang digunakan cocok/fit dengan model. Semakin besar, semakin fit model terhadap datanya. range -inf - +inf  |
| LL-Null | nilai dari log-likelihood model tanpa prediktor (intercept saja) | 
| LLR p-value   | nilai p -value dari apakah model yang kita buat lebih baik daripada model tanpa prediktor (intercept saja)| 


<br> 
<br> 

3. Tabel 2 menyimpan informasi dari koefisien regresi

| Variable | Description | 
| :--- | :--- | 
| **coef**   | Estimasi koefisien | 
| std err | Estimasi selisih nilai sampel terhadap populasi | 
| z | Statistik hitung dari z-test (uji parsial) | 
|**P > \|z\|**  | P-value dari z-test | 
| [95.0% Conf. Interval]  | Confidence Interval (CI) 95%. |

    
In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable.
    
<br> 
<br> 
    
    
    
Recall from Practical Statistic:
    
* $\alpha$:
  + tingkat signifikansi / tingkat error
  + umumnya 0.05
* $1-\alpha$: tingkat kepercayaan (misal alpha 0.05, maka kita akan percaya terhadap hasil analisis sebesar 95%)
* $p-value$:
  + akan dibandingkan dengan alpha untuk untuk mengambil keputusan
  + peluang data sampel berada pada bagian sangat ekstrim/berbeda signifikan dengan keadaan normal.
  
Pengambilan keputusan:

* Jika $p-value$ < $\alpha$, maka tolak $H_0$ / terima $H_1$
* Jika $p-value$ > $\alpha$, maka gagal tolak / terima $H_0$
    
</details>
