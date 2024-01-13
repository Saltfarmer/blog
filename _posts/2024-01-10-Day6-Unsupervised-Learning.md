---
title: "Day 6 Algorit.ma : Unsupervised Learning"
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
  - Clustering
  - PCA

---

Day 6, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Inclass: Unsupervised Learning**
- Durasi: 7 hours
- _Last Updated_: Desember 2023

___

- Disusun dan dikurasi oleh tim produk dan instruktur [Algoritma Data Science School](https://algorit.ma).


```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pyod.models.lof import LOF

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
# Set notebook mode to work in offline
pyo.init_notebook_mode()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from helper import *
```

# Introduction


Machine learning berfokus pada prediksi berdasarkan properti/fitur yang dipelajari dari data training. Beberapa tipe machine learning yaitu:


**Supervised Learning**: 

* memiliki target variable. 
* untuk pembuatan model prediksi $(y \sim x)$
* ada ground truth (label aktual) sehingga ada evaluasi model

**Unsupervised Learning**: 

* tidak memiliki target variable. 
* untuk mencari pola dalam data sehingga menghasilkan informasi yang berguna/dapat diolah lebih lanjut. umumnya dipakai untuk tahap explanatory data analysis (EDA)/data pre-processing.
* tidak ada ground truth sehingga sulit mengevaluasi model 

# Dimensionality Reduction

Tujuan dimensionality reduction adalah untuk **mereduksi banyaknya variabel (dimensi/fitur)** pada data dengan tetap **mempertahankan informasi sebanyak mungkin**. Dimensionality reduction dapat mengatasi masalah high-dimensional data. Kesulitan yang dihadapi pada high-dimensional data:

- Memerlukan waktu dan komputasi yang besar dalam melakukan pemodelan
- Melakukan visualisasi lebih dari tiga dimensi
- Menyulitkan pengolahan data (feature selection)

Note:

* **Dimensi**: kolom, semakin banyak kolom maka dimensi semakin tinggi.
* **Informasi**: [variance](#Glossary), semakin tinggi variance maka informasinya semakin banyak.

## Refresher on Variance

Berikut adalah data gaji perusahaan A dan B dalam **satuan juta rupiah**. 

Pertanyaan: Tanpa menghitung nilai [variance](#Glossary), perusahaan mana yang memiliki gaji lebih bervariasi?


```python
# coba bandingkan variansi kedua data ini:
A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
B = [4, 5, 5, 6, 6, 4, 6, 5, 4, 4]

print(np.var(A))
print(np.var(B))
```

    8.25
    0.6900000000000001
    

<div class="alert alert-block alert-warning">
<b>⚠️ Note:</b> variansi  bergantung pada skala variable 
</div>

Ada pula data gaji perusahaan C dalam **satuan dollar**. Untuk mempermudah, asumsi 1 dollar = 10000 rupiah


```python
C = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
np.var(C)
```




    82500.0



Apakah bisa dibilang gaji di perusahaan C lebih bervariasi daripada A?

> Ans: Ya betul sekali

## Motivation Example: Image Compression

Pada data gambar, setiap kotak pixel akan menjadi 1 kolom. Foto berukuran 40x40 pixel memiliki 1600 kolom (dimensi). Sekarang mari renungkan, berapa spesifikasi kamera handphone anda? Berapa besar dimensi data yang dihasilkan kamera Anda?

Image compression adalah salah satu contoh nyata dimensionality reduction menggunakan data gambar yang  dan tetap menghasilkan gambar yang serupa (informasi inti tidak hilang), sehingga data gambar lebih mudah diproses. Salah satu algoritma yang dapat digunakan untuk dimensionality reduction adalah **Principal Component Analysis (PCA)**.


<img src="assets/cat_pca.png" width="700">
    
<a href="https://www.tandfonline.com/doi/pdf/10.1080/09500340.2016.1270881" style="margin:auto; display:block;" class="button large hpbottom">alternatives on lenna image</a>

✅ **Knowledge Check:**

Dalam suatu gambar apa yang dimaksud dengan dimensi dan informasi?

- dimensi : Banyak kolom dari pixelnya(dan layer kalo berwarna)
- informasi: Variance dari grayscale value


Apakah nilai dari variansi dipengaruhi oleh skala dari nilai itu sendiri? jelaskan!

> Ans: Betul, karena variance yang tinggi dapat mempengaruhi skala 


## Principle Component Analysis

### Konsep

Ide dasar dari PCA adalah untuk membuat sumbu (axis) baru yang dapat menangkap informasi sebesar mungkin. Sumbu baru ini adalah yang dinamakan sebagai Principal Component (PC). Untuk melakukan dimensionality reduction, kita akan memilih beberapa PC untuk dapat merangkum informasi yang dibutuhkan

<img src="assets/ul10.JPG" width="700">

**Figure A (Sebelum PCA):**

- Sumbu/dimensi: X1 dan X2
- Variance data dijelaskan oleh X1 dan X2
- Dibuatlah sumbu baru untuk menangkap informasi X1 dan X2, yang dinamakan PC1 dan PC2

**Figure B (Setelah PCA):**

- Sumbu baru: PC1 dan PC2
- PC1 menangkap variance lebih banyak daripada PC2
- Misalkan PC1 menangkap 90% variance, dan sisanya ditangkap oleh PC2 yaitu 10%

💡 **Notes**:

- Membuat sumbu baru yang disebut dengan PC yang bertujuan untuk merangkum sebanyak mungkin informasi data
- Banyaknya jumlah PC sama dengan jumlah dimensi dari data
- PC1 pasti menangkap variance paling besar dibandingkan dengan PC 2, dan seterusnya
- Antara PC1 dan PC2 saling tegak lurus, artinya tidak saling berkorelasi
- Metode PCA akan cocok untuk data numerik yang saling berkorelasi

**✅ Knowledge Check:**

<img src="assets/knowledge check.png" width="500">

1.  Dari Gambar diatas mana data yang cocok dilakukan PCA?

-   [ ] Sale Price of Vehicles
-   [x] Blind Tasting
-   [ ] Logistic Machinery

2.  Bila terdapat 3 PC, PC ke-berapa yang merangkum variansi (informasi) paling besar?

-   [x] PC1
-   [ ] PC2
-   [ ] PC3

3.  Dalam PCA jumlah PC yang dihasilkan sebanyak....

-   [x] Jumlah variabel yang digunakan
-   [ ] Setengah dari jumlah variabel yang digunakan
-   [ ] Ditentukan oleh user 

4.  PC1 dibentuk oleh variabel pertama dan PC4 dibentuk oleh variabel ke empat

-   [x] Salah
-   [ ] Benar

### Math Behind PCA [optional]

<div class="alert alert-block alert-success">
<b>&#128250; Rekomendasi Video:</b> <a href="https://www.youtube.com/watch?v=PFDu9oVAE-g" class="button large hpbottom">3Blue1Brown: Eigenvectors and eigenvalues</a>
</div>

Untuk membentuk PC dibutuhkan **eigen values** & **eigen vector**. Secara manual, eigen values dan eigen vector didapatkan dari operasi matrix.

Teori matrix:

* skalar: nilai yang memiliki magnitude/besaran
* vektor: nilai yang memiliki besaran dan arah (umum digambarkan dalam suatu koordinat)
* matrix: kumpulan nilai/bentukan data dalam baris dan kolom


**Eigen- dari suatu Matrix**

Untuk setiap matrix $A$, terdapat **vektor spesial (eigen vector)** yang jika dikalikan dengan matrixnya, hasilnya akan sama dengan vektor tersebut dikalikan suatu **skalar (eigen value)**. Sehingga didapatkan rumus:

$$Ax = \lambda x$$

dengan $x$ adalah eigen vector dan $\lambda$ adalah eigen value dari matrix $A$.

Contoh:

Pada perhitungan matrix di bawah, salah satu eigen vector dari matrix 
$\begin{bmatrix}
2 & 3\\ 
2 & 1
\end{bmatrix}$
adalah 
$\begin{bmatrix}
3\\ 
2
\end{bmatrix}$
dengan eigen value sebesar 4.


$$
\left(\begin{array}{cc} 
2 & 3\\ 
2 & 1 
\end{array}\right)
\left(\begin{array}{cc} 
3\\ 
2
\end{array}\right)
=
\left(\begin{array}{cc} 
12\\ 
8
\end{array}\right)
=4
\left(\begin{array}{cc} 
3\\ 
2
\end{array}\right)
$$

Teori eigen dipakai untuk menentukan PC dan nilai-nilai pada PC.

**Penerapan Eigen dalam PCA:**

**Matrix [covariance](#Glossary)** adalah matrix yang dapat merangkum informasi (variance) dari data. Kita menggunakan matrix covariance untuk mendapatkan eigen vector dan eigen value dari matrix tersebut, dengan:

* **eigen vector**: arah sumbu tiap PC, yang menjadi formula untuk mentransformasi data awal ke PC baru. 
* **eigen value**: variansi yang ditangkap oleh setiap PC.
* tiap PC memiliki 1 eigen value & 1 eigen vector.
* alur: matrix covariance $\rightarrow$ eigen value $\rightarrow$ eigen vector $\rightarrow$ nilai di tiap PC

Eigen vector akan menjadi formula untuk kalkulasi nilai di setiap PC. Contohnya, untuk data yang terdiri dari 2 variabel, bila diketahui eigen vector dari PC1 adalah:

$$x_{PC1}= \left[\begin{array}{cc}a_1\\a_2\end{array}\right]$$

Maka formula untuk menghitung nilai pada PC1 (untuk tiap barisnya) adalah:

$$PC1= a_1X_1 + a_2X_2$$

Keterangan:

* $x_{PC1}$ : eigen vector PC1 dari matrix covariance
* $a_1$, $a_2$ : konstanta dari eigen vector
* $PC1$ : nilai di PC1
* $X_1$, $X_2$ : nilai variabel X1 dan X2 di data awal

**Contoh menghitung eigen value dan eigen vector dari sebuah data**


```python
# membuat data dummy
dummy = pd.DataFrame(np.random.rand(4, 2), #generate random value dengan 4 baris dan 2 kolom
                 columns=list('XY')) #nama tiap kolom
dummy
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
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.170823</td>
      <td>0.897426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.577542</td>
      <td>0.981163</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.685710</td>
      <td>0.688500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.735711</td>
      <td>0.914354</td>
    </tr>
  </tbody>
</table>
</div>



Mencari nilai [covariance](#Glossary) pada dataframe dummy:


```python
matrix_cov = dummy.cov()
matrix_cov
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
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>0.085857</td>
      <td>-0.008870</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>-0.008870</td>
      <td>0.033434</td>
    </tr>
  </tbody>
</table>
</div>



Mencari nilai dan vector eigen dengan fungsi [eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) dari library [numpy](https://numpy.org/doc/stable/index.html) 


```python
eig_vals,eig_vecs = eig(matrix_cov.T) 
print('E-value: \n', eig_vals) #\n untuk newline (enter ke bawah)
print('E-vector: \n', eig_vecs)
```

    E-value: 
     [0.08731741 0.03197376]
    E-vector: 
     [[ 0.98672189  0.16241896]
     [-0.16241896  0.98672189]]
    

**Note**: hasil fungsi eig() tidak berurutan berdasarkan nilainya. Eigenvalues dari PC1 adalah nilai terbesar, dilanjutkan PC2 dengan nilai kedua terbesar dan seterusnya.    

* `E-value:`: Eigen value untuk tiap PC, besar variansi yang dapat ditangkap oleh tiap PC. Eigen value tertinggi adalah milik PC1, kedua tertinggi milik PC2, dan seterusnya. 

* `E-vector`: Eigen vector untuk tiap PC. Kolom `eig_vecs[:,i]` adalah vektor eigen yang sesuai dengan nilai eigen `eig_vals[i]`


### PCA Workflow

#### Business Question: Dimensionality Reduction for Fraud Bank Account dataset

Kita akan kembali menggunakan data `fraud_dataset.csv` yang sudah digunakan pada pembelajaran sebelumnya. Perbedaannya adalah kita akan menggunakan keseluruhan kolom pada data ini dan hanya akan membuang kolom yang kemaren kita jadikan sebagai target.


```python
fraud = pd.read_csv('data_input/fraud_dataset.csv')
fraud.drop(columns=['fraud_bool'], inplace=True)
fraud.head()
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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>intended_balcon_amount</th>
      <th>payment_type</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>...</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>source</th>
      <th>session_length_in_minutes</th>
      <th>device_os</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>48.0</td>
      <td>30</td>
      <td>0.006760</td>
      <td>-1.074674</td>
      <td>AB</td>
      <td>3483</td>
      <td>5316.092932</td>
      <td>4527.956243</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>5.191773</td>
      <td>windows</td>
      <td>1</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>61.0</td>
      <td>20</td>
      <td>0.020642</td>
      <td>-1.043444</td>
      <td>AD</td>
      <td>2849</td>
      <td>8153.671429</td>
      <td>7524.130278</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>3.901673</td>
      <td>windows</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>0.370933</td>
      <td>70.0</td>
      <td>30</td>
      <td>6.400793</td>
      <td>48.520199</td>
      <td>AA</td>
      <td>406</td>
      <td>7648.434993</td>
      <td>6366.061338</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>3.777191</td>
      <td>linux</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9</td>
      <td>0.401137</td>
      <td>64.0</td>
      <td>30</td>
      <td>0.004651</td>
      <td>-0.394588</td>
      <td>AC</td>
      <td>780</td>
      <td>6459.224179</td>
      <td>3394.524379</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>3.176269</td>
      <td>linux</td>
      <td>1</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>0.720006</td>
      <td>11.0</td>
      <td>20</td>
      <td>0.032629</td>
      <td>-0.487785</td>
      <td>AC</td>
      <td>4527</td>
      <td>7852.258962</td>
      <td>5177.826213</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>14.626874</td>
      <td>linux</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



**Penjelasan Dataset**

Berikut adalah penjelasan setiap kolom yang terdapat pada _dataset_:

- `income` (numeric): _Annual income of the applicant (in decile form). Ranges between [0.1, 0.9]._
- `name_email_similarity` (numeric): _Metric of similarity between email and applicant’s name. Higher values represent higher similarity. Ranges between [0, 1]._
- `current_address_months_count` (numeric): _Months in currently registered address of the applicant. Ranges between [−1, 429] months (-1 is a missing value)._
- `customer_age` (numeric): _Applicant’s age in years, rounded to the decade. Ranges between [10, 90] years._
- `days_since_request` (numeric): _Number of days passed since application was done. Ranges between [0, 79] days._
- `intended_balcon_amount` (numeric): _Initial transferred amount for application. Ranges between [−16, 114] (negatives are missing values)._
- `payment_type` (categorical): _Credit payment plan type. 5 possible (annonymized) values._
- `zip_count_4w` (numeric): _Number of applications within same zip code in last 4 weeks. Ranges between [1, 6830]._
- `velocity_6h` (numeric): _Velocity of total applications made in last 6 hours i.e., average number of applications per hour in the last 6 hours. Ranges between [−175, 16818]._
- `velocity_24h` (numeric): _Velocity of total applications made in last 24 hours i.e., average number of applications per hour in the last 24 hours. Ranges between [1297, 9586]_
- `velocity_4w` (numeric): _Velocity of total applications made in last 4 weeks, i.e., average number of applications per hour in the last 4 weeks. Ranges between [2825, 7020]._
- `bank_branch_count_8w` (numeric): _Number of total applications in the selected bank branch in last 8 weeks. Ranges between [0, 2404]._
- `date_of_birth_distinct_emails_4w` (numeric): _Number of emails for applicants with same date of birth in last 4 weeks. Ranges between [0, 39]._
- `employment_status` (categorical): _Employment status of the applicant. 7 possible (annonymized) values._
- `credit_risk_score` (numeric): _Internal score of application risk. Ranges between [−191, 389]._
- `email_is_free` (binary): _Domain of application email (either free or paid)._
- `housing_status` (categorical): _Current residential status for applicant. 7 possible (annonymized) values._
- `phone_home_valid` (binary): _Validity of provided home phone._
- `phone_mobile_valid` (binary): _Validity of provided mobile phone._
- `has_other_cards` (binary): _If applicant has other cards from the same banking company. _
- `proposed_credit_limit` (numeric): _Applicant’s proposed credit limit. Ranges between [200, 2000]._
- `foreign_request` (binary): _If origin country of request is different from bank’s country._
- `source` (categorical): _Online source of application. Either browser (INTERNET) or app (TELEAPP)._
- `session_length_in_minutes` (numeric): _Length of user session in banking website in minutes. Ranges between [−1, 107] minutes (-1 is a missing value)._
- `device_os` (categorical): _Operative system of device that made request. Possible values are: Windows, macOS, Linux, X11, or other._
- `keep_alive_session` (binary): _User option on session logout._
- `device_distinct_emails` (numeric): _Number of distinct emails in banking website from the used device in last 8 weeks. Ranges between [−1, 2] emails (-1 is a missing value)._
- `month` (numeric): _Month where the application was made. Ranges between [0, 7]._
- `fraud_bool` (binary): _If the application is fraudulent or not._


```python
fraud.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14905 entries, 0 to 14904
    Data columns (total 28 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   income                            14905 non-null  float64
     1   name_email_similarity             14905 non-null  float64
     2   current_address_months_count      14905 non-null  float64
     3   customer_age                      14905 non-null  int64  
     4   days_since_request                14905 non-null  float64
     5   intended_balcon_amount            14905 non-null  float64
     6   payment_type                      14905 non-null  object 
     7   zip_count_4w                      14905 non-null  int64  
     8   velocity_6h                       14905 non-null  float64
     9   velocity_24h                      14905 non-null  float64
     10  velocity_4w                       14905 non-null  float64
     11  bank_branch_count_8w              14905 non-null  int64  
     12  date_of_birth_distinct_emails_4w  14905 non-null  int64  
     13  employment_status                 14905 non-null  object 
     14  credit_risk_score                 14905 non-null  float64
     15  email_is_free                     14905 non-null  int64  
     16  housing_status                    14905 non-null  object 
     17  phone_home_valid                  14905 non-null  int64  
     18  phone_mobile_valid                14905 non-null  int64  
     19  has_other_cards                   14905 non-null  int64  
     20  proposed_credit_limit             14905 non-null  float64
     21  foreign_request                   14905 non-null  int64  
     22  source                            14905 non-null  object 
     23  session_length_in_minutes         14905 non-null  float64
     24  device_os                         14905 non-null  object 
     25  keep_alive_session                14905 non-null  int64  
     26  device_distinct_emails_8w         14905 non-null  float64
     27  month                             14905 non-null  int64  
    dtypes: float64(12), int64(11), object(5)
    memory usage: 3.2+ MB
    


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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>1.490500e+04</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>14905.000000</td>
      <td>...</td>
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
      <td>88.974975</td>
      <td>34.377055</td>
      <td>1.044408e+00</td>
      <td>7.986892</td>
      <td>1571.105736</td>
      <td>5644.961419</td>
      <td>4764.243186</td>
      <td>4851.361779</td>
      <td>...</td>
      <td>0.541899</td>
      <td>0.400671</td>
      <td>0.883126</td>
      <td>0.213485</td>
      <td>551.910768</td>
      <td>0.028782</td>
      <td>7.701999</td>
      <td>0.559074</td>
      <td>1.029520</td>
      <td>3.293660</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.291264</td>
      <td>0.292755</td>
      <td>88.451892</td>
      <td>12.375090</td>
      <td>5.654084e+00</td>
      <td>19.702913</td>
      <td>998.577819</td>
      <td>3015.663715</td>
      <td>1486.594023</td>
      <td>923.966514</td>
      <td>...</td>
      <td>0.498258</td>
      <td>0.490051</td>
      <td>0.321280</td>
      <td>0.409781</td>
      <td>516.560244</td>
      <td>0.167200</td>
      <td>8.329340</td>
      <td>0.496515</td>
      <td>0.197443</td>
      <td>2.213049</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>0.000093</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>9.352969e-07</td>
      <td>-12.537085</td>
      <td>36.000000</td>
      <td>45.106142</td>
      <td>1328.410255</td>
      <td>2995.300345</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>190.000000</td>
      <td>0.000000</td>
      <td>0.039414</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.300000</td>
      <td>0.206239</td>
      <td>23.000000</td>
      <td>20.000000</td>
      <td>7.179709e-03</td>
      <td>-1.173150</td>
      <td>893.000000</td>
      <td>3402.021768</td>
      <td>3574.620499</td>
      <td>4261.751108</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>200.000000</td>
      <td>0.000000</td>
      <td>3.164021</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.600000</td>
      <td>0.472416</td>
      <td>55.000000</td>
      <td>30.000000</td>
      <td>1.498915e-02</td>
      <td>-0.834826</td>
      <td>1267.000000</td>
      <td>5329.868693</td>
      <td>4743.172402</td>
      <td>4908.851274</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>200.000000</td>
      <td>0.000000</td>
      <td>5.144863</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.800000</td>
      <td>0.748003</td>
      <td>132.000000</td>
      <td>40.000000</td>
      <td>2.610747e-02</td>
      <td>-0.204896</td>
      <td>1941.000000</td>
      <td>7678.860181</td>
      <td>5751.489671</td>
      <td>5485.543277</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1000.000000</td>
      <td>0.000000</td>
      <td>8.902307</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.900000</td>
      <td>0.999997</td>
      <td>406.000000</td>
      <td>90.000000</td>
      <td>7.582081e+01</td>
      <td>111.697355</td>
      <td>6349.000000</td>
      <td>16264.947756</td>
      <td>9341.329938</td>
      <td>6940.302005</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2100.000000</td>
      <td>1.000000</td>
      <td>73.909623</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>




```python
fraud_clean = fraud.drop(columns='intended_balcon_amount')
fraud_clean = fraud_clean[(fraud_clean['proposed_credit_limit'] >= 200) & (fraud_clean['proposed_credit_limit'] <= 2000)]
```

Pilih data yang hanya bertipe numeric :


```python
cols = fraud_clean.select_dtypes("number").columns
fraud_num = fraud_clean[cols]
fraud_num.sample(3)
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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>bank_branch_count_8w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9044</th>
      <td>0.4</td>
      <td>0.211194</td>
      <td>93.0</td>
      <td>40</td>
      <td>0.000542</td>
      <td>1900</td>
      <td>11286.768265</td>
      <td>7114.259835</td>
      <td>6790.848957</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>2.550744</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3438</th>
      <td>0.9</td>
      <td>0.177036</td>
      <td>61.0</td>
      <td>50</td>
      <td>0.006538</td>
      <td>1440</td>
      <td>4795.810553</td>
      <td>6281.811432</td>
      <td>5072.699566</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>500.0</td>
      <td>0</td>
      <td>13.379567</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10703</th>
      <td>0.7</td>
      <td>0.091951</td>
      <td>11.0</td>
      <td>20</td>
      <td>0.003008</td>
      <td>1970</td>
      <td>9637.374296</td>
      <td>5737.460384</td>
      <td>5114.294296</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>2.772705</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>



Melihat nilai covariance pada dataframe `fraud_num` :


```python
# covariance
fraud_num.cov()
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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>bank_branch_count_8w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>income</th>
      <td>0.084830</td>
      <td>-0.004591</td>
      <td>-0.612779</td>
      <td>0.510647</td>
      <td>-0.003152</td>
      <td>-22.873746</td>
      <td>-9.007355e+01</td>
      <td>-5.062088e+01</td>
      <td>-3.105886e+01</td>
      <td>1.463368</td>
      <td>...</td>
      <td>-0.001812</td>
      <td>0.000148</td>
      <td>0.000961</td>
      <td>0.007691</td>
      <td>18.224996</td>
      <td>0.000759</td>
      <td>-0.116759</td>
      <td>-0.009207</td>
      <td>-0.000441</td>
      <td>0.079395</td>
    </tr>
    <tr>
      <th>name_email_similarity</th>
      <td>-0.004591</td>
      <td>0.085705</td>
      <td>0.593454</td>
      <td>-0.276976</td>
      <td>-0.008924</td>
      <td>5.845287</td>
      <td>3.198269e+01</td>
      <td>1.878802e+01</td>
      <td>1.015023e+01</td>
      <td>0.444386</td>
      <td>...</td>
      <td>-0.012035</td>
      <td>0.001591</td>
      <td>0.001704</td>
      <td>0.001124</td>
      <td>7.589780</td>
      <td>-0.001033</td>
      <td>0.016699</td>
      <td>0.006064</td>
      <td>-0.002295</td>
      <td>-0.028691</td>
    </tr>
    <tr>
      <th>current_address_months_count</th>
      <td>-0.612779</td>
      <td>0.593454</td>
      <td>7824.656266</td>
      <td>154.588556</td>
      <td>-30.908551</td>
      <td>3918.276204</td>
      <td>7.973437e+03</td>
      <td>2.808161e+03</td>
      <td>2.017757e+03</td>
      <td>2419.540599</td>
      <td>...</td>
      <td>-2.937186</td>
      <td>5.020150</td>
      <td>-2.875765</td>
      <td>1.630877</td>
      <td>6921.295187</td>
      <td>-0.187393</td>
      <td>-12.739258</td>
      <td>-2.733750</td>
      <td>0.235010</td>
      <td>-4.112885</td>
    </tr>
    <tr>
      <th>customer_age</th>
      <td>0.510647</td>
      <td>-0.276976</td>
      <td>154.588556</td>
      <td>153.145865</td>
      <td>-2.615292</td>
      <td>-165.109768</td>
      <td>-9.137572e+02</td>
      <td>-1.304647e+02</td>
      <td>-7.711674e+00</td>
      <td>258.152863</td>
      <td>...</td>
      <td>0.065014</td>
      <td>1.077889</td>
      <td>-0.610440</td>
      <td>0.380307</td>
      <td>1089.333498</td>
      <td>0.000818</td>
      <td>4.356485</td>
      <td>-0.317894</td>
      <td>0.148603</td>
      <td>0.016601</td>
    </tr>
    <tr>
      <th>days_since_request</th>
      <td>-0.003152</td>
      <td>-0.008924</td>
      <td>-30.908551</td>
      <td>-2.615292</td>
      <td>31.966929</td>
      <td>52.986525</td>
      <td>4.321133e+02</td>
      <td>1.814850e+02</td>
      <td>1.387599e+02</td>
      <td>-53.041654</td>
      <td>...</td>
      <td>0.014440</td>
      <td>-0.143337</td>
      <td>0.051130</td>
      <td>-0.117430</td>
      <td>-189.537392</td>
      <td>0.002647</td>
      <td>2.033355</td>
      <td>0.057482</td>
      <td>0.024413</td>
      <td>-0.216861</td>
    </tr>
    <tr>
      <th>zip_count_4w</th>
      <td>-22.873746</td>
      <td>5.845287</td>
      <td>3918.276204</td>
      <td>-165.109768</td>
      <td>52.986525</td>
      <td>997047.382815</td>
      <td>3.959082e+05</td>
      <td>2.858011e+05</td>
      <td>2.694707e+05</td>
      <td>6094.917630</td>
      <td>...</td>
      <td>20.263025</td>
      <td>-26.639721</td>
      <td>4.915758</td>
      <td>-9.881681</td>
      <td>-8083.186676</td>
      <td>2.453612</td>
      <td>359.426268</td>
      <td>3.917571</td>
      <td>7.000640</td>
      <td>-600.642652</td>
    </tr>
    <tr>
      <th>velocity_6h</th>
      <td>-90.073551</td>
      <td>31.982689</td>
      <td>7973.437476</td>
      <td>-913.757229</td>
      <td>432.113320</td>
      <td>395908.153015</td>
      <td>9.096535e+06</td>
      <td>2.094797e+06</td>
      <td>1.140525e+06</td>
      <td>26029.215640</td>
      <td>...</td>
      <td>37.300862</td>
      <td>-37.627801</td>
      <td>-14.536191</td>
      <td>15.605739</td>
      <td>-40783.431792</td>
      <td>-5.237982</td>
      <td>1323.806011</td>
      <td>25.707221</td>
      <td>15.017745</td>
      <td>-2811.792603</td>
    </tr>
    <tr>
      <th>velocity_24h</th>
      <td>-50.620879</td>
      <td>18.788020</td>
      <td>2808.161069</td>
      <td>-130.464654</td>
      <td>181.484971</td>
      <td>285801.133198</td>
      <td>2.094797e+06</td>
      <td>2.209447e+06</td>
      <td>7.375738e+05</td>
      <td>26400.173477</td>
      <td>...</td>
      <td>14.965224</td>
      <td>-35.386047</td>
      <td>-12.725914</td>
      <td>-7.266532</td>
      <td>5069.770364</td>
      <td>4.123817</td>
      <td>864.137810</td>
      <td>5.405526</td>
      <td>8.775432</td>
      <td>-1818.947260</td>
    </tr>
    <tr>
      <th>velocity_4w</th>
      <td>-31.058864</td>
      <td>10.150233</td>
      <td>2017.756981</td>
      <td>-7.711674</td>
      <td>138.759911</td>
      <td>269470.666041</td>
      <td>1.140525e+06</td>
      <td>7.375738e+05</td>
      <td>8.534964e+05</td>
      <td>16891.001751</td>
      <td>...</td>
      <td>25.337173</td>
      <td>-23.035219</td>
      <td>-9.038828</td>
      <td>-13.015738</td>
      <td>15760.288993</td>
      <td>2.106544</td>
      <td>663.541738</td>
      <td>16.049966</td>
      <td>9.659638</td>
      <td>-1716.809159</td>
    </tr>
    <tr>
      <th>bank_branch_count_8w</th>
      <td>1.463368</td>
      <td>0.444386</td>
      <td>2419.540599</td>
      <td>258.152863</td>
      <td>-53.041654</td>
      <td>6094.917630</td>
      <td>2.602922e+04</td>
      <td>2.640017e+04</td>
      <td>1.689100e+04</td>
      <td>216745.571966</td>
      <td>...</td>
      <td>-2.142896</td>
      <td>13.835538</td>
      <td>-2.383647</td>
      <td>7.016620</td>
      <td>-392.652565</td>
      <td>-0.142042</td>
      <td>37.278437</td>
      <td>1.434909</td>
      <td>0.331983</td>
      <td>-37.211440</td>
    </tr>
    <tr>
      <th>date_of_birth_distinct_emails_4w</th>
      <td>-0.117235</td>
      <td>0.056748</td>
      <td>-78.551471</td>
      <td>-27.086848</td>
      <td>0.297179</td>
      <td>598.673730</td>
      <td>1.813792e+03</td>
      <td>1.165160e+03</td>
      <td>1.087195e+03</td>
      <td>-70.498481</td>
      <td>...</td>
      <td>0.068809</td>
      <td>-0.360390</td>
      <td>0.182675</td>
      <td>-0.058403</td>
      <td>-182.632372</td>
      <td>0.018900</td>
      <td>-1.417666</td>
      <td>0.159437</td>
      <td>-0.040432</td>
      <td>-2.643468</td>
    </tr>
    <tr>
      <th>credit_risk_score</th>
      <td>4.080329</td>
      <td>0.600302</td>
      <td>728.922657</td>
      <td>173.912361</td>
      <td>-33.283855</td>
      <td>-6990.585254</td>
      <td>-3.193542e+04</td>
      <td>-1.706834e+04</td>
      <td>-1.131608e+04</td>
      <td>-856.456390</td>
      <td>...</td>
      <td>-0.521268</td>
      <td>-0.377010</td>
      <td>-0.310680</td>
      <td>3.365784</td>
      <td>23870.161509</td>
      <td>0.489783</td>
      <td>-23.950210</td>
      <td>-1.874476</td>
      <td>-0.554730</td>
      <td>27.454425</td>
    </tr>
    <tr>
      <th>email_is_free</th>
      <td>-0.001812</td>
      <td>-0.012035</td>
      <td>-2.937186</td>
      <td>0.065014</td>
      <td>0.014440</td>
      <td>20.263025</td>
      <td>3.730086e+01</td>
      <td>1.496522e+01</td>
      <td>2.533717e+01</td>
      <td>-2.142896</td>
      <td>...</td>
      <td>0.248263</td>
      <td>-0.005578</td>
      <td>0.005232</td>
      <td>-0.005552</td>
      <td>0.929047</td>
      <td>0.001177</td>
      <td>0.146011</td>
      <td>-0.008685</td>
      <td>0.000374</td>
      <td>-0.076217</td>
    </tr>
    <tr>
      <th>phone_home_valid</th>
      <td>0.000148</td>
      <td>0.001591</td>
      <td>5.020150</td>
      <td>1.077889</td>
      <td>-0.143337</td>
      <td>-26.639721</td>
      <td>-3.762780e+01</td>
      <td>-3.538605e+01</td>
      <td>-2.303522e+01</td>
      <td>13.835538</td>
      <td>...</td>
      <td>-0.005578</td>
      <td>0.240177</td>
      <td>-0.044352</td>
      <td>0.026112</td>
      <td>-3.560998</td>
      <td>-0.000802</td>
      <td>-0.166718</td>
      <td>0.009658</td>
      <td>-0.000225</td>
      <td>0.060949</td>
    </tr>
    <tr>
      <th>phone_mobile_valid</th>
      <td>0.000961</td>
      <td>0.001704</td>
      <td>-2.875765</td>
      <td>-0.610440</td>
      <td>0.051130</td>
      <td>4.915758</td>
      <td>-1.453619e+01</td>
      <td>-1.272591e+01</td>
      <td>-9.038828e+00</td>
      <td>-2.383647</td>
      <td>...</td>
      <td>0.005232</td>
      <td>-0.044352</td>
      <td>0.103251</td>
      <td>0.000530</td>
      <td>-6.855258</td>
      <td>0.001017</td>
      <td>-0.027691</td>
      <td>0.005772</td>
      <td>-0.002789</td>
      <td>0.024970</td>
    </tr>
    <tr>
      <th>has_other_cards</th>
      <td>0.007691</td>
      <td>0.001124</td>
      <td>1.630877</td>
      <td>0.380307</td>
      <td>-0.117430</td>
      <td>-9.881681</td>
      <td>1.560574e+01</td>
      <td>-7.266532e+00</td>
      <td>-1.301574e+01</td>
      <td>7.016620</td>
      <td>...</td>
      <td>-0.005552</td>
      <td>0.026112</td>
      <td>0.000530</td>
      <td>0.167923</td>
      <td>16.362243</td>
      <td>-0.000643</td>
      <td>-0.268723</td>
      <td>-0.017893</td>
      <td>-0.002412</td>
      <td>0.029655</td>
    </tr>
    <tr>
      <th>proposed_credit_limit</th>
      <td>18.224996</td>
      <td>7.589780</td>
      <td>6921.295187</td>
      <td>1089.333498</td>
      <td>-189.537392</td>
      <td>-8083.186676</td>
      <td>-4.078343e+04</td>
      <td>5.069770e+03</td>
      <td>1.576029e+04</td>
      <td>-392.652565</td>
      <td>...</td>
      <td>0.929047</td>
      <td>-3.560998</td>
      <td>-6.855258</td>
      <td>16.362243</td>
      <td>266575.932029</td>
      <td>3.077936</td>
      <td>-2.286707</td>
      <td>-12.425298</td>
      <td>0.026084</td>
      <td>-51.424404</td>
    </tr>
    <tr>
      <th>foreign_request</th>
      <td>0.000759</td>
      <td>-0.001033</td>
      <td>-0.187393</td>
      <td>0.000818</td>
      <td>0.002647</td>
      <td>2.453612</td>
      <td>-5.237982e+00</td>
      <td>4.123817e+00</td>
      <td>2.106544e+00</td>
      <td>-0.142042</td>
      <td>...</td>
      <td>0.001177</td>
      <td>-0.000802</td>
      <td>0.001017</td>
      <td>-0.000643</td>
      <td>3.077936</td>
      <td>0.027965</td>
      <td>0.029076</td>
      <td>-0.001266</td>
      <td>0.000089</td>
      <td>-0.003614</td>
    </tr>
    <tr>
      <th>session_length_in_minutes</th>
      <td>-0.116759</td>
      <td>0.016699</td>
      <td>-12.739258</td>
      <td>4.356485</td>
      <td>2.033355</td>
      <td>359.426268</td>
      <td>1.323806e+03</td>
      <td>8.641378e+02</td>
      <td>6.635417e+02</td>
      <td>37.278437</td>
      <td>...</td>
      <td>0.146011</td>
      <td>-0.166718</td>
      <td>-0.027691</td>
      <td>-0.268723</td>
      <td>-2.286707</td>
      <td>0.029076</td>
      <td>69.376263</td>
      <td>-0.245622</td>
      <td>0.134616</td>
      <td>-1.542432</td>
    </tr>
    <tr>
      <th>keep_alive_session</th>
      <td>-0.009207</td>
      <td>0.006064</td>
      <td>-2.733750</td>
      <td>-0.317894</td>
      <td>0.057482</td>
      <td>3.917571</td>
      <td>2.570722e+01</td>
      <td>5.405526e+00</td>
      <td>1.604997e+01</td>
      <td>1.434909</td>
      <td>...</td>
      <td>-0.008685</td>
      <td>0.009658</td>
      <td>0.005772</td>
      <td>-0.017893</td>
      <td>-12.425298</td>
      <td>-0.001266</td>
      <td>-0.245622</td>
      <td>0.246520</td>
      <td>-0.007854</td>
      <td>-0.029529</td>
    </tr>
    <tr>
      <th>device_distinct_emails_8w</th>
      <td>-0.000441</td>
      <td>-0.002295</td>
      <td>0.235010</td>
      <td>0.148603</td>
      <td>0.024413</td>
      <td>7.000640</td>
      <td>1.501775e+01</td>
      <td>8.775432e+00</td>
      <td>9.659638e+00</td>
      <td>0.331983</td>
      <td>...</td>
      <td>0.000374</td>
      <td>-0.000225</td>
      <td>-0.002789</td>
      <td>-0.002412</td>
      <td>0.026084</td>
      <td>0.000089</td>
      <td>0.134616</td>
      <td>-0.007854</td>
      <td>0.038996</td>
      <td>-0.022020</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.079395</td>
      <td>-0.028691</td>
      <td>-4.112885</td>
      <td>0.016601</td>
      <td>-0.216861</td>
      <td>-600.642652</td>
      <td>-2.811793e+03</td>
      <td>-1.818947e+03</td>
      <td>-1.716809e+03</td>
      <td>-37.211440</td>
      <td>...</td>
      <td>-0.076217</td>
      <td>0.060949</td>
      <td>0.024970</td>
      <td>0.029655</td>
      <td>-51.424404</td>
      <td>-0.003614</td>
      <td>-1.542432</td>
      <td>-0.029529</td>
      <td>-0.022020</td>
      <td>4.896620</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>



Di atas adalah distribusi nilai covariance dari data yang belum distandarisasi (scale). Variance dari masing-masing variabel berbeda jauh karena range/skala dari tiap variabel berbeda, begitupun covariance. **Nilai variance dan covariance dipengaruhi oleh skala dari data**. Semakin tinggi skala, nilai variance atau covariance akan semakin tinggi.

[**Data dengan perbedaan skala antar variabel yang tinggi tidak baik untuk langsung dianalisis PCA karena dapat menimbulkan bias**](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).

#### Data Pre-processing: Scaling

Melakukan normalisasi pada dataframe `fraud_num` agar setiap prediktor memiliki scala yang sama.


```python
fraud_num.head()
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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>intended_balcon_amount</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>0.069598</td>
      <td>48.0</td>
      <td>30</td>
      <td>0.006760</td>
      <td>-1.074674</td>
      <td>3483</td>
      <td>5316.092932</td>
      <td>4527.956243</td>
      <td>4730.638776</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>5.191773</td>
      <td>1</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9</td>
      <td>0.891741</td>
      <td>61.0</td>
      <td>20</td>
      <td>0.020642</td>
      <td>-1.043444</td>
      <td>2849</td>
      <td>8153.671429</td>
      <td>7524.130278</td>
      <td>5341.758190</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>200.0</td>
      <td>0</td>
      <td>3.901673</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>0.370933</td>
      <td>70.0</td>
      <td>30</td>
      <td>6.400793</td>
      <td>48.520199</td>
      <td>406</td>
      <td>7648.434993</td>
      <td>6366.061338</td>
      <td>5431.786246</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>3.777191</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9</td>
      <td>0.401137</td>
      <td>64.0</td>
      <td>30</td>
      <td>0.004651</td>
      <td>-0.394588</td>
      <td>780</td>
      <td>6459.224179</td>
      <td>3394.524379</td>
      <td>4248.230609</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>3.176269</td>
      <td>1</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>0.720006</td>
      <td>11.0</td>
      <td>20</td>
      <td>0.032629</td>
      <td>-0.487785</td>
      <td>4527</td>
      <td>7852.258962</td>
      <td>5177.826213</td>
      <td>5942.104901</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>14.626874</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Menggunakan Z-score standardization untuk scaling dataset numerik dengan fungsi [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) pada library sklearn:


```python
scaler = StandardScaler()

fraud_scaled = scaler.fit_transform(fraud_num.values)

fraud_scaled = pd.DataFrame(fraud_scaled, columns=[cols])
```


```python
# cek covariance setelah di scaling
fraud_scaled.cov()
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>income</th>
      <th>name_email_similarity</th>
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>bank_branch_count_8w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>income</th>
      <td>1.000067</td>
      <td>-0.053847</td>
      <td>-0.023786</td>
      <td>0.141684</td>
      <td>-0.001914</td>
      <td>-0.078656</td>
      <td>-0.102545</td>
      <td>-0.116934</td>
      <td>-0.115435</td>
      <td>0.010793</td>
      <td>...</td>
      <td>-0.012490</td>
      <td>0.001038</td>
      <td>0.010272</td>
      <td>0.064446</td>
      <td>0.121202</td>
      <td>0.015582</td>
      <td>-0.048132</td>
      <td>-0.063670</td>
      <td>-0.007674</td>
      <td>0.123197</td>
    </tr>
    <tr>
      <th>name_email_similarity</th>
      <td>-0.053847</td>
      <td>1.000067</td>
      <td>0.022918</td>
      <td>-0.076457</td>
      <td>-0.005392</td>
      <td>0.019997</td>
      <td>0.036224</td>
      <td>0.043178</td>
      <td>0.037532</td>
      <td>0.003261</td>
      <td>...</td>
      <td>-0.082515</td>
      <td>0.011090</td>
      <td>0.018110</td>
      <td>0.009366</td>
      <td>0.050216</td>
      <td>-0.021103</td>
      <td>0.006849</td>
      <td>0.041724</td>
      <td>-0.039709</td>
      <td>-0.044292</td>
    </tr>
    <tr>
      <th>current_address_months_count</th>
      <td>-0.023786</td>
      <td>0.022918</td>
      <td>1.000067</td>
      <td>0.141228</td>
      <td>-0.061805</td>
      <td>0.044364</td>
      <td>0.029888</td>
      <td>0.021359</td>
      <td>0.024692</td>
      <td>0.058756</td>
      <td>...</td>
      <td>-0.066646</td>
      <td>0.115811</td>
      <td>-0.101182</td>
      <td>0.044995</td>
      <td>0.151556</td>
      <td>-0.012669</td>
      <td>-0.017292</td>
      <td>-0.062248</td>
      <td>0.013455</td>
      <td>-0.021013</td>
    </tr>
    <tr>
      <th>customer_age</th>
      <td>0.141684</td>
      <td>-0.076457</td>
      <td>0.141228</td>
      <td>1.000067</td>
      <td>-0.037381</td>
      <td>-0.013363</td>
      <td>-0.024483</td>
      <td>-0.007093</td>
      <td>-0.000675</td>
      <td>0.044810</td>
      <td>...</td>
      <td>0.010544</td>
      <td>0.177740</td>
      <td>-0.153523</td>
      <td>0.074999</td>
      <td>0.170501</td>
      <td>0.000395</td>
      <td>0.042268</td>
      <td>-0.051741</td>
      <td>0.060813</td>
      <td>0.000606</td>
    </tr>
    <tr>
      <th>days_since_request</th>
      <td>-0.001914</td>
      <td>-0.005392</td>
      <td>-0.061805</td>
      <td>-0.037381</td>
      <td>1.000067</td>
      <td>0.009386</td>
      <td>0.025342</td>
      <td>0.021596</td>
      <td>0.026567</td>
      <td>-0.020152</td>
      <td>...</td>
      <td>0.005126</td>
      <td>-0.051733</td>
      <td>0.028145</td>
      <td>-0.050688</td>
      <td>-0.064933</td>
      <td>0.002800</td>
      <td>0.043180</td>
      <td>0.020478</td>
      <td>0.021867</td>
      <td>-0.017335</td>
    </tr>
    <tr>
      <th>zip_count_4w</th>
      <td>-0.078656</td>
      <td>0.019997</td>
      <td>0.044364</td>
      <td>-0.013363</td>
      <td>0.009386</td>
      <td>1.000067</td>
      <td>0.131470</td>
      <td>0.192572</td>
      <td>0.292134</td>
      <td>0.013112</td>
      <td>...</td>
      <td>0.040730</td>
      <td>-0.054442</td>
      <td>0.015322</td>
      <td>-0.024152</td>
      <td>-0.015680</td>
      <td>0.014695</td>
      <td>0.043219</td>
      <td>0.007902</td>
      <td>0.035506</td>
      <td>-0.271856</td>
    </tr>
    <tr>
      <th>velocity_6h</th>
      <td>-0.102545</td>
      <td>0.036224</td>
      <td>0.029888</td>
      <td>-0.024483</td>
      <td>0.025342</td>
      <td>0.131470</td>
      <td>1.000067</td>
      <td>0.467295</td>
      <td>0.409350</td>
      <td>0.018539</td>
      <td>...</td>
      <td>0.024823</td>
      <td>-0.025459</td>
      <td>-0.015000</td>
      <td>0.012628</td>
      <td>-0.026192</td>
      <td>-0.010386</td>
      <td>0.052700</td>
      <td>0.017168</td>
      <td>0.025216</td>
      <td>-0.421334</td>
    </tr>
    <tr>
      <th>velocity_24h</th>
      <td>-0.116934</td>
      <td>0.043178</td>
      <td>0.021359</td>
      <td>-0.007093</td>
      <td>0.021596</td>
      <td>0.192572</td>
      <td>0.467295</td>
      <td>1.000067</td>
      <td>0.537146</td>
      <td>0.038152</td>
      <td>...</td>
      <td>0.020208</td>
      <td>-0.048580</td>
      <td>-0.026646</td>
      <td>-0.011931</td>
      <td>0.006606</td>
      <td>0.016591</td>
      <td>0.069802</td>
      <td>0.007325</td>
      <td>0.029898</td>
      <td>-0.553043</td>
    </tr>
    <tr>
      <th>velocity_4w</th>
      <td>-0.115435</td>
      <td>0.037532</td>
      <td>0.024692</td>
      <td>-0.000675</td>
      <td>0.026567</td>
      <td>0.292134</td>
      <td>0.409350</td>
      <td>0.537146</td>
      <td>1.000067</td>
      <td>0.039274</td>
      <td>...</td>
      <td>0.055047</td>
      <td>-0.050881</td>
      <td>-0.030450</td>
      <td>-0.034383</td>
      <td>0.033043</td>
      <td>0.013636</td>
      <td>0.086236</td>
      <td>0.034993</td>
      <td>0.052951</td>
      <td>-0.839851</td>
    </tr>
    <tr>
      <th>bank_branch_count_8w</th>
      <td>0.010793</td>
      <td>0.003261</td>
      <td>0.058756</td>
      <td>0.044810</td>
      <td>-0.020152</td>
      <td>0.013112</td>
      <td>0.018539</td>
      <td>0.038152</td>
      <td>0.039274</td>
      <td>1.000067</td>
      <td>...</td>
      <td>-0.009238</td>
      <td>0.060644</td>
      <td>-0.015935</td>
      <td>0.036781</td>
      <td>-0.001634</td>
      <td>-0.001825</td>
      <td>0.009614</td>
      <td>0.006208</td>
      <td>0.003611</td>
      <td>-0.036123</td>
    </tr>
    <tr>
      <th>date_of_birth_distinct_emails_4w</th>
      <td>-0.079916</td>
      <td>0.038485</td>
      <td>-0.176308</td>
      <td>-0.434568</td>
      <td>0.010436</td>
      <td>0.119037</td>
      <td>0.119399</td>
      <td>0.155631</td>
      <td>0.233646</td>
      <td>-0.030065</td>
      <td>...</td>
      <td>0.027418</td>
      <td>-0.146002</td>
      <td>0.112871</td>
      <td>-0.028296</td>
      <td>-0.070229</td>
      <td>0.022439</td>
      <td>-0.033792</td>
      <td>0.063755</td>
      <td>-0.040650</td>
      <td>-0.237180</td>
    </tr>
    <tr>
      <th>credit_risk_score</th>
      <td>0.191773</td>
      <td>0.028069</td>
      <td>0.112802</td>
      <td>0.192374</td>
      <td>-0.080584</td>
      <td>-0.095835</td>
      <td>-0.144945</td>
      <td>-0.157187</td>
      <td>-0.167673</td>
      <td>-0.025182</td>
      <td>...</td>
      <td>-0.014321</td>
      <td>-0.010531</td>
      <td>-0.013235</td>
      <td>0.112434</td>
      <td>0.632868</td>
      <td>0.040093</td>
      <td>-0.039362</td>
      <td>-0.051680</td>
      <td>-0.038454</td>
      <td>0.169837</td>
    </tr>
    <tr>
      <th>email_is_free</th>
      <td>-0.012490</td>
      <td>-0.082515</td>
      <td>-0.066646</td>
      <td>0.010544</td>
      <td>0.005126</td>
      <td>0.040730</td>
      <td>0.024823</td>
      <td>0.020208</td>
      <td>0.055047</td>
      <td>-0.009238</td>
      <td>...</td>
      <td>1.000067</td>
      <td>-0.022844</td>
      <td>0.032682</td>
      <td>-0.027193</td>
      <td>0.003612</td>
      <td>0.014125</td>
      <td>0.035185</td>
      <td>-0.035108</td>
      <td>0.003802</td>
      <td>-0.069131</td>
    </tr>
    <tr>
      <th>phone_home_valid</th>
      <td>0.001038</td>
      <td>0.011090</td>
      <td>0.115811</td>
      <td>0.177740</td>
      <td>-0.051733</td>
      <td>-0.054442</td>
      <td>-0.025459</td>
      <td>-0.048580</td>
      <td>-0.050881</td>
      <td>0.060644</td>
      <td>...</td>
      <td>-0.022844</td>
      <td>1.000067</td>
      <td>-0.281662</td>
      <td>0.130030</td>
      <td>-0.014074</td>
      <td>-0.009784</td>
      <td>-0.040845</td>
      <td>0.039693</td>
      <td>-0.002327</td>
      <td>0.056206</td>
    </tr>
    <tr>
      <th>phone_mobile_valid</th>
      <td>0.010272</td>
      <td>0.018110</td>
      <td>-0.101182</td>
      <td>-0.153523</td>
      <td>0.028145</td>
      <td>0.015322</td>
      <td>-0.015000</td>
      <td>-0.026646</td>
      <td>-0.030450</td>
      <td>-0.015935</td>
      <td>...</td>
      <td>0.032682</td>
      <td>-0.281662</td>
      <td>1.000067</td>
      <td>0.004027</td>
      <td>-0.041323</td>
      <td>0.018932</td>
      <td>-0.010347</td>
      <td>0.036182</td>
      <td>-0.043961</td>
      <td>0.035120</td>
    </tr>
    <tr>
      <th>has_other_cards</th>
      <td>0.064446</td>
      <td>0.009366</td>
      <td>0.044995</td>
      <td>0.074999</td>
      <td>-0.050688</td>
      <td>-0.024152</td>
      <td>0.012628</td>
      <td>-0.011931</td>
      <td>-0.034383</td>
      <td>0.036781</td>
      <td>...</td>
      <td>-0.027193</td>
      <td>0.130030</td>
      <td>0.004027</td>
      <td>1.000067</td>
      <td>0.077340</td>
      <td>-0.009391</td>
      <td>-0.078736</td>
      <td>-0.087948</td>
      <td>-0.029808</td>
      <td>0.032706</td>
    </tr>
    <tr>
      <th>proposed_credit_limit</th>
      <td>0.121202</td>
      <td>0.050216</td>
      <td>0.151556</td>
      <td>0.170501</td>
      <td>-0.064933</td>
      <td>-0.015680</td>
      <td>-0.026192</td>
      <td>0.006606</td>
      <td>0.033043</td>
      <td>-0.001634</td>
      <td>...</td>
      <td>0.003612</td>
      <td>-0.014074</td>
      <td>-0.041323</td>
      <td>0.077340</td>
      <td>1.000067</td>
      <td>0.035651</td>
      <td>-0.000532</td>
      <td>-0.048473</td>
      <td>0.000256</td>
      <td>-0.045013</td>
    </tr>
    <tr>
      <th>foreign_request</th>
      <td>0.015582</td>
      <td>-0.021103</td>
      <td>-0.012669</td>
      <td>0.000395</td>
      <td>0.002800</td>
      <td>0.014695</td>
      <td>-0.010386</td>
      <td>0.016591</td>
      <td>0.013636</td>
      <td>-0.001825</td>
      <td>...</td>
      <td>0.014125</td>
      <td>-0.009784</td>
      <td>0.018932</td>
      <td>-0.009391</td>
      <td>0.035651</td>
      <td>1.000067</td>
      <td>0.020876</td>
      <td>-0.015251</td>
      <td>0.002707</td>
      <td>-0.009768</td>
    </tr>
    <tr>
      <th>session_length_in_minutes</th>
      <td>-0.048132</td>
      <td>0.006849</td>
      <td>-0.017292</td>
      <td>0.042268</td>
      <td>0.043180</td>
      <td>0.043219</td>
      <td>0.052700</td>
      <td>0.069802</td>
      <td>0.086236</td>
      <td>0.009614</td>
      <td>...</td>
      <td>0.035185</td>
      <td>-0.040845</td>
      <td>-0.010347</td>
      <td>-0.078736</td>
      <td>-0.000532</td>
      <td>0.020876</td>
      <td>1.000067</td>
      <td>-0.059397</td>
      <td>0.081848</td>
      <td>-0.083692</td>
    </tr>
    <tr>
      <th>keep_alive_session</th>
      <td>-0.063670</td>
      <td>0.041724</td>
      <td>-0.062248</td>
      <td>-0.051741</td>
      <td>0.020478</td>
      <td>0.007902</td>
      <td>0.017168</td>
      <td>0.007325</td>
      <td>0.034993</td>
      <td>0.006208</td>
      <td>...</td>
      <td>-0.035108</td>
      <td>0.039693</td>
      <td>0.036182</td>
      <td>-0.087948</td>
      <td>-0.048473</td>
      <td>-0.015251</td>
      <td>-0.059397</td>
      <td>1.000067</td>
      <td>-0.080109</td>
      <td>-0.026878</td>
    </tr>
    <tr>
      <th>device_distinct_emails_8w</th>
      <td>-0.007674</td>
      <td>-0.039709</td>
      <td>0.013455</td>
      <td>0.060813</td>
      <td>0.021867</td>
      <td>0.035506</td>
      <td>0.025216</td>
      <td>0.029898</td>
      <td>0.052951</td>
      <td>0.003611</td>
      <td>...</td>
      <td>0.003802</td>
      <td>-0.002327</td>
      <td>-0.043961</td>
      <td>-0.029808</td>
      <td>0.000256</td>
      <td>0.002707</td>
      <td>0.081848</td>
      <td>-0.080109</td>
      <td>1.000067</td>
      <td>-0.050395</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.123197</td>
      <td>-0.044292</td>
      <td>-0.021013</td>
      <td>0.000606</td>
      <td>-0.017335</td>
      <td>-0.271856</td>
      <td>-0.421334</td>
      <td>-0.553043</td>
      <td>-0.839851</td>
      <td>-0.036123</td>
      <td>...</td>
      <td>-0.069131</td>
      <td>0.056206</td>
      <td>0.035120</td>
      <td>0.032706</td>
      <td>-0.045013</td>
      <td>-0.009768</td>
      <td>-0.083692</td>
      <td>-0.026878</td>
      <td>-0.050395</td>
      <td>1.000067</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>



<div class="alert alert-block alert-warning">
<b>Diskusi:</b> kenapa kita menggunakan StandardScaler bukan Min-Max scaling untuk kasus PCA?
</div>

> jawaban: Karena fokus dari standard scaler agar membentuk distribusi data senormal mungkin dengan data lain



```python
fraud_minmax = MinMaxScaler().fit_transform(fraud_num.values)

fraud_minmax = pd.DataFrame(fraud_minmax, columns=[cols])
```


```python
plt.figure(figsize=(7, 9))
plt.subplot(3,1,1)
sns.kdeplot(data=fraud.iloc[:,2:7], legend=None)
plt.ylabel("Base data")

plt.subplot(3,1,2)
sns.kdeplot(data=fraud_minmax.iloc[:,2:7], legend=None)
plt.ylabel("MinMaxScaler")

plt.subplot(3,1,3)
sns.kdeplot(data=fraud_scaled.iloc[:,2:7], legend=None)
plt.ylabel("StandardScaler");
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_42_0.png)
    


#### Principal Component Analysis menggunakan library [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)


```python
# inisialisasi objek PCA
pca = PCA(n_components = fraud_scaled.shape[1], # jumlah pca yang dihasilkan
          svd_solver='full') # implementasi full svd sehingga mendapatkan semua PC yang terbentuk

pca.fit(fraud_scaled) # menghitung PCA
# atau dapat menggunakan pca = pca.fit_transform(scale(balance_scaled))
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=22, svd_solver=&#x27;full&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=22, svd_solver=&#x27;full&#x27;)</pre></div></div></div></div></div>



**[additional] Note:** jika kita perhatikan bagian dokumentasi pada library scikit-learn, fungsi PCA menggunakan [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) sebagai reduksi dimensi linearnya. Output yang dihasilkan akan tetap sama dengan menggunakan dekomposisi eigen (mencari eigen vector dan eigen value), tetapi komputasi numeriknya lebih stabil dan efisien.

<div class="alert alert-block alert-success">
<b>&#128250; Rekomendasi Video:</b> <a href="https://www.youtube.com/watch?v=DQ_BkPHIl-g" class="button large hpbottom">hubungan PCA dengan SVD </a>
</div>



```python
# menampilkan banyaknya PC yang terbentuk dengan n_components_
pca.components_
```




    array([[-1.58193111e-01,  4.17231004e-02, -3.12993736e-02,
            -1.12354402e-01,  4.29029060e-02,  2.35194605e-01,
             3.53913770e-01,  4.15223133e-01,  4.79699339e-01,
             2.19143346e-02,  2.40314729e-01, -2.30827619e-01,
             4.97732176e-02, -8.09563599e-02,  1.64710215e-02,
            -6.01541713e-02, -9.13280517e-02,  5.58279894e-03,
             7.71753656e-02,  4.25029615e-02,  3.75431377e-02,
            -4.82409794e-01],
           [ 1.35352636e-01,  1.03067433e-02,  3.09083422e-01,
             4.30778257e-01, -1.08075190e-01,  5.94721124e-02,
             1.25566640e-01,  1.62480037e-01,  1.87053520e-01,
             8.44234569e-02, -2.81525038e-01,  3.82221586e-01,
            -9.67381455e-03,  2.00084404e-01, -2.35702837e-01,
             1.63752218e-01,  4.42144735e-01,  2.38701854e-02,
             4.36090971e-02, -1.18017066e-01,  7.11848597e-02,
            -1.90168955e-01],
           [-1.62305983e-01, -1.08967801e-01,  1.23175526e-01,
             2.28386319e-01,  3.22335412e-02, -3.00593250e-02,
             3.34595202e-02,  4.89731460e-05, -4.00993479e-02,
             1.30764843e-01, -2.96241363e-01, -4.46874904e-01,
            -5.13456494e-02,  4.40015716e-01, -3.79541008e-01,
            -2.12403957e-02, -4.57573328e-01, -1.08373972e-01,
             6.26680483e-02,  1.32022415e-02,  1.31232176e-01,
             4.50572823e-02],
           [ 1.34508186e-01, -3.58469858e-01, -1.46451454e-01,
             2.04504937e-01,  2.24077803e-01,  5.44249899e-02,
            -4.79294349e-02, -2.08365909e-02,  5.70973387e-03,
            -9.79628667e-02, -1.88221723e-01, -3.22019706e-02,
             3.13814042e-01, -2.91441741e-01,  1.12788730e-01,
            -2.53440640e-01, -3.64148140e-02,  1.22686183e-01,
             4.15692426e-01, -3.09373980e-01,  3.77976682e-01,
            -3.28759839e-03],
           [-3.12684289e-01,  4.44157306e-01,  2.09772604e-01,
            -1.79625729e-02,  1.70119360e-01,  3.20546380e-02,
            -5.35459698e-02, -3.39864159e-02, -3.72535154e-02,
            -1.19157099e-01, -1.49142409e-01,  8.34566932e-02,
            -3.38789799e-01, -1.26254300e-01, -4.45022130e-02,
            -4.88724839e-01,  1.28578546e-01, -7.10273043e-02,
             3.48378324e-01,  2.26440765e-01,  1.19418217e-01,
             4.25272151e-02],
           [-1.08824383e-01,  2.16070520e-01,  2.40439603e-01,
            -1.94369028e-01, -1.82808479e-01,  6.38008587e-02,
             1.45228927e-02, -7.58285566e-03, -7.24652491e-02,
             1.45266831e-01,  7.04409015e-02, -8.42914107e-02,
            -2.85944784e-01, -1.31275244e-01,  1.40387346e-01,
             3.48785217e-01, -7.93340219e-02, -1.10816075e-01,
             1.23957247e-01, -6.22821532e-01,  3.20216503e-01,
             7.04246420e-02],
           [ 3.67065762e-01,  8.04707653e-02, -9.07382431e-02,
             1.95796919e-01,  4.96506806e-01, -1.41648918e-01,
             1.56071646e-01,  1.07319850e-01,  2.31409554e-02,
             2.79655954e-01, -2.04781879e-01, -5.69564101e-02,
            -3.14285915e-01, -1.68211710e-01,  3.06921405e-01,
             1.57856324e-01, -9.56346018e-02, -3.28832405e-01,
            -5.52134765e-02,  1.07508285e-01, -8.67428891e-02,
            -1.54136302e-02],
           [-1.15805662e-01, -7.33062900e-02,  2.30573762e-01,
             6.42000152e-02, -2.53605294e-01,  2.16516131e-01,
            -1.06512723e-01, -5.43655035e-02, -1.79649751e-02,
             7.16482696e-01, -1.12244394e-01, -4.20518498e-02,
             1.40133592e-01, -1.30770788e-01,  3.12158954e-01,
            -1.43633521e-01, -4.42468610e-02,  2.23095926e-01,
             7.35231946e-02,  1.71116159e-01, -1.68418692e-01,
             2.83456276e-02],
           [ 1.51541768e-01,  1.21769995e-01, -1.48988957e-01,
            -6.92414249e-02,  2.75000584e-01, -9.87472132e-02,
            -1.18844066e-02,  2.79491526e-02,  2.05338325e-03,
             1.59174994e-01,  1.15892525e-01, -7.92084014e-03,
            -2.90031265e-01,  1.67325462e-01, -1.29272462e-01,
             1.06466567e-01, -1.18147760e-02,  8.07207126e-01,
             1.08115068e-01, -1.10014710e-02,  7.95638058e-02,
             7.48062815e-03],
           [-9.12279852e-02,  3.79810452e-01, -3.08530867e-01,
            -8.12691133e-02,  8.67875714e-02, -2.51218045e-01,
             1.62871773e-02, -1.08677696e-02, -2.79250744e-02,
             2.83103686e-01,  8.31392742e-02,  7.24579090e-02,
             4.53977699e-01,  1.98475045e-01, -1.08828973e-01,
             1.70852135e-01,  8.95164745e-02, -1.96460489e-01,
             4.70066746e-01, -6.67284824e-02, -1.51309341e-01,
             8.22066516e-03],
           [-2.77691643e-01,  1.10080572e-01,  2.35972110e-01,
             1.24801179e-01,  2.02527364e-01,  5.06805911e-02,
             1.30073404e-01,  6.06991491e-02, -5.96452018e-02,
            -3.71233550e-01, -2.73684628e-01, -7.52253905e-02,
             1.72102920e-01, -5.22902566e-02,  2.44814305e-01,
             2.30680758e-01, -1.21208342e-01,  2.59758262e-01,
             6.16472898e-02, -1.84046857e-01, -5.33951258e-01,
             5.42525824e-02],
           [-1.22833113e-01,  1.21150639e-02,  2.20380307e-01,
            -1.39574260e-01,  6.15788778e-01,  4.68633099e-01,
            -2.16613102e-01, -1.65395818e-01, -2.93554615e-03,
             1.21611818e-01,  1.45565133e-01,  6.44150540e-02,
             2.47935015e-01,  1.20974012e-01, -1.08183119e-01,
             1.04277816e-01,  1.08665576e-01, -9.09041993e-02,
            -2.50648491e-01, -2.09160068e-02,  1.44860391e-01,
             2.20914880e-02],
           [-1.69886520e-01, -3.89315218e-01,  2.81141606e-01,
            -2.45782147e-01,  2.26162133e-01, -4.77376136e-01,
             1.49677507e-01,  9.80140349e-02, -2.67335092e-02,
             2.18070980e-01,  1.18137494e-01,  7.02643808e-02,
            -6.28351349e-02, -9.93899966e-02, -2.34496512e-01,
            -2.67238960e-01,  1.25842852e-01, -4.78164587e-02,
            -5.98350819e-02, -2.76158939e-01, -2.52263914e-01,
             8.73903898e-03],
           [-4.62554577e-01, -7.38856713e-02,  8.03258864e-03,
             1.54395218e-02,  5.78997858e-02, -3.86764056e-01,
             1.90870487e-01,  7.01717977e-02, -5.11082716e-02,
             8.72037089e-03, -1.03451528e-01,  7.33498025e-02,
             1.56781729e-01,  2.69270420e-03,  2.76182943e-01,
             2.49822572e-01,  8.75973706e-02,  9.73352235e-02,
            -1.96610365e-01,  3.10162796e-01,  5.00743925e-01,
             3.45209659e-02],
           [-2.23369741e-01, -5.11654222e-01, -7.11377123e-02,
            -4.11392465e-02,  3.10148302e-02,  1.87383812e-01,
            -8.25468268e-02, -6.15836928e-02, -3.64176545e-04,
            -4.99733010e-02,  1.30027523e-01,  9.56396009e-02,
            -3.39921338e-01,  7.19387442e-02,  2.24269039e-02,
             3.70681432e-01,  9.49660633e-02, -1.23975097e-01,
             4.96672054e-01,  2.45413315e-01, -1.26601059e-01,
             1.93336014e-02],
           [ 4.68175311e-01, -3.58136845e-03,  5.95978025e-01,
            -2.26902472e-01, -9.51154429e-03, -1.51026960e-01,
             1.10373307e-01, -3.43227745e-02, -1.95178467e-02,
            -1.60614443e-01,  1.78403784e-01, -9.11328824e-02,
             1.98417618e-01,  1.83311927e-01,  1.67375637e-01,
             4.14503644e-02, -7.80939245e-02,  3.78094653e-03,
             2.84024629e-01,  2.62667492e-01,  1.11468129e-01,
             2.54100762e-02],
           [ 7.83152887e-02, -1.78698436e-02, -9.34491208e-02,
            -1.60375705e-01, -3.45020156e-02,  3.48067112e-01,
             6.65149958e-01,  1.42804850e-01, -3.73979288e-01,
             6.25167165e-02, -8.13174294e-02,  9.93821133e-02,
             5.65196194e-02, -1.17645493e-01, -2.28132673e-01,
            -6.64586079e-03,  1.82542603e-02,  1.84236665e-02,
             1.94829982e-02,  9.10335222e-02,  3.61519016e-02,
             3.65933593e-01],
           [ 4.13062126e-02,  8.41766056e-02,  1.45998381e-01,
             2.36021873e-02, -2.51677178e-02, -1.05888884e-01,
            -1.75816374e-01, -4.12023636e-02,  5.27897387e-02,
             1.59086168e-02, -5.72585508e-02, -9.37564701e-02,
             1.06868916e-01, -6.64014141e-01, -5.11552027e-01,
             3.43631489e-01, -1.12252488e-01,  5.42352175e-02,
             2.54448818e-02,  2.40443492e-01,  3.57171505e-03,
            -6.04381753e-02],
           [-9.47546453e-02,  6.00121165e-02,  7.37934774e-02,
             5.73853220e-01,  2.40695500e-02, -4.47642299e-02,
             3.28016057e-01, -4.63147151e-01, -1.97498950e-02,
             1.49432325e-02,  5.58471895e-01, -3.82748771e-02,
            -2.39235278e-02, -8.65953424e-02, -3.15663255e-03,
            -4.22733532e-02, -4.60323340e-02,  9.72243254e-03,
             3.37032967e-03, -3.47604114e-02, -1.78917569e-02,
             1.98387954e-02],
           [-2.77150334e-02,  2.68659749e-02,  7.66241497e-02,
             3.33450675e-01,  2.05342298e-02,  3.97969333e-03,
            -3.06046627e-01,  7.03402269e-01, -2.81011375e-01,
            -1.44045229e-02,  3.80397261e-01, -2.91231619e-02,
             3.47869602e-02, -1.38799841e-02,  1.21340707e-02,
            -1.95163457e-02, -1.59872716e-02, -2.83961148e-02,
             1.20106838e-02,  1.96084657e-02,  1.57240818e-02,
             2.53137800e-01],
           [ 3.79367168e-02, -4.62552726e-03, -3.57563737e-02,
             1.38222829e-02, -1.49299588e-02,  2.37238812e-03,
             7.90096240e-03, -4.34618863e-02, -9.79391139e-02,
            -1.51067967e-02, -4.16612302e-02, -7.18593621e-01,
            -6.41527048e-03, -1.92828140e-02,  1.32692228e-02,
             3.11551006e-02,  6.79839585e-01,  1.09951125e-02,
            -1.29286747e-02,  7.32565966e-03, -2.65712906e-02,
             4.65608250e-02],
           [-6.54458124e-03,  6.35716254e-03, -3.97031990e-03,
             4.33967158e-03, -7.91809576e-03, -2.01861765e-02,
             6.67420142e-03,  1.92196060e-02,  6.97213662e-01,
            -2.56640743e-03,  4.14850592e-03, -2.36899745e-02,
             1.29786933e-02, -6.99891203e-03, -5.28914441e-03,
             1.35884842e-03,  2.77141918e-02, -3.04669177e-03,
            -3.49762045e-03, -6.35362842e-03, -2.33000686e-03,
             7.14998950e-01]])



`pca.components_` : berisi nilai *eigen vector* yang akan dijadikan formula untuk PC baru


```python
# opsional
pd.DataFrame(pca.components_.T, # dibalik/transpose agar representasi tiap pca menjadi kolom, bukan baris
             columns=pca.get_feature_names_out()) # ambil nama kolom tiap pca
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>...</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>pca17</th>
      <th>pca18</th>
      <th>pca19</th>
      <th>pca20</th>
      <th>pca21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.158193</td>
      <td>0.135353</td>
      <td>-0.162306</td>
      <td>0.134508</td>
      <td>-0.312684</td>
      <td>-0.108824</td>
      <td>0.367066</td>
      <td>-0.115806</td>
      <td>0.151542</td>
      <td>-0.091228</td>
      <td>...</td>
      <td>-0.169887</td>
      <td>-0.462555</td>
      <td>-0.223370</td>
      <td>0.468175</td>
      <td>0.078315</td>
      <td>0.041306</td>
      <td>-0.094755</td>
      <td>-0.027715</td>
      <td>0.037937</td>
      <td>-0.006545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.041723</td>
      <td>0.010307</td>
      <td>-0.108968</td>
      <td>-0.358470</td>
      <td>0.444157</td>
      <td>0.216071</td>
      <td>0.080471</td>
      <td>-0.073306</td>
      <td>0.121770</td>
      <td>0.379810</td>
      <td>...</td>
      <td>-0.389315</td>
      <td>-0.073886</td>
      <td>-0.511654</td>
      <td>-0.003581</td>
      <td>-0.017870</td>
      <td>0.084177</td>
      <td>0.060012</td>
      <td>0.026866</td>
      <td>-0.004626</td>
      <td>0.006357</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.031299</td>
      <td>0.309083</td>
      <td>0.123176</td>
      <td>-0.146451</td>
      <td>0.209773</td>
      <td>0.240440</td>
      <td>-0.090738</td>
      <td>0.230574</td>
      <td>-0.148989</td>
      <td>-0.308531</td>
      <td>...</td>
      <td>0.281142</td>
      <td>0.008033</td>
      <td>-0.071138</td>
      <td>0.595978</td>
      <td>-0.093449</td>
      <td>0.145998</td>
      <td>0.073793</td>
      <td>0.076624</td>
      <td>-0.035756</td>
      <td>-0.003970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.112354</td>
      <td>0.430778</td>
      <td>0.228386</td>
      <td>0.204505</td>
      <td>-0.017963</td>
      <td>-0.194369</td>
      <td>0.195797</td>
      <td>0.064200</td>
      <td>-0.069241</td>
      <td>-0.081269</td>
      <td>...</td>
      <td>-0.245782</td>
      <td>0.015440</td>
      <td>-0.041139</td>
      <td>-0.226902</td>
      <td>-0.160376</td>
      <td>0.023602</td>
      <td>0.573853</td>
      <td>0.333451</td>
      <td>0.013822</td>
      <td>0.004340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.042903</td>
      <td>-0.108075</td>
      <td>0.032234</td>
      <td>0.224078</td>
      <td>0.170119</td>
      <td>-0.182808</td>
      <td>0.496507</td>
      <td>-0.253605</td>
      <td>0.275001</td>
      <td>0.086788</td>
      <td>...</td>
      <td>0.226162</td>
      <td>0.057900</td>
      <td>0.031015</td>
      <td>-0.009512</td>
      <td>-0.034502</td>
      <td>-0.025168</td>
      <td>0.024070</td>
      <td>0.020534</td>
      <td>-0.014930</td>
      <td>-0.007918</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.235195</td>
      <td>0.059472</td>
      <td>-0.030059</td>
      <td>0.054425</td>
      <td>0.032055</td>
      <td>0.063801</td>
      <td>-0.141649</td>
      <td>0.216516</td>
      <td>-0.098747</td>
      <td>-0.251218</td>
      <td>...</td>
      <td>-0.477376</td>
      <td>-0.386764</td>
      <td>0.187384</td>
      <td>-0.151027</td>
      <td>0.348067</td>
      <td>-0.105889</td>
      <td>-0.044764</td>
      <td>0.003980</td>
      <td>0.002372</td>
      <td>-0.020186</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.353914</td>
      <td>0.125567</td>
      <td>0.033460</td>
      <td>-0.047929</td>
      <td>-0.053546</td>
      <td>0.014523</td>
      <td>0.156072</td>
      <td>-0.106513</td>
      <td>-0.011884</td>
      <td>0.016287</td>
      <td>...</td>
      <td>0.149678</td>
      <td>0.190870</td>
      <td>-0.082547</td>
      <td>0.110373</td>
      <td>0.665150</td>
      <td>-0.175816</td>
      <td>0.328016</td>
      <td>-0.306047</td>
      <td>0.007901</td>
      <td>0.006674</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.415223</td>
      <td>0.162480</td>
      <td>0.000049</td>
      <td>-0.020837</td>
      <td>-0.033986</td>
      <td>-0.007583</td>
      <td>0.107320</td>
      <td>-0.054366</td>
      <td>0.027949</td>
      <td>-0.010868</td>
      <td>...</td>
      <td>0.098014</td>
      <td>0.070172</td>
      <td>-0.061584</td>
      <td>-0.034323</td>
      <td>0.142805</td>
      <td>-0.041202</td>
      <td>-0.463147</td>
      <td>0.703402</td>
      <td>-0.043462</td>
      <td>0.019220</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.479699</td>
      <td>0.187054</td>
      <td>-0.040099</td>
      <td>0.005710</td>
      <td>-0.037254</td>
      <td>-0.072465</td>
      <td>0.023141</td>
      <td>-0.017965</td>
      <td>0.002053</td>
      <td>-0.027925</td>
      <td>...</td>
      <td>-0.026734</td>
      <td>-0.051108</td>
      <td>-0.000364</td>
      <td>-0.019518</td>
      <td>-0.373979</td>
      <td>0.052790</td>
      <td>-0.019750</td>
      <td>-0.281011</td>
      <td>-0.097939</td>
      <td>0.697214</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.021914</td>
      <td>0.084423</td>
      <td>0.130765</td>
      <td>-0.097963</td>
      <td>-0.119157</td>
      <td>0.145267</td>
      <td>0.279656</td>
      <td>0.716483</td>
      <td>0.159175</td>
      <td>0.283104</td>
      <td>...</td>
      <td>0.218071</td>
      <td>0.008720</td>
      <td>-0.049973</td>
      <td>-0.160614</td>
      <td>0.062517</td>
      <td>0.015909</td>
      <td>0.014943</td>
      <td>-0.014405</td>
      <td>-0.015107</td>
      <td>-0.002566</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.240315</td>
      <td>-0.281525</td>
      <td>-0.296241</td>
      <td>-0.188222</td>
      <td>-0.149142</td>
      <td>0.070441</td>
      <td>-0.204782</td>
      <td>-0.112244</td>
      <td>0.115893</td>
      <td>0.083139</td>
      <td>...</td>
      <td>0.118137</td>
      <td>-0.103452</td>
      <td>0.130028</td>
      <td>0.178404</td>
      <td>-0.081317</td>
      <td>-0.057259</td>
      <td>0.558472</td>
      <td>0.380397</td>
      <td>-0.041661</td>
      <td>0.004149</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.230828</td>
      <td>0.382222</td>
      <td>-0.446875</td>
      <td>-0.032202</td>
      <td>0.083457</td>
      <td>-0.084291</td>
      <td>-0.056956</td>
      <td>-0.042052</td>
      <td>-0.007921</td>
      <td>0.072458</td>
      <td>...</td>
      <td>0.070264</td>
      <td>0.073350</td>
      <td>0.095640</td>
      <td>-0.091133</td>
      <td>0.099382</td>
      <td>-0.093756</td>
      <td>-0.038275</td>
      <td>-0.029123</td>
      <td>-0.718594</td>
      <td>-0.023690</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.049773</td>
      <td>-0.009674</td>
      <td>-0.051346</td>
      <td>0.313814</td>
      <td>-0.338790</td>
      <td>-0.285945</td>
      <td>-0.314286</td>
      <td>0.140134</td>
      <td>-0.290031</td>
      <td>0.453978</td>
      <td>...</td>
      <td>-0.062835</td>
      <td>0.156782</td>
      <td>-0.339921</td>
      <td>0.198418</td>
      <td>0.056520</td>
      <td>0.106869</td>
      <td>-0.023924</td>
      <td>0.034787</td>
      <td>-0.006415</td>
      <td>0.012979</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.080956</td>
      <td>0.200084</td>
      <td>0.440016</td>
      <td>-0.291442</td>
      <td>-0.126254</td>
      <td>-0.131275</td>
      <td>-0.168212</td>
      <td>-0.130771</td>
      <td>0.167325</td>
      <td>0.198475</td>
      <td>...</td>
      <td>-0.099390</td>
      <td>0.002693</td>
      <td>0.071939</td>
      <td>0.183312</td>
      <td>-0.117645</td>
      <td>-0.664014</td>
      <td>-0.086595</td>
      <td>-0.013880</td>
      <td>-0.019283</td>
      <td>-0.006999</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.016471</td>
      <td>-0.235703</td>
      <td>-0.379541</td>
      <td>0.112789</td>
      <td>-0.044502</td>
      <td>0.140387</td>
      <td>0.306921</td>
      <td>0.312159</td>
      <td>-0.129272</td>
      <td>-0.108829</td>
      <td>...</td>
      <td>-0.234497</td>
      <td>0.276183</td>
      <td>0.022427</td>
      <td>0.167376</td>
      <td>-0.228133</td>
      <td>-0.511552</td>
      <td>-0.003157</td>
      <td>0.012134</td>
      <td>0.013269</td>
      <td>-0.005289</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.060154</td>
      <td>0.163752</td>
      <td>-0.021240</td>
      <td>-0.253441</td>
      <td>-0.488725</td>
      <td>0.348785</td>
      <td>0.157856</td>
      <td>-0.143634</td>
      <td>0.106467</td>
      <td>0.170852</td>
      <td>...</td>
      <td>-0.267239</td>
      <td>0.249823</td>
      <td>0.370681</td>
      <td>0.041450</td>
      <td>-0.006646</td>
      <td>0.343631</td>
      <td>-0.042273</td>
      <td>-0.019516</td>
      <td>0.031155</td>
      <td>0.001359</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.091328</td>
      <td>0.442145</td>
      <td>-0.457573</td>
      <td>-0.036415</td>
      <td>0.128579</td>
      <td>-0.079334</td>
      <td>-0.095635</td>
      <td>-0.044247</td>
      <td>-0.011815</td>
      <td>0.089516</td>
      <td>...</td>
      <td>0.125843</td>
      <td>0.087597</td>
      <td>0.094966</td>
      <td>-0.078094</td>
      <td>0.018254</td>
      <td>-0.112252</td>
      <td>-0.046032</td>
      <td>-0.015987</td>
      <td>0.679840</td>
      <td>0.027714</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.005583</td>
      <td>0.023870</td>
      <td>-0.108374</td>
      <td>0.122686</td>
      <td>-0.071027</td>
      <td>-0.110816</td>
      <td>-0.328832</td>
      <td>0.223096</td>
      <td>0.807207</td>
      <td>-0.196460</td>
      <td>...</td>
      <td>-0.047816</td>
      <td>0.097335</td>
      <td>-0.123975</td>
      <td>0.003781</td>
      <td>0.018424</td>
      <td>0.054235</td>
      <td>0.009722</td>
      <td>-0.028396</td>
      <td>0.010995</td>
      <td>-0.003047</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.077175</td>
      <td>0.043609</td>
      <td>0.062668</td>
      <td>0.415692</td>
      <td>0.348378</td>
      <td>0.123957</td>
      <td>-0.055213</td>
      <td>0.073523</td>
      <td>0.108115</td>
      <td>0.470067</td>
      <td>...</td>
      <td>-0.059835</td>
      <td>-0.196610</td>
      <td>0.496672</td>
      <td>0.284025</td>
      <td>0.019483</td>
      <td>0.025445</td>
      <td>0.003370</td>
      <td>0.012011</td>
      <td>-0.012929</td>
      <td>-0.003498</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.042503</td>
      <td>-0.118017</td>
      <td>0.013202</td>
      <td>-0.309374</td>
      <td>0.226441</td>
      <td>-0.622822</td>
      <td>0.107508</td>
      <td>0.171116</td>
      <td>-0.011001</td>
      <td>-0.066728</td>
      <td>...</td>
      <td>-0.276159</td>
      <td>0.310163</td>
      <td>0.245413</td>
      <td>0.262667</td>
      <td>0.091034</td>
      <td>0.240443</td>
      <td>-0.034760</td>
      <td>0.019608</td>
      <td>0.007326</td>
      <td>-0.006354</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.037543</td>
      <td>0.071185</td>
      <td>0.131232</td>
      <td>0.377977</td>
      <td>0.119418</td>
      <td>0.320217</td>
      <td>-0.086743</td>
      <td>-0.168419</td>
      <td>0.079564</td>
      <td>-0.151309</td>
      <td>...</td>
      <td>-0.252264</td>
      <td>0.500744</td>
      <td>-0.126601</td>
      <td>0.111468</td>
      <td>0.036152</td>
      <td>0.003572</td>
      <td>-0.017892</td>
      <td>0.015724</td>
      <td>-0.026571</td>
      <td>-0.002330</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.482410</td>
      <td>-0.190169</td>
      <td>0.045057</td>
      <td>-0.003288</td>
      <td>0.042527</td>
      <td>0.070425</td>
      <td>-0.015414</td>
      <td>0.028346</td>
      <td>0.007481</td>
      <td>0.008221</td>
      <td>...</td>
      <td>0.008739</td>
      <td>0.034521</td>
      <td>0.019334</td>
      <td>0.025410</td>
      <td>0.365934</td>
      <td>-0.060438</td>
      <td>0.019839</td>
      <td>0.253138</td>
      <td>0.046561</td>
      <td>0.714999</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>



Melihat proporsi nilai informasi yang dapat ditangkap untuk setiap PC dengan atribut `explained_variance_ratio_`:


```python
# menampilkan banyaknya PC yang terbentuk dengan explained_variance_ratio
pca.explained_variance_ratio_
```




    array([0.1376811 , 0.09104272, 0.06854404, 0.05763219, 0.05041625,
           0.04800417, 0.04643708, 0.04563752, 0.04525991, 0.04340626,
           0.0428812 , 0.0418975 , 0.04056786, 0.03994442, 0.03862271,
           0.0349525 , 0.03072321, 0.02931932, 0.02269744, 0.02195474,
           0.01515074, 0.00722711])



Melihat kumulatif proporsi nilai informasi yang dapat ditangkap untuk setiap penambahan PC: 


```python
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
```




    array([13.77, 22.87, 29.72, 35.48, 40.52, 45.32, 49.96, 54.52, 59.05,
           63.39, 67.68, 71.87, 75.93, 79.92, 83.78, 87.28, 90.35, 93.28,
           95.55, 97.75, 99.27, 99.99])



**Note:** 

- Proportion of Variance: informasi yang ditangkap oleh tiap PC
- Cumulative Proportion: jumlah informasi yang ditangkap secara kumulatif dari PC0 hingga PC tersebut

Untuk lebih jelasnya, kita dapat mengeluarkan Cumulative Proportion di atas menggunakan plot di bawah ini.


```python
# Hitung proporsi variasi yang dijelaskan oleh setiap komponen utama
explained_var_ratio = pca.explained_variance_ratio_

# Buat scree plot menggunakan plotly
fig = go.Figure()

# Plot proporsi variasi yang dijelaskan
fig.add_trace(go.Scatter(x=list(range(1, len(explained_var_ratio) + 1)), 
                         y=explained_var_ratio*100, mode='lines+markers', 
                         name='Explained Variance Ratio'))

fig.add_trace(go.Scatter(x=list(range(1, len(explained_var_ratio) + 1)), 
                         y=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100), mode='lines+markers', 
                         name='Cumulative summary'))

# Atur layout dan tampilkan
fig.update_layout(title='Scree Plot',
                  xaxis_title='Principal Component (PC)',
                  yaxis_title='Explained Variance Ratio',
                  showlegend=True,
                  width=800, height=620)

pyo.iplot(fig, 'Scree')
```


<div>                            <div id="5d31e85f-48a3-4940-b8b2-ddd0830a4571" class="plotly-graph-div" style="height:620px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';                                    if (document.getElementById("5d31e85f-48a3-4940-b8b2-ddd0830a4571")) {                    Plotly.newPlot(                        "5d31e85f-48a3-4940-b8b2-ddd0830a4571",                        [{"mode":"lines+markers","name":"Explained Variance Ratio","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],"y":[13.768110073015585,9.104271939095412,6.854404038221358,5.7632186759071535,5.041625212674978,4.800417215325599,4.643707515691467,4.56375243315967,4.525990551147682,4.340626087445623,4.288119524572033,4.189750485888745,4.056785622973749,3.994442005593278,3.862271244722443,3.4952497608987323,3.0723213892830525,2.9319321220892802,2.269744443342226,2.195474137971751,1.5150742072412349,0.722711313738938],"type":"scatter"},{"mode":"lines+markers","name":"Cumulative summary","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],"y":[13.77,22.869999999999997,29.72,35.48,40.519999999999996,45.31999999999999,49.959999999999994,54.519999999999996,59.05,63.39,67.68,71.87,75.93,79.92,83.78,87.28,90.35,93.28,95.55,97.75,99.27,99.99],"type":"scatter"}],                        {"height":620,"showlegend":true,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Scree Plot"},"width":800,"xaxis":{"title":{"text":"Principal Component (PC)"}},"yaxis":{"title":{"text":"Explained Variance Ratio"}}},                        {"showLink": "Scree", "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}                    ).then(function(){

var gd = document.getElementById('5d31e85f-48a3-4940-b8b2-ddd0830a4571');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


**Transform PCA**

Menampilkan nilai di setiap PC pada dimensi baru


```python
transform_ = pd.DataFrame(pca.transform(fraud_scaled), 
                          columns=pca.get_feature_names_out())
transform_.head()
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>...</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>pca17</th>
      <th>pca18</th>
      <th>pca19</th>
      <th>pca20</th>
      <th>pca21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.985811</td>
      <td>1.769162</td>
      <td>-0.450567</td>
      <td>-1.204763</td>
      <td>-0.117952</td>
      <td>0.913919</td>
      <td>-0.725541</td>
      <td>1.175108</td>
      <td>1.759004</td>
      <td>-0.537196</td>
      <td>...</td>
      <td>-0.810912</td>
      <td>0.935328</td>
      <td>0.249804</td>
      <td>-0.067922</td>
      <td>-0.263015</td>
      <td>-0.442615</td>
      <td>0.653300</td>
      <td>-0.105641</td>
      <td>0.060576</td>
      <td>-0.063466</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.436214</td>
      <td>-0.073470</td>
      <td>0.229321</td>
      <td>-0.830563</td>
      <td>-0.100201</td>
      <td>2.323481</td>
      <td>0.927359</td>
      <td>-0.936612</td>
      <td>-0.827647</td>
      <td>-0.064329</td>
      <td>...</td>
      <td>-0.698201</td>
      <td>-1.541975</td>
      <td>-1.348272</td>
      <td>-0.053672</td>
      <td>0.511195</td>
      <td>0.031784</td>
      <td>-0.039432</td>
      <td>-0.396871</td>
      <td>-0.477543</td>
      <td>-0.203613</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.584369</td>
      <td>2.080913</td>
      <td>-0.033197</td>
      <td>0.575900</td>
      <td>-0.870996</td>
      <td>0.344936</td>
      <td>0.538653</td>
      <td>-0.454442</td>
      <td>-1.623096</td>
      <td>-1.124090</td>
      <td>...</td>
      <td>2.110493</td>
      <td>0.275672</td>
      <td>-1.110919</td>
      <td>0.196495</td>
      <td>0.736336</td>
      <td>0.427606</td>
      <td>0.865952</td>
      <td>-0.305482</td>
      <td>-0.585496</td>
      <td>-0.219957</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.245341</td>
      <td>0.854789</td>
      <td>-0.846922</td>
      <td>-0.526426</td>
      <td>-1.109341</td>
      <td>-0.393849</td>
      <td>-0.343671</td>
      <td>-1.335815</td>
      <td>0.108946</td>
      <td>0.625671</td>
      <td>...</td>
      <td>-0.262356</td>
      <td>-1.130105</td>
      <td>0.032108</td>
      <td>-0.351500</td>
      <td>0.061635</td>
      <td>-0.500656</td>
      <td>0.066543</td>
      <td>0.198758</td>
      <td>-0.302610</td>
      <td>-0.307206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.589848</td>
      <td>-0.145288</td>
      <td>1.091001</td>
      <td>0.032348</td>
      <td>0.029693</td>
      <td>1.672321</td>
      <td>-1.662417</td>
      <td>-0.280590</td>
      <td>0.326410</td>
      <td>-0.920124</td>
      <td>...</td>
      <td>-0.696854</td>
      <td>-0.764018</td>
      <td>0.425946</td>
      <td>0.366670</td>
      <td>-0.204808</td>
      <td>-0.079668</td>
      <td>-1.138763</td>
      <td>-0.004642</td>
      <td>0.955276</td>
      <td>0.085033</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



<div class="alert alert-block alert-warning">
<b>Diskusi:</b> ketika kita declare value dari `n_components` sama dengan jumlah dari fitur/variabel datasetnya <b>dan</b> kita menggunakan <b>semua</b> PC yang terbentuk, apakah kita sudah melakukan <b>reduksi dimensi</b>?
</div>

> jawaban: Belum, karena jumlah dimensi nya masih sama tapi persebaran distribusi dan informasinya jadi berubah

Reduksi dimensi dengan mempertahankan at least 90% informasi maka PC dipilih sampai 16


```python
fraud_pca = transform_.iloc[:,:17]
fraud_pca.head()
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>pca11</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.985811</td>
      <td>1.769162</td>
      <td>-0.450567</td>
      <td>-1.204763</td>
      <td>-0.117952</td>
      <td>0.913919</td>
      <td>-0.725541</td>
      <td>1.175108</td>
      <td>1.759004</td>
      <td>-0.537196</td>
      <td>0.074733</td>
      <td>1.481429</td>
      <td>-0.810912</td>
      <td>0.935328</td>
      <td>0.249804</td>
      <td>-0.067922</td>
      <td>-0.263015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.436214</td>
      <td>-0.073470</td>
      <td>0.229321</td>
      <td>-0.830563</td>
      <td>-0.100201</td>
      <td>2.323481</td>
      <td>0.927359</td>
      <td>-0.936612</td>
      <td>-0.827647</td>
      <td>-0.064329</td>
      <td>0.187772</td>
      <td>-1.227025</td>
      <td>-0.698201</td>
      <td>-1.541975</td>
      <td>-1.348272</td>
      <td>-0.053672</td>
      <td>0.511195</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.584369</td>
      <td>2.080913</td>
      <td>-0.033197</td>
      <td>0.575900</td>
      <td>-0.870996</td>
      <td>0.344936</td>
      <td>0.538653</td>
      <td>-0.454442</td>
      <td>-1.623096</td>
      <td>-1.124090</td>
      <td>0.150369</td>
      <td>0.294092</td>
      <td>2.110493</td>
      <td>0.275672</td>
      <td>-1.110919</td>
      <td>0.196495</td>
      <td>0.736336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.245341</td>
      <td>0.854789</td>
      <td>-0.846922</td>
      <td>-0.526426</td>
      <td>-1.109341</td>
      <td>-0.393849</td>
      <td>-0.343671</td>
      <td>-1.335815</td>
      <td>0.108946</td>
      <td>0.625671</td>
      <td>-1.456979</td>
      <td>0.262265</td>
      <td>-0.262356</td>
      <td>-1.130105</td>
      <td>0.032108</td>
      <td>-0.351500</td>
      <td>0.061635</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.589848</td>
      <td>-0.145288</td>
      <td>1.091001</td>
      <td>0.032348</td>
      <td>0.029693</td>
      <td>1.672321</td>
      <td>-1.662417</td>
      <td>-0.280590</td>
      <td>0.326410</td>
      <td>-0.920124</td>
      <td>1.185936</td>
      <td>0.362399</td>
      <td>-0.696854</td>
      <td>-0.764018</td>
      <td>0.425946</td>
      <td>0.366670</td>
      <td>-0.204808</td>
    </tr>
  </tbody>
</table>
</div>



> **Notes**: Setelah dipilih PC yang merangkum informasi yang dibutuhkan, PC dapat digabung dengan data awal dan digunakan untuk analisis lebih lanjut (misal: supervised learning).

Cara yang dilakukan di atas adalah cara manual, sebenarnya kita bisa secara langsung melakukan reduksi dimensi ketika membuat objek PCA yaitu menuliskan proporsi informasi yang ingin dipertahankan pada parameter `n_components`.

Kekurangan dari cara ini adalah kita tidak bisa melakukan detransform ke bentuk awal karena adanya informasi yang hilang.


```python
pca2 = PCA(n_components = 0.9, # gunakan proporsi data
          svd_solver='full')
pca2.fit(fraud_scaled.values)

fraud_pca90 = pd.DataFrame(pca2.fit_transform(fraud_scaled), 
                          columns=pca2.get_feature_names_out())

fraud_pca90.head()
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>pca11</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.156565</td>
      <td>-2.117836</td>
      <td>0.617836</td>
      <td>-0.297799</td>
      <td>0.133543</td>
      <td>0.108565</td>
      <td>-0.227080</td>
      <td>1.645738</td>
      <td>-0.358602</td>
      <td>-1.415333</td>
      <td>-0.624414</td>
      <td>0.677698</td>
      <td>-0.010943</td>
      <td>-0.166878</td>
      <td>1.618416</td>
      <td>-1.197033</td>
      <td>0.566608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.747659</td>
      <td>-0.773444</td>
      <td>-0.553069</td>
      <td>-0.479412</td>
      <td>-1.855533</td>
      <td>1.732876</td>
      <td>0.525852</td>
      <td>-0.714205</td>
      <td>-0.032657</td>
      <td>0.495306</td>
      <td>0.829078</td>
      <td>0.269277</td>
      <td>-1.283505</td>
      <td>-0.849027</td>
      <td>-1.081769</td>
      <td>0.575271</td>
      <td>0.610764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.591602</td>
      <td>-1.943262</td>
      <td>-1.016471</td>
      <td>0.348008</td>
      <td>-1.371603</td>
      <td>0.261071</td>
      <td>-0.084554</td>
      <td>-1.175479</td>
      <td>0.130690</td>
      <td>0.395665</td>
      <td>-0.344746</td>
      <td>0.218384</td>
      <td>1.778190</td>
      <td>-0.175649</td>
      <td>-0.805700</td>
      <td>0.712865</td>
      <td>-0.890682</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.170206</td>
      <td>-1.354921</td>
      <td>0.214172</td>
      <td>0.544212</td>
      <td>-0.430262</td>
      <td>-1.025455</td>
      <td>0.577854</td>
      <td>0.157622</td>
      <td>-0.686719</td>
      <td>-0.321691</td>
      <td>0.114822</td>
      <td>-0.708565</td>
      <td>-0.210811</td>
      <td>0.218148</td>
      <td>-1.013061</td>
      <td>0.797674</td>
      <td>0.574951</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.863529</td>
      <td>-0.685988</td>
      <td>-1.073506</td>
      <td>0.277979</td>
      <td>0.633735</td>
      <td>1.345577</td>
      <td>-0.185201</td>
      <td>-0.179568</td>
      <td>0.056330</td>
      <td>-0.486255</td>
      <td>-0.124916</td>
      <td>0.362843</td>
      <td>-1.208754</td>
      <td>-2.206490</td>
      <td>0.492851</td>
      <td>-0.771684</td>
      <td>0.585018</td>
    </tr>
  </tbody>
</table>
</div>



**[optional] Detransform PCA**

Mengembalikan hasil reduksi dimensi menjadi data bentuk aslinya. Tetapi hal ini hanya bisa dilakukan pada data hasil PCA yang masih lengkap.


```python
pd.DataFrame(pca.inverse_transform(transform_)).head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.617461</td>
      <td>-1.406485</td>
      <td>-0.463121</td>
      <td>-0.353717</td>
      <td>-0.183422</td>
      <td>1.914726</td>
      <td>-0.109015</td>
      <td>-0.159213</td>
      <td>-0.130894</td>
      <td>1.064934</td>
      <td>...</td>
      <td>-1.087580</td>
      <td>-0.817867</td>
      <td>0.363856</td>
      <td>-0.520999</td>
      <td>-0.68135</td>
      <td>-0.172179</td>
      <td>-0.301415</td>
      <td>0.887976</td>
      <td>-0.149544</td>
      <td>0.319350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.129349</td>
      <td>1.401910</td>
      <td>-0.316152</td>
      <td>-1.161812</td>
      <td>-0.180966</td>
      <td>1.279767</td>
      <td>0.831844</td>
      <td>1.856551</td>
      <td>0.530621</td>
      <td>-0.367800</td>
      <td>...</td>
      <td>0.919473</td>
      <td>-0.817867</td>
      <td>0.363856</td>
      <td>1.919391</td>
      <td>-0.68135</td>
      <td>-0.172179</td>
      <td>-0.456308</td>
      <td>-1.126157</td>
      <td>-0.149544</td>
      <td>-1.036425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.099295</td>
      <td>-0.377142</td>
      <td>-0.214405</td>
      <td>-0.353717</td>
      <td>0.947517</td>
      <td>-1.166930</td>
      <td>0.664322</td>
      <td>1.077426</td>
      <td>0.628074</td>
      <td>-0.391428</td>
      <td>...</td>
      <td>0.919473</td>
      <td>-0.817867</td>
      <td>0.363856</td>
      <td>-0.520999</td>
      <td>-0.68135</td>
      <td>-0.172179</td>
      <td>-0.471254</td>
      <td>-1.126157</td>
      <td>-0.149544</td>
      <td>-1.036425</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.129349</td>
      <td>-0.273965</td>
      <td>-0.282236</td>
      <td>-0.353717</td>
      <td>-0.183795</td>
      <td>-0.792364</td>
      <td>0.270014</td>
      <td>-0.921763</td>
      <td>-0.653084</td>
      <td>-0.391428</td>
      <td>...</td>
      <td>0.919473</td>
      <td>-0.817867</td>
      <td>0.363856</td>
      <td>-0.520999</td>
      <td>-0.68135</td>
      <td>-0.172179</td>
      <td>-0.543402</td>
      <td>0.887976</td>
      <td>-0.149544</td>
      <td>0.771275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.099295</td>
      <td>0.815273</td>
      <td>-0.881417</td>
      <td>-1.161812</td>
      <td>-0.178846</td>
      <td>2.960306</td>
      <td>0.731904</td>
      <td>0.278006</td>
      <td>1.180475</td>
      <td>-0.395724</td>
      <td>...</td>
      <td>-1.087580</td>
      <td>-0.817867</td>
      <td>0.363856</td>
      <td>-0.520999</td>
      <td>-0.68135</td>
      <td>-0.172179</td>
      <td>0.831392</td>
      <td>-1.126157</td>
      <td>-0.149544</td>
      <td>-1.488350</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python

```
**Contoh aplikasi PCA (bahasa pemrograman R):**

- sebagai metode untuk mengurangi multikolinearitas: [rpubs](https://rpubs.com/tomytjandra/PCA-reduce-multicollinearity)
- sebagai input untuk model klasifikasi: [rpubs](https://rpubs.com/tomytjandra/PCA-before-classification)
Mari kita coba bandingkan bagaimana kondisi covariance data kita sebelum discaling, sesudah scaling, dan setelah menjadi bentuk PCA. Silakan jalankan kode berikut ini.


```python
# alternatif menggunakan seaborn heatmap, sebelum dilakukan scaled

plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(fraud_num.cov().round(2), vmin=-1, vmax=1, annot=True, cmap='YlGnBu', 
            annot_kws={"size": 5, "color":'white', "alpha":0.7, "ha": 'center', "va": 'center'});
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_66_0.png)
    



```python
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(fraud_scaled.cov().round(2), vmin=-1, vmax=1, annot=True, cmap='YlGnBu',
            annot_kws={"size": 5, "color":'white', "alpha":0.7, "ha": 'center', "va": 'center'});
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_67_0.png)
    



```python
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(fraud_pca90.cov().round(2), vmin=-1, vmax=1, annot=True, cmap='YlGnBu', 
            annot_kws={"size": 5, "color":'white', "alpha":0.7, "ha": 'center', "va": 'center'});
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_68_0.png)
    


## Visualizing PCA

PCA tidak hanya berguna untuk dimensionality reduction namun baik untuk visualisasi high-dimensional data. Visualisasi dapat menggunakan **biplot** yang menampilkan:

1. **Individual factor map**, yaitu sebaran data secara keseluruhan menggunakan 2 PC. Tujuannya untuk:
  - observasi yang serupa
  - outlier dari keseluruhan data
2. **Variables factor map**, yaitu plot yang menunjukkan korelasi antar variable dan kontribusinya terhadap PC.

### Biplot Visualization

Kita akan menggunakan fungsi custom dari helper yaitu `biplot_pca`.


```python
# method dari helper.py
biplot_pca(fraud_scaled.head(50))
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_71_0.png)
    


Keterangan:

- **Titik/poin observasi:**
    + index angka dari observasi.
    + Semakin berdekatan maka karakteristiknya semakin mirip, sedangkan yang jauh dari gerombolan data dianggap sebagai outlier
    
- **Garis vektor:**
    + loading score, menunjukkan kontribusi variabel tersebut terhadap PC, atau banyaknya informasi variabel tersebut yang dirangkum oleh PC.
    + Semakin jauh panah, semakin banyak informasi yang dirangkum.

Visualisasi biplot (loadings) menggunakan library [plotly](https://plotly.com/python/pca-visualization/#visualize-loadings). Fungsi ini merupakan fungsi custom yang dapat dilihat pada file `helper.py`.


```python
biplot_plotly(fraud_scaled, pca)
```


<div>                            <div id="5d3b2c34-1402-4724-bf1e-87eb9d4a10e3" class="plotly-graph-div" style="height:600px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5d3b2c34-1402-4724-bf1e-87eb9d4a10e3")) {                    Plotly.newPlot(                        "5d3b2c34-1402-4724-bf1e-87eb9d4a10e3",                        [{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003e0=%{x}\u003cbr\u003e1=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0],"legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[0.985810708157095,2.436214299580105,2.5843686129589556,-1.2453407940374848,2.58984845567195,-0.6268959248913177,2.011817197047648,0.10187830806935487,2.2259455495047478,2.7525365131062873,-1.384309933981842,-0.05650429191560825,0.4420258024282138,-2.629845403256909,0.7953522513924806,0.48095985375333355,-1.4881954007413138,-1.2913650393057494,0.21632666530498618,2.035531165065254,-2.7243247685281102,-0.45957070672879186,1.5637537281384757,-3.7455728018921137,3.5940637125857937,-2.9171691277160057,1.267474346951974,2.319067786526711,1.1499998628313701,1.3972022187727946,1.628503691501174,0.22698418990016997,0.5272581824009106,-1.0288710453588672,-2.959142174467093,-0.45892956708047344,3.076727218749205,-1.5503359523312334,-3.2185561317485165,-0.8573673108801645,-0.9386005042964849,1.5421897461435927,1.0168875475730288,-2.151350236783365,-1.0728526430536245,-1.719235641550976,1.1005891648266337,2.6158018880351097,0.37873269045092156,3.939739257989442,1.0672995745728497,1.4628046182460281,-0.22807763031296666,1.3879779825990806,2.48048436787781,0.9252674124954077,-1.4321189925491173,-1.810393933372328,-2.84809734381564,-0.2571028285595112,1.0004317591918839,-2.8985023646505965,1.1444496607240358,-2.7888711020029993,-0.7226850060340358,2.932872621353717,-1.0404554537470496,0.6909837717835736,-2.9561579572728194,-2.7745857783610286,-3.575086426251834,2.865320689233716,-2.2154956595035795,3.4635332310419034,0.8277063000995794,1.0752011634943985,-0.48346975083528565,-0.35988207688910345,1.287653620695373,1.5683114573386296,0.5988930205560746,-0.7241152505061325,2.08458553123725,-1.3770331613273714,-3.58938528003171,-2.676276182393563,-3.1052559617316704,-0.8409735197110941,-0.5942000346083995,2.5958921628621474,-2.447034718596211,0.39208699075885456,0.02427508004868455,1.13750168789427,-0.7830951909567686,-1.7865922657891025,-0.9937184275282259,1.4271665254895147,0.5388875575806069,-0.15017970470891934],"xaxis":"x","y":[1.769162265068691,-0.07346951480876397,2.080913350255794,0.8547889684880439,-0.14528786722985107,-0.6395979476658106,-1.0946650904995339,1.2340879311914594,0.005821279624604247,-2.4575154352189954,0.3627689843204974,0.7943144266955282,1.8134652555031434,2.220278832932891,-0.07695677462059357,-0.6253242092584836,1.3192317685001376,0.6479984686305873,1.723657966352754,0.2874084376325079,0.20499746676957156,1.3760603816651693,-1.829583514809756,-0.05047722547260783,0.24928484022312986,0.8347300238519798,0.9221207024807896,-0.10383470893896511,0.9754418953140004,0.445471572447048,0.7520887108146855,-2.155370956174571,-2.98532536891007,-0.7353167030726925,2.479969402039471,-2.1740518013973946,0.5971607285524939,-1.3233083700475698,0.07447538840280071,-1.3417007933277134,-0.004129879824132912,2.875965631475363,-0.7587870465966822,0.32320664459403803,-1.5153929944503541,-0.12649609415159188,-0.568507643852495,-1.598045440806501,-0.9517514684142007,-2.833755485130499,-0.7033476136066129,-0.24592361710877375,0.31141005945498496,-1.0516621755217395,-2.9894438172881497,0.04997412545357597,-1.5093750142726998,0.794999338560007,-2.2943235982077956,0.9397737049248944,0.12781754295951597,1.4167257551463193,0.10357790836636936,-1.3119333542952087,1.284865501529103,1.330151285133971,-0.2771160924677139,0.06908848610787748,-0.9239692990491776,-2.4196133293071256,0.5634743814741769,-1.1801434085985771,1.3056701974398304,6.979070501398151,-0.1840078664263021,-1.0042680586584494,-0.5986967253476728,1.2897733171589107,0.3340643119640657,0.5867394389653157,0.9463141489529396,-0.23484973601700382,1.5514302091842174,2.2214928217140746,-0.7578112260649845,1.8736720882459195,-0.8429087422506792,0.5661418796885872,-1.4223698615140745,0.487511667467078,1.779535365850745,-2.0545913593103204,-0.19924938827358255,-2.2027251230125335,1.346320075342313,-2.7023051776852487,-1.4335537678779586,1.6785183454162733,-2.849297910846424,0.39915481596223656],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"PC1"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"PC2"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"shapes":[{"type":"line","x0":0,"x1":0.2014617650165673,"y0":0,"y1":0.7218047814855328},{"type":"line","x0":0,"x1":0.3372675680438581,"y0":0,"y1":-0.32069118563398485},{"type":"line","x0":0,"x1":-0.8353291557120788,"y0":0,"y1":0.1810946532047793}],"annotations":[{"ax":0,"ay":0,"text":"days_since_request","x":0.2014617650165673,"xanchor":"center","y":0.7218047814855328,"yanchor":"bottom"},{"arrowcolor":"black","arrowhead":4,"arrowsize":1.5,"arrowwidth":1,"ax":0,"axref":"x","ay":0,"ayref":"y","showarrow":true,"text":"","x":0.2014617650165673,"xref":"x","y":0.7218047814855328,"yref":"y"},{"ax":0,"ay":0,"text":"session_length_in_minutes","x":0.3372675680438581,"xanchor":"center","y":-0.32069118563398485,"yanchor":"bottom"},{"arrowcolor":"black","arrowhead":4,"arrowsize":1.5,"arrowwidth":1,"ax":0,"axref":"x","ay":0,"ayref":"y","showarrow":true,"text":"","x":0.3372675680438581,"xref":"x","y":-0.32069118563398485,"yref":"y"},{"ax":0,"ay":0,"text":"month","x":-0.8353291557120788,"xanchor":"center","y":0.1810946532047793,"yanchor":"bottom"},{"arrowcolor":"black","arrowhead":4,"arrowsize":1.5,"arrowwidth":1,"ax":0,"axref":"x","ay":0,"ayref":"y","showarrow":true,"text":"","x":-0.8353291557120788,"xref":"x","y":0.1810946532047793,"yref":"y"}],"title":{"text":"Biplot PCA","x":0.5},"width":800,"height":600},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('5d3b2c34-1402-4724-bf1e-87eb9d4a10e3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### Individual

1. **Outlier detection**: observasi yang jauh dari kumpulan observasi lainnya mengindikasikan outlier dari keseluruhan data. Observasi ini dapat ditandai untuk nantinya dicek karakteristik datanya untuk keperluan bisnis, atau apakah mempengaruhi performa model, dll.


2. **Observasi searah panah** mengindikasikan observasi tersebut nilainya tinggi pada variabel tersebut. Bila bertolak belakang, maka nilainya rendah pada variable tersebut.


3. **Observasi berdekatan**: observasi yang saling berdekatan memiliki karakteristik yang mirip.


```python

```

####  Variable

**Korelasi antar variabel** dapat dilihat dari sudut antar panah: 

- Panah saling berdekatan (sudut antar panah < 90), maka korelasi positif
- Panah saling tegak lurus (sudut antar panah = 90), maka tidak berkorelasi
- Panah saling bertolak belakang (sudut antar panah mendekati 180), maka korelasi negatif

**Variable Importance**

Selain melihat berdasarkan variable factor map, kita juga dapat memetakan 


```python
# Dapatkan loadings dari PCA
loadings = pca.components_

# Buat dataframe untuk loadings
loadings_df = pd.DataFrame(data=loadings.T, 
                           columns=pca.get_feature_names_out())

# Tambahkan kolom nama variabel
loadings_df['Variable'] = fraud_scaled.columns

# Tampilkan loadings yang signifikan (misalnya, absolute loadings > 0.3)
significant_loadings = loadings_df[abs(loadings_df['pca0']) > 0.2]
significant_loadings
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>...</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>pca17</th>
      <th>pca18</th>
      <th>pca19</th>
      <th>pca20</th>
      <th>pca21</th>
      <th>Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.254536</td>
      <td>-0.184049</td>
      <td>-0.052353</td>
      <td>0.048724</td>
      <td>0.181880</td>
      <td>0.553324</td>
      <td>-0.406133</td>
      <td>0.319775</td>
      <td>0.330375</td>
      <td>0.055199</td>
      <td>...</td>
      <td>-0.171046</td>
      <td>-0.024820</td>
      <td>0.163752</td>
      <td>0.043083</td>
      <td>-0.031187</td>
      <td>-0.092707</td>
      <td>0.033165</td>
      <td>0.092057</td>
      <td>-0.016102</td>
      <td>(zip_count_4w,)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.278197</td>
      <td>-0.167477</td>
      <td>-0.065016</td>
      <td>-0.001018</td>
      <td>0.019048</td>
      <td>-0.250079</td>
      <td>0.144120</td>
      <td>-0.034537</td>
      <td>-0.181518</td>
      <td>-0.292714</td>
      <td>...</td>
      <td>-0.173220</td>
      <td>0.020450</td>
      <td>-0.039390</td>
      <td>0.539321</td>
      <td>-0.439740</td>
      <td>-0.138476</td>
      <td>0.152361</td>
      <td>0.048419</td>
      <td>-0.144075</td>
      <td>(velocity_6h,)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.418412</td>
      <td>-0.175674</td>
      <td>0.032164</td>
      <td>-0.015764</td>
      <td>0.094908</td>
      <td>-0.016151</td>
      <td>0.174177</td>
      <td>0.001748</td>
      <td>-0.106125</td>
      <td>0.004612</td>
      <td>...</td>
      <td>0.065990</td>
      <td>0.036640</td>
      <td>-0.163368</td>
      <td>0.214036</td>
      <td>0.616769</td>
      <td>0.294909</td>
      <td>-0.002323</td>
      <td>-0.278603</td>
      <td>0.064113</td>
      <td>(velocity_24h,)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.422678</td>
      <td>-0.161588</td>
      <td>-0.076163</td>
      <td>-0.011424</td>
      <td>0.126181</td>
      <td>-0.093132</td>
      <td>0.015535</td>
      <td>-0.075685</td>
      <td>-0.067948</td>
      <td>-0.070653</td>
      <td>...</td>
      <td>0.148771</td>
      <td>0.146961</td>
      <td>0.088360</td>
      <td>-0.470691</td>
      <td>0.017705</td>
      <td>-0.203896</td>
      <td>0.094607</td>
      <td>0.063541</td>
      <td>-0.619253</td>
      <td>(velocity_4w,)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.328022</td>
      <td>0.285483</td>
      <td>0.169881</td>
      <td>-0.030157</td>
      <td>-0.058191</td>
      <td>0.270602</td>
      <td>-0.033609</td>
      <td>0.131475</td>
      <td>-0.202543</td>
      <td>-0.016625</td>
      <td>...</td>
      <td>0.330536</td>
      <td>-0.206888</td>
      <td>0.101545</td>
      <td>0.303949</td>
      <td>0.032425</td>
      <td>0.101370</td>
      <td>-0.003186</td>
      <td>0.017381</td>
      <td>-0.069864</td>
      <td>(date_of_birth_distinct_emails_4w,)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.264609</td>
      <td>-0.239203</td>
      <td>0.203804</td>
      <td>0.202757</td>
      <td>0.252504</td>
      <td>-0.128731</td>
      <td>-0.084717</td>
      <td>0.187801</td>
      <td>-0.378717</td>
      <td>0.071309</td>
      <td>...</td>
      <td>0.024587</td>
      <td>0.228687</td>
      <td>0.114639</td>
      <td>0.105978</td>
      <td>0.391262</td>
      <td>-0.294474</td>
      <td>0.002183</td>
      <td>0.367030</td>
      <td>0.034964</td>
      <td>(credit_risk_score,)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.437990</td>
      <td>0.118262</td>
      <td>0.077484</td>
      <td>-0.032829</td>
      <td>-0.126963</td>
      <td>0.036287</td>
      <td>-0.031599</td>
      <td>0.127081</td>
      <td>0.100576</td>
      <td>0.135952</td>
      <td>...</td>
      <td>0.024086</td>
      <td>-0.055217</td>
      <td>-0.087107</td>
      <td>0.224116</td>
      <td>0.141136</td>
      <td>0.149134</td>
      <td>0.211955</td>
      <td>-0.169378</td>
      <td>-0.702388</td>
      <td>(month,)</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 23 columns</p>
</div>



## Pros and Cons PCA

Kelebihan melakukan PCA:

- Beban komputasi apabila dilakukan pemodelan relatif lebih rendah
- Bisa jadi salah satu teknik untuk improve model, namun tidak selalu menjadi lebih baik (Untuk kasus overfitting data)
- Mengurangi resiko terjadinya multikolinearitas, karena nilai antar PC sudah tidak saling berkorelasi

Kekurangan melakukan PCA (sebelum pemodelan):

- Model tidak dapat diinterpretasikan, karena nilai PC merupakan campuran dari beberapa variabel

# Anomaly Detection

## Local Outlier Factor with PyOD

**Local Outlier Factor** (LOF) merupakan salah satu algoritma umum yang digunakan untuk kasus anomaly detection. Teknik ini bekerja dengan menghitung skor berdasarkan kepadatan data berdasarkan jaraknya (sangat mirip dengan konsep k-NN). 

LOF dapat menjadi pilihan yang baik untuk deteksi fraud dalam menentukan anomali data, berikut adalah beberapa kelebihan dan kekurangan dari metode ini.

**Pros**

- Efektif dalam menemukan outlier lokal: LOF dapat mengidentifikasi outlier yang tidak dapat ditemukan oleh metode global, seperti outlier yang berada di dalam cluster yang padat.
- Tidak sensitif terhadap distribusi data: LOF dapat bekerja dengan baik pada data dengan distribusi yang tidak normal.
- Mudah diimplementasikan: LOF dapat diimplementasikan dengan mudah menggunakan library Python seperti Pyod.

**Cons**

- Dapat menjadi lambat untuk data yang besar: LOF memerlukan komputasi yang cukup berat untuk dataset yang besar.
- Memerlukan pemilihan parameter yang tepat: Parameter k (jumlah tetangga terdekat) yang digunakan dalam LOF dapat mempengaruhi hasil deteksi outlier.

Secara sederhana, LOF akan menghitung jarak antar data dan data yang secara kumpulan lokal terisolasi akan didefinisikan sebagai outlier oleh LOF. Berikut adalah ilustrasi sederhana dari kumpulan data dalam ruang 2 dimensi secara lokal.

![LOF2](assets/lof2.jpg)

Pada ilustrasi di atas, C1 dan C2 merupakan kumpulan data lokal. Titik yang diperhatikan adalah O1, O2, O3, dan O4. 

Pada kasus kita ini O1 dan O2 dapat dianggap sebagai outlier lokal untuk kelompok C1. Sementara O4 kemungkinan bukan merupakan outlier untuk kelompok C2 karena rentang jarak per data di kelompok C2 cukup renggang/tidak sepadat C1. Sementara O3 dapat dikatakan sebagai outlier global.

Kita akan menggunakan data hasil PCA yaitu `fraud_pca90` untuk mencoba metode ini.


```python
fraud_pca90.head(3)
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>pca11</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.156565</td>
      <td>-2.117836</td>
      <td>0.617836</td>
      <td>-0.297799</td>
      <td>0.133543</td>
      <td>0.108565</td>
      <td>-0.227080</td>
      <td>1.645738</td>
      <td>-0.358602</td>
      <td>-1.415333</td>
      <td>-0.624414</td>
      <td>0.677698</td>
      <td>-0.010943</td>
      <td>-0.166878</td>
      <td>1.618416</td>
      <td>-1.197033</td>
      <td>0.566608</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.747659</td>
      <td>-0.773444</td>
      <td>-0.553069</td>
      <td>-0.479412</td>
      <td>-1.855533</td>
      <td>1.732876</td>
      <td>0.525852</td>
      <td>-0.714205</td>
      <td>-0.032657</td>
      <td>0.495306</td>
      <td>0.829078</td>
      <td>0.269277</td>
      <td>-1.283505</td>
      <td>-0.849027</td>
      <td>-1.081769</td>
      <td>0.575271</td>
      <td>0.610764</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.591602</td>
      <td>-1.943262</td>
      <td>-1.016471</td>
      <td>0.348008</td>
      <td>-1.371603</td>
      <td>0.261071</td>
      <td>-0.084554</td>
      <td>-1.175479</td>
      <td>0.130690</td>
      <td>0.395665</td>
      <td>-0.344746</td>
      <td>0.218384</td>
      <td>1.778190</td>
      <td>-0.175649</td>
      <td>-0.805700</td>
      <td>0.712865</td>
      <td>-0.890682</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




Fungsi `LOF()` dapat digunakan setelah mengakses modul `model.lof` dari library `pyod`.


```python
from sklearn.neighbors import LocalOutlierFactor
lof_model = LOF()
lof_model2 = LocalOutlierFactor(contamination=0.1,n_jobs=1, novelty=True )
```


```python
fraud_pca90
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
      <th>pca0</th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>pca11</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.156565</td>
      <td>-2.117836</td>
      <td>0.617836</td>
      <td>-0.297799</td>
      <td>0.133543</td>
      <td>0.108565</td>
      <td>-0.227080</td>
      <td>1.645738</td>
      <td>-0.358602</td>
      <td>-1.415333</td>
      <td>-0.624414</td>
      <td>0.677698</td>
      <td>-0.010943</td>
      <td>-0.166878</td>
      <td>1.618416</td>
      <td>-1.197033</td>
      <td>0.566608</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.747659</td>
      <td>-0.773444</td>
      <td>-0.553069</td>
      <td>-0.479412</td>
      <td>-1.855533</td>
      <td>1.732876</td>
      <td>0.525852</td>
      <td>-0.714205</td>
      <td>-0.032657</td>
      <td>0.495306</td>
      <td>0.829078</td>
      <td>0.269277</td>
      <td>-1.283505</td>
      <td>-0.849027</td>
      <td>-1.081769</td>
      <td>0.575271</td>
      <td>0.610764</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.591602</td>
      <td>-1.943262</td>
      <td>-1.016471</td>
      <td>0.348008</td>
      <td>-1.371603</td>
      <td>0.261071</td>
      <td>-0.084554</td>
      <td>-1.175479</td>
      <td>0.130690</td>
      <td>0.395665</td>
      <td>-0.344746</td>
      <td>0.218384</td>
      <td>1.778190</td>
      <td>-0.175649</td>
      <td>-0.805700</td>
      <td>0.712865</td>
      <td>-0.890682</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.170206</td>
      <td>-1.354921</td>
      <td>0.214172</td>
      <td>0.544212</td>
      <td>-0.430262</td>
      <td>-1.025455</td>
      <td>0.577854</td>
      <td>0.157622</td>
      <td>-0.686719</td>
      <td>-0.321691</td>
      <td>0.114822</td>
      <td>-0.708565</td>
      <td>-0.210811</td>
      <td>0.218148</td>
      <td>-1.013061</td>
      <td>0.797674</td>
      <td>0.574951</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.863529</td>
      <td>-0.685988</td>
      <td>-1.073506</td>
      <td>0.277979</td>
      <td>0.633735</td>
      <td>1.345577</td>
      <td>-0.185201</td>
      <td>-0.179568</td>
      <td>0.056330</td>
      <td>-0.486255</td>
      <td>-0.124916</td>
      <td>0.362843</td>
      <td>-1.208754</td>
      <td>-2.206490</td>
      <td>0.492851</td>
      <td>-0.771684</td>
      <td>0.585018</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14895</th>
      <td>-3.006352</td>
      <td>-2.208084</td>
      <td>-0.244070</td>
      <td>1.140636</td>
      <td>-1.137031</td>
      <td>0.261961</td>
      <td>-0.182638</td>
      <td>-0.192580</td>
      <td>-0.584322</td>
      <td>-0.187782</td>
      <td>-0.009381</td>
      <td>-0.214395</td>
      <td>0.692088</td>
      <td>-0.435719</td>
      <td>-1.041518</td>
      <td>0.198371</td>
      <td>0.314340</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14896</th>
      <td>-0.500000</td>
      <td>-1.009571</td>
      <td>-0.074096</td>
      <td>-0.329787</td>
      <td>0.704215</td>
      <td>1.900702</td>
      <td>1.150522</td>
      <td>2.356263</td>
      <td>0.672031</td>
      <td>1.162215</td>
      <td>-1.075699</td>
      <td>0.109144</td>
      <td>0.877142</td>
      <td>-0.504097</td>
      <td>-0.403844</td>
      <td>-1.671751</td>
      <td>0.931399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14897</th>
      <td>1.321106</td>
      <td>-0.978755</td>
      <td>-0.035949</td>
      <td>-0.577802</td>
      <td>0.821491</td>
      <td>-0.492155</td>
      <td>0.222857</td>
      <td>-0.138077</td>
      <td>-0.258987</td>
      <td>-0.652570</td>
      <td>-0.021578</td>
      <td>-0.929279</td>
      <td>0.012119</td>
      <td>0.839235</td>
      <td>0.426793</td>
      <td>-1.340530</td>
      <td>-0.576510</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14898</th>
      <td>-0.064242</td>
      <td>0.870179</td>
      <td>1.870764</td>
      <td>3.899863</td>
      <td>0.642990</td>
      <td>-2.117461</td>
      <td>3.541267</td>
      <td>1.985717</td>
      <td>0.629173</td>
      <td>1.760697</td>
      <td>0.402671</td>
      <td>0.632329</td>
      <td>0.249250</td>
      <td>-0.006183</td>
      <td>1.404148</td>
      <td>-0.428410</td>
      <td>-0.790182</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14899</th>
      <td>-1.084422</td>
      <td>0.650284</td>
      <td>2.226155</td>
      <td>-0.978600</td>
      <td>-0.422600</td>
      <td>2.157075</td>
      <td>1.396626</td>
      <td>2.711047</td>
      <td>0.905026</td>
      <td>1.918226</td>
      <td>-0.180522</td>
      <td>0.340268</td>
      <td>-0.696036</td>
      <td>0.352001</td>
      <td>0.614355</td>
      <td>-2.447711</td>
      <td>-1.459189</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>14900 rows × 18 columns</p>
</div>



Objek LOF di atas dapat langsung kita gunakan kepada data yang sudah kita olah sebelumnya menggunakan method `fit_predict()`.


```python
lof_label = lof_model.fit_predict(fraud_pca90)
lof_model2.fit(fraud_pca90)

lof_label2 = lof_model2.predict()
```

Karena merupakan proses unsupervised, maka metode fit_predict akan langsung menghasilkan label. Tetapi sebenarnya terdapat skor anomali untuk setiap data yang dimasukkan ke model. Skor anomali ini dapat dilihat menggunakan method `decision_function()`.


```python
# Menghitung nilai LOF
lof_scores = lof_model.decision_function(fraud_pca90)

lof_scores
```




    array([1.18030495, 1.11967908, 1.19635871, ..., 0.98605162, 1.54397575,
           1.12362949])




```python
# Menghitung nilai LOF
lof_scores2 = lof_model2.decision_function(fraud_pca90)

lof_scores2
```

    C:\Users\SaltFarmer\miniconda3\envs\algoritma\lib\site-packages\sklearn\base.py:465: UserWarning:
    
    X does not have valid feature names, but LocalOutlierFactor was fitted with feature names
    
    




    array([ 0.00680207,  0.06742794, -0.00925168, ...,  0.2010554 ,
           -0.35686873,  0.06347754])



Karena merupakan skor setiap data, maka untuk lebih jelasnya kita bisa lihat distribusinya menggunakan histogram ataupun boxplot.


```python
sns.histplot(lof_scores)
```




    <Axes: ylabel='Count'>




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_93_1.png)
    



```python
sns.histplot(lof_scores2)
```




    <Axes: ylabel='Count'>




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_94_1.png)
    



```python
sns.boxplot(lof_scores, orient='h',)
```




    <Axes: >




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_95_1.png)
    



```python
sns.boxplot(lof_scores2, orient='h',)
```




    <Axes: >




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_96_1.png)
    


Sementara untuk label, kita dapat dengan mudah menghitung masing-masing hasil label menggunakan `value_counts()`.


```python
pd.Series(lof_label).value_counts()
```




    0    13410
    1     1490
    Name: count, dtype: int64




```python
pd.Series(lof_label2).apply(lambda x: 1 if x==-1 else 0).value_counts()
```




    0    13410
    1     1490
    Name: count, dtype: int64



## Parameter on LOF Model

Objek model LOF memiliki beberapa parameter yang dapat kita gunakan, parameter yang paling umum digunakan adalah:

- `contamination`: mengatur proporsi estimasi anomali pada data (default = 0.1)
- `n_neighbors`: jumlah tetangga yang dianggap sebagai 1 kluster (default = 20)
- `metrics`: metode perhitungan jarak yang digunakan

<!-- Selain itu kita juga dapat mengatur metode perhitungan jarak yang digunakan dengan parameter `metric`. -->

Nilai contamination ini dapat kita isi disesuaikan dengan kasus yang ada, contoh:

> Apabila kita ketahui terdapat 1% akun bank BRI merupakan akun yang digunakan untuk penipuan maka kita dapat menggunakan nilai `contamination = 0.01`.


```python
fraud.columns
```




    Index(['income', 'name_email_similarity', 'current_address_months_count',
           'customer_age', 'days_since_request', 'intended_balcon_amount',
           'payment_type', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
           'velocity_4w', 'bank_branch_count_8w',
           'date_of_birth_distinct_emails_4w', 'employment_status',
           'credit_risk_score', 'email_is_free', 'housing_status',
           'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
           'proposed_credit_limit', 'foreign_request', 'source',
           'session_length_in_minutes', 'device_os', 'keep_alive_session',
           'device_distinct_emails_8w', 'month'],
          dtype='object')




```python
lof_tune = LOF(
    contamination = 0.005,
    n_neighbors = 15
)

lof_label_tune = lof_tune.fit_predict(fraud_pca90)
```

Mari kita lihat dampak penggunaan parameter contamination dari jumlah anomali yang dideteksi oleh model kita.


```python
pd.Series(lof_label_tune).value_counts(normalize=True)
```




    0    0.994966
    1    0.005034
    Name: proportion, dtype: float64



Selain melihat plot distribusinya, kita dapat menampilkan persebaran outlier kita pada bidang 2 dimensi hasil PCA. Berikut adalah kodenya:


```python
# menampilkan plot anomali (___ diisi dengan nama dataframe PCA)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=fraud_pca90['pca0'], 
                y=fraud_pca90['pca1'], 
                hue=lof_label_tune,
                palette='coolwarm')
plt.title('Hasil Local Outlier Factor')
plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)');
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/6.%20Unsupervised%20Learning/output_106_0.png)
    


Atau untuk lebih jelasnya, kita dapat menggunakan fungsi scatter dari `plotly.express` untuk mengatur posisi legend yang ingin kita lihat.


```python
# masukkan nama dataframe PCA ke ___
fraud_pca90["color"] = lof_label_tune.astype(str)

# Plot hasil LOF menggunakan Plotly Express
fig = px.scatter(fraud_pca90.sort_values("color"), 
                 x='pca0', y='pca1', color="color",
                 color_discrete_map={'0': '#a6c4ff', '1': '#ffa07a'},
                 title='LOF Results',
                 labels={'pca0': f'PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                         'pca1': f'PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'})

# Menampilkan plot
fig.update_layout(width=800, height=600)
fig.show()
```


<div>                            <div id="d854eca2-25d4-465d-9b50-ae35e58d1be6" class="plotly-graph-div" style="height:600px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d854eca2-25d4-465d-9b50-ae35e58d1be6")) {                    Plotly.newPlot(                        "d854eca2-25d4-465d-9b50-ae35e58d1be6",                        [{"hovertemplate":"color=0\u003cbr\u003ePC 1 (15.99%)=%{x}\u003cbr\u003ePC 2 (10.31%)=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"0","marker":{"color":"#a6c4ff","symbol":"circle"},"mode":"markers","name":"0","showlegend":true,"x":[1.156565140076827,0.8291323730332444,-0.8608890720473867,-0.37151678869880833,2.9021102970979586,4.130179791712732,0.5154172649604241,1.3953117914824982,2.280479561135549,1.1839536205024503,-3.6876164754569274,-1.4715794806592208,0.06529351466830963,-3.7463047918902497,-0.33450175104422525,-2.2085861958788238,-0.05234504095112187,-1.6510285321265752,1.6886591704450997,-1.4112352405136457,-1.9859089616559662,-1.8246041022628288,-0.10393560816542739,-3.1887025822339234,0.837573557863576,-1.4977600074375355,3.6816769494949826,-0.3313330276737615,2.4506803057686786,-2.450203799964302,1.3115154560535314,-3.570490453245418,0.08683931279639953,-0.8058964838141998,-0.769414736993558,2.0577595651433285,0.24340739442507528,1.0688311865406388,-2.3084203838223383,1.235713938460101,-0.5152811658869548,2.1361747844841177,-1.1901056785167499,0.23220843537688837,0.9999199787245646,-2.673899036174976,2.9560192337064928,-0.17393528353252197,1.32610118077197,-0.176371686119227,-0.8532514696418267,0.18811576091540905,-0.4205607024918795,-2.096362391906963,-1.6631996849483353,1.7850297032509574,-3.0210550182054705,2.228134035507008,2.031667656755872,1.5205866590276202,-0.7199100636792342,1.8610158608358693,-0.431141513434749,-1.789138745214202,1.789535770084921,0.1274539482479184,0.778331190769578,-0.46206631876268456,-1.2115408521505333,-0.7554311401508572,-0.9940825893183212,0.4800445079096284,0.6874718247129367,-1.558074632873361,1.890765023148715,-2.931085892577335,-2.099468842899475,2.796860706497006,-2.4167018911421625,-0.6827165109812122,-2.3361783177843365,0.36394590566311624,-1.5583257455894948,-0.8118319786884903,0.011726397341910482,-1.1803718821007319,0.23359543559254684,-1.080765841558893,-2.1937014149035887,2.1546691981632096,0.05296963748993469,1.9433182510370504,1.6949019481815615,-1.1033070312302797,-1.9560834206987472,3.968027100540743,0.6926950136536247,-0.25924188008909194,0.29573915585230454,0.47186574080164734,-1.6553517195025385,0.8086123004074882,-0.8892221655894403,1.0487219430211652,-1.2242720789168102,-0.7612240504877515,-0.1709274311116933,-0.2978143684776611,-2.9418091706310907,2.182147332703269,3.5239721354300095,-2.906606501266806,0.21593374356956638,0.23501042903563132,0.3273218170985019,-0.8568853381534152,3.381763501418858,-0.14974096249679922,-0.10511819121513978,-2.0494784808963162,2.7477627579211505,0.4730174701454633,-0.3142464469897945,2.724292928349483,1.2224599568693812,0.5055114935898095,-0.4242699855515361,1.471315300154624,-1.6832297791639785,2.299350030536957,0.5009113326859659,-0.002790848633457556,0.33995269567606373,-4.149424110111154,-1.1967267478935228,-3.252789126086604,3.1065252495433615,1.2756725750894304,0.26657210170315604,1.387549146435443,-3.4057440779734347,1.9393049776602382,-3.186787154378191,-3.4694707214394116,-0.8796773493851409,1.1385088723062822,1.205593816724677,1.9838069748566634,-0.2589010382812506,-0.031598475820044325,-2.511717548602104,0.20208621638698904,-3.0808374158371925,1.6037945555717064,0.36850860726375084,0.7755669015331293,2.4699193129815598,0.7026765687308789,-1.9523763094200015,1.3291096569171634,-0.9948027064905721,-2.1673033801648596,0.29019928417624796,0.6549867827600735,-0.8797764551444726,2.3600924529828173,0.8797326844000254,0.8256732571748245,-0.9888937219850703,1.3120268283013137,1.1424027645310046,2.0293688681220536,-1.4539969611149308,-1.5104196881967442,-1.6182211871699552,1.7527805827972707,0.8155846891488185,-1.2971993764140441,0.6532662231559307,-0.10788675769337318,3.8666471865115546,-2.3746384151379694,1.4947105098811313,-0.9339070410110932,0.15700173361139338,1.9687464638564804,0.5871854762933495,-0.7760842229863856,-1.7992225697223907,0.1398133905763222,-3.642985263177929,-0.18794447577020182,1.7031895368600054,-0.08980588554712007,0.4926883822344106,-2.432138826148111,0.2768213303328944,2.013980503990908,-2.6891266673920997,1.165846835098337,1.8965132575943442,2.403325331365193,2.370351996801596,0.9480027200476233,3.9976842796260055,0.7452863671577354,-0.5740064716153068,0.22781100677495622,-0.9758746007866665,-2.2467239878287084,-1.0627598885207044,3.341238535364532,-2.687197509875949,1.5423716417761475,2.7097548776204903,0.7645873061902936,-0.09509080537434676,0.33855303560862804,-1.90158709128407,0.08948328983973007,2.653471229334816,-0.7051099373617361,-1.5290013257555284,0.12135005028777378,1.2045343168060645,-1.8367468218099112,-0.895366720853512,-0.7299077985632182,0.14189531738092181,-2.567594708618185,2.309081909601536,0.8397233759275162,2.846402394856196,-3.444450017243642,0.8550212000010231,0.19897162512215227,0.051101084130802665,1.6583980686033297,-1.3999777812451197,0.2980953937177641,0.4228309696253891,1.4952603254856545,3.971908959303112,0.5255684071448545,2.6097562200552122,2.232607781997722,0.2066360742573058,1.4654471523302866,-3.5781648706529,-0.49950669703706624,1.6994735750073278,0.0670739085686283,-0.9956036703885061,0.7073784844836948,-1.9793645900849055,-1.0230676314518272,-0.9914130095412668,-2.333007369224787,-0.43696204325330834,-2.195161804495845,2.1908415326906066,-0.04546577171456015,0.8983673697180274,-0.6038055096139048,-0.9349324558079725,-0.3849830339240086,-1.7658036822432084,0.6925409346344833,0.7435453364040872,0.49838777432073367,1.5094793829173676,1.6860247579562746,0.8074024371829345,-2.0635611399046656,0.25023949428113057,-1.257019115241545,-1.0481494448044317,1.1565903008837624,-0.5620935925783186,-0.5381577797399851,0.08286463436960617,1.559480396651534,1.5567778501668021,-1.6996062033365662,0.46866686977612476,3.2130477394730224,3.7091068692726954,-1.0077528088713346,-0.03469175903976599,0.348868540051783,0.4702737998012487,0.5573108281812644,-1.5517041776486324,2.109447416917359,-3.135354096288763,-1.5231977083863832,-0.8051568942036926,-0.07534547979486517,-2.431083835606214,1.8278981191388004,-0.9449348730785454,0.8687609886541473,-4.263497357247265,0.05763122035613297,-0.5650896116612462,-2.0191771303813506,1.4666942644350283,2.049340224415653,3.0788208513790805,1.7233366132220693,2.008726421325357,-1.664898146544666,-0.4178106071013907,-1.2604077018847621,0.35343926007844867,1.0649155801559433,1.2303598361472394,0.9717501263693996,-3.1437462227046624,2.1916633290695695,0.9326289155245164,-2.56122076260182,0.0004118392016812479,-0.6592596798913555,-0.3436553817618952,-1.6979132698478325,1.4713516732606968,1.5184102649136473,1.2953599789705013,0.14345178037646375,-1.6156297509205435,1.5020607176750673,2.557911124893331,1.7607821569487607,0.6252221130773858,-2.5570556604206023,1.6352005591564571,-0.8069376774556418,-0.7578401994320514,1.5003550815912565,-0.8295173445162553,1.9226295108115188,0.13842398275020731,4.605846049322562,0.663454156342941,-1.429028048554905,-0.14118583425087092,0.7427475951703723,-0.22439075021736424,-1.9512954651341152,2.2958875087212767,3.1339451208354663,-1.6368893174961712,-1.4535050115426387,-1.2479295046709622,-0.9173285931882071,-1.2980570281663824,-2.7427454007037344,-1.1245109557760606,-0.17414452269170122,-0.8634390775382172,1.1089104944421424,2.8792706157649826,2.3438867527217915,-0.6699258026942531,0.6854084861665279,-1.7636691128401687,0.8326745736847614,-3.6384399806663517,-2.1743204199462536,-2.008830183399497,-1.5004172565171139,-2.0787524032704927,-0.7521155026555325,-2.4769789847192674,2.137972130783848,-1.1158028346261044,-0.00039706533424590585,-1.2827568483885716,1.4922885706948412,-2.27875963650431,1.0840516538050884,-2.733349150159772,0.8337146516064208,-0.5163911005805826,-0.29519330315868036,-1.348423855893065,1.4588480803036277,0.9132831360425805,1.3109440914466983,2.560574525229018,-0.5519583710202374,0.6107165479207184,-2.795914828639031,0.3490181832224884,-2.7263963144006103,1.9952937320613757,-0.3156467923955339,1.8024320409840562,-0.5006572106694591,-1.784241782352648,-2.387525213567676,-0.1943537933479646,-1.2056337751717687,1.3391735542979908,0.7149155322440742,0.2876320537274888,1.0481691849619297,-0.8598381067035089,1.7073096260096712,-1.6879185073509677,-0.5424686684014629,-0.1054236316681534,-0.5739120771361967,4.123807697416654,0.02927168261811969,-1.6387074823101853,1.940150956963545,2.0538614506426933,-1.103628365323426,0.7074933028745629,-0.8895873915852738,0.8107704007015548,0.9042127988862035,-0.463746712519957,0.7816814292636133,0.6436373337597711,2.0699403252402635,-1.9826993691679942,-3.05106647819141,-2.6156177937601703,2.608907601699001,1.364389861540978,1.7844026468846672,2.3094811195521863,1.0743277890966632,0.6633756219694028,-0.25669567537664134,-1.3388468194467718,-0.2931253225464027,-0.9230983517093834,1.393938842241457,-0.20229684380220547,2.1584791878852574,-0.7325590256980948,-3.2717385739838436,1.013577067792632,-0.13458162856154302,-0.7992628244636139,0.195303531188883,-0.22480159614553205,-3.1093379855873113,0.36596638195099335,-2.3119654898518136,-2.802925674554871,0.9703454420841253,-2.3227373750769726,-2.304237387380758,0.9586557316078242,-3.9339746196411567,2.8791729815501625,-0.8466244899266164,-0.37731630033489705,-0.07732864224748702,-0.27336317031503954,-0.29742964374767433,0.9469088953925513,2.0161265468991574,1.655908291579813,1.2002401187532281,0.23230152620394728,-0.5663975082543156,0.6509235071394818,-2.0961473709277656,-3.2976784525153175,1.0956441017904166,2.377416445057894,2.2334063564529103,-0.5119765556181378,1.8623513068527477,2.6706922030602347,-3.216308486978367,-2.527853931302706,0.7148454555861095,1.6799643559621036,0.6671576069290877,2.32144584414442,-1.8588050885804155,1.057417870674277,-0.6859968184955959,2.0425186868442258,-0.6757907588406964,0.02092832628675702,0.4800731970469633,1.2215001214890129,0.8569064881159372,0.9386940372048751,-1.21872112117077,0.5262503393894448,-0.7066508765502315,2.01503299714639,2.9934808824860197,-1.203212744593035,-1.7578890158353269,-0.7139666227859021,1.7206150752993532,1.2097534527889615,-1.116059576900034,-3.342234961392945,-0.21098203114141864,-0.5907275607362822,-0.08668023309398883,-0.48676260890780476,0.21955520525232727,-2.613260784109151,-3.5573323057810633,-2.1676507516997146,-0.5482052828781676,-0.15976662279803194,-0.43347852435643663,-0.8655828717147951,2.060932151368405,-0.9857669601278976,-2.5794490321749377,2.5659986776649975,-2.892604993598012,1.8212256124869346,1.3679556506180768,1.4339085875230961,2.1244263008535484,0.25567356169406436,-1.400430879887916,-0.9652086826505236,-2.278510865048209,-3.3416553938570845,2.735481879625287,-1.3480362458345374,0.8019637421764241,-0.48580211184318844,1.4979070586063077,-0.4872000869772188,1.6629816564607658,1.8154449381635451,2.1495351966969536,1.879949918584838,-2.3029321813096097,-1.1678781657260924,1.4140398905303229,0.12485608090741171,-0.7906128957711996,-0.9463520778770637,-2.2455949835066584,-0.797023840476356,-0.12722618844372072,1.273689889026915,-0.25313088035719716,-1.629688668148046,0.19749162432038966,-3.729290896361309,1.564420185318126,-2.7418513339122086,0.42984126766446334,1.5516706904655015,-0.6887970668682045,0.9936835134054491,1.1563416302109735,-0.7969568446969348,1.824532090119909,-1.8704993538517867,0.6223409643263984,-1.1644788331630007,2.57356741203888,0.6893325132788332,-0.5106549398137672,0.6842768948901834,-0.5127108153877986,2.2545749654696303,-1.1433370217293992,-1.1005123560154535,2.8999767532115683,-0.07238603219237999,2.5050307046143105,1.1605074037722602,0.6971313729146997,-0.49929901110421493,-2.652531387827693,0.20307148078795167,-1.947466732885052,-0.600788125650772,1.8874711099170949,0.6915548189461285,0.5835321673334808,3.082408788777196,-3.5156213628203212,-0.23371279123820043,2.3860185094012865,1.965131950361482,-0.4842784217945686,-0.1959443822104295,2.5851796849499498,-1.7444655111992513,1.7585612576039298,2.470334246978231,-0.0551034982073565,-2.6257199100011372,-2.6015518159007223,-0.2669202052100548,1.1739500234284308,-0.5859545976777428,-0.984025226713728,-2.0706206043987536,-0.9793114792625867,-1.994992489049048,-0.5986691370369684,-1.0450866846881361,0.3952701248749784,0.9726387456400983,-1.2506058645162397,0.37740981895328385,-0.8143724507368238,1.0157435880678574,0.9180718175872745,2.5005886710891003,-0.2431306790634634,0.7944785580052229,2.375967758712715,-0.08204801673861692,1.3439226560381432,-0.22553900374120403,0.9101017346717595,-0.09757978591224267,2.3304901107273195,0.06587814295502542,-0.2630431850939912,0.898822678962966,-2.7329579126377554,1.2714713230369925,-2.319043561327758,-1.6362217215785892,-2.235936718148161,-0.6652622313437123,3.8789968254401925,-1.300493099558096,-2.682748111701043,-1.3739454645762885,-0.4589963403055013,-2.090144475867775,2.893859110893232,0.9977127642150263,0.24036184794287324,1.3431042196784917,-2.8603475907786353,1.2036364991251636,-1.019973004621653,-0.6365836138089559,-2.607760127447364,0.9786321733234463,1.550058272504533,-0.8269435323200625,-1.3609112842343085,-0.7299035832541267,-2.7806232136474875,-1.8655940946577736,-3.366884467648739,-0.16702105207554901,1.0404502500512336,-3.1238076377759505,-1.3538691037257722,-1.0339168359638078,1.4184879133817996,-0.056442918666721784,2.5402862190012843,0.30238788459494365,1.2971672264678074,0.373827821914648,-1.8530414849846992,-1.4560853332784556,-1.6376810869716385,-2.6717289714410457,-0.18665391357046493,-1.0402597051099012,-0.7958958399024854,1.5359627227861559,1.6326361308452109,-1.3260187830630836,-2.1293784598517402,-0.8211262653954806,2.1277075910913674,-1.9692867642877148,-3.1067493077019286,0.8697930611306093,0.06065143644302855,2.086046478713084,-2.8393471995478743,1.7918409444915266,-0.7393870055166303,-1.7000877454649392,3.0538668286927826,-1.4038914710336978,-0.28926714518785224,0.16984175120939063,3.6361640944673117,-2.1130975685436018,0.2077033654853387,-2.9966285163767568,0.7280882746215159,1.7228598344345918,2.4028293274636288,2.497976009189934,-1.91309606538158,1.7837921291071925,1.4729250060712487,-1.3339888658644778,0.6220159528123554,1.0638302026234425,-3.2257766233084806,0.6444617450863275,-0.18556584261245085,2.915333396138119,-0.5613191592109869,-0.4882664192763648,3.6098906907548303,2.2258895268188414,-1.4326340651636473,-0.7614018857710001,0.5847052825516583,-1.6990712112596726,0.1166779932644383,-1.9494743838088848,-3.11224174339812,-0.26481618735212553,-0.2894051139099642,-0.9030865187378126,-1.572593590808833,0.6370752009598152,1.1408259644227223,-0.564140291876882,1.0140975311661913,-0.5227644717009035,1.169639569475778,1.3516816449122997,1.9575372371705264,2.691042070124921,1.8011767324716363,-1.379869936100329,3.8206937358108286,2.1593946156253248,-2.3172276240040346,1.5342460545553553,-1.675580903720412,-0.6008460533822885,0.19917431604760139,0.10516774276758288,3.1727380728442265,1.836868741015077,0.42991383381229603,-0.6479030007972445,0.6718334388333067,0.2822498034075552,-0.8382752809963097,-2.7943620830838403,-0.8800476968621841,1.0210974494608278,-0.020779559256967017,0.5440015963252304,2.0956055079468063,0.33336878757123145,-1.1342142983699914,2.6549145220775534,0.21330533551580927,-1.1367552100896072,-4.695199261051229,-2.350351790900105,2.9699452635432535,-0.7152580227070975,-0.8624242872735617,-0.49752287680526164,-2.3481862395384736,2.4230640976994335,1.1799911787417106,2.7949155758399673,0.7858074253482064,-0.025234225138042347,-0.4083013611172543,3.109294175663568,0.32057534704372287,2.5762753526456614,0.0219590596103296,2.099308005794797,0.8473224400541793,0.7996668980764011,1.3197306594066083,-1.2824546518108586,0.1530268407449228,1.8080090125160115,0.6684198979537865,0.2666611190841655,0.08009565622188074,0.37268025471551874,1.3076871245456108,1.0451024675916412,-1.3309193993594781,-1.570835602821352,-0.5747850644438648,1.623278170029533,1.831816781490927,-0.7969864756525887,-1.1616282231750437,0.0936111999348745,0.16016303084553624,2.9716905015741855,1.1043943847182338,1.592638308346977,-2.7851803823642682,0.43422808480277025,0.33672583290780816,0.7751605396025433,3.4252393358564284,1.4387369448994027,-0.8407474261005616,-3.4864054570675775,3.0959227131063276,0.05657296842827397,1.3513524896176103,2.2838357721157054,-2.962938520903309,-1.0032349528417541,-0.059128370457534815,2.026591599773945,-1.9865520566233827,-3.1541486898305715,-1.3866859463506485,-1.543713824340239,2.7061867586271084,-0.7550729641652075,-1.1908006655242902,0.13206415950461725,0.8511855448582332,-1.0783927250438599,2.7375117623589986,1.6681352227861495,-0.84436955732211,0.6484248217421584,1.0437126095307165,-2.0732801486049826,1.7477319683756318,-1.2426102559694505,2.6430340190365627,1.0486369658324577,2.4300682970319456,-0.45522697781131605,3.2625216691953502,0.5788775262093562,-0.8802755387794123,2.2508694577832546,5.089048822380284,1.8349877031627169,-2.6266226416714686,1.207354063728301,-1.1181049174187117,-0.1913542166082053,3.4632870723331686,1.0920151386732044,-0.4386920297719191,1.643823353285419,0.18786933999323463,-2.2224001311620367,2.36150746083912,0.9631771387207949,2.013177517469614,1.2279329857583257,-1.202110977732448,3.4728233863998113,-0.05644889773685888,0.26580293234296876,1.564116492748351,0.45848854470793476,-1.1300671609284294,1.6708548357557051,-1.955359598922531,-0.5218643645301595,-1.462924252647955,1.8757649938873304,0.6510827215971949,1.337273194313247,1.0082764584090207,0.5038428866429373,2.547833472577145,-3.009741854896017,-0.46096911091032977,0.9082791572271862,2.1457242755684973,-0.6225837251300457,-2.693640437027239,1.9494807367415554,0.2955064751730799,-1.6937769700569556,0.5216001836276946,1.5723320721752718,1.2241097186201984,-0.30547582828078035,-1.022372952089456,-0.4582509904641804,-0.19134869185205552,-0.2939561606554581,2.3174388449443435,1.5443935473970234,-0.3812656959923116,-2.4106401371261685,1.789405226672049,1.175368028077509,-3.061231864813717,1.096149309859967,3.3305949384870774,0.15949585891216156,-0.9731638284825,1.1566437358829915,0.006756017698224155,4.835289792917515,-0.8899188299744227,1.5670586344937578,1.7811777363253711,-0.15993342069830987,-2.001704062764479,0.9352917598083157,1.6474146175193747,3.4714030991416323,-1.0327424203132065,-0.03368792993698065,-2.3308044177714815,-0.7848690121553114,-0.010403933704499515,0.04641079329166412,-2.0757049594711385,-0.03686681337853067,-0.37718513170197476,-1.5871008514895253,-1.8878155459480175,0.4895577083315263,-0.499978813093129,-2.1288459408382714,-2.4805358026675832,2.070223398781402,2.180193180243025,2.9940773207120754,0.24152944955227534,0.26346448238529996,-0.19099888780619964,2.087487148540594,-2.390276426232206,0.7054683706992434,-0.3065148517028667,2.129256247206269,-2.724437441288833,-0.7589167225069997,1.4895325964983401,-1.2148371761099308,-0.6210519844869036,-2.646273135176113,-0.7272612610636647,-0.9025539229301098,1.6112874862967126,3.003140720528243,0.5017286682500812,0.3087658816181819,-0.3089338000051922,0.9059583761176617,1.5260452453548425,-2.0393234552481707,0.4181625868330613,0.20390145713526697,2.1418438526318804,1.1642006274878993,0.4753053851304044,0.9014934150683125,1.3033617819887848,0.8919709981313205,-1.4057221884697813,-1.3034987603413657,2.490952113667354,0.5232978607660885,1.7211626249758056,-2.1074400315387716,-0.7990465923192561,2.9395482338900285,-0.7082589950740322,1.8799905610076257,0.2702737478041357,1.5091089309141208,2.4292319214897278,-2.343782433752515,-0.8010890803721155,1.7917665512672307,0.3329977279422032,-0.7685776028605384,-3.2687115764907024,3.201423662489496,2.2177100940183196,-1.0047431727979106,-0.9276011054285668,0.1970804784685744,4.972468526305595,0.326688056892487,2.57820259676564,-1.9267634505201046,0.9009305340316109,0.5037749854321937,-2.743033644703379,0.36034805519230534,0.4648624855198443,0.645597513690739,1.984898677966237,0.06858323942215179,2.0391101782225256,-0.8794157362487428,0.4578358898588114,0.4583420231594277,0.6610427478729899,-3.5608832255101563,1.734856579800703,-1.9277232167489096,-0.04607575327593429,-2.3380171358563753,-1.7485107918186413,1.3214343159483812,0.7939881824258347,-2.2242681511162625,-0.5114120664688792,0.14836574479352868,-1.949934943327421,-0.3296441293528564,0.12558016640985703,2.474092893676218,3.6049420710579785,0.14942672298006957,-0.5582050633323825,-1.5686050675650947,0.926649709734176,1.8995289590281312,-1.9164804830608941,1.8387273837973663,-0.8690538134737409,0.29138307295631843,0.6009738087012645,-0.1523433074803381,0.2518487983979971,-0.2938470065339969,-1.6488339166749308,-1.5837116550548476,-0.03662508139217818,-2.421187711380503,-0.25718755804636406,2.051123289706547,2.905235656389517,1.811736664336285,1.9343688165983248,2.6543621945512355,-1.4335803568731542,0.007303979037593059,1.8387582912010614,2.2814406291935403,-3.0258273979358483,0.9237317137037517,3.1446059021370427,1.5627525405157368,-1.0734367507927067,0.5224097999060977,-1.1367597373008247,-0.19361443812549442,-2.510996613168483,-3.2808452283706195,2.6635177183069985,-2.190071556123887,0.6241574569173034,1.0816936560675965,0.4074750815314932,0.5942230947687096,0.8251545832335657,-1.4842648533354654,1.8262366314811374,0.66452816096807,0.8605238442556328,2.8900642015503784,3.15677107238593,-0.7869484732625501,1.0426406547807239,1.5028596445881313,1.0237969956396038,3.4383909085058715,2.5451541102265347,-0.15886802975681852,-3.204277502319965,0.016280857365959325,-1.3612135280852833,0.31714198742437977,2.3597629774180513,2.0467936771438553,2.4800970908967486,-0.5810012097665987,-0.4928498145483838,-0.09947587107345884,-3.6590022169442067,-1.7152815535166015,-2.9998200814775586,1.8472054484830622,-1.0765517352158385,-2.0113102446804705,-0.6367885623393112,-2.399399639317675,-1.5945496000197543,-0.7898893447983794,-0.561181101185952,-3.2467719544477367,-0.008941970744613486,-2.727597926503525,-1.4538699530993069,-0.15690253732487894,0.7806291201496708,0.2605412953669607,0.7779259530905639,2.6611933665893,-0.19189876052786226,1.631017302143746,-1.671558573868721,-0.1459065606351123,0.5313921550547173,-1.4983032432717125,0.12586794145283425,2.536228719287275,-0.4781387358823155,-2.6763345599743302,-0.8524318192279631,0.7629002847780334,1.5461617353366786,-2.606761045490756,-1.9138146551586614,-0.9482177616388547,0.659908695363287,-2.793656563971053,-2.5699498331946207,-0.5986599653043407,0.14336857189870073,2.7869371733513923,2.2611409547958563,-2.5984918780893933,-1.029045381162244,-2.936724256811747,2.1168197788277987,-1.3655954549192655,-1.1766151058509293,-2.290865446796415,-0.5669197171228326,-0.4112758088956478,-0.9487636723062031,3.8737066614645785,-2.088263731136816,-2.365895599107706,1.473786587867537,1.383869267796253,0.38709815298979816,0.539244222031863,0.1740674699729725,-0.2231563829810739,2.118440145487981,-0.6968441107762441,0.971023193806503,0.035552221570059335,-2.092487006056565,-1.7262979794963043,-2.2340869617049597,-0.3507648485296283,-1.810784690636878,0.557738732633472,2.2313814639733205,-1.0276044106826094,2.1662660699964325,1.350729898948265,3.220879323646903,2.793403401483255,-1.1083326760300123,-1.6062545892222009,2.50186564467141,1.7320385780542311,-0.33180376816185453,-1.9488362860650428,1.6891546775769266,1.0194774319848063,2.8417170042214215,0.39766654452964634,-0.907688336899224,-1.8089928606074999,-2.4494853568480193,2.7423926024647685,0.6123401980150845,-1.5760956147758654,-1.053396626696517,1.5673175466220322,1.8340446290389485,-2.903538957750124,0.004038824645761546,1.6708175514349315,1.6650597868900596,0.41885560384176823,-0.7959991762887317,-0.4882125458479173,0.7922389536628426,2.1203055102622397,1.6545238857259859,-1.3691534707327795,1.1317102046962486,1.7164570346556247,-0.9162987550924496,-1.3103143762286122,1.7210411013198172,-1.3462392226441582,-2.0313061437367073,-0.20275357073730335,1.8871974279279462,0.3622821392645961,-2.6343792731565854,-0.9051443346936219,0.8489326048876998,-3.0816733184247944,0.48211204951634357,-0.1853411850056241,0.746849499379013,-2.694876986316801,-0.1827229078273176,0.3974730560224077,-1.6078261581806927,0.21244936828141783,-0.28642995811151595,3.135600038695384,2.7918901320470484,-0.577735609077681,0.8471012603762227,0.5662415062822962,-0.8055144986217833,2.22444864642336,-1.8215432388831152,0.3490314954775011,-0.7396562385453254,0.6072827732656197,-0.42192721965072416,0.10679083982703187,-0.5573255696299454,-2.0556260743204993,-1.7922188031342772,-1.6716208746319858,-1.663963772026198,0.622427526477963,0.31030393572741566,-1.0111534212986049,-0.657180156008137,-0.730922051502656,2.5234879906845027,3.693973357371598,-0.957582955260812,-0.07369162062509241,-0.2998754891457906,0.9918807256420669,-1.3038250579642017,-0.11546434501868763,1.548247035610285,3.9987812595334415,2.737386109363888,1.8825631166399053,-0.8210447567739028,-0.6511947968916311,-3.8864310489154916,0.9064419318172621,1.1475914319505531,-2.935993907431492,0.10014977772852596,1.2915677349342771,0.18963845794599082,-0.31791228664807375,0.863358439886503,-1.7041858730804371,-1.7039039191586747,0.24718331210272707,-1.712389869631865,0.3383034433115652,2.231366538607031,0.7548389991763058,0.4671348302463991,1.514006765079231,2.0505340325336183,-0.18930698719970768,0.37344347534258693,1.1301192770961415,0.9294077199844339,3.8081925286711478,1.7186073718010275,-0.5800286565617897,-1.2187540176212017,2.277817642160675,-0.9145750154074777,2.143393037736042,-0.5536895971416995,1.0572316779170203,-1.022222266850879,1.539146318194848,0.15243237259518352,-0.1839123933744156,-0.6556543771457191,0.8480633867079999,0.6486725396176861,-0.7548332141466274,-2.268587522628583,3.9026583276660327,2.162458316655086,-1.174571444218093,1.0976076679017828,-0.8135649039648327,0.3512704549062137,2.401102025387249,-0.8622825935208712,1.547246880401,2.644067931819904,2.6115350215459903,-0.45708379671656313,2.0596917551261433,0.6639796016324115,0.2690116450217988,-2.1486000203705506,1.1155793638394063,-1.004933928314972,2.0385007329317517,-3.084894893149918,-1.2469628224196763,0.9774582469362149,2.6441969759162305,0.10653940531563569,-2.07579461483515,0.7304593418817797,-1.8913411000718736,3.7284972650838113,1.9604988578689717,-3.01991793279277,-1.480117945252172,-2.079490017219696,2.1305287891871645,-1.3006753810263145,1.3422591425690773,-0.7259422333772028,4.268298768095912,-0.34553252907761545,-1.052037853870322,1.4002630277982446,-0.37498741218117454,1.5086587291113052,-0.6581586658853913,1.9575004561796323,-0.5109431564332015,0.7099695688872156,1.0931236809409375,-1.3974167691263786,1.5542680834675806,-0.39236632217529005,1.8421787892122918,-1.9673271110379593,-2.532702452630401,0.7465412019362432,-0.46865127272387785,-1.9542934498537943,-1.6266358145547242,-1.2263319003846298,1.459031364050973,-0.7951365882194851,1.5442156170961332,-1.78843912523079,-0.17075941682832077,1.1906675198921568,0.10691568400511624,-1.8475700659901442,-0.9604906814427965,1.012870835144064,-2.023448602434945,-0.4433612705896715,1.368637616972523,0.4587458315083498,-3.608699151036853,0.3254703244445985,1.5131299547232337,3.288591524392275,-0.6262941278489961,-0.9334057153284663,-0.6547829622577596,0.9939065979185672,-0.28388123411441823,3.3166260600353343,4.045741331986815,2.5578064636064193,0.49024126786866423,0.4981087514099844,2.129629746010015,0.24535935634202427,-1.8375951014940508,-0.7661144369504856,-1.2962318282447227,0.4710805454623881,-1.4386833381116326,-3.3594379169184445,-1.7177882231083934,0.9994881758831909,-0.9200753654484297,-2.4771010848346737,-2.5059351033912463,-3.014644844199958,0.2987616437235434,-2.4552020601353086,2.053461978633476,-0.04754207929933934,0.2499568303136023,-0.01114417065105029,-2.330779741262225,-0.967310116288717,2.0992280812487034,-0.5769273998674298,1.083110276203202,-3.391349798076668,-0.5018413223734947,0.11189375410693378,0.38926300044508777,-2.4883617961340296,-0.8781773800482772,4.231302812411101,-1.4100990096649615,-2.1650979050169274,-0.6331757637308235,3.841173551683807,2.1642546151326782,-1.587741946591979,-1.6428606482161932,1.0903457517245183,0.17331556365240394,1.7785391922923988,1.5998668501485056,0.6002267596008198,-1.8286272342064318,-2.017739155527714,-1.174412274715463,1.6763145621260624,2.804933724032938,0.6395956156975168,1.3258786896609773,-0.7729808350469436,-1.3669027681242725,-1.4453775102061648,3.519114657067703,-0.756939486603759,3.074954184639515,0.2389401127544149,-2.0032394062675296,-1.43152939674654,1.369272464700502,0.25109344246818743,-1.4784242142488808,2.4284282062529323,-0.6474734426033106,0.9444775258164422,-0.5093403317749278,-1.0873433502252128,-1.5760417633098018,-2.7568272724704013,1.4808347217393902,-2.1109714393499086,-1.716556236379125,-2.653374857251489,-1.2783611295951338,-0.6162840610006303,-0.060066205108111226,-1.9448361949022117,-3.8786560401169825,1.3658963265093287,0.8214555521696366,-1.424138369747615,-1.3210333073473983,-0.08993373005461164,0.6236285373528873,0.03666902864754877,-3.6158373203836036,-0.23668613866592167,1.4611565329174037,0.026313761249897906,0.1201805743750086,-1.234624288723854,3.6091150628470365,0.7088843970140879,-1.6202365194060417,-1.2800758510096573,1.9720762127307656,1.3331130484785714,-0.0008890195430720875,-2.9189926388146787,0.4117104467921971,1.5092363455002036,-1.2328543182130876,-0.10290249675135532,3.162047584748561,2.699030352824848,0.3193645160936481,-1.6193552274345822,-0.5593433923009897,-0.5770121756376797,1.1509502956014817,-0.8610559637192388,-1.5733802932770822,1.0302027248223624,-1.6286906387228275,3.1004062566319592,-3.037578455241911,0.6719031186847658,-2.2033042802754195,-0.06271859651831792,2.152214836218256,1.7509420493011476,-0.7048678837701003,3.479358746370063,0.29204509579508575,-2.7073564004098225,1.3538570407910262,-1.8004376250011578,-0.6538113647975949,0.3174593962423279,-0.9080225549183996,-0.22911759368808973,1.237217547880034,-1.5153309517197913,-1.9069668033043299,-3.1188333530674592,-0.6905063832448822,0.6024549266629535,0.8332611563490787,1.564195265218838,2.2890009296978735,0.9619963967265628,2.0917787839088238,-2.8148020615216884,-0.2737347240557217,-0.4920263504649564,-2.8336900834522027,1.4183807725774327,0.4865837249809156,1.943215813428571,-0.13202949923708016,1.49055196336166,-3.177171264810454,0.8617340824058739,-0.021606566650349456,-0.17452665090956934,-0.2229025830531693,-0.044139106323118675,4.0392398503514,0.5933566176395717,1.6052021669155239,0.35566699987716804,1.580906786354912,-1.7319199080441152,0.33945461936990945,-0.4494011368870137,0.04808588631894434,0.7721569312337548,-1.9502089246059966,-0.05901445264446614,1.0018392980517896,-0.08075624961303177,0.9765419085024909,1.31669408968812,2.6831032423722476,2.2450631876172467,0.23596029952580885,-0.37310677760790145,-2.1449418946560055,0.42738826640158184,0.8387778146964203,1.53042082219538,0.5035218159767187,2.5166840551242724,-2.851290543887432,0.28710354889224377,0.7514143707672757,1.1157411568789972,0.9063161499096036,3.2288401680697407,-0.31041224377971804,-0.1801864246730243,-1.985894306214358,2.5031930912057088,-2.0276292652415147,-1.4674072524732928,-0.05299167793228493,-1.7454562943332672,-0.11255915461387653,2.7829204650518977,2.0860825379634917,1.3350598386036694,1.8858377126368362,-2.174823582331205,-2.504677148042281,3.010960420461675,-0.2632261206025071,-1.695776586448773,0.7416893259905646,1.0090108033036067,0.29402370768561115,-1.0071057943051678,-0.4715442791045533,0.060939201136587975,1.0706566398888466,-0.17112563566073064,-1.579940806526937,-0.7435181555213306,1.4686811420889685,2.862758399082522,-4.033266036711511,0.6128204559858696,1.651947065582429,-3.2835510559894434,0.9863348006383827,-0.6853073952023078,-0.8227294650167444,0.8711984382072218,-1.0266458537392842,0.9847891466193223,-0.10601742373437067,-0.3002423584938409,1.5694246926235091,1.2775073824482281,1.5541923690942874,-0.8715863348689218,0.5039151044511737,0.48771079718599586,0.29112841289306046,2.488256918621503,-2.0345642375956667,-0.4560542091801353,-2.4805139580110263,-1.5733927407384245,-0.9803846502754024,1.7733662263100889,-2.7334474661795056,-1.367738286459265,-3.5770359447672972,3.590436217260258,-0.6319345280355986,2.209148060171626,0.5890216127986768,0.6931011252852577,3.1389174618487004,0.41869619631593763,0.5206383248303967,-2.8045477268401817,2.536328930322965,2.1133943122964336,1.7951654906066583,0.9771058839529955,-0.10176434732943551,2.6978475198198573,1.4505839294629634,1.2063574944228392,-3.2727596256623843,1.686679102644424,-0.7172435824484581,0.3965767078687894,1.6543288817353192,-2.5095610453269184,-1.614498009522594,-1.7038230095216238,1.9645682559693058,1.5561460839315848,0.4321805801031448,2.4385006229290997,0.501289415755946,0.13955997501881887,1.8112922804487481,-0.37450620389672135,-0.7529886781593675,-0.49957918819216907,2.6050891391343796,-1.3050984557022256,0.5241501457424584,0.712937209637868,2.7302513338333956,3.0937927263588514,1.5687420772442158,-0.09924798155474321,-1.9223857051096134,3.4981907545811577,0.46034389266318154,0.8914477969857613,0.011415683740177998,-0.8897425801108676,2.303558108218575,0.807287121519177,-1.4195452332990381,0.2724335662492007,1.1027785454033587,-2.6307978833942647,0.49537635017221054,0.6086644122933967,1.1310605668460192,1.4325754081429338,-0.13798307134526047,2.0786238667244294,-1.4922111708940877,1.4088527728978895,0.0677079793588445,0.8418404765364971,0.3647925051706889,0.16980037328296163,0.9888300972698472,3.8402671480904766,0.70843704975395,-3.3581895524363463,0.45146602062827895,0.612567377760121,-2.23807325666846,1.057102082579416,-2.0304745850435273,-1.030878170294311,1.1533546323107102,-2.143415753586158,-2.507204973172665,-0.22978473600297689,0.5195759302870874,-1.0341485580071184,-2.3117029340085207,2.9150004480588425,-0.6959431700434994,-0.28727868138654816,0.5623609960315767,-0.1416471193484799,1.4720910941903012,1.9988447312121795,-1.8845015855818135,-1.0712017712203772,0.5867081321919603,1.3486925064335546,-3.2220212068071814,1.8006332585679585,-2.5202399791680103,-0.763502288683744,1.6699278598990133,-2.177813147718079,1.8309122367364437,1.8798771752049332,-1.2962138255025413,-0.5188399176333248,3.147297528357638,0.974854933474416,1.837939558737755,0.18263119027432856,1.1435246228553424,-0.5783999007536071,2.02998376046952,1.243023709999271,-1.918592583408899,2.827157519613928,-0.48276089073271705,2.159696796081061,-3.0268207694866627,0.030880012373682604,-0.15093475991142136,-1.925664533081857,-1.3649016250079269,-0.7716821538955886,1.9581891230224149,0.5689146668633004,-2.494480546180208,0.837419695699002,-1.3150812696529457,0.967724829728841,0.5382073224062219,0.38883445918439863,0.8985183225774219,1.625036005252631,-0.7597796191953218,-1.7729831214317913,2.7862952174086684,-0.740403463913756,3.937393789105851,-0.05127936101392144,-0.5826997668777604,1.3572541148512445,-0.2378985968751799,0.48549357169086554,-2.057442875813797,1.7649390312101318,-1.8094164985179306,1.1979587544833046,-1.5151155366428006,2.35221714701895,-0.5598131951645049,1.1454413586276235,-3.655367162079342,0.8678458440990025,2.498620526160418,0.21048410087008515,1.7944227498793581,2.2728010547995843,-0.560666254620777,3.12718275600677,0.060649159147397595,-0.12241242788784484,3.048676980447073,-1.2895922204997976,-3.69050169873788,1.1031137598518916,4.098255323010946,-0.00548356766001715,1.9818411102922782,-2.870094184089725,-0.568124317118847,0.7728911111653968,1.1216191277244956,-2.236568705765174,1.759366800103985,1.414762880798483,1.13625922241129,-1.0189615032246357,-0.3849973330271076,0.545071029694801,0.18166161410677098,-0.6975186876562812,-2.42848941533585,-1.2459018811564644,1.1257752411674056,-1.9999521719531534,-1.0496283644385362,-3.7571098388096544,-0.4260627852744338,0.4161514500080098,-2.7547479324247104,2.4305092502672383,1.393927406419692,-2.4218859048591876,-2.327984298234083,-0.9596904737628895,-1.9651561656861147,-0.47770398220455845,1.80171001182399,-0.5940112078790676,-2.6780838631070667,-2.345725657210525,-2.4107097019436967,2.0706858320904304,-0.41226949975219046,1.3510607045554042,-1.1377124656684603,-0.7036342025526469,1.2134223763226792,-0.8145986855737968,-1.2548921161144893,-2.016347986444656,2.1497681936995288,-2.6172383897187257,-0.3517546545000953,-2.885819519180597,4.2914651354718965,0.2721182280881682,0.1619110444819732,-0.3038900150677962,0.3160957232987729,-0.0718469881293687,2.770174744012247,-2.0460140766257906,-2.029664084149493,0.6616012628832594,-0.07137105638966533,1.0553733114234944,-0.5752684012787269,-0.8671646820580384,0.7857256081242533,1.5565214391646833,1.3277159150774833,-0.6186631847280885,-1.0080122256230581,0.7727080939214485,1.000941966016213,0.8705322190254711,0.59032948883831,1.54530325414314,0.5657283080010572,1.2867080608320067,1.2024984925523794,0.29253284922781336,0.0091766249803389,-0.9563315516760844,-3.38421921545294,2.893602004927498,0.4406537585539707,-0.9099007403483739,-2.6797015808610283,-1.552313457678333,-3.1295264068098954,-0.10199909687255011,-1.7433290268446262,-0.8579369871012336,2.4754667286029273,-1.675190218958149,0.7733197593223267,-0.5982975844109482,-2.6511741067999965,-2.139512593027435,3.093050777265998,-3.8615343924961065,-0.018482305332454554,-1.5742430987776228,1.1582990039191219,-1.4823434073803323,1.0630618107209302,0.4559854727334233,1.5034952149386802,1.0260789441042797,-2.2877714398753732,2.113887325656427,0.29751786844167966,1.8219433518852564,-2.4352345105825193,2.0067611909179193,0.7411152885980739,1.2752982050296675,-2.2322378495775492,1.0359769725737968,-2.8574567185799444,-1.0303143128421282,0.7433916741694706,-2.0910434645262144,1.681170427237423,0.09788183601237604,0.326351930841695,-1.5212313833637667,-0.2484036996440768,-2.1522092302365845,-3.289722070809023,1.6854547313202328,1.0643330428581594,-1.1600883360806369,3.6904877514024723,-3.1773856597822916,-2.7476872748365997,3.689624739895709,-1.5656762693211574,-1.2086887786883975,1.316251390222595,-0.13217371650018944,-1.968024921693376,-0.3638122286877215,-0.2830614900494257,-1.994661812811945,-1.0805456222907543,-0.8590434981106583,-0.03162418168642062,-1.9188641188815352,1.2840205577412636,-1.3398806388174767,-1.4219738129613189,2.0880361790899293,-0.5183816384638515,1.7213483927159976,0.06191168541359687,1.726489512568496,-0.9986218374333143,-2.0689504203679516,0.046035424885716984,1.387653996372935,1.0120940941803567,0.8410002828237642,-1.672545175400526,-1.6657732645819916,1.6955912144020346,0.5051126654276876,-3.189339527090453,-1.3383784874744065,2.43359383823044,2.2735871874413536,1.3998128780954293,0.9811491406457614,-2.915417377488484,1.575898058439248,3.6256696549654337,-2.8658088258935273,-2.796325482885894,-2.5973156062143454,-0.5927181061403295,3.381106939802581,-0.5500462992365678,1.4319084940048763,-0.6297614928377517,0.7874137768279773,1.2113345372824527,-1.5775686884379592,-0.33144904964143107,-1.9931250866682024,-2.6225268874937666,1.1663621303964602,1.6546647598823407,0.33299899880198197,-0.7377740803742237,0.24591162810876296,1.2179805876217944,-0.573304608576684,0.6804365314815299,0.9196535058843092,-0.15803288744607802,-2.9733574918786116,0.4542970334784494,0.9815192816085089,0.3448044613410392,-0.506281416769762,1.4774848872244393,-0.9809769111916061,-0.2523951542649658,2.187391310062689,0.7943420344692851,0.6692864390558091,2.548522300880841,-1.8398492538497038,-3.6193404107980123,-0.6243320483096854,1.8666322198106469,-1.6494299217184478,-2.198190817783873,3.176938803512523,-2.400022074668983,-1.6581438365821193,1.4444707788074622,-0.8377919917497394,1.2328581488043207,-1.0334103559595138,-0.7298632834842383,-0.5571881165552619,-2.2226323003712345,0.0797722704146979,1.818545457041598,-0.029623250754843844,-2.4127663771265513,-0.6200988680231679,0.33399054541386675,-1.6320141241293626,-0.2623078688288101,1.705103695383923,0.08793579931248655,2.3939275541945793,2.5366240438997276,-2.210035595587724,-2.3407406733829177,-1.1852953137684845,2.804728533903172,1.7120685725436067,0.7832275245858638,-2.6332260528821108,-0.7857946014787796,2.3261165457824156,-3.186352172977506,3.48999279971397,0.6754085994671628,-1.4735777780221562,2.162328969651994,1.6702782045029052,-3.8617141867275357,1.6980876528674445,-0.4756077715578834,-2.4471827195730014,1.3972981532825095,-1.6372244161775642,-1.8802059866713656,0.7879033242335794,1.32906781195999,0.6446602395785064,0.938624914713898,-0.7732926562028892,-2.135420956956736,-3.2108811348351405,3.993487031617697,1.0882924036828654,-1.5739521931271552,-0.0019381055305437958,-0.751276770041563,1.6506930772777828,-1.63823074171386,2.8060184969341466,1.416617507836987,1.021286012603897,1.520835213218623,-0.4579857168234025,-0.5926914234555136,-0.8588937128190492,0.8279688934746997,2.5961914942671296,-0.7361405035395255,0.48682978222525264,1.3452202535110207,-1.2610596801319531,-0.527277137278196,-2.4472878932226148,-2.8756595901586337,1.4408726933971343,3.0415144443327207,-1.71503606806768,1.7602586006909053,-1.249541164921773,2.3485646041183403,-0.4669603352141342,-2.505154629160821,-1.4875586565532464,2.0710495756776277,-0.357543874383088,2.3358767084364462,1.9969273505886784,-2.871649660763653,2.6838271720573776,0.6104177904063118,1.4000506230384009,-1.1933595550156977,3.1114160066055696,-1.0976552600664156,0.34207813780681134,-1.9982447360664746,-0.07900891430945682,-0.779702439071063,2.810224586169447,-2.397400498275759,1.098649499906506,4.641345909760013,-2.4195362010631576,-1.288918587650748,-0.6842956300035499,-0.2542791206607381,-0.6571578426459537,2.198302139124452,2.0034173120071066,-2.8008394413383275,0.12821148606357974,-1.148554173721594,1.5466605164782372,2.125052450800879,-2.30939522300859,-0.4368173679925318,-1.8764165077542756,1.7954801580861792,0.7026338827307472,-2.470285908384603,0.9153241391515867,2.2580298481560535,2.704479138576161,-2.2523137442356287,-1.99981964246849,1.299383625835086,0.4522290197116921,3.4849750066633165,0.8841530958918628,1.741671440390529,-1.8471910823901754,-0.04986369186328572,2.325269560766651,2.19022458483791,1.1842147200385431,-0.5821767628534692,-0.7234096616412671,0.7174729842644059,-1.811038257804185,0.7791098136896297,1.5229608885195796,3.1306625419090044,-0.34742686172017684,-2.3814404593155327,-0.1378697760992848,-3.2426348130223883,0.08872710598369207,-0.4417648850645806,-1.2081521557364818,-1.6686515326017513,-0.7223573870692735,1.2986949584656287,1.8656959532428061,0.24169002894012664,1.6577677716665258,-0.45065549308930375,-1.2638234021169974,-0.32797263928365833,-1.1100870876785112,0.6117254426707837,1.7827070325808578,2.616042890572513,-1.5284292348012092,3.634695742912846,0.04520198681807503,-0.3671554084641238,-0.7665712142612696,0.1030004234197941,0.6900431588974819,-0.5457084115713737,-0.4582941476249118,-3.0946526032167823,2.2510491887714625,3.039128902209497,0.18194720790878308,-0.4393178592218462,0.46563004898607335,0.7960501516079374,0.3530381813456886,3.4878123975174717,-2.3241849815476665,0.35345471043479115,-2.4857424704789985,-2.616289690561947,1.322691616045201,-0.8125602462989078,1.3886795016502522,0.45263323848772513,-2.622693666267985,-1.0095446023431236,0.603323430859407,-1.0889667620798538,0.6743833338446148,-1.186598970576486,-0.384442448724161,2.2700966274325767,2.0729157148463155,-0.05911385259185774,-1.22901040354301,-2.665455614627816,2.072916660729743,0.5192743594593243,-0.7494840451793368,-1.032853363044364,-2.935070870863572,0.21655337461073026,-0.1480933810926183,-2.0873457557257216,-1.3286780124037918,-0.8899786836190294,2.000584005935105,-2.031201649055498,-0.9112044178091465,-1.2483166415694937,2.4224673916127215,3.1589251173849773,-2.232412902804664,-0.9910246847040902,-0.9336534447902208,-0.40789883692307066,-0.6150402203440111,0.16649086563884097,0.36739193986003976,-0.20712754795693133,2.5114905335444906,1.7421259672759117,0.8383183602879245,-0.04501280879731676,-1.1886396381164686,1.0628966335722114,0.20138907766711425,1.2955480716535932,0.49426179024359046,1.0995097420661084,4.474523668859883,-0.5276320155242187,-0.07398166442379601,-0.9829157697270204,0.29812128129168136,-2.8696702353551955,-0.22602585069177297,0.2635145542877713,-1.6296159509029666,0.9355166937396302,3.7593933963280355,2.3082013608059357,3.138998023631046,-1.631470327731274,-1.7557292928038886,-2.280870296349135,-0.2862927998648168,-0.674130549775082,0.4672176849706812,-2.421886499220925,0.6678704829995284,-2.007574532895073,-2.211874005849556,-1.1076329557021614,1.6818173937754637,-2.060673182619517,1.6871695914273435,1.577553455962722,2.7363342243061446,-1.5812914099335742,-3.3563307778046654,-2.8897194967364617,-1.1882283747277662,0.8591395546355703,-1.1433154626710689,-2.869873291526416,-0.8750165037357551,1.0231480947602185,0.5239779140514612,-2.183680970365105,-0.6823662068235873,-0.30810292071290196,2.3129026149001186,-2.0165567340941144,0.6568273782124253,-0.24169428788718464,4.013710546285348,-0.736818697981873,-0.06264407972325582,-0.3512056220645502,-1.9136021083703152,8.552697758825384e-05,-1.991568911247698,3.7967509182924606,-1.7423163890888789,-0.7361041861837908,-1.8603382488466023,1.3591274936624071,0.8922118570553198,-3.2626917338857586,1.481895305778196,1.2634218040450282,-0.2191526541966879,3.691392651258459,-1.2267255768672107,-0.20778698327603076,2.042506656178712,-1.0233534291722568,-0.36471314233300073,1.1826081244982927,-1.3179503237221957,-0.1411608716631208,-0.9255622704492652,-1.1205702888072295,-0.2926864689312402,-2.881017672207574,-2.4628013649546507,2.5805818237879437,1.733358235955424,-1.8136902981459477,0.20785723522487115,0.6167433450228101,-2.2391156147257707,2.3101645101881485,-1.4800440818981793,-2.679974744305857,2.8654957150076794,2.106433235606502,0.44569147326468156,0.7813944118567846,0.2811545432588662,1.049641325182823,0.15100354423790058,2.7342667540419403,0.44287279538220387,-0.4310627963528427,1.1514929802470737,1.9591597814279014,-1.4585903084074725,0.08776431319812938,-0.9297377348077713,-1.9377309250046644,-1.1084525095189253,0.5779783682623338,3.3941323206457197,-0.20512325464540956,-3.4075222594188266,-1.016754795408303,1.2997604780734386,-0.06254832880657096,0.0022897604887526122,-0.660487965079239,-3.0223059336306246,-1.6832072986656863,0.8008262560760638,-2.8392434192554044,-3.236811855743032,-1.92525922135664,-3.0578805000929896,-1.6191186766261627,0.1845052513924333,-1.15414704257165,-0.8972288359752127,-2.017229043245894,-0.2834212006487386,0.46660897282031205,-0.8591566197509712,1.5776226618441704,2.3753467543514075,-1.45818994773511,0.5240956076812203,-1.2812491795955963,-0.5953556629080434,1.0974663593304588,-0.8979015627985568,1.2715141751511112,-1.2416078094037373,0.8797173768042782,1.071430150077029,0.9151126940237888,1.2488718602696378,-0.22315049099772122,-0.7827320406521737,0.23243974328391315,-1.2217423857830858,1.4075755471681137,-1.8834778436609256,-1.0641245777934678,-0.03477334257383016,3.269528041275238,0.3241415062434691,1.0552077691478992,-1.3590882399717341,2.133914896845793,2.7341456093440315,1.9018064445340375,1.1955038290115887,-2.934878923564419,1.6215006672081729,2.4842979400921443,-0.0661613855564104,0.27232689673359595,-1.6612193336605277,-2.5453707415037994,-2.08962406289311,-0.06550797162896667,-1.244796217745868,1.4971000834542199,2.003954264885767,-1.043561607887642,-1.7429178698588383,0.3210596929032739,-3.628979531440715,1.4366086613346116,1.6237789868382162,-1.197839342751956,-2.72591081879674,0.004513600894286967,-0.2021607297910579,0.15169828874436328,-1.0284858244671085,-0.07478964682906757,1.6235739957435165,-2.112081972148462,1.8854754768129622,-1.4027740821605612,-1.166123063962088,-1.878586666530205,-0.5451979325434283,1.4553086039617662,-2.9433119086994477,-0.10048478043221096,-2.882087053949352,1.7257190286898338,2.1496961699291135,-3.1380266691874943,1.646185651606003,-0.5542119908529085,0.15716529811245164,-1.8119610728132698,1.131589338944577,0.15274317552587882,-0.7679485201610742,0.7663686624322481,-0.8217168057736076,2.1908060719694653,-3.2109540088063313,-0.6456620052778105,-1.0577269331761154,-0.8327691921026954,-0.9899894341237105,-2.346854226805386,-2.9944031044347934,1.0423083268839617,0.7270183906747145,-1.391524570368712,0.26064541536584857,-1.1079416484530953,2.6659406543846824,1.199038720257417,1.5757698660523705,-1.9551950330620764,-1.3248527328742805,-1.431718368408113,-3.0062701132584113,2.384802308208823,-0.94039766333757,-0.7901256252880298,-0.7643501334322709,2.1735311963830513,1.1320125557998286,-1.894723750641403,-0.7609546206955217,1.1536678741138724,1.9743239353697328,-2.828627954247882,2.437948038908214,-0.9897495648169217,0.18772736740878293,0.009553341335799204,0.3921263710150135,2.8284083702356644,-1.4201018387347464,0.7065847518459779,0.5025452630574347,1.4920811139140824,-0.8157423867946004,0.5041462624674511,0.0054514166091973884,-0.2144240546232771,-0.5064849057439146,-0.8609203301678736,-0.04207831081277307,-0.16729670246476425,1.1269251656053025,-2.0391292122469578,-1.401213242280876,-0.1492517916548946,2.4029505673001132,-1.1206023246569534,-1.674069390767383,3.827987981911029,-1.7832048928527455,0.3465463192367434,1.4162387473716223,2.185896927061653,-0.49701565684752363,-1.6689477669445205,-0.17548940675744346,-0.4456531775642488,-0.7778612675221658,-0.35346663967314224,0.387256291985014,1.6261286486050077,-0.6652839225517977,2.2522094911442494,-2.1219175093456393,1.1346155225998946,0.5356823513267782,-1.6034189461697659,-0.35806720671081965,-2.3675537335413583,-1.3554296820099345,0.4587786504039468,-0.5375733568560495,-3.0620079263640907,-1.1597888068249291,1.5691511895900074,1.98375849878994,-0.06769496239779542,-3.9569714540684835,-2.7728526147619763,-2.483420321855395,-2.467869724828749,0.2701781568749571,-0.9421270279653353,-0.6135568631343539,-2.873158749461289,2.079229425166474,2.027290597318685,0.2379248016478497,0.8426199737059884,-1.786685029030734,-0.6600662890640272,1.5941657853913338,2.0588759815991438,2.368328863378176,1.5125477132506797,-0.560670345485477,1.498873099021227,0.23525299289369953,-0.9382393394587644,-1.6233483006250786,-2.189167981117174,1.6314350833336133,0.4276208717692948,-1.2205090261820757,-1.8508990366920004,3.115938923885094,0.3792665727634188,-1.5183821869389722,-2.6975032219516195,1.8562740516263776,-0.9791427813619671,-1.1077601634066252,0.07434635243799041,1.6056934821175002,-2.2539028627795643,-3.6874131309659615,-0.7937619143813738,1.2480375497561549,0.3721580824909672,-0.42454363953258656,-2.4784206546181884,-0.8991645870844696,-2.377516240878282,1.400923765739874,-1.0699223550392174,1.3102665278167853,1.5141070238282084,2.8808511187554227,-1.118133688651577,-2.442874806019611,-1.8278879174877753,-2.902019695809084,-2.175262746890008,2.9565157620308864,1.1002023789507789,2.3999632883711732,0.12911222448377774,-3.312109454742872,0.08814526892502161,2.5692089877658795,-0.4097872903772058,-1.5582262917240186,1.162338890868774,-1.383802285152909,1.015586491622273,-1.3301078605073262,-0.2177991782583274,-3.723166978967485,0.48407090567377764,0.34814982951946905,-0.604621893423201,-1.7977240377445967,2.175548821894766,1.4717748555064936,0.026544518273582514,-0.2825771286883979,1.7063406670119143,0.7559975467792128,0.13642975628915677,1.0397494983502649,-0.13897405168741422,-0.010982298996643122,0.6825835513611783,0.6517710421509785,2.0576370322711743,1.674345188102935,0.7379462495654386,0.5414140318495259,-2.9459113419346936,0.9258630252247604,0.7344086139086011,0.9725900353950042,0.09541540937798675,-2.8516704430134703,-0.51965857330962,-0.3882898798196033,-2.240359136513701,0.5992275214526206,-0.6755981473329687,3.4734348502120955,0.4204416546505482,-3.8989302537974138,-1.5775976665212896,1.582004746373573,2.2354338425715388,1.2643470112340907,1.2551421627763144,-1.6340087252952593,-1.3208209881855222,0.36068704782255595,0.41908222582687893,1.7497584639207455,0.13253481255123914,1.6754472238292488,-2.5462468440790653,-0.917930815241851,0.09171831245657529,0.9928945944944043,0.49905105557882884,-1.2223159390400828,-1.6304687486312954,2.6606392830635874,-3.1380501718235445,0.17626040635706064,1.419675696829686,0.17992390562786603,0.8645803613201561,-3.3143812817840757,1.8398438107164705,0.6950602598088236,1.0433872403866684,1.03658503253572,3.6918754508581073,-0.06255313343561697,-0.2618996299588075,2.358510138732816,-2.499094545941998,1.4220285767800787,0.4188832402449833,-0.23601730574303573,2.56581843933223,-2.1781637251689716,-2.2716730542302623,0.7911441348041932,-0.6117453178147485,-0.8005232425719413,-0.3204567993729016,2.2133101226942844,1.8536352692141513,-0.019416793882812117,1.9526109656415027,0.20406023571696777,2.043880719246015,0.08526968200510779,-0.40054229556380344,1.7250217179452758,1.0640028824980472,-2.4351135660875616,0.6765142460911774,1.6405798095981952,0.383097936185699,3.653005830134818,-0.7283738609976401,1.1402584571452419,1.938032382629846,-2.1370499891765227,-0.06753036166534321,-2.6033292582372334,-2.047177793438963,0.03028245650378279,0.5286798726779471,4.969234433060556,-2.2934406256366047,1.1836996230773562,0.4371031457601983,-3.1130423368708797,0.5153604270523543,-3.0879748056697687,2.4740208859293173,-1.1690340041676763,0.6814234169478446,-0.34632892221872624,0.3137184455251195,-2.5004849645870055,1.342103277457448,1.1180485351551082,-3.567292394509555,-1.067364712234772,1.8635689104838808,-2.1125439154447285,-0.5838535404526353,1.3986900170161292,2.6176107186383075,-1.5576460555634803,-0.7438863002324392,-0.4408493499955957,-2.3227815908100893,3.0765153796935754,2.277942912494563,-0.16428304480239414,0.5966182728263034,-2.4508311505210436,-1.6161498508172727,-0.995975949306802,1.2918154816258505,-2.983068993404518,4.056739018400471,-0.6724488304797932,-1.2309169106004878,0.38134994907325703,2.0960312499049616,0.6004251758120316,0.8993946227782841,1.7673814649850188,1.5746862704592235,0.3626484603485188,1.0274760511115686,-2.228705549319015,-1.1843669295906172,3.350947412014704,-1.013882817071707,-1.9600241868836266,1.0066371299022647,-2.6989608246535646,-2.8464028285453122,-2.4580402916244393,-1.8251953651318236,-1.2840280501389072,-0.39987138245952636,-2.2906577194309024,-0.8147484189741873,3.15399964138271,-1.4894625675794453,1.878220956022612,-2.364500285889246,1.3094627378958226,3.840158428641388,-2.1703974201123404,2.1219580124664637,1.9850296817802282,2.140477315579694,-1.6902228979920644,0.7129830365966219,2.158124366147141,1.2259590081113871,0.6766797400196353,3.226288556517794,-1.7888980401188406,-1.7752878270736252,-0.13831599075541848,1.5703267707694992,1.475456394162504,2.6084472196749915,2.157600237687009,-1.1584707513956236,1.3682859112592771,0.09037110834714303,-0.9332073732482324,-0.6548920978096179,2.3635282946581087,-0.3115483923468145,-0.315177432489334,-0.05121187705726963,-1.443431005239469,1.997169231635441,-3.346234270107261,2.220019130400077,-3.6081428280394827,-1.8987449226548276,0.055851221056706514,1.0117341125045574,0.8106590482327111,-1.0979249940305094,1.9564839861508763,1.4599950742508814,2.947924943666117,1.4987048325942873,-0.8818655285788872,1.0004465144416446,2.7189285260600307,-0.05759640762704257,-2.085856666699062,2.95355119343197,0.6086078600293008,0.5655264452878412,-0.25228515421942543,0.5248601032042555,-0.1725211196905447,2.6053007159772985,3.1300877086214722,0.9852670446020858,-2.0671822320572137,3.5727849967607517,-2.275255443586933,-1.0696327798586633,3.9113497973748825,3.317027331596133,1.8625904462613498,0.1842170913827803,0.7473328962694891,-0.4885751492173746,-0.36295082521945066,1.7783389378536938,-0.473601220079912,1.9385883955835048,1.5610187156057846,2.4841674037458343,1.128448339024245,-0.5956351783790136,-2.667287137166638,-0.7665992770841483,-1.269734694172885,-0.6377051565590482,-0.03804457908486529,0.6029892374531093,-0.9454242472423113,-2.2206627371727548,0.7542911586772272,2.730232656578581,0.15356682215848044,-0.021003816812939956,0.9169523901992136,-1.2438481903418197,-0.8692302627756622,0.11668634016497974,-1.7645255214134954,-0.6489448408986466,1.411653345473695,1.4401713805954848,1.2402296588269905,2.5112805488448036,-3.281093032995697,-1.9439802855075323,2.116667294212741,-1.1102498197293817,-2.029706371776171,-1.3578060606577225,-0.42390802627698265,-0.8503480786437965,-2.152151651996455,-1.2706255696407394,3.0032312757684134,-0.04177680565762159,0.7086135488105249,-0.9748832998270337,0.1495075687793028,-0.6691951001194262,-2.0047544610023964,-2.454218548896981,-3.0123964644728822,0.42397778157368515,0.6195370419350194,2.762358464876817,1.7205791282319496,-1.316342767389507,-2.2012441989583906,0.7250074119465276,0.23106731793030497,0.5886558870267017,2.7694707343762963,1.041741369813396,1.5227023395885504,-3.2706073664429147,-3.017029232268795,-0.5877167184512069,0.39240049291835355,-2.7536588430115807,2.1524066664814865,1.4178055624479307,-2.651247022679578,0.3580036462958715,-0.320776078127883,-2.7171584016729984,1.9740390041659885,-3.6025571786122548,-1.3570050061979673,-2.908922919531066,-1.4639876622098191,1.1392849545618007,-2.8324021245678175,-1.5631543039582718,-2.566560638795444,-0.5420844109176853,2.7202932076065065,0.10406722228501877,-0.5944204012046771,1.7785347230329833,-0.787061608086225,2.403346085925284,-1.4116711686968422,-1.2062970550602647,1.0690574270447961,0.5583610377511945,0.8017981606776443,-3.846008609385073,1.013607093179472,-2.0969003243162967,-1.0531656637598379,0.9580316396847163,-2.6364123004205697,1.9891801409815222,-1.0341566044048536,-1.6925651295504611,2.7881928980461486,2.678619480100754,-0.5402266451487817,0.62259596547363,-1.0580386154925963,0.3816579455826414,1.646820121783821,-0.10755441340064938,2.37053304801119,-0.8641624725779253,1.737438534166673,-0.6381303629562591,0.023661808333041715,1.3232506075240615,-1.2585870348735015,-0.6148053941405556,1.0563356185489672,0.7064861223144657,0.6801494191934473,-1.5357882902545645,1.087637671613569,0.9441867578446661,-0.5169122169439955,-0.5925430807742428,1.851400852423076,2.777846449648277,-0.2830150569582003,-2.835721493017951,4.370686977971425,-1.5724071982827657,-1.3560292570962744,0.06576212335683113,1.220023698379239,1.4303717525250643,0.597152620691496,-1.1261067147860784,1.354159630268488,-1.223635997596951,1.3870600561450364,0.47048168201509133,-3.619109448404754,0.5076512002527275,-0.9193441565923627,0.6669846257325072,-1.9716176379795796,0.0022329885571333637,-2.8714041570216153,-2.2329388592302486,-0.9786969496357762,-1.4678424041464053,-0.8532874994151249,2.5436241737478484,1.1230469430446273,0.37626078380605227,1.01585948776528,-1.3157082235904927,1.6463885183728104,0.011962437787835041,3.35215019314186,-1.4135444785875717,-1.3187360696575288,0.3784296545653077,-1.4947150676935652,0.019528396190400865,1.690645276331283,0.28556008563222723,2.672712939336609,-1.3470776474755934,0.8937292991765564,-3.0826829775467264,0.08940150361365766,-2.8290232534995563,0.24970955978042988,-1.7850850247391377,0.8058665178477612,-0.1761381970959778,-3.6643860777868045,2.059796199391692,3.5506556055000758,2.729230065486749,-0.11969913211763696,1.4559226640620824,-1.7984964926423976,1.2253094609288557,-0.6574277709376147,-2.392389549583853,1.0094729673046268,0.7588416839970801,3.142019281118626,0.18173634430132876,-0.23850892471531546,-1.2557680782693526,-2.2092642661328634,-2.3558994085598695,-0.8449137672932241,-0.6224912185276305,0.029416021989192707,-1.5403422319203064,-2.687577362528285,0.019539837518523652,1.1346802041215638,0.1840686188158778,-1.52410051034051,-2.2502238663487244,0.978448047926683,-1.2441314541517163,-0.551579655118925,1.411035866733277,0.22784655029268283,-1.0206319295418573,-1.4567192922824281,-1.320790112271592,2.1027195598626496,0.5397432963233751,0.8932170757599749,1.035280723208276,2.088763468742693,0.6542807204375578,2.1510699046895043,-2.538015225551329,1.0304407184711872,-0.7002571129260748,0.6308744290959011,-1.9877869426840562,-2.5778618677229863,-0.2560029308761559,-1.6436611844466444,-0.78051362914715,1.6268547024545634,0.7083756296768161,-1.3358813783517716,1.072919310205523,3.309649203399266,-2.631364892674377,-0.9386608472194987,1.1139920537611383,3.583743107019435,0.7581924063137199,1.388779016843277,-1.2318859762253678,1.2769149885004372,-2.0808822197125156,-1.3167646776872595,-0.890784256223299,2.2542765980392057,0.9773174059218214,1.0973573017870286,2.23027928491068,-2.127359554154882,-2.147785658750217,-1.7032143725581612,-1.0191576947925822,0.9014892051428904,-0.696432677461709,-2.6092509142125975,-2.2544073565156904,3.907559659844157,-1.1681380719168681,0.5466106823515001,0.21667034480276245,-0.19896348062151822,-1.2052457413293853,1.431201865103064,-0.5436325945665019,-1.7015156245119685,0.7744735765517332,-0.177352399899136,-0.5447604719073357,0.3100783825671723,3.0376965985929525,1.6430002839845135,-0.005107844210889385,-1.8060265684528032,2.4694283085115933,2.721220525523015,-2.0465529893583585,1.9386223667407796,-0.574735762155732,0.777727034211133,0.6945787164974833,1.5967767367867305,1.219837388897289,0.36055719891350485,-1.0471531441758766,-2.2688064165105213,-0.2304734167092646,-1.8279056797568047,1.352262209696766,-3.331487104751283,1.2814269118589343,2.5127306827704246,0.4876226143922397,1.2183581824248073,2.3573065715028667,-0.9605403545766279,-0.10840325339431073,0.2624795476345959,2.706005687704776,-0.04957408011536994,-1.2404297077101054,-1.9063839205568978,1.8914959173785908,-2.1714050967959415,2.3617369435644755,-1.774870980168071,-1.0669738090474168,1.0521318388497363,-1.3323201437020193,2.506864139238275,-1.9930385295985917,0.2783572133265365,-0.6013792092137876,-0.8375234056880053,1.8529697913786185,-0.5379928161056703,1.9813058814784814,-0.1530624385758311,0.7862795354237269,-1.281861772807798,1.2148010770349593,2.293657394119759,-2.3225266979303374,0.287289812190243,-0.7660986366062208,-0.2934103579904475,-0.31395213359486673,1.405635398737127,0.9702619935032131,-0.6627831291949511,-1.5116480285554497,2.4628241789110077,1.719524892782149,0.27046043598114483,-0.9292535614807772,-2.7474244631760985,-1.8378699786278314,2.1536844706961773,2.1447242015099346,0.8871838647013078,-0.12360979805301144,2.1405869650530094,-2.188088106792417,3.901129232633487,-1.9268095306542976,-1.5905797038027152,-1.3140046185267487,1.3702080583163485,1.6491780704341545,-1.478690087716488,1.177968697640684,-3.7430470022241167,-2.248443124305841,-0.4231822905866558,1.5810643324711164,1.772118090341706,1.2857472426768273,-0.5317426299951017,-0.27226381632033964,0.7635343706445706,2.306751395192539,1.5400268848013718,2.0339935929636477,-0.7638969786986282,-1.313999181053112,-1.805593840786473,1.7453571466310736,-0.11312770267234093,-2.266085340420594,3.43711204006391,0.07040656720720173,2.1093304100597767,1.9350847811002334,0.28474401404731936,-2.2312773653704916,1.2645583169684136,1.938606653918414,-1.415653129543951,-0.9016462896734704,0.073482968161033,-0.0340235869732752,-0.4682444090059687,-2.267825291974406,0.45507294463200104,0.03438129508472488,2.7553496780920437,-3.861349064461417,-1.9115313350673901,1.0564120466219482,1.4560870353463407,0.7264779604154484,0.36933232622580686,2.052230380699108,-0.3334280614226621,-0.9630600305658146,-2.707317926332671,-0.6334583544432301,-0.8708281317618508,0.299093382194801,0.39672644100940213,1.3751162519737063,1.3516239210707564,1.169208259757291,-1.091856282320785,1.5500379217998241,-0.5308286433250259,-1.8714024783562189,1.1880852444054637,-1.0340645177809977,-1.986660094622154,-3.0581913205338176,-0.15938680951227824,-0.857168766373634,1.037097974942903,0.646864236721566,-1.6934242829895685,2.013373520515404,1.531916238517079,-2.410238447410243,-0.7806808360871009,2.8282824661131793,-0.5371830988270586,-0.2832323957888771,1.2907718000419284,-1.153579641864882,0.8117630807524928,0.6938838244010593,1.1881066425672209,-2.9061190708259184,-1.4419984125360996,0.20515721839762618,-0.5349678195656895,0.4967379402598009,1.6236175806884532,1.5387188589165892,-3.5689938739454017,-2.96553621458962,-2.9118528177308147,-1.2454292414080461,1.6223703882423677,-2.2244722689060206,1.4848326441939352,3.451092300825895,-3.121227360360165,-2.9072639393180135,0.7609101014771639,-0.31552902930809223,0.18326990870697427,0.5675141726437122,-0.7066043228992601,0.40031665875476374,-0.7296703838907632,0.5383876397993709,0.9565671669685816,-1.4980606916629278,0.48821697792402924,0.8050176226694007,0.1757234442801662,-2.014278975264405,-0.9583680151277953,-0.9579628037577868,-0.5197147971746353,0.5442422178511898,1.6352556578154533,-1.582171491004904,2.0257027572786894,-0.8009708529727962,0.71765021962736,1.1206808386320408,-3.269564612820706,-3.784133422636888,-2.2444160363436567,0.8412121460166507,2.116099576868761,-1.0222975593410688,0.3737494114336314,-2.264601045708914,-1.2025969643276848,0.27153107329819975,1.2345252019470525,-0.148707629676518,-2.939383411549798,1.428561503789559,0.6452032575826909,2.230608871171468,-0.21308603678994736,-1.603263500980292,3.5156913360986333,-1.762232834706318,-1.9452294186902688,-1.4400678638624724,-0.28728614476161235,-0.5015071261949433,-1.4133961025323047,1.672689833220503,-3.770401417836372,0.5842774271122959,-2.3687320352638097,-3.0967235839568135,0.9102010578974963,-1.0808008928821882,-2.962444697197775,-1.4069453160071648,2.543941218382835,0.5381743438909264,2.854588464795279,3.5078581480836046,-0.6935703797811447,0.7940474851292244,-3.559420957877243,1.0241826515483443,0.9223374066314869,1.5825463090144802,-3.6536753454799205,-2.532910174667001,-3.7565860188456974,0.5041226232246113,-0.5094692245044428,2.0919392289611127,-2.621444770040763,-0.7942319090174864,0.37031790338447623,1.840707334723719,-0.08861146797058153,2.8803347566239066,0.45363280940125933,-0.26679184813558793,0.3172093938976195,1.2988265468798574,-0.04974333253981353,0.5037753449551017,-0.9762335626189095,-1.5527593392379104,0.4549159151253001,0.18259843187335495,3.392569754106979,0.8852982126155317,2.3340949821038923,-2.0515107214648096,-0.6423874023603278,-1.3188085709179034,-2.023391578316004,0.0255174694044717,1.9316628342942737,0.5678081635274322,2.294439485272782,1.3585560591515418,3.33595133806344,-0.6390383438912495,1.1116983301910395,-0.026900490749815396,-4.094680932520925,-2.408119178255609,-0.9106526643956199,1.086346664016741,1.6514167247777158,0.42244412341562465,-0.5487577066129059,-2.9638551311111425,1.5986432889862054,0.31397418820733913,-0.24629167052699893,-0.44039780984906063,2.997839098459214,3.0167710622084725,1.4649374600505487,-1.26028207334637,2.2963151057099154,-0.7397106202622902,0.4978380413862232,-1.9308423392997651,-1.240570805335175,-3.4803835978234354,-0.6155160409005757,-0.853871064486258,-0.7555696242846127,2.6919801066364046,1.4561449989980073,-1.8894897683523975,-0.910894875847709,-0.2269804562242426,-3.2643817390823133,-1.8120144991308962,0.20027382379919798,1.7276031621244272,-1.4483581607020708,-1.337997028035266,2.0474804618040223,-2.6152398467446365,1.2000196144995494,1.2843476859277072,-0.6232159558054661,-1.224713348532886,-3.4938033849396057,-1.0363562048561699,-3.449446941033366,-1.2063039489731262,-2.1285512352970146,0.9263873610596765,0.6154394385279877,-2.2606314033959816,-0.9196513941475625,0.5466652668258605,2.3767320448860425,-0.010184057441819203,4.212595192592156,-0.019995397726332416,2.3753905117353544,0.6693787879482554,2.1661988749998553,0.6135726210951539,-0.5773676805829006,0.3455824125041239,-0.2390610340310228,1.6399424958379598,1.9719179944924208,-1.3377780317045285,0.39828646880887286,-2.940450151866344,-2.3746050299579835,1.2230064700599739,0.85263695920993,-1.0060736512646742,-3.7915177920659793,2.337959074030456,-1.0108253236037201,2.341596307608255,-0.49776248000217926,-0.9773421593991054,-0.4863934104608446,1.5282513064003727,1.8791510193282184,-0.6462316613473854,1.701773199939187,-0.42765128053550294,-0.3944211098825557,1.7073030405036345,1.0607439148814428,-0.7016396589172693,0.2798578893028609,1.4114131075068328,-2.3367260282355327,2.617393484345203,-0.9322715760703713,1.2636803991947152,-3.643430512513383,0.7046617232669312,3.0408502090438536,-1.9976075493986318,-0.1675861923403716,-2.2472010858869265,-3.025152236590498,2.1881905064557055,-2.2239071064235607,0.8139960595591821,2.0064869988838656,2.06699766219364,0.709319798390876,-0.13128692284306576,-1.2232315657921442,-1.1632887831501142,-0.8833069512084081,-0.36737806942614143,-0.20806431384835716,2.406674336911661,2.0200965330006593,1.6296693206796966,-0.7187536557901427,1.8461862221015937,2.3456426208662626,3.2446132234375793,-2.344929759113608,1.8168348597360815,-0.016240386929518073,3.373205093983601,2.499576233327297,-1.4918502586905826,2.177420905090075,0.522187384901842,-1.9876012555727363,0.06528344237755038,2.084105571832428,0.9754425779872271,1.2344737161931536,-0.4812130578406837,0.7718092864965354,-0.19102589553128538,2.560030958821593,-3.0562365250995103,-2.410868723060843,-2.089758865628419,3.634633507429525,2.916781463490895,-1.1969522285160334,3.116493427981453,-0.6738009553239226,-1.2560626069725493,-3.1132959256418133,0.14969070747473814,-0.7226241988234003,-2.3091733336883546,0.2728685544850632,3.030936130368733,-2.852002650254167,-0.17554145456011006,0.09312080022853433,0.8631149585676334,0.09829215036497105,0.5690592865799003,1.1408442203450837,2.438074815216825,-2.6349260620591886,1.8299487166701631,-1.5818957476358853,-3.380753643321591,-2.5084323062164056,0.32180533488335045,3.9396272539823483,-1.3629452450434434,2.130433726422917,-0.9263582651989429,-0.5051756557860175,-0.18134973140354083,-0.14026896921877277,2.3260695098904764,0.3041981102827821,2.866440684962392,1.3022083817361976,-0.7399329275681691,2.830058748976543,1.2956008753528692,-2.54776426720042,2.646189639980148,-2.761751112508745,2.6351834923574793,-1.7971797562074954,-1.2526753260076509,-2.562823498923875,-1.5945025210802848,-0.7432845902740363,1.1728063339767896,2.093115992650471,0.9318682722453435,-0.9943762706479143,2.297841875847306,-2.816031430557506,1.124973308943082,-3.4914850655351177,-1.9436884713135227,0.13619566189121324,-2.0025477903498796,-1.8070702256167421,2.4568199316656023,-3.5201893474832997,-0.23973798773075455,2.1754284102014503,-2.373911126331141,3.226682291493337,-0.6376880957218175,-0.08268702392514761,-2.2984777789889184,-1.8540148001947732,-2.8254497453769507,-0.6571732832260265,-1.3263231677344247,0.5554460867475834,-0.9591120834947657,2.502185175774206,-1.1899803060290364,0.342982041847223,3.537724467022128,2.9064512997786176,-0.3123118551438354,-1.0894060456398909,-1.4366508442100834,2.5949691530850796,-1.4529970790612838,0.4716080696128102,2.34423759406315,1.7482452717125097,1.322728255551831,0.3323266127459472,-3.0001425594557074,-1.5599554189268234,-2.0077663417644946,-0.5907696273548463,0.5466852294081197,1.903085400382695,0.6866678306725771,0.2733889281696693,0.6913673276736946,-0.11504257319634444,-1.109077724784319,-0.6583928169076471,-2.0239865776544868,0.2785351776594631,-0.9477530500217004,2.561082792395801,0.7432902699383925,-0.9263093039665876,-0.5005343739432828,-2.808293289461228,0.5813613496974099,-1.4722770278439015,-1.9131666010819794,0.25197467393804085,-0.5361627687910169,-2.96784556506883,-1.7279543244939926,-0.22509769723291262,-1.6007227729644071,-1.057967393856114,0.9830975702826728,-0.4234221612018228,-1.9741208678816302,0.9798800226771859,2.6988720128455377,-2.6480300878821854,-0.8177180251121146,0.6198874969736123,-1.6695500058527852,2.0932899982235478,-1.9856910090535032,-1.4315689276509704,0.16190039400033676,0.4840854951254812,-1.632026664167604,2.35543337005565,0.8487122003408534,2.2859525716317295,-2.164853637280001,3.1599259208572987,0.212247627739175,-1.7221902616294558,0.9154366989592835,-2.749662231326146,2.5079140300365537,-0.491628275324199,2.0203997590686846,0.6491792537820152,-1.0207350745884558,0.5237037676049453,0.16532719506791052,-2.0124468709873757,3.548033653843809,-1.5645502827438327,-3.089127559424885,-2.140042734099705,-1.1199455605783795,-2.2958300490198007,0.8570631111664705,0.7717770821987008,3.102984223353727,1.5939590246629578,0.7983271503757886,1.5751002498034328,1.5810687689659597,0.7511516052675415,2.842951323191498,0.9683711017882867,-3.869013967291767,1.5593712746496298,-3.2902397907706202,3.236199052075972,-2.8963114740685083,0.1746743506025319,2.237833466341414,-1.3268570885204605,-1.160292162452898,-1.5097542038880247,1.33704164988919,2.837716243237091,-0.8000004804889566,0.3597693283692347,1.1305829176524262,-0.13294709082081774,1.698791376337044,-3.0040921604869855,-2.5289485806817735,-0.7662151417631439,1.2525076315577441,-0.005748109805293267,-1.3194674876661217,3.2902432013484577,1.7352205974780253,3.1380218413079564,-2.345670228248265,-3.0706942302120366,-0.2209354807889764,2.572032051074003,-0.09418148895896099,-2.7240776388557264,3.0552275335966326,-1.5320976252480356,-3.929256124655291,0.39198629244262456,3.381456297717479,-1.9888312236847716,2.034991953922004,0.3124093455800505,0.4913624695024797,-0.8310837181201072,0.020970186125383396,-0.4963505297159703,0.6803792994199921,0.35824789571367566,2.5303390978233833,2.2151886838128405,2.1487615235250916,1.1902363562960505,-2.0685677597055183,-2.9753446582727436,1.8727098043406691,-0.2642729614157509,2.539512739872274,0.18783500832385325,-1.032845419574384,-3.1703718639031546,3.824822509699144,2.8389355493003716,1.030305394100214,2.11214876032764,0.7223965567771753,1.5149476898732535,-1.7789272347636753,-0.9309602496514015,1.7594563030259045,1.758781985585737,2.138439958036903,2.4283683889760157,0.06035164999733482,-3.0966811386987456,-0.05487802482612003,-2.7144841916843676,1.8043136023805877,3.0082026873581666,0.398588482527829,3.242015616882752,0.956362014876711,-2.4573057522654222,0.2910148890082846,-1.8711902705915375,0.24256778478881677,-0.36060187979937364,-0.2291620104192005,1.2682042161567089,0.9912133017920918,-1.7662513985711388,-3.622213524070915,1.7707754843972898,3.353892303965622,1.4296183043088053,2.3676096943697296,1.8321204972273193,2.517717487077792,-1.1070122605529653,-2.8086887330387267,1.9980062218430046,0.2652286607327767,-3.1205860828844627,-1.4726917179873455,1.0675232528010583,-0.7378864568584611,0.09783470881939942,0.13828911166879404,3.3000476662200797,0.2821455266150733,-0.9206920418094091,2.7553110687607627,1.2057151273566755,2.629106649268006,2.4246897961441656,1.9044384357200872,-0.5477624082760392,0.7203315017197127,2.129326600309708,3.796225206101905,-0.5563515837779119,2.078772393287237,3.596385345169846,-1.0030844304187407,2.1654032018562606,-2.5472356296165257,2.093998876914269,-3.891877501072468,3.550682300320683,0.5736866333581296,1.0660997132870533,-2.7446912428465486,-0.008775424320743885,0.7568077704897868,0.943367090073005,-0.9229576893205951,-2.710631417667595,-0.14395686793826598,1.1397917839797893,0.35829952321222297,1.8891670372966163,-3.0811831822823086,-2.3365329782213204,-2.3746537813159887,-2.4573835191857323,-0.4280063520412289,2.089867270010421,-2.0126470765726365,-0.7148266581354071,-0.9605614538397915,1.8539157439549474,1.01972913159129,-0.4537313001222796,1.2617666001040289,-2.9947458190303755,-0.3925948645956972,-0.4366596140504633,1.4835097523022405,-1.3934893961286363,1.9405311850624756,1.73031165259482,-1.0215452717104256,2.0358025527620995,-0.8597544852364766,0.6182098657999485,4.521722513061208,2.8343425249303773,-1.6409134845745361,-0.25372008445015576,1.4333089655000768,1.9733566754027279,1.160534516522331,0.7482061390843535,2.0709556489154832,3.2666321602334385,2.1724026454616356,-1.0226841803393336,-0.713848272824504,2.158791246591998,1.6261915208459676,-1.7057583285043925,-1.3737163979942848,-1.6442931498482527,-1.8969074569113873,-2.0486592359777918,0.7779228292834489,-0.322636265831795,1.8544837454631116,1.5487122087017342,0.9320113168848115,3.839838418699443,0.7534618467247307,1.6535486521358935,-0.1669530290558907,-2.4866044258368616,0.3381073496325764,0.07824213613996298,0.48160153624239527,-1.257772566610746,1.3209747168721127,0.9335264898974889,1.445485847786388,0.36070361957412983,-3.1250118183721614,-2.2990580867768196,0.2579761990857601,0.44803523150476043,2.405791918473063,1.1314329555411797,0.5327158396074548,3.629987330452764,-2.1324648421404735,0.7301763539963452,2.5615924058892294,0.6783764125662183,1.9290536088585946,-0.8113665087148211,2.781414316656932,-2.3640678922628737,0.11860902203568602,1.073348154424732,-1.1204328171735591,1.7262269682877682,0.34591755749102204,-1.3950914614392043,-0.5006852690658729,1.8631270026204776,-0.24283078099046873,1.126625613960058,3.054461855440943,-0.28649798336749277,-0.5088139644163552,-2.7451358380162842,-1.9091471805304192,0.4592382765962111,-1.7799869146016076,0.9787477810631913,2.1129483401215516,3.540640634185373,-2.569769629872084,0.09930528041645867,-1.5944767038796643,-0.37269601476713066,-0.9027980834076272,-0.18830434521923217,3.1161838678064124,0.12024757435955766,0.7021512722827878,-2.0871491698615077,-3.826807341962478,2.024732265149297,1.0632298095668677,-1.2555884644763649,1.977908401642617,-1.6516966684185979,-0.8609704828434798,2.0677960294911264,-0.3954435781822465,1.551816416133089,2.5160058507027054,-2.080958405674849,-2.6989040314165633,2.15061941571021,-0.47392492722493657,-0.10595944875167729,-0.6766086132111574,0.660478829843723,0.05315299356587737,-3.1240536145619036,0.046387189627113926,1.741321263642222,-2.3636484269652067,-2.396406272804119,1.6398649817924524,0.6269620611727632,2.1416191517125385,-1.7450725472565736,0.7976836718284079,-1.629286756480561,-1.2911840917323032,1.392534453086283,-0.47027081708387325,-0.46521431740103664,0.3706328593367479,0.7313549327227498,-1.5309865406010035,-0.4819157136862044,0.8942432782388006,0.3398107471611677,-0.046375947623907086,-3.7826869505058576,0.6148673973926873,1.3710028314346054,-0.2930066135046055,1.425511038772605,2.6443355408034446,0.5228532688661777,-2.968330398137916,1.311970915168358,-1.1499229472794021,-0.627115463929619,1.0920581196633707,2.1027282058509367,-1.2299051000600525,0.25280765604449934,2.3801026410030963,1.9630488185290933,-0.2923652035551304,-0.7382687011554994,1.427531673255996,3.030563672451703,-0.7543412557206584,0.6321820532665138,-0.48704150307002664,1.3711502879358106,1.1297572723528115,2.8160115705328184,-1.9254727696714338,0.07257719792900114,-0.6383220421585071,0.12139242406837962,-0.12199484444627383,-0.7118234642980269,-1.233853162101926,-2.307968065685275,3.120194252434334,1.1733932063090498,-1.7753967120885692,-1.733415347671185,0.5844194116695308,-1.344506290838944,-0.3644318255074698,-0.6868033923828409,-2.0907712796160953,1.2119410451628247,-3.6817041887732316,-1.0324940744933053,-0.25687534264185136,-0.8596436107341683,1.0886752387965641,0.5602115758795938,-1.605128826659563,4.09016002972218,1.7622842666217329,-3.0012588986224786,1.1552991996532702,2.709672763197575,0.30747753261087013,-0.5904566077282357,-1.7537249016622503,2.4749074156144455,-0.9368992303927652,-2.8236657513894063,-3.5696125596789687,0.9073573046552954,-0.9698101125458566,-2.4324079762255555,1.056617525071199,-3.705728575744619,0.38448006106396004,0.1627314801069336,1.5135025612303021,1.3355733767280193,-2.6930028875112852,0.2582043429836682,-0.6160312839148265,-1.423537073654434,-2.3591984388452074,-1.128517908566639,0.35534950257999004,-1.1511918846899107,-1.7053873227918808,-0.24133350760518082,-2.410879185607477,0.23611552714866665,-0.21231556271357385,-1.2552398761992967,-0.9637301601458858,2.877329926406258,0.22995028695631506,0.5185756687256976,-0.8722344877918032,-3.2740370403410974,-0.1834340874735816,-2.65261714289511,0.17179026703834363,1.7843315126865826,2.470578972744123,0.9613174897636523,0.8303940962214614,2.067405382346051,2.2333258876777444,1.6442901442095028,0.00767295698836087,-1.7640800534171583,0.37805059392206497,-2.0896977417716913,-1.1837067180615064,-1.49416746583243,1.3762101247126133,-2.8336017182216775,2.387692825115747,-2.4049719956942974,1.7217516139557845,0.99603271063587,-1.3181247976952906,1.355811692823589,2.2072776101668783,-2.417773027253359,-0.0245638558445837,2.019762676395675,0.08770542935068643,-0.9693136900282493,2.9357708251941457,1.220212628850952,-2.9416568157313625,2.1002638760591315,-1.5110697917382332,-1.1603677325318595,-3.0457464496696023,-2.312180880781617,3.375591682994386,-0.4663510127835821,0.373388161250295,-2.121077439766598,-1.3152469225719705,-1.9584888442688624,0.34567905967224677,-3.2572146466313203,2.0413445087581548,3.9285168597855287,-0.3975256040647865,-2.2928616694056956,0.19601119285936966,-4.105388785665383,-2.198568085041092,1.1156100489614618,-0.7025339219058501,1.3215850164644316,2.248456687646053,0.14998663952813124,0.6505776275373489,3.626640286225894,0.3718116957766724,-1.3450578926618344,-1.4200460236324797,2.629241303980889,-0.20045640708373,-1.0034496814953018,2.9287351540139,-0.3865414203081432,0.5921305744405075,3.3497128503716165,-1.0434440093254032,2.7372654708951503,-2.8810698940116035,-2.990332863204567,-0.4098215570215493,-1.4469133128486322,3.015991279738335,3.0191545513071474,0.5299507172380318,1.286394296597523,0.10651630308097022,-0.28586491553316884,0.24484142243331136,1.0574736564771328,0.34049420547713877,-2.143527951879131,2.1028077829717264,-0.07127089935810171,2.742446259923535,-3.968197866009392,2.739734315854745,0.14388376630024027,1.0117940783810717,1.4598937608438007,0.4280246280257035,0.17116853306315558,2.6741250840072563,1.777708760021943,0.14060272744055174,0.5118186469205989,-1.669623100430223,4.859322985898154,-0.06327207238248765,1.553767893638557,-0.43069989375132095,1.907624046989971,2.084430249679819,-0.02138591392406592,1.0499580405759013,0.4047025104677374,-2.2983408232382696,-1.6218933440152454,-1.6095072547422347,-0.22346221380555034,-3.8871739124447484,-0.1127963887216033,1.8659896283528226,2.663934303936203,-0.48651754389130325,1.047111469813292,-2.7235525445669477,-0.6082480969145847,2.1161652331090144,0.46440211015710897,-1.4273306333538718,-1.5429917182214221,0.2443265290759487,-2.3135401337437727,1.4150773842270248,3.2159649005854187,2.4865658398839177,0.49425936911765883,-3.143870443126486,-4.040514464433635,-2.536576433608325,0.5013010767011901,-0.590164695295255,2.33480309272706,0.5073689344879145,0.11462571414131138,0.40842123984684664,0.7201702285662711,-2.38290044148158,-1.0928540652015892,3.894985248368382,-2.7811348705645105,-1.870838826564397,-1.7634979172531966,-0.018861612949866918,1.2805032214065473,-2.075488365256107,0.47672077895159415,0.32178009418539594,-0.21115712001074252,-0.8798203722958333,1.9222286493171368,1.2827748683490334,0.22091062718848872,-0.9474202685268688,1.693530499285457,-0.9495617195033572,0.13124807595220644,0.297681765406982,3.8121101880081296,-0.08473123742332148,0.3817536825979663,-0.17088556746527722,-0.6690614482007232,-1.76759561691937,-3.1324617177221263,-2.5289013137207723,-1.8497220573688697,-2.934238294494433,-3.0064646290251575,0.6955030281565178,-1.4828336040466366,-3.29948434167462,3.9934848780896974,-0.06837602247093402,0.19304100880790095,-1.160206709553596,-1.7013328228021114,-0.13330073967880948,0.1845683400087428,-2.8101929073869862,-0.5040654150104908,0.6807574937327223,-0.6277956412371765,2.3098103395369374,1.4769135660015356,-0.24971401662233225,-1.4331043325689885,-3.1764535957707327,-3.077126971819803,0.8101170298728056,-1.5712848841227673,1.8057494269630214,1.252246300799158,1.1976492694408425,1.9515967770067886,-2.01176239808608,0.3859998814643129,-2.0103407386426095,-2.1870868107897943,1.519400622070143,2.503375440444401,1.6868991646843148,-1.2565570111719608,-0.10642492967674187,3.6607545728582496,0.9261315291844012,2.80865462577879,-0.031237262978645586,-2.16688766172568,1.7695372906025149,1.6893085573512547,1.7024766923454637,-0.4868449319088915,0.5955639822476363,1.0764143997306073,-0.09954533191438995,4.043217160593806,-1.4207106801378002,-1.172164546499304,0.4169863685952708,2.797739474543857,-3.422340581311011,2.837238349352818,-0.8698474828697027,3.0717563805598824,-2.3840339824483245,1.8628609713564492,2.523689651526779,-3.2956119328990954,0.45076437438412037,1.2227894841584417,-2.053966757591046,0.5762580916873293,0.44406184845320623,1.6970130378184027,-3.079428370011241,2.4007845991424186,2.250310186284252,-0.06496738133148547,-2.4210133620563967,-0.5369043264206199,1.304909041762355,-0.5833745955281907,-0.860835657736295,0.42836697855540523,-2.8539999257172433,-2.4783146664737457,0.6986917232369834,0.008678087512428727,-0.7148155348955333,-1.8881562877018003,3.654918066950832,-1.4909608251090618,1.5740095830323713,1.9069700094866417,0.8068613692474893,-1.345686529889403,1.221243107838776,-0.013412432521931882,-3.3416263329728,-1.2459829279342218,-1.8809692871602417,-0.3211337225215904,-3.686360861038996,0.061868502923848434,0.719589250472571,-0.1705027425071881,1.1997332774753884,-0.6739236909186304,-3.055778454469032,-4.118114137860346,-1.716862710982834,-1.2502637964874295,-0.32325341979650535,-0.43202094107430306,0.7638282883364049,1.1364210871953226,-1.4021526325459537,-0.6497535387764761,0.5983360322888255,-0.5952051377064806,-0.305481836562538,0.8622958794923398,0.129738828347363,-0.8926070579121488,-0.28959527750194486,1.211326362011807,0.565972393306105,3.525390593718652,1.3743789711488403,-0.48137867402404,0.9026838865817459,-0.3204884857424056,0.10747837065873592,-0.011172114523219647,-1.9122243825966208,-0.5390790029440217,-1.84121258788776,0.25052109250851706,1.5907790146874898,-0.6361710258732124,-0.26842313120848793,-0.3315864572116574,-1.9215836537216566,-0.09213409483371988,-0.1581579544179722,-0.09583081160790939,0.5398379975925128,0.26095364026268714,0.4922700179510119,-1.9085745999127528,-2.15285698745049,-2.311714875645837,-0.32726820124144795,-1.4838360682272707,1.0236400387498037,1.3003185554004837,-1.505620543604253,-0.9112916845859793,1.0198860426177538,-1.7488792035331884,1.9781339140693515,-1.96008528183844,1.874338585526889,-0.5590769195315429,-2.1282962956470817,0.9265434428798776,-0.3984517242495501,2.9242101660649484,-1.6122907850633585,1.2264223383351573,0.8499746869931588,-0.30670635725729983,-1.626074612590153,-0.15749219926766178,0.4251546127342177,0.7058675964293682,-1.8700970780802963,-2.6730944811773103,2.1244132869992876,-0.5796294633008541,0.12947055605509486,1.2633507124706065,-2.5479973221575967,-1.884233321771921,-0.8150397283519509,-0.9984824741868066,-0.9768131057606392,-3.27987765833074,2.31958763590997,1.8451469137820264,1.0529674979790766,0.8895267098608536,0.7259638770466551,-0.14701045443224764,0.47614240466138397,-1.0011259846788914,0.1400207396125958,-2.6016483595088005,-2.926204211618752,0.29247368148872677,-1.2249767347630065,-2.3988984540733456,0.6778657437636197,-0.930453709044566,-0.8423003627685244,2.1065600161383333,0.8533343477918869,-2.3393598964787135,-1.2185501717411698,-1.7491079864018577,1.6771436483859543,-0.5897725676254931,1.185996497508463,-1.4173960425731404,-1.3588207987115692,-0.28393926989718715,0.2945345481248371,1.4694803783726866,-0.36765489792546996,-1.5178048027094775,-3.562344358435851,-2.726572970026497,-2.309007817721122,3.931470646867153,-2.303027797819539,-0.37925298607513885,-1.9357068729636606,-2.5191105130055313,0.5617823101445917,0.9903277214413377,-2.65301107989935,0.9311702354778066,0.30271799421491097,-0.5931212516940961,1.0233357573497066,-0.668594135380995,1.4052252414847433,-0.13017417040483673,-2.363090767337844,1.9500721863784325,-0.30449021062166964,2.1814501716148524,-1.0229316769818102,0.23301427223169294,-0.4968322010533602,0.9459657414264248,-0.26981100304854666,0.8202251906077136,2.5818724159382507,0.4623921948119183,1.0897684277511719,0.5063337061910849,-2.3147157729641537,2.8838331842073575,-0.5438956070410831,-2.5559829582006146,3.235852078869402,2.0776195259662074,1.8299893438524353,0.21258809160868763,-0.6083470539519013,1.8357677228713956,1.5186009826064244,0.04752865395469909,-2.8572049372769297,-3.549407402991103,-0.20503943263554733,-2.5818845796586882,-1.1389293125529356,0.9478757961031172,-2.4350501692211264,0.12232017257634543,-0.8403478694213177,-2.4504485496118975,0.4515395151114922,1.8877877408545496,-1.208222329908568,-3.1169126216484315,1.6632015774300668,1.0187297659977468,0.6761775559383613,0.11388879137851927,0.11727692597330877,-1.032051951447708,-0.952076626674962,-1.5316847999750929,-2.6726371882000874,-0.6466497150506572,-1.3828520601930914,0.6625531106918433,0.5100068009910603,1.9914492862772581,-2.4897535036720417,-0.015033162932211514,0.8937862898687207,0.1100216207142991,-0.7366411775944536,-2.8075383458194856,-1.5734056238768386,-2.2430018960120037,1.7340925074890752,-0.8900373741560614,2.998290746752556,-1.7432665871964739,-1.1584553045419959,1.317158613896622,0.12818906896521634,0.39319154935137374,0.3360020123672966,1.5292673285463363,0.38618342306850323,1.135473237862381,-1.1867668079453813,0.12950923209929055,0.9502545775205229,-1.8224646670648488,1.3530670084364584,3.461724476218563,2.6093639030058546,-0.44398988214194257,-0.43506049663284996,-1.2550884751298752,0.07974271167402,-1.6225720880832546,-0.29841937008540803,2.435684278544502,0.7883485023529295,-0.9544023595642146,-3.0784522655414848,0.7110774785913827,1.2213637298431907,1.463968086879599,-1.140729327227746,1.106623279585819,4.3122414092267976,0.41663016293859373,-1.7649936110514233,3.750759273266131,2.972502649653753,-0.05014314098731017,-2.2165489328530024,3.3191859805658455,3.9551383634076216,-0.19698316466178165,0.951144087804547,2.6034809885064707,-0.07503256056944686,-0.4418768756985197,-0.4765923721141184,-1.1086504514087905,-2.0940390116303074,0.5592166590213862,0.7653959071447847,-0.5026232020046327,2.221385361919457,2.597164847473922,-1.5130368871090998,0.6492912223174017,2.4957514908098983,-2.6014683435078205,-2.192277692237635,2.3748658826781637,-2.118579973531935,-0.9375670869661834,-3.432616231220477,1.2617162392109824,-1.9110284821884551,0.3209860847000241,2.8763234477601225,-0.5895172141837698,-0.10058666065946464,1.1304242082064082,0.30654984680371633,-3.5889404212620386,-0.3975995107380577,-1.6999818279782137,3.196362828632445,-0.04928704680397907,-0.9753059466698816,-3.105584514275027,0.11137482730852412,-1.9795514047088616,-1.46605721987202,2.5506097709897104,-2.3137359283919094,2.797398891400633,0.03743722833713161,-0.7335632066522023,-0.8261369109209196,1.4563562557380685,0.61642471399849,1.6995413200216194,3.2138974988588993,0.3069558587481303,-1.524195463063444,1.0822606447196732,-2.536594708740511,-0.6365145380935179,-0.395327942451321,0.5190413561479861,-0.23467211317531445,0.8145637526054788,3.2625732121323603,2.3379961492280033,0.8598827372348204,-2.6090856884769607,0.11248997833796308,-0.27591732490380316,-0.1747781179895401,1.6303153350114026,-2.9047508352366798,1.4630631450984848,1.2104449964060517,3.053882578112483,-0.054260470331317534,0.03831595925724786,0.9126726683250287,-0.7585123642080533,0.48930289152488937,2.0378303592356892,-0.317076925161845,-1.94077151869013,-2.8065881111178475,-1.7373102556969617,-2.1993240758967594,0.21788462987866203,-1.4993207708000074,-0.23990454439980835,-1.8075365303224977,1.8601971166210458,-2.946973727976802,0.9538443128175392,0.6900916144853322,-0.8345358456894963,-0.6036389075586596,-0.4675353459521653,0.45775907506204616,-2.634543248150804,0.3613495599544603,0.11743656547051103,2.200511943468783,-0.814134159343226,-1.7429690057168359,-3.222084752584736,2.0278605761383646,-0.6316609953473171,0.10882423888026226,-2.251304950087266,-2.5926812429770383,0.4810209064605728,2.5137743327474635,2.1290640673885473,-0.9987896040560575,1.0265468919410186,-0.32984352264637606,-1.7641089646368346,-2.3108827688387636,-1.370639799658813,-2.305632910411914,-1.4105884577740326,-2.275502329607587,-1.3272616060006062,-0.3407072073840875,-1.3313876419200643,2.6270415799780253,1.54962140872762,-0.4010731122831006,1.1985044602423605,1.2709666117590177,-1.162412989375648,0.45142165036999954,-0.16035678537652515,1.0630709607850553,0.6002715308989097,-0.5111113928484139,0.1359072489758204,4.2296554222027565,-0.7395981469470505,-1.0774366952890653,3.139678469518958,1.4039161089405798,3.2823277151094135,2.8546441267822833,-0.05122965751325705,-3.8386185433132782,0.648159051519433,1.5029779883930519,-0.5128536318441137,2.8597937036737986,-0.6858997815035549,-0.8728428903417463,2.498693030854128,-1.6406395866184265,1.33281720896901,1.2903978309235553,1.8137923250430916,0.4099803386804149,1.654881054452912,-1.1323900825976756,-0.5451235987835136,0.9135076574281464,-2.043276816478644,-0.032205056160308755,-0.7704659190822317,-2.540790560721719,-0.48755465515995866,-2.364144760501896,-1.6763574695111847,-1.109273966104269,-0.3084357476083054,-2.932575038726138,-1.1310170317451735,0.1898855422053841,-2.5953204650496655,-0.7271906124799608,2.9805594281485135,0.8288544702395713,2.7144908835306154,-0.5805688122510404,0.10892852629487922,0.9019215283997108,-0.011915211866768685,1.7389393925907761,-0.9208423346591532,0.6217979635894205,-2.0649466442453774,-2.6433167442422922,-0.36778860799198637,-1.4417916013863465,2.9787915301696217,-0.3762334308170544,0.1611680345872011,-1.4648561256065549,-1.8098750547431766,2.574006127882615,-0.585534684060655,-1.740593007698955,-1.1489326064348873,2.5802092774994665,-1.173857472901455,-3.659987129430706,0.9538559541690244,1.5780622719967758,-1.3796798671650041,-0.7221068472945017,0.6534173636060256,-2.0427096797403674,-0.07651413426575628,-0.778781293733046,2.066169541347761,-2.252712313223033,1.1080407315040353,1.0621191136242543,2.5572420508201423,1.8535310059212187,1.1070876696266765,0.5632855518335207,-1.4763831313300901,-0.04459700748898764,0.8320809779257777,1.21775170694762,0.8165479288468833,-1.4911120328723373,0.04863486374936586,-0.3068420313790246,0.8593199325550854,3.0737413387850108,2.3479679367631183,-1.149309097011962,-1.0739099086359747,0.05470693648547078,1.1783363932110698,1.8054743778531839,-3.052817993573073,1.6536187839705836,-1.7738414715305741,0.9733503188389087,-1.0749700876899924,-0.9915710214254697,3.02347142970649,0.22711777775662947,-0.8853733231145579,-1.7290174360817077,-3.2652340780807907,0.6496477467702652,0.8732554950487842,-0.9259119013061252,0.289099948153428,-2.5253039216098374,3.562553413443679,-1.4450358702761172,-0.3858787393271394,-0.9067036010087313,-1.0551759215437793,-0.36661939709721747,-0.0638681801249195,-3.9196117196180147,-1.981097170699775,-2.7086979423731505,0.5105909026924194,-0.02798569151409392,1.3764802559573452,-0.5978255296562234,2.779647417490591,1.4015737531244457,1.001100751951109,0.7902716390307152,-0.8997116384717838,4.167153630362911,-1.0294976559485418,0.12159567878798601,-0.21139610853020815,-0.4811645944822262,2.0200927615068185,-1.0028664442105464,-1.9542030161057076,2.3149744544755926,-1.0442683897360214,-1.0848294943134935,-0.7363540347195482,-1.2940934084234021,1.2434097393000945,4.270648156882858,-0.7082788552792323,-1.8796601839899918,2.396250961748129,1.8033136931222724,1.1817581183806318,1.934707569739519,0.6916185066703636,-2.3937732446936684,1.2743389001788632,1.6184223126915287,1.9785055277576726,-1.053876853959308,2.7628394577085658,-2.9359433453306556,-1.1007572070404739,-0.953204459769551,0.14361465116640204,-0.9691629272280079,0.30823351450779124,3.8081588025394844,3.8582781191609974,-0.665562773558706,3.8218863269369665,1.4579135982059515,-4.154772882980945,2.8180710374897306,-1.3231420574982813,2.2895837900387983,-3.2149230959981625,-1.9606415396539176,2.9211014098274237,1.3914220817765632,1.4310547839011918,2.1150480348428085,1.1362662748432564,-1.747256779496556,0.028275163019772967,-1.4342781592340799,-3.2795389194097884,-0.26502380964209604,-1.1330650583402224,1.544185122701647,2.734666338072088,3.5134123906019603,-2.331128624208546,-1.5642220976228425,3.160729616880648,3.216877775220648,-3.332086606740993,-2.339900580340157,-0.4122846578379003,2.304935359540058,0.9066606365660561,-2.305696807644132,1.0530883740664865,-3.7360555615780733,0.6637766325290912,2.3035781756614133,-1.9505943544936621,1.2807291089864408,0.04020621754263394,-2.43465673513017,-2.1983403694239985,2.0314781929901558,-0.9784479615279168,-0.2728175864337116,0.8364768257258836,1.1399861299823795,-0.6152599808300909,2.7726840881036066,-2.907025723248125,-0.7277776501275824,-3.006351724798408,-0.5000004232481577,1.3211063741409796,0.3572547986876326,-1.260163343125305,-0.4003249334530155,1.929390790595352,2.4660795369838087,-0.10041129254502178,2.4188276617912803,3.172310045887268,-1.6589912692463336,0.7170192312635246,-3.157076449924348,1.5746347303177182,-2.4602956936860387,0.9370089001982795,-0.9465550432074294,-2.9042460427320744,2.3874192417626148,0.5777686545606999,-0.26795976303629027,-1.9215514868538424,1.5542702329978058,0.3494672342881531,-0.002390527796728511,-0.32551672909007984,-1.8358121292990635,2.5630422040042196,-1.6270312954045134,1.5296433216633263,1.6136776740350423,3.257444402325317,-0.2281253755345703,-0.4294398653249505,-2.7687553322618763,-3.095473465780088,-3.791306189011916,-1.917165454107357,-1.78589748960378,0.09084890402123658,2.1208287826024006,0.8588692604415923,0.5135828963805495,1.2207898551708682,-2.3750384619280522,3.402418372091303,-2.5090776662540835,2.669819714794018,-3.3073662588656694,-0.7139488211304302,-2.6443891091497282,2.4465529377278425,-0.14358099944548144,2.703932562980582,2.321287261749284,1.2136839886457484,2.0531011709749,3.2529587032081935,-2.7797881391677755,3.2846197977404032,-1.1973962522937445,-1.0372232546749085,1.1648847467339847,-2.126184569500117,3.3882724926594068,-1.5038552826046259,-4.023432086723154,1.4331645234951038,-2.4278781946523567,0.4960696857868283,-2.0460270438833006,-0.6214076463671191,3.4731653630974373,-0.3705231576519545,-2.6393468142115317,-2.789703836862123,-0.14212125892591393,-2.223434427086236,-1.3863350659897933,0.9560779145789255,0.5969393239702614,1.7899562508851368,-0.9848775847747218,-2.915001794544993,1.1747818530981025,0.03751660705195566,2.210136864271675,-0.4001751499645682,-0.6074539354352555,2.087533902899085,0.83890665602599,-0.6440135570987211,-2.11171858079028,0.010997360166996952,-1.4825860058796068,1.2602730999439005,-1.823904513511264,0.7632716027011021,-2.584537897110823,-1.4396065217239373,0.6099905741762656,-3.8600486334357504,3.063015275238228,3.2281416815148405,-1.064345467994255,0.8299722916111775,0.35805799945303357,-2.4505569823749798,0.9984834356336606,-1.7157093711028801,-2.508318157347601,2.7826864124412354,0.9069562182541234,-1.9322697392808914,-1.7376695034591023,1.9699323477091701,-0.9473608543012432,0.5617045623697089,0.05190334888926758,-1.2786409263751026,-1.773248603772377,0.385167708236675,2.143926828311592,-2.666656001931654,-0.32952099987899025,1.8129413530401421,1.1233177767437104,-1.507868465754149,0.0010952887576180365,0.522856734909836,-1.4811384579138527,2.286787137437737,1.281091516364812,4.394984249741109,-0.5884484433825908,1.2052620712626423,0.04282880680481871,-1.210230235474211,-1.8725409156594635,-1.2007859105168757,-0.3548499577745093,-1.2116766155458392,0.8294626514304038,2.2495467184834865,1.9447693691532306,3.116066947265013,2.5360338411947114,0.5891927014959489,2.4243905086839,0.44796553531451927,0.6340868237469058,0.034147579192801564,2.6423704334862017,3.0695154885843383,-0.23563295575037685,1.9806938005217691,3.4846983474735302,3.3832179365547286,1.3432133588318818,2.192676172866487,-1.6253484613839975,0.9671965014611895,1.1261526379946871,0.009698782224895697,1.1873249868545033,0.7695072530267201,-0.5316426770078461,-3.1291081536392102,0.43945684643922894,0.35886100299479656,1.8887321747449755,2.5409185075711362,-2.510859117668209,3.522280810457542,1.8323637381303104,1.7635188575559768,-1.3250731468939592,-2.0377098445053528,-0.611329959500377,-0.41227057039851117,-1.7972117728445367,-0.33049227752021054,2.0388720656001103,0.15195182329425513,2.628850099626904,-1.4450110657490778,-1.5385065816350103,-2.170943709526438,-0.030120149928346582,-0.5305050487570732,2.7578088598262513,-1.8045983935280392,-0.4921319572082292,-1.6927408281305294,-0.1107944406985068,-0.9612892425778177,2.1865710848608813,1.9468380438043364,1.73184284493355,-1.2391078560591158,-1.3179016792250215,-2.30808552210772,3.3029850042575917,3.913273702575822,-0.7907366999134882,0.41200926462601445,1.249466465632002,2.8459376868576154,-1.5884612928532955,0.22446169387926232,0.32053773821431825,-0.8223386860039865,-1.9864351728490592,0.340794774109664,1.569103295341172,-0.5830263654955752,-0.24183558607951908,1.5481197173972931,-0.4043883020843854,0.8190143333254802,-1.675126401829638,-0.9780752836073195,1.5741809533624629,-0.42170339084910197,-1.630576460510734,1.7314915957027144,-0.4560532777051387,-2.290315950936779,1.9813239531476114,2.3523634652885588,2.590935840628883,-1.0557845881560122,-0.5614277850105507,0.3323800687652367,2.9920490383681955,1.5306650760932392,0.5728855849704798,-1.175825171653519,-0.8102742318497338,-1.5244462972765611,-0.28152060240060534,-0.49908678173818843,-1.1956267301459134,1.1818231449171885,0.7705141382909515,-0.32348722494780985,-3.447994897975126,-0.8060356038531792,3.3698057023327266,-2.7052987722754596,0.11057127400660899,-0.5440317524521827,-2.536256338592973,-0.7056100562505191,1.91352153431578,-1.476599587497105,-0.9843286886431667,-1.1879672154373118,2.212643571621641,-1.5357280385436867,-0.6392755470679112,1.0652254246851882,1.784387771024467,-2.5018777874143683,2.7793071368183186,-1.5753444607565348,2.5323909214479503,1.9080524143817743,-1.8409467634238899,-2.1153989038770993,2.5394056365651445,-2.441968485661993,-1.4189009248788094,-2.1494085792260305,0.634613469657428,2.2604999049143277,-2.8384489966387,-0.615758003843182,0.3293119295204651,1.2630213324197817,-0.015198428786404355,-0.10493095267847592,-1.0410667357836416,-0.03193852576077115,0.8778817895668153,-0.23844257547598885,-2.928265521411093,-1.0269243809671547,0.6431785813350692,0.9826182236649295,-1.0717686587242203,-2.9743590668821,0.15951464054640224,2.371768089492561,-1.2732303643593799,-1.5491045280691624,-0.7039870958666212,-3.7929247517869222,0.35396168332578426,2.7948958357026195,-2.039368457095533,1.8615138188185827,2.4948897755140416,-1.6765038004968928,-2.361011336579301,-2.237778272324231,-1.3696660331169286,3.5014979943035245,-0.9573888903333524,2.548300349006162,0.30443966812014106,3.2666431397400695,-0.6880977588108256,-0.6435958443587666,2.8270069500989154,0.8880758878333592,-0.3558183485439729,-2.0347974421046224,2.889634336877386,0.867016582914803,2.014380235012401,-0.8083728677971715,0.24218380347623422,-2.4606194506848356,0.2010150472463124,-1.178507135807873,2.172714894492735,2.302696042312714,-1.4103685830782373,-0.9420014482391901,-1.270403588942569,1.3285573501781933,1.9027825179747324,-1.0814175205565733,0.19831554199515855,-1.9934036164852063,1.0055814667024412,-2.2486865663886864,0.4306976818406852,-1.1119167959671146,0.7631654474650369,-1.3607345022673072,-0.6327666758552422,-2.841613530503014,-2.977037032120504,-1.3961006218980754,0.5630449341518818,0.2835717474353246,-1.645538553805871,-1.4225034096022435,-2.480395710577645,0.3685782050435462,-0.36735363386607184,-0.7837653065688244,-0.07085783775148613,0.12448602927277627,-1.3709936594474814,2.362135035897699,-3.9709293326552424,0.6946877804763713,-0.10563537522278721,-0.5553572218889381,2.5095496857170336,0.7845483700886725,-1.5996369037216143,0.12311785224091835,1.9670372204370747,-3.6356501215963384,-1.8051258396142733,1.2641718265517734,1.6704449458681014,0.08117056784756135,0.4184088950920476,1.597718354153662,-1.8695865765803488,-1.3611279393126703,-0.42530579355392645,-1.8490883506763585,0.9817719266571749,-1.1165637838326097,1.0751570610900842,-0.17984371444366498,-0.44167746938963653,-0.3311453328843604,1.3033009529230446,-2.656512503606973,1.7832797862470393,-0.970257948888376,2.1448894342741047,-2.0394325180468247,3.305144738681169,-0.19063480630430604,-2.478238981903307,0.5538758156254602,1.5184360313860896,-0.738658444186721,-0.5963720819365502,0.4925912827976574,-1.4950455315242759,-1.688398605446298,-0.9424545097790424,2.3202450003262918,-2.2504896446657585,-0.21301685778645488,-0.6583432605020237,0.6371702043958577,-0.9340207572737648,0.9633099652486397,0.2837133323178411,1.0101890249055552,1.1941746039854144,-0.8631035054391925,-1.5698119495198648,0.8253349250403242,0.8636301795243342,2.5780983616815982,-2.675548800616654,3.013324622899604,1.6311484886831755,1.1540593740597296,0.37675922689906155,-3.10995792850396,2.356593280396287,0.9021700082454311,-2.006237299367136,-1.576964565438727,1.4023544783078827,1.0156404319360552,-2.5657803584396452,-0.20966946088439153,2.7091156068839393,0.7482354737556244,-0.8706119375266793,-0.3162421743655893,-1.8325308927657888,0.8069604369155553,0.7778677567911295,-1.083384633740037,1.7957311154065172,2.3921182521421174,1.3842068703489439,-0.8441864781821536,-0.638671225094948,1.4272007104327589,-0.10361630820103633,-0.21232131857004072,0.63439369125733,0.35221396911350406,1.1921172325200964,0.3587348079129007,-0.8187764802948264,0.48227603647726425,-0.8649936357825576,2.1893952291492793,0.721558470455927,1.3035771862771637,1.788586546136783,0.15463635893533412,-1.8226726221447906,-1.5868886058978444,-1.1974284210933566,1.0925533392520699,-0.7294459219114128,-1.1513064765290144,-0.3347889208705896,0.21648630436934166,1.503025603595374,0.5468668976530463,2.1095176566076637,0.4271281513519614,-2.1936477321536962,0.8240032389813562,0.6104709417250166,-3.7244471412194176,-1.3771101302971853,-0.5786563215548093,3.1397216173841294,-1.6778526644723797,-1.762577821808185,-0.36809306872192377,-1.7082133913088178,-1.2249082827263624,-3.7926443005410517,-1.379382708816453,-3.1879262384623916,-0.3115794541236614,1.0893914050972417,-3.1528181625695866,-0.16427783304471114,-3.0483950986892103,1.3436292802636431,1.3354951164334983,1.2933174366230331,0.6428169309288001,-1.9724343109074312,-2.0128487600433114,-1.7493640081479238,-0.2905617744050389,1.067077313499997,-0.3300385564874143,1.1530169914457533,-0.10236580334756121,2.0929083911093844,0.4424049774681514,-1.4514467526947739,-2.4486586935686354,-0.06832576232123637,0.7745247672896051,1.5807563337234014,-1.051862253256043,-1.0033675515351137,1.68988053672931,2.2332036011212244,2.8298615262193505,-1.4864274375984647,-3.204458543206576,2.508425242639876,-0.5615551046373907,1.3345299907730475,-0.755095002341162,1.5828088754605565,-3.3600225372463455,-1.0584030145071226,-2.8226808783520405,-1.4846783966215908,1.140416852160703,-0.02962722647743729,-2.1761591668846973,1.4868237340946706,-0.24689408870986826,0.7666618695252884,0.26940403227731935,2.1736632643702745,-0.005514173270212729,-0.9500587936080653,0.8012478788672072,0.2612255583717853,0.18224179960988918,-1.595085322665827,-2.130213787667136,0.8139729720283961,0.9435376119407134,0.12620569078568347,1.440885296815173,-1.738439147910621,-0.3022856477333147,1.1120734717151706,0.08940701564271171,2.2381958570604543,-0.5795076655496334,-0.2594982417635092,-0.3792992312653163,0.15651522293556114,0.22782073991494287,-2.1392619374519204,2.593125206605605,-0.825499833314924,-2.411242082637438,-1.8359273157996217,0.6953943885426805,0.7825875881141732,1.193860600521813,-0.0950045854490538,0.9727312236686191,2.9337733003370614,-1.1197702974374426,-0.47794606335004497,1.268816364089673,-3.203159746719401,1.0372197336224158,0.7091057329095599,1.9465220486690205,-3.273755553692715,-2.102238816154047,-1.0815168012182317,-0.9012311407652754,-1.2168967189835207,-1.5657732005760505,3.144207631388272,-1.5053128490301266,0.3123596714663812,-0.8542739030611489,0.6706027424115361,-0.95830564705407,1.0930233481211908,0.3642732978589739,-1.2413581226294137,-0.24920259217089968,-0.9129222351293379,0.5035155465657223,2.4797052219085916,-2.633064460447875,-2.5729484581435913,0.030489929879201776,-1.1633594218529566,0.7467879973320339,-2.2730722437661584,-1.172604623047625,-0.13281244561930167,1.9950075662847921,1.7504937221343293,2.578546768504864,3.2599277202848156,-0.7305754587942607,0.6213786061503639,-0.4766976141273665,-0.9359877066567182,-0.14192384394698673,1.095875575658916,1.7533389699472843,0.6854392066543871,0.47193407537955595,1.436637033660404,-0.3884893661155436,-1.5698207341523305,-2.4938213618316767,0.10453702969169336,-1.8570071577701206,0.27504654768466774,0.24780064106924166,0.5495274642257608,1.0169279992271294,-0.057330023164640496,-2.5881516831095075,-1.7519590733770418,-1.05447468657139,-0.35949825154814996,-1.409104560187642,-0.6379294827994797,0.8734101644938429,-0.8438023881034512,0.28403183654011266,-1.5881785833839395,-0.02669179514938125,-1.7127495367712284,1.6493745669515036,-2.0071130789465963,-2.5653264570676106,2.0457074328182148,-2.8759465472895247,1.5169306676661278,2.5089695895665454,-1.1603586707869937,-3.504890430453148,2.790256802261468,0.23478395444169223,-1.0378726999655588,0.02221394031274265,-2.1929216119966912,2.612274758854785,-0.38164758385071784,1.2148439577534464,4.219951593889507,-0.04806190931596838,-2.6807376903563824,-0.27201328889795984,-1.3287236961247728,0.4380795521017947,-2.243932372730277,0.9558509763954097,-3.2494611035313694,0.8116843529761348,1.2666292451284007,-2.1401057336987708,-1.6381794606341837,-0.1457196187791755,0.898311557607976,-3.3224485774526706,-1.234540922070171,1.841805218960262,-1.8906310889666345,-1.6065244278242339,-0.8969290561048847,-3.4550237416546343,0.36141598221392535,0.3959503164380871,-1.0535548405761417,2.3435665608603973,0.6896294874752956,2.030975877192207,-1.3128290139267833,1.3151033441119764,-1.689378236681288,0.0782090080370332,0.39167346932530667,0.21292067973151138,1.8361544761363637,-0.6351170697114591,0.9381681798637772,2.0034559673004964,2.491700006830859,0.5812317420312743,-0.9489031380242572,0.6684320344101014,-0.361833962255539,-2.507089066113376,1.525676194487939,-1.7898001817048566,0.2683455838891614,2.5054514069742413,-0.6077886394144224,-1.0418864902589857,2.6564332437087463,-0.5802715309557417,-3.1961728317900646,-0.04516985080261311,-0.23824620935864577,2.182151041345582,-2.353238976713885,-1.4462006131275023,-0.4083010474339799,-0.41211109844547184,2.2159891533546805,1.7800901708059271,1.0878781731119185,0.5769745993988814,-3.262777243508255,0.8919381298147946,-0.7628976762547246,-0.6685981190969679,-0.4079909924211218,-0.5759001024144423,1.241536448457167,0.22746821508688486,0.3099356465075227,-0.506250168935145,1.3685506848541245,-1.5812779339293306,-0.8672468430530936,1.462399951967255,-0.7438853446550449,0.25229706471523755,2.0141278391080286,-2.8182193443198673,-0.10584100148654926,0.825579930247194,1.226036325975476,-2.265378618803673,0.3963369577226404,-1.6083417925925538,2.039549744740005,-1.7020047880345743,1.0309797293700533,0.8111133723626754,0.1591818579556412,-0.3025545854484451,-0.60121594043613,0.5196333621567141,-0.13101523516356928,1.6043538174259306,-0.8778508243941504,2.3144085424426692,0.2164842141118278,-0.5333150653913266,1.342013095494129,-2.878477136125164,-0.9161899845906641,0.34096918521864633,2.2306934535959164,0.967879609900518,0.46474480870682255,-0.26434775186752846,1.632436987812079,-0.3294712959579424,-0.8821334497966955,-1.6549564622216928,1.5028340777440552,-0.4285197363372001,1.1207299937809203,-0.46615563981395014,-2.5510786430749928,1.9060939604564333,0.3742781622457714,0.3878079213733985,0.8128057930526826,1.3172131882242446,-0.2501196945342443,3.5031534147169996,-1.1332828642357196,-1.7487014583577376,0.10400511120259301,-0.7408615653597358,0.8519594822554272,-2.116024109329725,-1.34730492319449,-0.5411023598835216,-0.2636646700243202,0.19417482249095758,-0.21000897670245594,-4.354776223468825,-2.1876752777777004,-2.2975495304014117,-3.1532121554274974,-1.4544694469469053,0.7121133306073097,2.8071258411930327,-0.7269784437240561,0.9373296693977932,2.3733381026194182,-1.5678849366650074,-1.5231955723414539,-0.5123138938862325,0.5979865565967087,0.2122713425336597,1.9875089672452342,0.33540994784876577,0.300116532315017,2.597408918173119,-3.329979597489778,0.037742039714955496,-0.5312442576273516,1.544495732345964,-1.4018160541793518,-0.03431701697081763,-1.0876334281608175,-1.6495306396143086,1.9967337290735876,2.787235738386078,-0.7611883968539185,0.4266617000383811,2.1591465198172,-0.34502721556684934,1.6078969085071126,0.4963305371370159,-2.9005086628816783,2.1729351168789637,-1.6553280812140905,0.32713684820109207,0.32051342489851276,-0.6597880478092357,2.8338334827744993,0.5916644967836776,-1.0201607812778863,0.07733121746281586,-1.9295704193629186,-0.6603560425742036,-1.9421562145507831,-2.9763987549376023,1.8232104777461318,-2.068167668831263,-0.6290304818194943,-0.7177838603876454,-0.11707900573020248,-0.5192978239172839,-2.450868338641372,-1.667077134008211,-1.9842159406889,-2.543103072748852,-3.038443905912568,0.5228961736309808,-1.5941781397685058,2.0320916990017004,-0.8074256026660362,3.995492702469384,-1.6486760643963247,-0.2671673174243614,-2.3905526567747457,-2.71596444617377,0.0020455173223825215,0.9037005620337466,0.9860951289808497,-0.48007235923834213,-2.0315267810482833,-0.4112421108471926,-0.49125779359316635,-0.42738463857743597,-2.5654867233153937,0.5918740884879935,0.04381842381637603,0.4733141231689965,0.23003832168976968,-1.249747708284603,-3.5807590109062764,-1.3823338394303337,-3.0753762983657915,-2.0003045304564333,-0.6613042287753054,-1.1264984087893317,1.114872219491866,-2.1057093465900465,-4.0565084479098905,0.6871119592400353,-2.1016163059985495,-1.5139341619583713,0.24477858349887033,1.0406231857425599,-0.3086705462544851,-1.9785515109684144,-1.7729669953571177,2.869593891657788,-3.3389985569812746,0.35691998874008507,-2.856644134078325,0.2642233239481568,-2.2854313483639985,1.2763051472598967,1.1716128757714221,-2.8379345164566656,-1.8039624357711979,2.718293918115714,2.2728256521273944,1.4939061598372092,-1.9430822886097507,-1.0554565462014611,0.2884217824770616,0.21274062876537764,3.6215606504548075,-0.20757762439079305,-0.822840699616801,-1.3169829451356097,-0.6635871585733975,-0.8387758687672943,1.7946868508397729,1.2749470968007586,-0.16392128919575477,1.171261910209262,0.9633186735217937,1.2916225119065325,-1.9003559265051018,0.2668053954130257,-1.5367032490057317,-1.474501712422064,1.9398693303202268,0.9499051448018193,-3.45385843697437,1.5748751628150337,-0.5444785093837103,-0.18648114474684094,2.0709657883665007,1.5676712628062435,-3.9299485077113894,-0.0019094322709239215,-2.643343860216254,0.21100702772065993,1.0577931438631494,-0.7123618129601491,3.479138683140985,-2.099535344526886,0.2330493658909451,1.9910554582055013,0.7898627801204356,-0.810079556535198,0.11951068073139141,-1.2140341689852265,-3.7532399984382345,-0.13822773432671479,-1.5081898136476632,0.536176212641034,-2.3169982938297395,3.04490767104539,1.4427407434850184,2.4078062137912273,1.3633728991726628,-2.308518655236414,-1.537280500976692,-0.9063246603356766,1.9168596437556027,1.898663551853165,1.3863235189694916,-2.3966403992282213,1.1155245417470014,-0.14439149377082647,1.7619319321206874,-2.422609155486739,-0.8896520079529384,-3.1482520105635183,1.5678207323193583,-0.8283835123759073,0.8958955066665796,0.18743452263256968,-1.8820230342857105,2.6354971542796797,-1.7786259520873609,4.1662549588353714,-1.5990942343532064,-3.6678626899327966,1.539103382391292,-2.903951186648938,1.1975495184757137,-0.3631850652379254,-1.1855889456507298,1.2539065819239998,-2.6311833123068653,-0.23556963151949914,0.7043262186023723,1.1976626873901122,-0.5705106132583893,-0.2520973845645538,-2.3903943683962696,0.4370802832128067,1.697161488926498,-2.586463840599973,2.730423466507659,-0.4777140329451245,-1.1848836164000327,1.6616240535176194,0.2847185234125483,-4.116611785617329,0.16179681578497312,0.615863266882215,-1.0873567142519074,2.264044321922857,0.6135389884525415,0.9604069057405817,-0.6919771137334905,1.9366604787627897,0.776234726631318,-1.1009550075479868,-1.8369381716220663,1.968168524033519,1.7223945179757691,0.6125330861793052,-0.09783098144620408,-1.9038029891981387,0.5038022839203093,-3.473387552548372,0.2880453500972988,0.2957226981700897,1.1107726117235244,-0.7957699702808521,-1.7459826691776632,-2.329880926581946,-0.7445228258909052,-0.4454195267625979,-0.7895268013337924,-0.21966661856337574,-0.8722524278972039,-1.665067001745748,1.7223496456622358,1.5501007142087668,-2.8365951753521705,0.695874811970266,-1.042192961140673,1.0854474359579367,-1.8777783837496078,1.2668867751851443,0.4372048748604074,2.9536354432368395,0.22325348931366018,1.4698538022132401,2.298464535098409,0.5672422801018633,-1.340125674685879,-3.0633825784031554,-1.1944955523013514,-0.2690302625090722,0.19169571640941027,1.1996473215503167,0.7569302382493259,-1.8622843550582215,-2.9258172222802092,0.711814534225832,-1.4516608044395602,-0.06307648790785475,-2.2983627078794764,2.8948498870523416,0.3346792462232106,1.3408696211603102,-1.377288622898901,-2.6338557036607098,3.003196525436522,2.3056939476408447,-3.239165588182137,-1.190553837588886,-2.1114515336315174,2.7674841581061655,-2.407594354081465,-1.0972631697777366,1.188021782684787,0.8438113570098911,-1.0497548352256652,0.15406452361992984,-2.355750615329826,-0.9714489961172524,3.0734657290464633,-0.47954489039674947,0.054230307352065625,1.0150316064190101,-0.2645593665214408,-1.2733110852478553,-0.6834990287188715,-2.360132920706354,-2.42821257546515,0.8901981551091023,1.302280324901667,1.2846213395479384,2.873265848192208,0.42885625741260164,0.8052436935431668,1.4207931938813967,3.810927370726608,-0.0713589664160548,0.21033216067884333,0.6914509871636031,-1.2392302269482034,-3.234053423861045,-0.03527852121589349,-2.524316893088088,1.633102616771529,-0.24491008323644475,3.225835473200629,-1.3145797419220364,1.1606890209793508,-3.76749418400531,-1.3499495248523352,-0.32794881546866356,-3.576398316664637,-0.7941405758321884,-0.026270280824430704,1.382818609471136,-1.7501357514852292,0.9838168279223165,0.3854936540008191,-0.5452349304324643,0.9845315205801605,1.076237295751597,-0.3683466756373987,0.35934355190994494,-0.5588126486950582,-1.2797638541218228,-1.1367183955826003,-1.5405110159984143,-0.7129474383425711,-2.286164415829883,2.0728789614575533,-1.362416490647765,-0.8612515610165988,1.4408390275506935,-2.874430277522079,1.1359916396447578,0.14063792482532309,-2.2716698386116922,2.6755937732400055,-3.437822120418587,1.8176887363045304,-2.3252381330699636,0.39102162835400966,2.763729846299036,-2.3874442030495353,-2.4889283578140353,-0.20998383126674883,0.2894216777365981,1.5318128052099176,-0.1996415414311489,-2.6572491512897147,1.1761148918748503,1.6776556794396449,0.5215946753310928,0.6753356366755181,-2.8751500484233237,0.3771389249323443,1.4052782433746513,-0.44046103261786607,-0.765569091789858,0.16363888617745512,0.24346507003838033,2.1037746676999336,-0.4674442329624959,2.0067794410204853,3.914531245475465,2.011033781237356,3.940276146118227,2.5259160173880075,-0.4460224256687852,1.1923144939363652,0.8698324360024894,1.7572292244375085,-0.07824483254068626,-0.271586971331775,1.1576703439781961,-2.004447587412567,0.24425962183917435,3.3451321228836988,-1.0068369056488577,0.13922716669755805,-2.3461917566286252,1.8657125625012096,-2.2664522411967414,-0.273752080619771,-0.9459737368176148,-0.17337587124804352,0.19811020241195595,-1.8942353376137764,1.6764052837562797,1.6961591474106996,0.19624130529458336,-1.0315163042482591,1.099677572008007,-0.035250470111145,-0.3831993019365649,1.1673922122511378,-0.348434153615054,-0.23022344215979676,1.685370293375495,3.5385838326514523,-2.7673688201690947,-2.038167442840859,2.1820650345618167,-0.5082082563928692,1.9197803002089855,-4.032957667727825,1.4479310983074458,-2.2704405731579613,-0.5965659495038032,1.1977356095579725,-0.9130686772690436,-0.9875921111860085,-2.625984288867125,-0.6150444445874056,0.3058752252521556,0.41681349192069406,1.6370613064967692,-0.9458157460894682,3.305315487080152,-1.3067417667558283,-1.3702249255274188,2.662940553099139,-1.159296914382653,0.7683627618219473,-2.617294370337686,1.0140982626113804,0.5895350276300729,-2.7691034184161176,4.215734028285329,3.7375546784415223,2.005667201269625,-1.059976080223384,-2.0006916652535973,0.14409423041600725,-1.502672024925978,0.2922429787269574,-2.5343749998137306,1.061604306296875,0.4334683480734293,-2.700926882507534,-1.4123908717837668,-0.35374629211205005,3.0358628299081496,0.2161042890939222,3.383026714410238,0.22011575640725753,-0.5122813207839718,0.07998957429501312,-2.5353913906006404,-0.0781528685098192,-0.7299580687576394,1.7919001253271907,1.8879086763998614,2.2743250744144863,0.09440451556129245,-2.0755224413278244,-1.1444498983339848,-0.32587421083658497,-0.2723090238443704,-0.615848611414674,2.700513940774143,-0.01867608814227206,2.9124231815900283,2.344716038778174,-2.518791744416739,0.29034481616390223,-3.1014474831218197,-2.839451851472425,0.6039508422393938,-1.3769574377338594,0.38232842150616325,0.5761132746766908,-0.11055071399427392,-2.4927638964234378,-1.5072961286475821,1.5442876571668909,2.5170307988315535,1.3050823060805015,1.3871784796986206,0.47892224033578473,-0.015735317876477802,1.9098769307762165,0.4345242197156921,2.5522090568155567,-0.20006226003387437,-3.3112548816856817,-0.000579481578659994,-1.728016406311162,-1.6373169921473072,-2.563585743853397,3.5405939171496597,3.07822221183156,1.6580852578262544,-0.6429297692398079,-0.10545347467180168,-1.883113325797468,2.139961231485858,-2.3278308322858043,-0.40763038335535584,-0.026577672379365452,-0.5380661761831032,-2.5370909225760996,-1.242098871867467,-0.5108445083827857,-1.074447134635727,-1.0954196774835185,-1.0897206360619813,-0.4394455745070453,-0.12344880812643744,-1.2317084057238243,-1.772930256066408,0.437000541638482,-3.045832200010199,-1.577680757059484,2.6997372218804756,0.610893066733281,0.7396064828326085,-3.212776226803731,-1.5262456793386454,3.4394741045200847,-0.9135262142838279,-1.7377170610261548,-0.972129440605454,1.5644424088234794,-1.2238365232863158,-1.0601918647801387,3.1917720768806275,-0.9264882779226747,-3.511492394646068,0.5168814801186886,0.9440895501037685,-0.06815670152341939,1.784755251516554,0.8579284280614996,-0.06549351057163158,1.2229224065166366,-1.486978856311469,0.411912402560796,-0.6127967455837006,1.6381665198760473,3.495525813190612,1.8212747526857873,2.188941871730261,-1.3113952295247402,-0.36564320557176955,3.591278287027527,-2.5246373180450576,3.3239925341077545,1.831396855572124,2.973185040565656,-0.5358592574903585,-1.76772252390081,-1.826559687824174,-1.1687589702231802,1.9184843859709257,-1.1228448322067963,-0.9759204862346689,-0.9997380191229032,-0.1313720960070738,1.7011235602395076,-0.9876490227703176,-0.29013927997458744,1.635342468002145,-0.4895790309414913,-0.4557747079415049,-0.5162584315717377,-1.5101565846202019,-1.2593699040337374,-1.9602576873917812,2.9624158093606745,-2.002972878939565,-1.9892788705512345,0.6190512576732228,-1.2687887425801159,2.5377408221567044,-0.16193712890059622,-0.14655801037990804,1.9944272730104005,1.3690677984273876,-1.9839395989025512,1.1652746614815552,-2.823483486526267,1.0402337587040609,2.3374595826404296,-3.7610386635613167,-3.325482135047407,3.1837012443425445,0.06655055608411746,-2.035451764048813,-0.33381611652360366,1.152756724769655,2.929275340669724,-2.4701966919678093,-0.19482772163907977,1.3973820593700235,1.8112054383883918,2.4649530551104615,0.4077855664505458,-0.8450769408022922,1.2679836926738863,-0.45337009544040596,-0.3297971084917122,-1.9692710633765023,1.8793542676128165,2.7139191165592496,0.5386914029427011,-2.112954078737712,-2.8902893174813866,0.3632038376060325,0.3281038620868265,0.992051738466355,0.1883477084316358,2.6706442526502614,-0.42336996008985456,-2.2107087655302844,-2.1404797296666183,1.3583659799166854,-1.7460375410783522,0.18948353720021582,-3.5335958976608337,0.0008601762420728477,-1.0065977840246725,-0.32424960109138407,1.0765096469220712,1.443441258563549,1.2365901393494838,0.1402746208522222,2.5126077812505327,0.15470962794704782,0.7175601930614548,0.21851815404438063,0.6635250543329607,-0.5189735887511978,-2.943120721452598,-2.717613985281377,-0.20652595681537605,3.349350672394475,-0.0745900386495115,1.3215905904074154,-0.7419802354706106,-2.2189971819269445,0.9164158302586488,1.0042012886484517,3.9683362553868924,-0.09895625080937838,4.867387894880853,3.542272987945956,0.07145929005415885,0.05573717787469853,1.4178542948645427,1.1997486393197998,-0.943238153148591,-0.7573972479378784,-1.1709788557609895,3.194214104772866,0.014096570485442616,-3.297352225333839,-0.7053759130115769,-3.0554961886847787,0.6118751086172257,-2.3511385396898747,2.07766116609985,-0.9304044139842562,0.9890958521214237,0.24809072900964446,-0.19674233702966085,-0.8605196474295032,-0.20018743891550864,0.07082862890074304,-0.17852751977948836,0.035643284004915246,2.4379194558460306,3.4548427557726016,-0.5536924965554988,-1.225691068988486,-0.3752351882643241,-3.1812094915231097,1.3727822930211038,-1.267202187337569,-1.5563102095366395,0.9898187102781142,1.0028511333737995,1.5834020996384863,0.9850385816344402,-0.2606756012840044,-2.4301244932777433,0.7290521986903131,1.3705437092261885,-1.1097310223694214,-0.9127626419450191,2.4548762002917566,-0.3053425626559188,1.7341841914592184,1.4287166204446677,0.8748789575738085,1.6650416438927982,0.27458221933239435,-2.7643474870826044,-1.0868153138429264,2.4208818197943587,0.8301816012757856,-0.1006874143888497,-0.8389730685234674,2.4695865853024808,1.9312007916937362,1.7677652442439025,1.581442815778423,-3.1422666909236745,2.282119887265343,2.127718994938568,-2.0641689439842796,1.436074272235113,-3.0167975194732874,1.1323095884439138,2.2522935798681853,2.4646237208052675,-1.7189448259522409,-1.3523928232430222,-1.3860367350583127,-2.0760403145908746,-0.9263561221443297,-1.9229141832851457,-1.8716036347195437,-0.5414281393709064,-1.9912844709844326,1.087066120307106,1.6304976100909876,-3.4106726044897133,3.8275496239525295,0.9439924837476679,0.956357433047211,-0.6003662868906192,2.1639647044371952,-1.3910175763391288,0.8277508886097182,1.003442729446394,3.3534652507745726,-0.17365932694827108,3.5167595240669414,-0.4665466663056324,0.06269605102603931,2.3877180046189728,1.5532481334166734,1.8643623619063405,-1.7128249613825783,1.3097718600116395,-1.0527812409039017,1.681355187982878,1.3380761084795672,-3.781661604217157,2.5347590014057237,-0.09921563295909105,-0.5352646139340438,0.3494342780239058,-0.9781133049578192,-4.335558884023895,-1.5044608140371623,-3.2152289915118475,2.9902939692479213,2.4111896869038913,-3.521785715315012,3.5055753854787097,0.12608843236990347,-0.2933769799278075,0.6163930465844569,-0.49885400461710677,1.6123496022987527,1.1005170313502726,1.1038158926734083,2.4099798332298277,-1.944407035140815,0.5704729591497933,-2.677677624935666,0.002057145730589677,-0.45051123666964915,-0.24313322423503642,-1.5634434808397737,-2.7860163657169292,0.9852770829988937,0.4193173337165039,0.5861720066207842,0.7694463006833174,0.37548759129000914,-1.0407595833495578,1.766825958219294,-2.063749613001646,1.2491162917921563,-2.4429747800734107,0.8483699048260027,-2.707195829793027,1.1745082856081797,-2.430162164384537,-0.4352271722374061,0.7503379789452073,2.7533676347753278,1.5006114716777688,-0.9270698198409837,-0.8738681144744559,0.7717270053669344,-3.39948314349052,2.023803420043538,1.2903545726057588,2.5487119877357913,2.817371560185605,-0.02281353449631474,1.8864556049787629,1.5258900630604832,2.685617702546263,-0.5518975644926465,1.2991693336369985,-2.0193872589731092,-2.506407453697003,1.9342551551601501,1.1966282494574776,-2.731719589001813,-0.9005527088323562,0.7840143643293703,0.8229674373055024,0.4823681316946404,-2.860119445141209,-0.5457704715583797,0.17687776092911023,1.001083797616312,-2.736650470182002,2.639451326710236,0.2934024664053858,0.968503080587559,1.520544154628172,-2.851974469936051,2.7933285327013015,-1.7621214728775274,1.4673312957865985,1.6527734357489547,2.7897248587381256,2.910432559379017,-0.6137835795193951,-0.009534168791917414,-0.10691102740441578,1.071434173411401,1.134367244895351,-1.2254902710930309,-0.5216294617678412,-0.035819773459900936,1.536163256290371,0.6276898072360374,4.191163207201812,3.046091674746038,-3.3948709519373756,-1.4944869560215879,-0.9395355894100962,1.2035794011016896,-1.1758076565124538,-0.9881563722254373,3.681301375796649,3.0603779884620725,1.49362632490118,1.4252911579077119,-1.1811956374051622,-0.4258056114315058,-0.018142503743705136,2.1674370499204927,-0.08454490900482961,1.384302664603729,-2.4266218852614,0.6591626373505116,-0.19038851243138588,0.4942082094388038,0.9033556026077677,-0.001129007984923243,-2.3308654585906377,2.9507956442569814,-1.4682021159721632,1.257321502397477,-0.6615175597981061,2.1668421809176275,-1.1565202216377224,-1.5378827824523433,-2.2052081078289216,0.01848361052646593,3.012484445726108,-3.661623563712466,2.1053441852951993,-2.943642828554033,-1.5242849094451778,2.0980593947064428,2.1549477217883974,1.8552590677081362,-0.5867603916336477,1.2398737385679766,2.712886493762031,3.146576971629084,-0.734462570942574,1.9956609064707023,-0.22055480289613041,0.7801919534343853,0.771251954221943,2.6457006304835184,0.9397022685236358,-0.26331020544356676,0.7520227399936484,-0.4924872456540561,-1.198641690165224,-0.9626002060293701,-0.3664151574327339,1.7469506106759485,0.17297910519471388,-1.998852484748983,2.0419075545771643,0.2842746951162827,-1.8283126374581895,-1.0587838514091708,1.7629687000720389,2.2269443907692072,-3.0275151893511567,0.46641379886970113,0.16108196436328412,-1.1722227980083388,-2.695966838017173,-1.5871072026205917,0.5494714693809102,0.30954511085735376,-1.2308888758737198,-0.03436412093661268,-0.7995021353879107,0.9266579591026024,1.9479122721746929,-1.8275160267569486,0.7964856502069887,-2.7008525003429837,-1.3106736580204499,4.5251896068641075,-0.11531259228495187,-0.609527336484405,0.9881163344925666,-0.9502365089838871,3.0144988934655514,1.5529496967457421,-2.4565455499818603,-2.3853531596849407,-3.151954791827494,3.811998297259339,0.5394932902918207,-3.1339433417330507,-2.5595752790342114,-0.12563437376795997,-0.9085898539700672,1.6770469684165248,-0.965578408720042,-0.2320384109917852,2.1474491753854137,0.6080815631968862,1.726797069505563,1.9730577200356811,-3.0150771127146023,2.4796030703205116,-0.45932602358149033,1.873437369301147,-3.6668416780088426,-0.32492212366359746,1.179135836483892,-1.1452794438694314,3.726090475072178,-3.2515766145205585,1.314497176964411,0.5425539575503295,3.020082514297661,-2.600959243820231,3.1405128237752504,2.78531019130831,-0.11160178427102424,1.6760107186902995,1.897386203708992,-2.4459785862158494,2.741674846987158,-0.7093689667941965,2.100497828274168,-1.1838419447505428,1.3514739546899215,-1.1195838384008194,3.426539187939255,0.3768568672194334,-0.7491019172390854,-2.8433673848415153,-2.775681744544968,-2.366141902718371,0.32397336294245704,-0.3927462597921888,0.37791363025419605,-0.6056691797910807,0.004257424353410018,0.954838660869786,0.22052472260122555,0.6422796129939713,0.771971799646408,-0.30942531533854367,-0.8964622671488724,-0.5270008637158989,-1.3201278182656133,0.2970603044149118,-2.7880868379154697,-0.5357780044501327,1.6531256018220155,2.333727335121343,-1.6459671432769902,-2.2272290859311026,-0.5566030293877687,0.0008990723593161082,-0.8851086640151292,2.6690521645820793,3.686123599397128,0.6438855203408861,-1.3870919175130905,1.6289460626733345,0.8730040895357435,1.77879382769965,-2.782898784737369,-3.2082661661930714,-0.9738633514627139,3.3923088574729587,0.05589375144020196,-0.07219990486848135,-1.5353638155495366,-4.070253737346584,-1.65262257640678,3.6699436645294035,-1.0237218756279365,1.6096792866659175,-3.016893908571177,0.8182601876705652,-0.230343480520245,0.9216551231888035,-3.2055247012486596,-2.5446212800492765,-0.9384756227462987,0.3915140395158116,1.1615438507976192,2.645588591215353,-2.6479556184737,-0.24060548180999144,2.8977233392201587,0.5748582480557067,0.8647034627823128,0.7432568501388742,-2.8301345792310726,-0.7380023445508408,-2.1168831525601126,1.0303005709083817,0.2934647824595237,2.10318490638523,0.2512851957426128,-0.9918670188143355,-0.5444938551235532,-0.37281277659957013,-0.5506043999279425,0.6510841096745789,3.595816673588519,-1.3323329563917492,0.8730535602760537,-1.2352192833418167,0.2739968017216249,0.7631732678981498,2.81565252824409,-0.3595882917624247,0.5033678961033462,-0.4887026933598268,1.536554319880745,-1.2229378669200783,2.970989502033927,1.4637470320177108,0.18731187653080608,-0.8260095704350278,-0.08853626667480816,0.9054656725431499,0.04316051597365926,-1.8623296919724908,-0.3866627598945744,-1.7768796627572534,1.7871358686384684,1.6467007698666873,0.7216968574186631,3.655690551892283,-1.9477408673344376,0.005866235628284043,-3.8895670923656422,-2.6533202069215864,-1.3699690325719678,-2.0569802489254387,-1.726450678839808,0.5105316187907452,-0.5742142865647836,-2.3298975641127693,2.237724874288135,-1.890584382781149,-0.5465960047042014,-0.7506266633891276,0.8128253042046455,-1.3403535566870346,1.5592420904283228,-2.8936740679188717,-0.551881671854992,0.402895150580358,3.165279031718744,0.23816326909965838,-1.3066767096851584,1.3618515877178454,-0.6560284494381455,-1.7615282473231586,2.0945971224719115,1.803856991917774,1.5184801516228028,0.38121366544858776,-2.3217713202253285,-0.858808604887885,-1.5328128168715476,1.3771938557530505,-1.169942141753113,-0.45448251925312116,1.6107805923133058,0.6866495131306671,0.13565769162263677,0.34428260507900876,-1.4470115830822472,-0.7615585986767583,0.49932748239159447,0.4809084216697899,2.014537722506671,-2.7172061364961406,-3.3069846676252688,-0.7759520377979001,2.0903943333267816,0.2148784197610289,2.6367337954900396,-2.5089417767841535,1.216133538950286,-0.42893014539866653,-0.4360715295306414,1.2856306021542012,3.139612091389177,-1.8408314332779503,-1.45702546298682,-1.2333036770433623,-0.6321918089294021,2.071792797499044,-3.277490639347634,0.039938100190260416,-1.9489000332935513,-1.0404826014298274,0.31151057312674085,0.6017375990040911,-1.0233903575706238,-2.3954677451566693,-0.9900275074445012,1.3498188239260231,-3.078338397393511,0.15212839407828188,-2.1614920813272582,-0.7923173789637908,-1.4883258538035253,-2.2866460797626913,3.281570769660714,0.9941647141661839,-1.9919330241528823,-0.6903357965586886,-1.4940825381641731,-1.5242888428738035,-0.10920002560005249,-1.3335424129774187,1.0589841612824344,2.3804772893909436,-1.4928211221021563,-0.01957323137638585,0.6846210823542437,-0.16796156759751768,-0.3151337999824642,2.7244626704068984,0.8526931258832218,-0.8705069499382351,1.0915910738117065,-1.2220377633248183,-1.2986898155131055,-2.3121458801792065,-1.4786710296762398,-3.4823767733513864,2.3136491773386787,-1.2433009758003135,0.0723208096419775,2.6512003006009563,-2.084215199015213,1.6581999617173362,-0.0031151380409961955,-0.48821773800549356,-2.800240337511264,0.6595715251759245,1.6565515042625973,-1.3588187634260014,-0.7536228734162137,1.6245444603422659,0.7369758203945684,-0.26851260281972855,1.1282714408972727,0.2710625743645271,-0.6766223406936273,0.8457725322360619,-0.09453062801837928,0.9472570944595854,-2.3192478592878714,-1.0987532570067102,1.607863154971328,-0.17324646278010009,0.28489307050866197,-3.2393590079998065,0.2573112010269588,-0.2601344227310895,-1.2261311011474607,-2.9779585387298493,-2.209376194219139,0.7255868517880182,0.5865952027289154,3.410470054325635,-1.090386537904387,0.0029914387543602647,-1.2491400803694586,-0.31712556324537144,-0.4971255801770207,-1.3969350647389147,1.2725084025798987,0.8648406604595524,-0.644982692729001,-2.957437941560759,-0.03477525756309073,-0.21368946035576136,0.27297645736276877,2.397266662796013,1.454619207608062,1.0183769234409024,1.5534463635416482,-1.1567627557297315,-0.7633609606493806,0.9243448213026627,1.8377409541788314,1.1739250558706908,2.7938862144045267,2.678555408100636,0.33400572523068905,-2.425427942457298,0.3918966333139637,0.1788103706242497,-2.6542611841429444,0.660723214504661,2.065447775347503,-0.9382949074955863,1.5281906208711955,1.3189489713181213,1.2756931954577342,0.6188428521443792,-2.2167472030691484,-1.437938141260871,2.3606791692743005,1.2185619447420628,-2.5662225266159244,-1.8579088217738242,-2.094610918178732,1.5251403316561838,1.3888937274415385,0.8630101913178406,2.6953811609622416,0.004653513607243172,0.9645775827924279,-2.7529735713293446,2.813385561621009,0.6317382597852484,-2.238385469745527,3.791971052961234,-2.1869863036003383,0.3188538771931539,0.3293642461604931,-0.41062334523252514,0.4655555493660694,-3.064594756647216,2.0725948059180443,2.3838875208299655,-1.148101786575039,-2.9891834006008744,2.633262079083905,-0.11165960775158097,-2.9768431198567082,0.5040648960407896,1.1965702011182082,-0.05279069391974696,-2.399530328985783,2.085745261637667,1.6943315705796367,1.0048989781025368,-2.536843758671553,3.5065828244132913,0.4189322282487049,-0.6151983372057247,2.0792876241540115,-0.8472782611965188,-1.4610961458116238,1.6367042616666545,-1.2216705440183953,-1.078168645390581,1.1349781798985017,1.9396057150844197,2.225112706587778,1.2609022737241682,0.9914388731422781,-1.6699298217334884,2.0416431088863614,-2.5419670858383037,1.0454542785057621,0.20931675831520669,-1.5560466556312156,-0.6415253885427622,1.604593445220725,0.18983333980933023,-0.893467550313643,-3.7677571577825177,-2.5439516943137965,-1.8605920606367916,1.605178816898683,0.3592682839797249,0.2607638528165271,-2.2061888024632026,-0.579019894158773,2.4680597798468855,0.22851590768174032,-1.6816185966282409,-2.7395004448930056,0.9584073708636546,1.6474949958868745,-0.953542810885012,1.0748031245992384,-0.6066873808310927,3.9163207291652875,0.7526004493732378,-0.9877217046435431,0.4629219308670721,1.493973062276466,0.6649825368818425,-1.2136187994033352,-3.074550083081358,2.0251835811018477,0.26805271982916423,-1.838648640484399,1.7145730909700287,2.3967150991545365,-2.769003974730892,1.7298143693315255,1.0066845425353226,0.9374446079548229,-2.161427437825838,-0.42139523308119753,-1.2739918113500626,-1.4012864632081803,-1.93660941311579,-1.4621863098779662,-2.407496069104777,2.3355739493272196,-3.1470891235087275,-2.334544948690485,-0.06519183186740125,-1.810625521831984,2.822399662806373,-0.1211344422582483,-1.5660605895969493,-2.546967366421612,-2.020979442697274,1.8548659204782099,-1.3085676894017966,-2.3930016804364818,-0.48560789181538205,1.6785740718018753,2.8247992416083902,-0.8530487576996728,0.2772776642556081,-0.6972043223928724,-0.11327871900757595,1.8044711907140372,1.1825386507888043,1.3221975413271674,-2.3534910715237407,1.8428946426532522,1.1385726760769512,0.30655248768971,-1.0975026728194603,1.2518487035936243,0.9561445250122308,-0.021657370718344865,2.8991857281352327,-0.5465201954408153,-3.1796247070580987,-1.9448833881861682,1.1173748327149726,2.329092610783208,-0.5871371326787008,-0.9238299572447024,2.0330801023877196,-2.901500879782973,0.28602754593184493,0.8304135815660438,-0.4802110938472246,1.0935836950949238,0.1379381285857157,0.2644427892020099,2.255722909608958,-0.1619440350361193,0.2542216591005962,1.7149147179155242,1.085786310768036,-3.3156602268280735,0.8051043719357978,1.642219940838726,0.9484089696709201,1.1010664274017552,2.636756067900561,-0.6264153992124792,0.30709719322058393,-3.7154672802534123,-0.06531719731720438,-0.18573024806121471,-1.665335965634752,-1.756167819654215,-0.21459865230417474,-0.5658019726220277,-0.3618059926778747,2.644444659545835,-1.483624381119723,3.0659421723333593,1.6418940433455664,1.6735723342039166,0.21378791826580887,0.8444416457982873,0.4343701470930966,3.0697646608523828,-0.536071469760269,2.050226894998194,2.7572526080372013,1.5734958736287217,-1.874802707729951,3.980085635840754,-3.1271613364914606,-0.17392432581686537,-1.476137069598938,0.11683329026722097,-0.9143071240686711,2.782153989477646,-3.779735532747167,-0.3873247992115611,-2.7218353522571324,-0.524702847565614,-0.35234133577693055,1.8683665317655935,1.1991753973454404,-0.05437230089269782,-2.2475608273293326,-0.6687057168817466,-2.1704213274893713,-0.8257718627355466,-1.7124959147611154,1.415582909286196,-1.8259185417770665,1.3109344907932734,0.8663720717877103,-0.1157645967473159,-0.7176762222411841,-0.763213030955269,0.5014395598032212,-0.5488780334409316,-2.2666628099244015,-2.445098014556405,-1.127842513263249,1.9922832574073925,-1.1301696614951777,-0.9943637835562258,-3.3105502710772274,-2.2292624780061168,-2.4349105196270986,-0.3836546248861182,-3.8153410551092404,-3.548448112093216,-3.382540107283328,-0.5262330070904879,-1.6078723559160313,1.635841749224406,0.385344349009892,-1.5251625403432434,-1.077820057872135,2.3971390553788194,1.2980870048835687,1.3313091741029652,0.5862976771010145,0.722905190006816,-0.034919580949033475,1.8579714047691445,3.186732162774163,-0.994460274688207,-1.5283320237715694,1.042329429676944,1.4440427452319418,-0.10816283678663212,-0.9040227287524331,-1.7459915462464366,2.363184171212899,-0.9258272088345538,-1.641121942427657,-0.6382617287131757,-3.3270346029213695,-2.4554050038498283,0.24193727482257238,0.5046876452087153,0.9445689623277709,-2.9681635987658574,-2.8342489905092396,-0.5766877181527649,3.7187812597974363,-0.5731290513154084,-1.6188185679886695,-1.4914799023040273,-0.08211730866914575,0.353563141802705,-1.7527552878824757,-0.6439832396812557,0.7193236906290676,1.6196980402717789,-2.013566070342567,-1.1626210581247214,-2.993753215700826,-1.1188461902673055,1.7732323726986459,-3.1679922373216436,-0.2106275704140781,3.3365028444584968,3.037952488898372,3.0311712296821836,-2.9547498578319287,2.684218238246879,-1.3604724871293883,1.5341462319127375,-2.034208353192087,1.7443592434486326,1.768135438524238,0.7872344843492629,0.32016106344941536,0.329344731136135,-0.40625488376484203,2.2357422635907827,2.203530392306296,1.3507383615696051,-2.55907513360084,-3.549992245682579,-1.3891505481179633,-2.1277125701867496,2.569794420830386,0.16063859756557236,-0.1621517591521019,1.371596715000288,0.031284552128228615,3.2098854553540708,-0.07064333521504666,-2.0043769181197884,1.5916238232018287,0.027941566228510697,1.3958929222244163,1.6129134073665155,-0.08300792916866832,-1.5006685303840548,-1.3083717677626996,-0.12620915620282214,-0.5336603847122668,-1.1426452434025018,2.3312504911115988,-0.640354360374315,0.3918026773108788,0.1512719041547944,0.8866293533800533,-1.7386051118129306,1.9276299261860022,0.972766657161461,2.094900071329021,1.0101312277841559,-3.4907902974497387,0.021263572214955038,-0.7428642324289874,2.921307051690517,2.8127661921076497,2.5358281290777547,2.4218184898650197,-1.016978439719717,-1.0141733691389965,-0.022068637327070033,-2.4115473185237524,-0.24924965383044861,0.527713811704677,-2.087109190574819,0.9536301290128225,0.889252100221812,2.630736553406965,-1.9909455774298297,-1.048724820744754,2.4041392601717373,0.8377989892160767,0.22069315679944498,-0.31106335440906047,0.05589231333157492,3.1503337648054965,-1.3922478869002386,-0.858032735971426,0.3764735340005694,-1.5806893910140472,-2.09692861264143,-2.1811358922183706,1.7877075671044573,1.0090074255231243,1.9971903889639908,-2.553178619319118,0.27786243693547824,-0.0693683863300293,-2.7856281561535683,0.3983964112492115,1.3224145361585187,1.8614566057722959,1.879746570353856,0.2624264579808055,0.3602929239210949,-0.16349568286093838,-1.3934661014068592,0.23709445067811594,-0.5681382371571992,0.1832139215958642,-2.0615944230664347,1.8745003772838398,1.0989382696712837,1.5232213411854634,0.0761624766640867,-0.5187402595496554,-0.855304381702493,1.9064087627257966,1.0681675243707158,-1.9073709595522004,0.1046167525489354,3.688120913523113,-3.044764604096657,1.520157423733536,1.4441863069245215,-0.6015001321774992,-0.9014668459079537,-2.0720505740030233,-0.5556742923461251,-1.5837325270060856,0.9196540598825965,2.5082014734358684,1.1151829615555886,-2.1836957871530265,1.6069918470065825,-0.8353589816813278,-2.2453043463691826,0.7935957073562071,-0.546714607054966,1.6934372524469548,0.6680654806584891,3.4760697255773687,3.242304190682111,-1.9393789644931259,-2.1780594873706933,-1.7459180263297838,0.3193676271982902,1.1601849257375405,1.3402588308857941,-1.616179612716533,-0.05818789267942572,2.7694049242446837,1.3024855693182344,2.50589627210565,-0.36477452747453337,0.3775503333434135,-2.3176730203977893,-1.982641139835295,1.2050991273338465,1.2659213269981267,2.0769160310428845,-0.10732006571681287,-2.547668364155002,2.5879770160343574,1.0994702370152256,-2.9260394792472297,-0.333639605939936,-1.2430236803188368,-0.7507425891144693,-2.252677716008072,-1.7840414081420388,-1.413355544957366,-1.006958652982433,-0.8536130547281888,-0.16484081530453565,2.5534565374974423,-0.03217280506295137,0.6695574187710933,-0.8093274477867437,-0.568273432126402,3.1187026880506252,1.6120491548317757,-2.1572716299221852,0.265706277322899,-0.6516871774984331,-0.14389873385393837,0.4984325292447112,-1.5856720103051547,2.2194182197615895,-1.2137904101288535,0.11673768438847781,-1.219665718145925,-2.3627388480708493,0.5115021388722591,-0.13479573043161167,-0.5292153749974386,-0.8956162261161028,-0.6639583302615475,1.5584414462857539,-2.540889440822596,1.4041284519931128,-2.2294682959143,-2.641734505137662,3.185276518348678,1.9760661754316042,-2.7185544310292538,1.1833393321370826,-1.3595859396036931,2.6577242059743322,1.3484222849070402,0.7075682816408289,-1.2409397421347494,0.014957644817805683,1.091323400657201,2.1362749775619267,1.488908009522397,-0.6427404706686106,-0.9876386589149846,-2.4115839203964793,-0.6263121414233422,-0.8278030144899199,2.8634506370716735,-0.293798310368937,1.9383134985126143,-1.4702440935995995,1.003312801606179,1.0215090717926198,-0.9466273112525312,-1.5871487410441552,-3.5117410445079567,0.19587148034828716,0.8468692835290272,0.6563089027825143,0.6692006533424392,1.9351796176689056,1.3201066846393765,-0.6518219464785028,-2.346442785857679,2.0786419108541994,2.9327669146206965,2.0289862081892442,-0.5327505823365078,-0.10818777314536263,-0.96220953933416,-1.5788989840140022,0.5459197195719304,2.1499564995940736,-0.36404336743340215,-0.2832814699010879,-0.8674300647688692,2.992292423005209,-1.5954944565879914,-1.6166390566476654,-0.23482760621076734,-1.3313334229583433,0.46936189457393257,1.9824149491430156,-2.5577920722716945,0.1509972176563989,1.1346252692525658,-1.821210570825658,-0.2389175205464272,2.5476388672980645,-1.2080066665979623,2.5689820373287735,0.237789914560646,2.914449579344849,-0.1656467121876596,-0.10882864037829251,1.4800624641005613,4.787807522897276,-2.5458751000024007,-2.171762203453776,-0.5174006857303594,1.794612511523271,2.2693073217118664,1.9978276172404197,0.5665849481984888,-2.347555793763856,0.5340358712127397,-0.8662491778537429,-1.3291033576707318,1.9969985783501694,0.6667335164956519,1.4519075846525653,0.32093385026835997,-1.8584316642222507,1.9964715778343065,-3.023711089974115,-1.3555008699046915,0.7713638143121104,-0.8282678300116272,-0.8445042694223132,3.0552493726965797,-1.0124596816862501,-0.13515219569202763,0.47500974716435507,-0.6336694896382006,-1.0369646528496403,-0.4676526373294588,-2.637025442998958,1.5835715459762467,1.072831494569439,-1.6234848620729738,0.1551744602652959,1.2012161332165512,-1.493702059212279,-0.06508942271867978,0.3383641736028308,-0.1107980651688828,-0.6028361831236011,0.10826164869514474,1.1657854820829023,1.9290731285527405,-2.4914559374667,-1.2853868234139616,2.0517422453229943,-0.1403363380138567,1.6881207064078256,0.49659933383674654,1.5562185354432765,0.22790308047135005,1.4822602961440658,1.4582161640280082,-3.3317647904733008,-0.07842411458939967,1.9428182623478036,0.9508231567832228,-1.0991767867618005,2.8224703309179273,-0.797129797928724,0.4327177505578662,1.2298212163718394,0.5330028611375055,0.7144857721071233,0.25357712553291606,-0.2021684779539229,-0.206316505664935,1.0435640055483422,1.0149161860697018,1.3601225325911186,3.036063045782016,0.3300283756222194,0.7953887511756992,-1.45551220060194,-3.6314714536896706,0.3003187713899132,-2.8860763925153416,-0.3899882482629593,2.4427080177774543,0.18431847855428204,0.07732010652175875,3.494741608624687,1.5404301171287387,0.1812051121212602,3.6757268057356227,0.46112968511750263,0.10008375395197326,1.4571827641598045,1.2765539549517633,-1.1453617974634105,-2.346238373478614,0.3336218155525196,0.9558193624019019,-1.580167032267008,-1.079048057654627,1.7968428466610156,0.9644702619122294,-0.704279521773856,1.5640259011926623,-1.4473944404676173,0.6603624930970515,-3.1304954209109255,4.166443727187777,0.9395497281951826,-0.1545312406143545,0.438222704756921,3.332717515841301,-1.0548373937521465,1.3562811159069768,0.624190402574013,-1.1532212190066067,0.3778889664549424,1.4592248799635967,-0.19298761668291495,1.112361509034428,-2.5038498765700994,-2.878114043970814,-1.7751971678031704,1.0783539873442323,-0.3757564024571344,0.029910572611675634,0.9973364629734722,0.9328236947346481,-0.8101219627107832,-1.0396879347657542,-1.7228286608246475,-2.5749037437308955,2.7290306739259305,2.240823193730306,-0.02401246891699224,-0.4486442100481106,-0.036308802603417,-1.4328447200810543,-3.946441710808014,3.364965168800918,-0.2836115113284646,-1.0255730684518494,-0.37265739348864835,2.425484462558613,3.4595769362504036,2.5652545152216253,-0.9783116189853831,-1.5064053424411845,0.31451811307213273,1.2621187136281773,-0.423778680569014,0.8172583431924669,0.5374556777732407,0.7426910422844905,0.453097367540909,0.8370081367521596,2.312957636843977,0.073883817563131,-1.2475588828261035,1.0447404623253402,0.16647924790518082,-2.9100316327470472,-0.7313058971231049,2.7356683794514094,-2.9623196064696713,-3.226376553933186,-0.10118100137958208,-1.0661706703726253,0.7808590804759706,0.2773561889652556,1.6784316194958877,2.4931175683649505,1.1186054047906675,-0.051824237929273896,-1.686172430702531,1.5054831533414892,-0.16149401882116274,3.0963732043025223,-1.0336980846875126,1.2607956169095889,2.4640067751265673,2.264131604338322,0.5196262779857577,2.1903627057578894,0.5553807294246094,0.17197198286728857,-0.2911366646023183,-0.6824282846776785,0.05058378511207846,0.1000975040166199,0.8392369165775534,0.9593261549775957,-1.8526115466793067,1.1121655566756627,2.6461267798322474,-2.8814789266069805,3.4502818514256655,0.2815954465944928,1.5092296958561444,1.2546088675454696,2.2717601327186503,1.2875901624283048,-1.1351796769599682,-2.099647773657015,-0.5035100731375118,3.6720916797188092,-1.6548363739000842,0.6917686968374952,0.8978913282312246,-0.42440651975681903,-0.6979511734938684,-0.35356256523611457,1.4830255049420307,1.8849652088298792,3.8419234382468384,-2.0386644589460645,1.6440148491032096,2.2824916013460657,-2.597466030939925,-0.8004796268701513,0.1773059053442177,-2.397016629669718,1.3979571267060285,1.5348077086397318,1.4864110815516387,-1.2626449576562486,2.612519432227126,-1.4443381778459605,-1.6328529157028333,-3.2843928159331983,-0.29443698149484465,-0.439125036031469,-1.2247126451086754,0.42826181527884516,-1.2432018304428865,3.414270824571276,-2.1670779333688506,-2.521091827116646,1.6482565283027857,2.65914084981714,-2.0811475946320375,1.2710996778955423,-0.8307675767618241,2.927997937744523,1.913276422492455,0.22676916716323847,-1.717624892374742,-0.6442588865282427,0.06604699306337938,0.23710884314163233,1.1045955288002494,-4.113281727391948,-2.7495493048291872,2.642856050865893,-1.0627116670860663,2.5461512979177026,-0.07964374180000203,2.0801547263611218,2.785988812521583,2.4269774941904023,-1.8570827079394643,-0.43717697348896917,3.582494974141486,-0.38648155706305537,-3.1460349785654245,0.7771428652676077,-0.7694489130722701,-1.877223753153337,0.39470236364000455,2.2770245237621696,-0.9690600578524923,-4.185139913300336,-0.04547327449914177,0.8347103628961233,1.2325981364561214,-0.4691968610697285,2.320002279875301,-0.5466037627191848,-1.007004261176534,2.95104183574779,2.6575103394696695,1.3195494565885437,2.897468519526589,0.991157640574951,-2.3923327707862625,0.9600855798482059,0.6040449951933994,-0.6383558857740542,-2.0624676715571066,0.8296172882142185,-0.16791584087863815,-0.4392986753463036,-1.862487484391063,2.674753375161767,2.4228245264984274,0.6506593794791222,-1.3215167643022434,-2.3838050393464223,-3.2193199217608788,0.7792986759503282,0.24856240597281296,0.7989735971063581,2.364652274489214,-0.5589848596764746,1.7269346427442809,2.478232818411885,0.1613912660052225,-1.6585449340340996,1.4011297208455535,-1.0919936747383947,-0.6350829763060819,0.6847685531390004,1.4991380957099616,-2.3108845378610043,0.15771439021422062,0.9018850723574449,-0.72849460327735,-1.3313362415272967,3.8627720873608467,1.361197185540007,-2.677277648365794,1.7523859742717276,1.1894104819650175,-0.4475698490890851,1.3188931390717695,0.9099491807010682,-0.2058527820085387,0.7703668172975452,2.78962517926982,1.0397832310018789,1.0043702767154261,1.2789242460179069,0.46383594381852394,0.6478882796858255,-1.1526443018735586,2.36821345908028,1.2513347823568586,-1.0650965610589924,0.6874860976253703,1.996110660750447,-2.9053031164896006,-2.7294381535147854,0.057070972618107674,1.0432194527518035,-1.0605174804598037,-0.09606707315923656,-1.5804149791875093,-0.0736688539033195,0.7161774596727665,1.8724607896097631,0.6338533723027996,0.7428184447229833,1.5349181989248784,1.7644145475573836,2.973872346674843,-0.3349507825490612,-1.181344863796778,1.0816339687540668,-1.2316876093340294,1.0645337626784717,1.1508058202677631,-2.508024072541918,-1.0739411191437969,-1.4893105727200462,-1.8264885600081546,0.6139181059030082,-1.2137122189827203,0.2789411933010082,-2.3491568381551264,2.1634846945963297,2.4963938714203846,-0.2815044762117078,3.422451719092284,1.5321287272393758,1.8387180999780879,3.382080596278901,2.39841864730109,3.7232234282721466,-0.3788199499769593,1.1543305284228325,-0.4060381144074462,-1.7043389735307328,1.2634531915370748,-2.206602548191947,0.18305586261602616,0.7879912727070496,0.5294476903870294,-0.5471239670901129,-1.0267196813352888,-0.1743611346554091,-2.567238761265879,-1.5861469857255657,-0.7314873434250566,2.606489369074707,-3.5430193582805236,-0.5072342343273232,-0.43796269325767595,0.29069268753184185,-0.8568505302148678,0.5932483093134446,-2.1508194772095486,-0.5607164568877245,-1.6938431442089135,-0.39671195741155935,-1.0515053590161212,0.4445618969979085,2.139319501067734,2.1863699282479496,0.07114852933824009,-1.0448901441169454,0.33431410525033073,0.9613587261613936,0.5842654194213105,-0.4767703224846371,1.2883042957660384,1.092752208900911,1.1200659111376294,-0.9601281004811905,-0.5681588212248893,2.9355958030231095,1.5134053112219854,2.744994246337656,3.3470463770471937,-2.0685670357776016,0.7738097362557975,-0.20725296424430045,3.367442815578599,0.19507353133457825,-1.9019961475640748,1.818002100881018,-0.5105424223887008,-0.87765045533368,-0.8951227970851381,3.160212538115652,-1.4006397108247817,0.4689239147126712,-0.050015928342477166,-2.6421218446935697,-0.5687589305917167,1.6743232534997465,-2.525173227295921,-0.7756687269678201,1.0943741196278323,1.9722525556589918,-1.7919066671036918,-1.8700728488887066,-0.735502754269338,-0.8311032681713603,-3.056133918349606,-1.7068449343596943,2.9122424527532322,0.6862387885087584,-0.4013345938894649,-1.9826893702775257,1.167386471241466,-1.453296833916852,-1.4098733826180614,-0.14536116304491026,1.3083022816610934,1.3552597162716984,0.9657660224523864,0.6801170729881095,1.8724411037080957,1.412287422323525,-0.6862182055610225,-2.435349248919269,-0.5484287724603024,2.3769183998779515,-2.118257480587849,0.6815066146091182,0.23922895403460404,1.7331155427394696,-2.0199783260870436,-0.8933070603811322,0.21256085746156364,1.26013170651024,0.6009923697729953,-0.4057644825784316,-1.6503742534429406,-1.3681697544769644,1.5293517112950157,2.370163886791873,3.6239186739829248,0.8904297791018925,1.76961441517871,0.0907874956434672,-1.8317167236692773,-1.7173744300095155,2.5128855844971114,-1.414865436857926,0.6206921061339845,-0.643799103374647,-2.520231407779768,0.1550117341554703,-1.1397429105388406,-2.560673877927539,2.186544242260973,-0.4672268256074341,-0.8085572915039649,-1.434259484836167,-3.400193186520252,2.781957140392919,0.2552208508447559,1.4816581003003002,0.2946505907134949,3.733616349448828,-0.7138190358866059,2.3929617839131754,-1.3696272292774196,3.2551621389024796,2.2984520073170542,-0.7962384276482883,-1.9608410552765279,-2.8628071416990215,-0.0015423088390852692,1.7142379140137673,0.3986044337381147,-1.1036774337313615,-2.5259018795381376,0.7600002590686761,2.823322810827091,1.4370751847898602,-2.1827005109584885,-0.8454256592847634,2.4159974536307973,1.4557804663739136,-3.0404305466732517,0.44871612593174975,-0.8919498207727371,-1.9080537363787105,2.333079228586599,2.3777901801595758,-2.7790838628683274,-3.089602932133744,-1.4155431120561268,1.8231567351627957,1.0910358304419763,0.2597580256042947,-1.0787118485889005,-1.3038154234316768,0.3370732336021372,1.4269618190090703,1.5328793639169276,2.331956319883,-3.0141328947347743,-2.622187682499601,0.8229445271998062,-3.0167308481999733,0.4301154061488161,0.696878488733185,-0.853068748379519,-2.430997876049822,-2.4481038442578,0.9595191710361707,1.293927772224634,1.3660698574474393,1.1376759275795805,2.6756811751403036,0.8434051218938525,0.06271481947339345,1.7714316183225822,1.336205782284927,1.3468694439662086,1.3635038010631562,-0.2877659770471687,-2.8574086925990834,-1.5782422777658511,0.8236665137615645,0.636617293839928,-3.281835637044022,1.094802491468863,-1.6466570955393374,-0.7458296196817963,1.077909336053281,-0.45586978733508565,-0.9949961212794968,-3.2642070670214394,-0.38933983950505974,1.0560101203348653,-2.5848392987094604,-0.31763976807652156,-0.22083526391725572,2.1838968299854202,-0.8866654930265814,1.9913051124188785,-0.17178722129394128,0.9045526541322383,1.7210915870479329,0.42676683512194774,0.7837136136698071,3.7274012714815785,0.6607745176681672,1.3872614208299163,0.34896539014134337,-0.1575768630740532,-0.6881983334140069,1.0723573937639512,-0.6381442094529836,2.6613672653467892,-0.2041863777184067,-1.8123430809735366,0.9095811217497909,-0.27253715156769215,0.25828483603535224,-0.615286575753193,-2.4322200372289986,2.1223114201290896,-1.2170087217583783,0.24560186462975697,-0.8933591489820577,-2.752841397509749,0.9537672385803362,1.5157514640666507,1.8346395646124816,-0.5348384409222691,1.7181109868241515,1.726169308276835,-2.1469768348201246,1.0434887257342083,-0.06852911692038006,2.0464628557487567,-1.2020404739439066,0.21850705651485428,2.902951303492896,1.2210996709288136,0.9514019390385577,-3.3179135542021365,1.1171052565147737,1.955221099548493,2.330921875074903,0.2479352858054504,1.3271515394216467,0.2811637656237267,-0.9085393175702715,1.4956332056393082,2.419552286985759,-2.5765427184444443,2.031284951652418,-1.876580410568648,-0.09494914803793435,3.6952978953470494,-1.1894090890365452,-0.21846213706680306,2.038964688479113,-0.6367212759740638,3.520765700004676,2.1024133516818924,1.6434176345634557,-2.341071146327416,-0.19270488279058703,1.5916141020876688,-2.0821313437566116,0.7417231268482539,0.6791378401127623,-3.1574807997829724,-1.6410006111012578,2.465332129344312,-2.0035828973844314,0.39680198016145835,-0.640646575206632,-2.3160143371432795,1.42403244689098,-2.837628734287414,-0.9700672837410902,-0.10766416152093373,1.7094577008873832,1.5221799434894285,-0.7803366372106325,-2.296408124795674,-1.0843844972587537,0.9946318378516678,0.8123804568569786,-1.230454544126837,-0.25798812157368084,0.8377231942374115,-1.9841200180911285,-3.4519457974123293,-0.09316273626123042,-1.1809439232978183,3.206470798622008,-0.4701097472158981,-1.5694583960352013,-0.30141571114088295,0.8415934522230956,-0.22675319776487166,-0.24189167990023847,-1.2178226709329334,0.15880922072856896,3.400723973137037,3.692815927804021,0.943651076861909,-1.7611529340112557,-0.19550291291548208,0.6892540865433285,-2.1945072345765877,1.682707269000614,-0.9802367109798653,2.377949202253414,0.4384488942543136,-2.6915316258838407,-1.5533348539486354,-0.02438452632916925,-0.27003080789726785,-2.433106460941493,-2.7777121702398286,2.025155434355288,3.519915627966696,1.6783018139967407,0.9065541709128797,-1.046636459852516,0.014104362015225184,-0.6310353998756005,-0.4701102482120643,0.5765861839675779,-2.871861382713565,-1.9902159702699218,-2.3254413535056395,1.2279917049316729,-2.042996840823422,0.7080500750992647,2.2988010102002887,1.6324648818119702,-2.123822948397591,-0.24280494130175,-0.0003163008012484396,2.81220464373527,3.358106749918216,-3.5032019070561478,-1.805406769323465,0.47662705678591627,-3.5505847026379334,1.2786480985549071,-1.8721486541821502,1.92719393152672,0.9522886150341182,0.2628753668727002,-0.1642188183685333,0.8143918657606737,1.574238527364508,-0.7933024137529414,-1.8170672005677548,0.7981804046895842,-0.6056724040497758,2.258481518075223,2.414825919770181,0.16298727504170823,2.2204980391374054,-2.1335936120295664,0.4736999883420109,0.703267109807271,3.5695319479924095,-0.6167099907113673,-1.9401113750487344,0.7890024525388838,-2.69342228927165,-2.719157753172775,1.805725591707594,-2.2668703266457837,3.7672305086818105,-1.8333824758255262,1.6449617993257877,-3.457681204315066,0.3950648746842371,1.771589151474583,-0.7438150331908671,-0.25186116464220376,0.9543210589790686,-1.023153657977628,-0.5228372973054197,0.4976172326103097,2.0302946166054086,2.112610695945148,-1.9836226948943154,-1.7581121112458415,4.417977449721752,0.23571547859129666,1.466162543384971,0.8045635481151686,-0.24367065904354465,0.6058459117136098,-0.779076828773128,1.2157393601467843,1.2550897407599855,-1.7566046491113503,0.23508167246350517,2.5608537631886246,-0.1099690882623724,0.6707777051310175,-0.27722931593403927,-0.4348653346564092,-0.7133378401707994,-0.7588917026632911,1.5203849070550375,-0.11467590592635132,-3.0050020130293658,-0.4384642418746191,-0.6586418843015375,2.0248178614048475,-0.4819189143648627,0.5480270127159396,-1.4434972487131337,2.7487994792896444,2.660168659763537,1.475564356921047,-1.7917847064004935,0.9456862044187193,1.3550156018812072,0.6489208781067982,-0.6823896784759853,0.6809419854688155,0.7557014529567045,-1.4041870639861365,-2.5202535832846404,-3.585659203016368,1.5505986306664379,2.2199678203906004,-1.8676956077209816,-2.022039374353819,1.870786470475572,3.7170896715874555,-0.6603030185635144,1.5618587429446713,0.45857952927173484,-1.3379150812526188,2.936425452838496,3.7918193425285036,0.3318922205956912,0.9991437779488036,-3.1457252098600965,0.17812158600221117,-1.269918962557141,1.44359320479979,-2.825304817974154,0.9149638002304461,1.0281106477053579,2.45335285455934,0.01976350721321546,1.6276483022407655,0.9691627635556536,-1.2261095242809439,2.6815272449670235,0.0055861918511456105,-1.4675878020330808,0.6749875925067799,0.1562067914549314,-0.47413918361119756,-1.679090814838428,0.42281294220421217,-2.2529237179334447,0.6916380529369492,-0.8601210516121003,-2.4135691346761363,1.78810403120429,0.26176903006922086,2.8027134228823085,1.0196975559715393,-0.7249033079601505,-1.9803700989113358,-1.548971035448571,1.109829867864511,1.698352763179285,-0.5046601521795123,-2.3325542617229353,-2.873680844249633,-0.706093543998037,-0.4735888385540876,1.143956887535384,0.049887708913588964,-2.4118764480447528,-0.6469975749307326,-0.8196995007141664,1.7937251137091172,0.28378691411230994,-0.7605397334198424,-2.385312621500402,0.24769663484989995,-3.097990693749487,-1.715578193261147,-0.8135759036275396,0.5987949037493442,1.5059545307268305,0.7323064431838984,1.7479201244150295,-1.099065291061513,0.16274062580982768,0.5161598031531309,0.7591919093919794,-1.4065102913511034,1.4564795592596342,1.9308771327760068,2.6254488992854017,-0.13746975970177347,0.07996349865647953,0.41926612141702396,2.830130852825643,-0.4491902674924259,0.08341574307122217,1.0199014901048435,-1.882942325995724,-1.199989809908867,-0.2751265414516046,-3.1859675491217385,1.0775406102316518,-0.30868937904769883,-0.6651293298818169,1.1174836753231632,1.2114053449895597,0.6423067522756825,2.969603707353098,-0.26487007138829755,0.10473163154760401,1.1332189241076047,-1.858820195001254,-0.7729858802290381,3.0155363453522877,2.3642615134997027,1.0619169080891018,-2.176473771756853,-0.31228973906596974,1.9468917961963155,-2.9468531111314213,-0.647921557508623,-0.05071872274155585,-0.41714608174772466,1.1264931423113491,-0.45916832187113144,0.9581166884293913,-1.4406740617993263,1.5875307122616165,-0.6833739078647272,-2.4051812416410305,-0.3751087498358798,0.41230377022134934,-0.48661740444675744,-0.9575536148700023,2.80263901816126,1.4157459776698582,-1.1718460658846614,1.7238711977099572,2.830806536494001,-0.2643430248609906,-0.06908785991831703,-1.233786741639355,-0.5162472356068744,1.423873465571174,-3.1324207910710786,-0.38755011493760494,-0.4055813125948151,-1.9599794635703895,-0.047077391773360774,-3.527863332716331,-0.4842201089662199,-0.33029375454576426,-0.3287111807630712,-1.6352057952142893,1.8748848282096113,3.049530246127233,1.929568258778238,0.3789033475154003,1.3886271141159934,0.5614924485332313,0.7834092997317688,2.085086424670537,-2.8662638210080043,-1.7925452137451914,-0.0017555573717466322,3.1023764818925805,-3.183942317803469,3.4103066642855397,-0.7206853904244521,0.0028946578558309696,3.5637443400701896,-1.9851980777601144,0.5513398305379537,-2.2937707616059675,1.4312825638451938,-3.8110884020082976,0.4972591542633667,-1.707648010883881,-1.7576510180607066,-1.020300628948373,1.1128115322778243,-2.3017399316761966,-3.1106095892968675,-0.9339516700557967,-2.3094816874324677,0.4601223924398294,-2.481915775832632,-1.1809832112693892,0.955859088635349,0.9723635520248778,-2.255250361415865,2.862478203910889,2.5918271401589195,-0.4756900909047877,2.8969563052774747,-3.0509313397672653,2.154357862282449,1.9494929387363598,4.045343415889594,1.4383045665195149,-0.010639269810052233,0.24904897577084836,-0.859735502819102,-2.1998271466318258,0.8783318604645145,1.733164607057025,-0.22376958801145297,0.23848830401299959,2.5720409228077634,-0.3204491289325827,-2.105777427498722,-0.5359987781877182,0.8870443065700109,-1.9528602476279595,2.283242952325813,-0.4916142357149653,-0.9696315709172404,-1.439278218864536,1.5952878146356597,1.3561708182482315,-2.2699547383933045,0.5425495440440292,0.6599888812745703,-0.7374423585877579,-1.3943767294634004,-0.196106162750489,2.654777008041068,-2.594147502503161,-0.45344939514940635,-0.9959650741163102,-1.4261565735676611,-0.03925220680441983,-0.09940091006890511,0.4339371151370994,-0.09502752301886831,-1.124876352355228,-1.1331220425435222,2.7010803006648048,2.697413108362335,0.7615754231108828,0.28499778232361617,1.9127871295757166,0.7767835113688578,1.3257770866094347,-0.6816572184349361,0.5742597770052463,-3.253469668001257,-1.6319473872915171,2.1163919842017718,-0.18336497093240803,0.6752218683656072,-0.027070189715208928,0.5208075674234642,-3.0731525711972005,1.6057099495231573,1.463507498062831,-1.574778021406103,-0.7014994390200381,-0.7100293342992985,0.11919893907393697,1.6736523499563114,2.8770133683832753,1.9600349436251816,-0.4517965255633002,1.6695015504262758,0.13475299410517452,0.44619973469689717,0.97548599432783,2.564673597827487,2.325903023409257,-0.025810247676851476,0.7227667191481022,-0.19838896862089297,-2.8362853576819096,-3.447986685054839,2.078470620524031,0.4610971826230799,-1.7906129340874322,-2.557790015345394,-0.8100366065819343,-0.7623071153256402,-1.108083319583765,-1.0849722456511406,1.907940773056654,0.41957698495618767,-0.4059577591429291,-0.7115525144347036,0.5938391772872705,-0.3402419714397837,1.9972434831588608,1.434660000450524,-2.1130748891811724,1.9448940611800185,-1.4778928003231564,0.2789614827619925,0.3694851702163435,0.5980987417080457,-0.04577090256123693,3.360443222972719,0.7430284126798734,0.6028869756756999,0.8913568134552795,0.45917350944811813,1.3125967244339827,0.2623663089639637,-1.7161759030985524,-1.2108847751268734,2.5891166436739583,1.2131883437507867,-0.5358025950538519,1.6575583263171465,2.0209326101934297,-1.222272859124809,3.2747274175945753,0.9246594400373921,0.6305356266889902,0.7045361681180891,-3.230027858615949,1.0923257478767985,1.0478307547673473,1.738626559833037,2.8061637346973165,-0.5840986068099027,-0.4031323929653608,3.123216170630625,-1.8685855571531125,-1.8096311868326775,-3.630660844866638,-3.0638903122466137,0.3304136192408009,-0.7611953426437595,-3.045797843400349,-2.0727124573836524,0.5722030831425433,0.33919839916077005,1.2552615493716568,-2.813209630353744,-1.3173446101112734,0.8516025829820751,-1.3702262489362247,-1.8368508459956534,-2.8710195427261316,-2.81158853249136,0.16294205808620765,-3.6349074235434986,-2.465367580615799,2.75807952293976,1.7121934627669104,-0.827993875912066,0.7630741897945431,1.2018875059502543,0.3012242015390892,-1.4625697211141409,0.8236075677394024,-0.35288220940996257,-1.50776579092918,-0.9532485746137893,-2.2113685788271993,0.11693635505177344,-2.5207427438723693,-1.0045734881863628,2.577988445938155,-2.3546953453387647,2.053047833510367,2.7822392542414325,-1.3989299045917125,0.841899706270972,0.615060074658758,-1.1041259040612952,1.229083217704513,-2.298122216699128,-3.2136481851881893,-0.4757451664649361,2.075909618452221,-0.8712909636879754,1.8547017391769085,2.8584528916056793,0.06409372675395235,1.6951458514632503,2.0386552513822083,-0.7708038752281092,-3.464991191581467,1.791891539628675,-2.407559363332601,0.597303219299708,-0.5096228619462404,-0.5710638993856779,0.2960809423822917,-0.16205373933038986,1.3307022522617584,-1.2072633846710639,0.8904617731522527,-3.0759180177691743,-0.20009282792615696,0.6478629441498184,0.5419266245450655,-0.19837430866210867,-1.9359200236324978,3.1271520857210904,0.22782536561549996,-2.5993066219353658,-1.4702030859986457,0.077412716436419,0.31817106204010553,1.22447734470053,-1.719597138025585,-1.4746769230484054,-0.06698729499888022,-0.19987671399669554,0.4456619117632616,-2.278775929559009,-2.564378406687351,-0.3339873182106432,1.2701793960441092,-3.7554102185770324,0.7049671318268553,-0.6418725263576097,-0.6468081944646502,0.6404988322576434,0.6807208274579416,-3.3367643646256386,-0.19483277833952864,0.7395956307227767,2.693722680524081,-1.4186451409290333,0.40295980091886324,-2.0120885352236377,-0.9743918179454342,2.4614948412254205,3.054394128746183,0.0674001117134557,0.014278797932470975,-1.7792646683463902,2.1455956698845555,1.5918724645077218,2.650673537800932,2.429469056039803,-0.496795192623799,-0.0689395478075918,-0.4729819469691802,1.277769516971819,0.7161548380291773,3.7556605694237555,0.20452813647255771,0.2006187328393419,-0.38447162931049184,-1.561390517037166,-1.0608551926697118,-0.5668696448465607,-1.3516364575573268,-0.748672032165978,-1.3024950669486863,1.7058319133499165,2.645005766299684,2.3785358844104008,-0.9902409543392486,-0.11339813374663506,1.1881047995703524,1.2395681999551313,0.8816558692918118,-1.8606075199927237,-0.5163418598470514,0.973419972183097,-1.2291072457186445,-2.1203359313859305,3.076673508331687,1.5528906479488582,0.2808459764727237,-2.6673904108928888,1.073870320939583,-2.367754011506648,3.4968592564871077,-0.026047219760940505,-1.7089563333554627,-3.781611750245187,-1.7447148356344269,2.9836821682991634,-1.9819505549518859,3.1577051530805194,0.6299205062081306,2.41601922036811,1.2923549721766034,1.239345340348102,2.069765227933306,-1.035019275607185,-1.4535534916487405,1.5468601249023877,-1.1939544954020402,-1.331771303978018,0.6965389112534023,-3.341251540374439,3.376982431774846,-2.018676493686295,-2.393716784754537,-0.2152543015963771,3.499147960207785,-0.9645678684900834,2.899006294136923,-1.188402840986386,3.0775121248026758,-1.440130678578507,2.248043172973845,1.7544092688947985,-3.0656196600290606,-1.0248274419233874,-3.5113450365341694,1.3064329181883723,0.9512855913669275,0.2760170630674131,-2.130123708118644,3.9198529672284113,-0.5824854195677241,-3.562005801825264,-2.5161889644459783,1.9310750146999354,0.3236619404932608,0.3318152084431852,-2.6533879040750366,-0.2450760071538679,-0.1955943166298228,-0.7502142772507986,-0.7285398175791321,-1.346268691894818,3.0464001142135513,-0.07578146077347982,0.521956849549782,-2.4143789584939173,-2.537503770384538,-1.5040992909792859,2.704678115538146,-0.6354493915461756,-1.2429013945616176,-0.4557113541172458,0.34548201788524907,-1.2986041530291814,-3.73310828711337,-0.0401332220226747,-1.6929102155221236,-2.2921459588111963,-0.33769676380667674,0.6383892038676882,0.6622331074853133,1.3728544377140002,-1.001204440700415,3.1659637751033207,2.9536850765168325,-1.2024554256754245,0.16551452032871444,-1.4161177890284704,1.1445941929814991,-0.6351679557038055,2.8562609445902,0.09660177799080555,-1.7747060724784947,3.6033045357624567,1.6137775744943572,1.1998184971290924,0.009492224358016988,0.8301001375387448,-1.866137056093423,-2.4358494368175,1.6081997771581664,0.4636402780186623,0.33352051488028883,-0.3927270333002942,-1.470199716701783,-0.13409299062107813,1.4224272795022617,-2.4974834510788937,0.6642058350901202,1.891708862445406,1.2549288701570698,-3.0659775132396994,0.9576941322460315,-1.5670593554304753,-0.0018461442461507691,-1.1478143271601513,2.013511457814281,0.45220245439030443,-0.9445245109321453,1.004472889755887,-0.5840432891796634,-3.170774689038469,2.5474899412418814,-0.1522957993020395,0.7036629865112266,0.865386682347216,1.875382966545238,0.5941310184428716,-0.9983862866636449,-2.041174952990142,-1.9031241372022063,-1.8281317060295263,-0.66564806355106,-0.14955991895706833,-0.22910836096379156,2.1460882836577597,1.78660627671093,-2.122325433781848,1.4165802224935216,2.9663692072429058,0.7664992173040227,-1.8814881457450385,1.4471193520099952,0.6932182412406567,-2.9763279956183544,-1.3937031785876248,-0.5803509972217222,0.8460684119992757,0.30736511605774575,3.591426430741455,0.8175886325056606,1.599699614246664,-1.71340253816115,-1.2671369184490688,-0.39781514453213473,0.4512952601975635,1.4517614626741777,2.332001538773863,2.3296000878481564,2.912965452057654,0.33698964291599487,1.5193398131348488,0.8217033542269607,-0.055248522451910816,2.384689332429924,-0.027594877881724524,-2.8135485061289804,-1.9944696066317313,2.7054332629331705,-1.8064466462814621,0.12205870124753825,-0.5191269140528427,-0.5233211331301925,-2.502599735312203,-0.25761927586413436,-1.802628774085578,-1.2966937307595199,-1.6524346500628093,0.4364577384176877,0.36074783227329427,-4.005923664608169,1.0111770650573277,-1.224273133968606,-1.1267287715902778,1.1495576016933555,-0.03703204124591531,-1.2062194433365618,-1.238035408752359,-2.8726615764276486,2.2098180791548616,-0.5851468437789967,2.187122384519611,-3.4066676850537987,-0.7242994636045293,-1.01481746429809,0.9925774384170623,1.4103442212089647,1.018334292922512,-2.0945437014780794,0.5532969749020217,-1.0808962194992218,1.114269144176385,-2.980004256182643,-0.34793350228646236,-1.256989060490447,-0.9181544698134563,0.6859082223141256,-2.514025712190676,-1.3184491297950718,2.6485629422912984,-1.090488493233115,-1.9377437336902328,1.2147856167766575,2.4757163809563343,1.3308418740528498,3.474738230894036,0.2814010272481164,-0.8620249246941036,-1.598301824289057,0.3446773514230904,0.690155781856078,1.118721851469989,0.8966194422776724,0.5949119918879772,-3.1341046017869534,-1.1441844800889482,2.0213892677700387,-1.7535976766479755,1.1275009123858173,-1.0146328714171673,0.7788006770345621,-1.6979390234557727,0.07451855848968772,-0.6442448800235667,-2.542978669691917,-0.05590499036702033,3.3597237689292596,-3.332007345731853,-1.3450380468592111,-0.5779477209534698,1.078917700774422,1.7204354144575864,1.3459772879859124,1.230078196631134,1.3443632622175996,-0.48922631522555526,-0.6409742263545368,2.793293059095719,-0.3090097255726149,1.0160306524189566,-1.0546821996941897,-0.10164621365425552,3.588033273913631,-3.1152597460505236,-2.361936823522366,-2.1359225357950815,3.619237798930744,-0.5574168607010649,3.519690678115296,-3.5899221719023564,2.4859199290435106,2.703042084843705,1.4094488783131385,2.258877896613164,-1.7715335319859575,3.2337005684502858,1.3892791012285226,1.3610638571863545,-1.4190279374614678,2.3779480410286093,-1.289634716260244,0.13581371037355097,0.1223579190799539,0.8457576235363824,-1.0402014803735793,-2.6144035639551895,-1.1373016243610519,0.6181042704325662,-0.2334353379564443,1.629815513046824,2.131113305260808,-1.4118013007998078,2.9974499331878484,-1.743455703333226,-2.5962433203831274,-2.4540192093067907,3.258083168204038,-0.44404808046606653,-0.751117452839145,-1.2249990128924124,-1.0311450646854459,-0.8407900931425444,-0.0008739543217531632,1.433909871620768,0.26074845188558105,2.597372722245744,-2.6301202494960902,0.8126805926687282,-1.4595711494051276,2.1631382110305584,-0.8563355038669093,-2.8052082049359632,1.324978269062536,1.2894894095462783,-0.4668865745289191,2.326935150061905,-2.1638057117736884,-2.326120654630772,1.9802543984032766,-0.4395557288763195,-2.006774745836087,2.6061529932555407,-0.25841768261464304,-1.8490311022312187,-1.9231012643029768,1.7148928458631008,-1.0681412124252527,-2.7832815470919265,-1.4326738609346599,-0.994088964487659,2.4162090802563654,-2.471748769882065,-0.007525827086982159,0.9154036578013431,-2.1306032504081447,1.0682612742961006,1.3784322145936705,1.0129336215554225,0.055479931845889174,0.8903625883763985,1.2217647582602722,-1.5432158847614783,2.5888100949076707,0.8894306917921109,0.5316567911188876,1.0534910537103845,0.9888102714295328,0.06582815324878297,-0.24856899277958644,2.4058067266944407,-1.91973616303867,1.940389685516755,0.0705994212161823,-1.9268115328785846,2.870657311495887,-0.4801217080942407,0.9994065270454744,0.7329347273539353,-0.7877700071013647,-1.5897598219695308,-0.6772192774532632,4.383306832120905,0.10467201673408477,2.2316998995996586,-1.4797401217036636,-0.64841062740311,-2.5097024968013533,-1.2508700333690534,0.18749096390137374,1.3098615739786446,1.862693523053395,-1.5781375014271015,0.8893362485264499,-2.417415368807307,-0.11387455931050974,-0.2391467050980752,-0.3540182504153475,-0.6412279964441314,1.4429441775706293,-1.3824820202366872,-2.8500008424104784,-0.2673829559711595,0.3012742248613283,-2.0158044845795655,-3.0414994338767034,0.9216853912434467,2.7214803316326086,-0.7080536253890226,-0.844382270672674,-0.26893108991164505,0.9896552422591456,-2.3942085454526745,-0.5675825976113691,0.934373230436313,-1.1359661891619415,0.7049963100974895,-0.3664199787989329,2.1746073105530686,3.0309245928863446,2.8257230252767225,3.6639772298604107,2.997416616761887,2.7168823447703954,0.13560744233521924,0.3918399563174649,1.8452007486568114,1.8517438984042147,-0.6171336924048768,-1.1091275300872507,-1.795076142338854,-1.7349288330922639,-0.12939335197655935,2.0105129493998612,2.4759348505813867,1.1817879190708258,0.6552483781526794,-2.6659236751308946,2.3428925159115384,0.7701147836844777,-3.3806072956329487,-1.7038000488136023,-1.1907780092785893,-0.028469544675232553,-1.2809084307204714,-0.8494575320282317,2.2523482711621465,-1.9952050835779627,-0.6084364454691109,0.7050917058905579,1.1731039346974124,0.06722444215840556,-2.0420973552799833,-2.7515226346293122,3.0550047731732133,-1.4221821804177066,0.5426871300282187,0.1097193059986648,4.40773612104693,-1.3574122350606233,-3.257418131456721,0.7442535118579422,-1.5752463473515397,-0.9925247557973751,1.2677328363929412,2.793759353933752,-0.9852622246612804,1.7755655986370216,3.028238352452494,2.8600965067760904,1.2480271853866003,2.6808474378764173,3.5408376358240887,0.22443491368741456,0.14123993854909636,0.9701468828622638,2.762743018936653,-1.3947840741556359,-2.2233236835906744,-1.6670548977724569,0.11193873858014111,-0.6182720520601805,3.2649998891737635,1.227833926548657,-2.416283573863712,1.0738909557237486,-2.171388600861608,1.158068275613779,-2.576721893512036,-0.08252456356599501,-4.008731430473993,2.0279200145217935,0.8729682450096766,1.8515198005155271,-2.417236168174073,-0.38628920957112023,0.39191009346710187,2.123803296012475,-2.450043598344983,-1.155469111191558,-0.13511992382147175,2.092995183132868,-0.73136238012548,-0.5037690000977701,-1.2105855881026455,-3.7996336989383273,-0.4982436309812873,0.45260588100646476,-0.9542410561036948,3.2580842606352967,0.8237778804611229,-1.281806393456599,0.05347449839143577,-0.3408980469016134,-2.9881263500652846,3.00807627724614,-1.0265098498562115,0.7254092007376443,-0.5306251796420973,0.948833736859648,0.9146599648121708,-1.3948630974746303,-0.8974101650196953,0.26086527621050604,-0.2627900030407237,-0.39686723340380087,-0.1254788561437172,-2.253579516545528,-1.4771984562290124,-2.159042616788747,-1.0126038161651367,0.0033311581954260172,3.7549556794330687,-0.21754518474134824,0.8984613697473512,-2.2352022926653685,-1.591120627090151,0.45851962910435634,0.9132955084699517,-0.4865008314620914,1.0370451234830125,0.9483842925003926,-0.942875827323133,-0.906434709826941,0.8197540915340115,-0.5656521144419626,-0.6202543604934899,3.948450550661128,0.6316539454174852,1.2843690185039929,-0.9592256859300349,0.4857095180714282,3.7832055769064885,-1.0160554122760952,-0.012293556441131927,1.1228916920807879,0.964798118056897,1.7945537260989857,-2.8719971593920532,0.8887940223152917,-0.7392948712511552,0.8320460244063076,2.251519685956572,2.164276059437454,-2.467354556469096,1.2515292569889238,1.7463713368909506,0.7598668201014205,-2.730546156753509,0.619776342406233,-1.1883961618444374,1.4018813368399667,2.5108116259573476,1.1551770139638353,0.9832148285997061,-1.47209912034659,2.3686600283700607,1.1640463594756223,1.9685520502383564,0.777984576164798,0.3394976002179673,0.374959059489631,-0.4149248002333142,1.8821340808732316,-1.2572631536156598,-3.631275670189696,-2.9614727181412994,-2.4651448259306976,-0.704075237756371,-2.1731210196192925,2.460803308028917,2.092992077932646,-0.7816265252085364,1.2438551203518717,1.7976527932496627,1.656827550687228,1.1669298453213575,-1.1037236849372298,1.6720516679096116,-1.8698130603697882,-0.7044097617797537,1.4988016827332749,-2.182054037897087,-0.7684408253173938,-0.07952489288298481,1.8785734284586533,-2.824589048960954,-1.8081183767094813,2.7206679583317213,-0.5833831618589851,1.2499673344160735,0.83877057926178,-0.7538590984410235,1.1187102569027518,-2.631262570711308,3.1896588900898597,-0.5878445648191054,-1.49053108620457,-0.09277799868541581,1.5194879696949182,-2.8731021482751675,-1.8498837477781762,-2.5246318729280937,-2.4779461979123667,-0.1896976231476704,2.0258102867976557,0.032593784935625696,-1.5218288569919252,-3.5757621007421423,-0.9920030550413349,0.3237259594984433,0.9714134822952245,0.8490747753681501,1.8263652471767897,-0.24819956083517536,-1.5759654757283557,-0.42711930535251635,-0.6129473113313169,-1.4283569107215401,-3.308915772355357,0.8245593847580995,-0.5683506133017414,1.7836808573127123,0.7939550237493535,-0.44153504783121994,1.4563095487280469,2.997507794901567,-1.341346024956396,0.2665130302249678,-1.4669148803651229,2.048956572002897,0.9250115657445619,-2.265422486377178,0.47190342402422764,0.4395337992797669,2.2066406869948794,-2.7516690807249216,-0.4230433875205494,1.195213718633142,-0.30263361956817264,-1.2144759645688803,-1.3940842045294044,0.6562901938583902,2.1305585234875433,-2.9884269990285386,-0.3652192477111344,0.4753635819308598,0.5871960124964459,-0.7129568605441622,1.0545624509297409,-1.6814566550817767,2.590935971557496,2.527379931709627,-0.16679339490115005,-0.6294093392191464,1.7210832529545714,-1.3231566812105309,0.07692753063028963,-0.7054755569200808,-0.2499014393855627,-1.9812824024423639,-0.5411761892522071,0.8527278949184448,0.8039883544219,-3.3749295661092513,1.272135208232459,-0.9598748679029006,-1.2525809976374853,-0.8246270069285488,-0.33561952760543107,1.3291322535018617,-1.4216773396062514,-0.7032048355122338,2.5288032210300164,-2.7855649000257885,-0.21605165785519026,2.5461595453264945,-3.399317388333225,0.05123901566535826,-1.6620804585961138,-0.8648533574940187,3.0132578979894453,0.19397440170742444,0.1177479122685691,0.041945964237910485,-2.068096789580307,-0.08348953903197052,1.4340550564844492,-1.4068926276596354,1.8329484940681486,1.4677797098294916,-0.05387938319575708,0.2043446163670411,-1.3655358554135937,-3.903456170514771,-1.7987867717569213,0.31746707446077,2.351616176847766,-1.7187734531982086,0.9781316434013568,-0.5750780329390482,0.916916866895332,-1.4662684206482495,0.19185391711668803,1.7845601444024821,0.5475209013024425,-1.9943271258584983,-1.6131887635185826,-0.17881459810436606,0.31601759022723563,0.10730958834472654,-1.6459032720686997,-3.267850275360924,1.2957235823207682,-2.466859965424434,0.1044743505819104,2.496722173473923,1.7395805203698684,-1.503664375697516,2.7181673531173822,-0.8103055859760461,-3.188817826993966,0.1751123828112851,-1.550943431850035,-0.2543302180623197,-0.35777694497972756,-1.2945420766398308,-1.7967463523478675,-1.3493628670806006,-3.4933045227307877,-0.08055583127595792,3.324709716414962,-0.030667879726911776,-1.9911026259048223,-0.45652773920721224,2.3708301814696413,1.141652209476892,-2.1407582801964535,-3.3025073851318005,1.0170146988672353,3.9492831630840333,-0.19902965438646605,2.2816037980922053,-0.622348566512307,-3.5162023232186304,-0.5825781344730031,-0.9688836817709151,-0.3838416818868492,4.126299802322464,0.3815778162566228,-1.5025716592983511,-1.410817131697318,-0.6855555968689605,-1.9196684913348365,-0.04961651151323103,1.8127968102621577,0.5217361758982956,1.0512350187427857,0.8210518125227345,0.7346312355162065,-1.5287103220116756,-2.9360973993217856,-2.0081145451225373,-0.09014282784085802,2.835575607175595,1.9131457967836745,2.8702537130167465,3.358763791970074,-0.1380186063078352,-1.3330833905397212,-1.01666214532881,1.0820255961350342,2.5582657195463487,0.10883482707978867,0.5634556361634522,-0.7358747433006394,-0.46072795247387954,-1.9545435358380487,0.656378922289955,-0.31680605961752556,-0.387030832754277,-2.4250184228996097,0.22839848327366313,1.3691776918792182,-1.9108920903526447,0.9218238489023528,0.14521156755861117,2.3858610053034845,0.255628405454281,-3.580994880251579,-0.7197503966735107,-0.687698184937061,-2.2846615191832345,-1.5863967981458817,-1.4629338645046706,-0.010730129533265109,-1.841484372833113,-1.7562758770331575,-2.4014959623275325,1.4071830907069292,1.0506994114911021,-0.4179889066107459,1.632901708943113,-0.10449066789119636,-0.5539180192500213,1.4560594459768625,-1.395975003719548,-1.8863121974860106,-0.4818359140113136,0.24852094224358157,-2.925989353887872,1.0578543918616363,0.680661088036699,-0.7658804559504379,0.7330163592419435,1.5028528147685536,1.2067746835821473,-0.32014821236199875,-3.9681989913537206,2.0487618647874197,-0.20491500309333943,-1.3745044234444064,-1.6485790778653984,2.1247369071558184,0.9503823059547946,1.4739427959610414,-1.1704857715637,0.7097622582721689,0.9642838574618684,-0.7406493834652328,-0.3199331242126307,-2.264868686655599,-1.119190833409904,-1.2628218546582934,0.4232247136936828,0.10238270004820081,-1.7309606425237418,1.4870744870459054,-0.48757689042133395,2.4267817865827754,-2.7824226089526367,-1.3926965204835458,0.33753197931419543,0.8204395819828785,-1.8796176220799625,-2.93292334348358,-0.09873141452909522,-2.261706593039914,1.237768407119455,-0.5420667918771064,-0.6683537407277383,0.16467547015760528,3.145675878804629,1.5578904043762127,3.1985626477015283,-1.4525191712649494,0.06921207265925218,-1.568986260569963,1.4462478551733444,-1.8449719584705524,2.2746983642002907,0.8439419714569117,0.21862310284943104,-0.38932760877892775,0.5463857732745933,1.4560249648039176,-1.593720007480549,0.1102294136745199,-1.1082408285318384,-1.204342346770932,-1.8784155819432802,-1.702617469823751,-1.1265693684778566,1.2783313865070323,2.0643873310975644,0.6016786764096792,-2.3186213904534934,1.9257834053748921,-1.0272687924932422,-1.2353401494192218,-2.4034337775263435,0.002773738930671584,-1.0705089917149213,1.5385973255462395,-0.008292700500090244,-3.8009851243770174,-1.9799831230768214,3.5909597975050884,0.19206514839589592,-0.14302946734711558,0.2719974340825106,0.7503732806311916,-0.4674476102620166,-1.3913989291210727,-1.3270173729853048,-0.8422886518301884,-1.2831890366907668,0.6773156206845125,-1.628732859833635,-1.2347661490217858,-0.015986702339147372,-0.9160092828597204,1.018309476395643,0.2246248839284473,-1.542594311313978,1.9532980470218693,-2.900243858983825,-0.6863361575501246,1.454251608804418,-0.8719155519865616,0.8945323419124811,3.409186889617061,1.5158234415452725,-0.3291208546728173,0.05331228359968951,-0.9485016479463886,-0.08021878320251641,-1.2341176787859935,2.3012129331921725,2.1884746845285323,1.0979431351247841,0.6401989367162696,2.129839795721685,-1.1841689960355986,-1.744445492201182,-0.3447128264365027,1.3984511742564953,1.7451738675595123,1.2815847072743938,0.08181094339478835,0.3883528980983858,-2.412238166610344,1.5457572994596698,1.4721302247921408,-0.8837057152297455,-2.098617014813328,-0.2542804504067528,0.7705209680631853,1.0413396192738633,1.0675793584554136,-1.5490826995472087,-2.2786395467959144,-0.19749264192673804,2.6562072521957867,-1.4114024915253098,1.5254723293903971,0.7416160552178729,0.16367900830673088,-0.5563984262259897,-1.1958603293890713,-0.8178389988376898,-0.6775020180950826,2.685192802011947,-3.63139290322259,0.3834293516307214,-2.0781297178440705,-2.6155455831512278,2.1886559070981577,2.719357317546991,2.8567429693297317,-2.076875227876929,-2.3783449763942266,-1.1624909318620897,1.855919121611455,3.4420587037085917,-1.5811924643621353,-0.406547816034419,3.6893660085626183,2.865246048310671,1.5995818510418616,-1.2512612211827818,-2.6491818208960822,0.06327036333557687,-2.3494247307769753,-0.969935448588094,1.8159083097798177,-0.8535331426467566,2.0363969539472158,1.4385210199250125,-1.7396009438619757,-2.044757929936261,-1.9984530817353525,1.5460915526326025,0.34067165674187494,-1.7324432742930271,-0.16267156009972744,-1.7998585933741182,-2.2033827749586323,-1.1159789704918874,2.305554137239205,2.137828825772748,-2.980357145742691,-0.5954464115442548,-4.050109671859428,-2.533169020836347,-0.7059436845832026,1.3087258643022748,2.3371764300974784,-2.511654472175127,0.3406784813897942,1.0213207363349395,-0.7691737349243353,0.02355809422586245,0.10025934276006829,-2.683892553246814,-1.948466050288036,-0.7889835354465757,1.4426323565315868,2.285664302909165,-2.9937991992659394,-4.327699865223282,-1.2666386095158788,5.242050269020031,-1.0966253835980784,0.8767106058144774,-3.011386683006101,2.621988705502782,2.0511646178378213,-1.9948782381244017,0.04649954814706196,-2.22460763110507,1.863406347641376,-2.3315315705409203,1.078421210067833,-0.9752899908940101,0.3032665599449111,2.6148275623043693,2.7119509958990844,-1.5902882446636484,0.5154857776863406,1.8651792507950258,-1.2297358170273993,-0.43559167541623,2.6990881154651136,3.5761533523965987,-2.310506252852423,-1.5274450278318117,-0.08881896534138599,-0.27789605872771644,2.721428021656884,2.2936562401633798,-3.1456009371444105,2.772702147908437,-1.3935409654288937,-0.8500933420881045,-1.7867466858104428,1.2126054685348737,-1.9152898659904571,0.02966500683781529,0.7184380787671837,2.515260217007804,-1.0333343875585241,-1.6868557891682034,0.05810174589433656,-1.9401713879104878,-0.35317757738191297,-1.6058735003158038,-0.9205178084887624,-1.8671449667706226,-0.3339912652814373,-1.0222868832638525,0.5720605471694801,-1.272285548388745,-1.3073469744785176,-1.7233687781021572,-0.18921392822825633,-1.289287691628495,2.41455688785998,1.8830136014739298,1.3833328236713953,-0.999938299406038,-0.3912155450349987,2.855861928789874,-2.2945722720026867,-0.5378946887653401,1.4592352948525347,-2.8491242892512476,1.5173560846563536,0.4880638986079216,-0.06450861732336702,-0.5406443400906505,1.5239108011930167,0.5635180658949117,1.1030854716522553,-0.8393595860219044,-3.0289526815714045,-0.029279117534430382,3.5361231492049274,-1.0847215037339581,-3.342128136554969,-0.7183379187520126,-0.4323829787973058,1.7362932469937151,1.5987294306715811,-1.733302314948299,-0.3689558703614987,-1.4399064629045675,1.4584002983774798,2.783148530741926,1.0896106427766157,3.3140674840148923,1.582131167696014,0.8342664516167894,0.535435932348776,1.6990405014415075,2.666469380283385,1.5247192399661014,1.2868870179966079,1.617004138358362,1.5893435272284402,2.7476586266420786,2.591602082026754,-1.170206223817969,2.863529215026597,0.13863561657859738,1.8646356339776187,0.14183873947230724,2.4938772003818155,2.9063701217278637,-1.1908021570436382,0.2653492258078555,0.7154848033986374,-2.233235426599459,0.9259187483436913,0.921650774795907,-1.6348280723502213,-0.49918692794598607,0.5739482974716511,2.2684446010269923,-2.18314037378633,-0.26538174715699336,2.0838966737417426,-3.695960920336405,3.4493872320767256,-2.7009505354656165,2.751643735412531,-0.09553148118859679,-0.8750790358022076,-2.6204466999689457,-0.19085674306680234,2.8211243251182294,-2.090307556394665,0.9321407790604128,0.8536581260954641,1.9342870039526452,-0.43391784605900535,-1.5063657543840936,-0.1634960548458713,1.692862780945012,1.3020296512775866,0.14739876497640164,-1.6323047249230924,2.733873603500054,-1.6687892739252985,-2.511792008990384,2.6423831456443456,0.6691395781267365,-1.4649190508491285,0.9862998132647596,1.5837315409519648,-1.332886276226067,-1.2061136308357818,0.29668426203973786,-1.9727510206066687,-0.29286871893634603,-1.443389795668135,-2.8363161387638773,-3.012687110094033,-0.3435079270694623,1.2172539758074703,-2.421580304785184,1.432029071931937,-2.124406365216976,-0.475115321165468,2.860161736390532,-1.0181946779190172,1.3689342573623433,-2.5548916558607764,-2.3877394424939804,-2.745249063618553,3.3492406336553873,-2.1305702173023238,3.1333837695963767,1.3871262244608666,1.6806573587147433,-0.005564682370574861,-0.004965071895925896,1.040359747202806,1.6390884394073304,0.8950793538778867,-0.14954703201729158,2.3396356436790113,-1.0793885872876208,-2.403753426747579,-0.954367401239964,-1.8171398488152004,0.7910388825272319,-1.9990016663390198,1.910193517812818,-0.539130116845912,1.0069598514903886,-1.5371055402767122,1.5787525567111365,1.097996975369472,1.0375852340662108,3.5102686798293585,-0.3219312897961916,-1.0397879949656668,0.9567962934042819,-0.7694543398363972,-0.7237951210868506,-0.5105621228900301,0.6835859984746813,-3.3930334966605056,-0.07717762023286859,-1.433403810480987,-0.35831787766531714,0.39116668981975494,-3.302577859180986,-2.164535892831425,0.1288737665468331,0.7037646541286261,1.9083343200569751,0.5411659817937006,-0.32160495557160446,-2.953507887767663,0.24826552592012824,2.293621979768724,0.8002000412466981,1.381607115895452,-0.4826161022554693,-0.49871877483736315,0.6597082090254061,0.6105443173968794,2.8220373963883034,0.3495036281234393,1.256948549342223,0.6708862776973161,-0.22599306427419563,-0.824555017206869,-0.1892120774826912,-0.41442085369718984,-0.49160368734844345,0.9047030718963112,-0.19733835004730646,0.9451625129185681,-0.4352950379384451,-1.7919080015736017,-1.5635346867306283,-1.9236358928115929,-2.1854022370558073,-1.0618424179403516,1.108138618590863,3.0756846946777574,-1.9869050709586034,-0.13745725404934866,2.086586080871407,-1.4032576110756936,-0.48042184818348177,2.103219883720222,3.3175633784436798,0.3774383662598104,-0.9104835498636098,-2.56393223848957,-1.325355734750079,-0.19852066682574684,2.6185084699817525,1.827443827394127,-2.4955692784226766,1.8699821409138053,0.4237849842591296,-1.4462143159475387,1.941267259413372,-1.6748003954376036,-1.9598208167839015,1.631331975458107,-0.3707174809955149,-0.6798405927314481,-0.2603384301865978,-2.5067610995907086,-0.09430600147167467,2.264587478620891,-0.1302882232340541,-1.8988140635778312,-1.9378989138792668,-1.1556082686210136,0.09335902114966957,1.3431519209059446,1.419642231501066,-2.214458182029714,-1.869268626601984,0.7252371441600425,-3.8907392422752918,0.5489137359362524,0.2351374631228053,2.781367819528447,-1.09201120744775,-2.789576421855915,-1.3614647918180194,1.0922549513017368,-1.1716256171989887,1.2468292431131265,-2.934391722658089,0.6179513382796844,-1.8712000715165609,-1.7233596741453867,-0.15083993454515496,-1.4262144865819804,1.606972188571024,-0.007623731069031265,1.6659746140929694,-0.3533413354824356,-1.9353890896285306,0.2726045656526281,2.609024151062214,0.8181097984235354,1.1497701949203196,1.0888856642459084,-1.150008036265747,-2.9321866219972486,1.7464316022504003,3.909219690640858,0.8561663965681746,1.0384959723329301,-1.2103234536122534,0.8452848737317431,-2.2105641350062477,-0.9827180337306977,-0.13186958254632605,0.5900394944663662,-0.8830332138390988,1.1159951321707196,2.017575894019091,-0.39579488185248374,-2.6534294140505086,0.8856534267296254,0.6278380740144573,1.895475197833141,0.24518184784224464,-2.3083383218233546,1.5252495273864939,0.8357934999010717,-1.6302495614123331,2.328854532812761,0.3215545294927113,-2.0727389208640115,-1.1873415342825666,-1.0367820258781988,-3.021167999400164,1.37418816745147,-1.0621949643320234,2.3257545029607196,-2.5894930040462154,-0.10660529470995862,-1.1339371677469885,0.8054295530200626,1.7840793974208482,3.160345684551598,-1.51098384790367,2.7307848207178456,-2.729665813082601,1.8657932557641668,1.6742920770177736,-1.6243523342903956,0.0296845674665872,0.90092593544785,-0.184278163325251,1.3661508940570382,2.632447444580469,0.8472446747938347,-1.0136200431127225,-0.32674422648890533,0.7280744579690028,1.4100601808842053,-1.784235134911898,-3.5493012837950086,0.4788163521065944,0.3799396577795528,2.845492536917789,2.6131943935105557,3.2050918857530295,1.6987421032015804,2.8635826592422475,2.532374811490591,-0.9832723284148722,-2.724690840538007,0.897604196545459,-1.3880265744177587,4.369175111068052,1.8619684355455635,-0.3633166340787935,2.2993160282096756,-0.2869442576917077,-2.8826843293443316,-1.4449389544698263,0.4072958957243528,1.824208169512476,1.0119038180070625,2.2667825874779863,-2.953699927412656,1.5361667574615625,-0.5743179258926051,1.4117174173830809,1.918474263717965,2.178219310851891,-1.7659318762256528,-0.25123727601327117,-0.5802819500208842,1.91010162641251,1.8649342449271205,1.9408254987695126,0.3144397501985936,-0.6626636913367402,0.023134648985559746,-1.3902636256263061,-0.7456791258943004,3.1174134732011005,0.3441799158227588,2.9696850550550495,-1.0385220323801965,0.9280665833868263,1.4382528113250888,2.2385465557869697,1.4791527564783093,-2.833431038305082,2.723602072188198,0.9713216748871437,1.0755704662445,-1.2026439669633247,-1.4443207572444023,-2.3606039246471324,-0.4053781553680136,-1.3951456076451865,0.39973853832897244,0.4773425431912195,3.4026070090189933,1.3098160681517264,1.9401929861654807,-2.9608060598614645,1.3811907068496636,2.235698371257102,0.28489585327577954,0.371441947795946,-0.05580776087065483,0.3970262645564381,-0.1714433141275157,-1.3500037273613177,-1.9461144812430136,4.111693005886627,-0.7350554700168875,-1.8565316419474314,2.447856640976947,-2.3691973894215574,1.2465403765160152,-0.9699695924857444,1.4605207671969618,-1.153488305357548,-3.728776120163675,-0.7523461133139062,1.3692593411165828,-0.6576765290705188,0.2966608317452893,0.8017853720438332,-1.1985807008107505,-1.761236749887781,-1.826425300892303,1.3615268369041902,-0.0839723391073103,-2.4517819623591564,0.49441720536026995,1.8887488030306006,-2.1982658242875717,1.1524955071366083,0.19067850379026238,1.5886508883315422,2.150118220526201,2.2659162133819164,-1.7056983716103855,0.06923950935977319,1.1063926351331101,-0.1602641156899429,2.3660654635100435,0.8812722632166445,-0.5586637792585811,0.43236858719161886,1.1110918945719626,-1.3024134558764933,-1.9894846831165167,-1.0582514207950466,-0.4122386466342558,-2.5585542890112833,1.0715409919115184,-2.5440435445728706,0.2729738410945934,-0.512297119647484,0.8329123881704085,-2.52173882935174,-0.7620121701423502,1.5213130727726725,-3.7555025668308555,0.2639981546184679,1.8728651176890438,0.17419368699142912,2.290297467896921,1.3408963459918537,-1.7832082863643526,-1.2240456522555225,-2.5751366307975614,0.28489291846741305,3.4077898752820435,-1.7579314895275107,-2.9031931592034015,0.6143551618603752,0.24554777598490146,-0.5834981361297429,2.1162747775055943,-2.293331431644396,0.9008476961840731,-0.5255334203826977,-0.43198235876650976,-2.5007023601917973,-1.6367173596506643,-0.34386825817186756,-0.679839436324244,1.7991423592758433,2.8187010947278166,2.6860568817908117,-1.5116841464436939,-0.8445329098546767,-0.9306437808565613,1.0691791596067761,-2.853690834491739,-1.89005485187894,-1.295965572215501,-1.2996411126487069,-1.101954921094931,0.8201444095225625,0.4417266525323558,-0.7251983009721689,1.9619268662265046,-2.17581699458219,1.719655515715859,-0.2665913021781864,-3.1860852046810835,1.0173048170298051,-1.043431376121055,-0.6930946856811752,-3.72589141382737,-3.822431661804861,-0.2727588394511974,0.9913723706088411,-1.470837092471927,-3.3894834071152276,1.9430989968352192,-0.6061038296358744,1.3095163062512658,3.7180201442472534,-1.3900865297522504,0.8306387355195124,-1.130441782974374,0.1860275841249007,-1.46461925629196,-1.8011993715213568,-0.15061899791830857,1.0833674175788714,3.56510534867569,-2.295188706209014,-2.0368919372796235,-0.33871361379647624,-3.805208052919914,0.612843263529602,0.6214888798965107,0.6972495216130031,-1.73859296355499,1.377983079041898,-0.009427498867643463,-2.241843409792005,0.2561554755076899,-1.9470590053261867,-1.842134012673321,0.1789040627239895,-1.0071078179398685,-0.5563043561906316,1.348104721198243,0.12393106900011075,-2.0232046765380445,3.283803085371237,-2.721719704859503,0.495128703290432,-3.0030777425985287,-2.3049456910059405,0.7773346819436117,-3.9416357877878516,1.0125153983230808,-4.471737131142381,1.0005147220413115,-1.4925193547491047,-0.8865212344736034,-0.23269994853952852,0.8880201679324924,2.0356820555411566,-1.2362416871032589,2.3209249958912817,2.7755339103958443,-1.4504246311237028,-2.1054281341662406,-0.20724779882331337,-0.5546309344463299,-2.993231607769677,-0.2595545637006462,1.3442159348264198,-1.1973806447466182,-2.9329558909280764,0.8815228193812501,-0.7773716988425229,-0.5349235294630394,2.6117137036847717,0.1885836941552718,-3.384877593038001,0.9775024925429889,0.12676782346490467,3.41449135265218,0.534208736349491,2.0136380691655784,1.9660505009975597,-2.3480549897936753,0.16336712937202616,-0.5982988302612476,2.2691902644361455,0.18539112070607844,-1.6315123053898415,1.1948604826323426,0.8553669292243418,0.5592263156075672,1.1970903012061869,-2.0557834431289272,0.8684591084395499,2.211719472583358,2.838144365258183,0.8344758428483202,-1.7828014830760865,2.094053020373092,-0.2276697114248621,1.0279724881977048,2.8373541548854297,-1.2601744326347464,0.5918994752531348,1.9432736562961177,2.4682492695253018,1.3748025231129595,0.5686772420465904,1.147863090357311,0.042384407110272596,-0.6498881405097375,-0.9686047631380392,1.2558570901683814,0.45000702453372216,-1.7135152132389893,0.09722600488010602,0.0660474928072287,-1.336078322971761,0.7579681838280232,2.536372694045154,-0.3527533137144736,0.02129941131664318,0.7508652111178137,-1.1616435514624366,2.794189417254497,-1.0647135686424862,1.3239288171011852,-1.4508878412753017,0.40276822400919043,-2.1142800497073475,2.8130804168878196,1.9603616445964611,-1.549917063620163,-2.45689092422575,-2.0420151795724113,2.5209403541492232,-0.013758539529640818,-3.4846641502577063,1.4897760048426982,-0.8744965027562152,1.647216708984363,0.35371155587714015,0.2774629269701508,-0.661443736300879,0.5201518834785466,-1.6448001060988229,-2.058739323481129,0.49225921816011764,-0.053047577542485985,1.3864940884284804,-0.28323144028469477,-0.49928441137471163,-3.2017285567730065,-2.28426974658436,-1.2041241085577126,1.4341338610188363,-0.6370102757469484,-2.64359958671377,2.694689432994464,0.460027649271089,0.07313724544439423,-1.1625583582115888,-0.4762777094668353,0.37451376636097805,0.2551631887016305,-0.2552284934803337,-0.5041569649599729,1.333846788861875,0.9848611573609072,0.4994547282090849,1.6863504775308413,-1.438804014825226,-1.8165703526877541,1.797234196699206,1.9135404570911974,0.632417905089796,-0.8416753497582639,1.4135986297173964,0.6253254767941585,-2.6525619480940583,0.3208041854093053,-0.056446632642587216,-0.4018823615110965,-1.0681861942361497,-0.08183048063879236,-3.4802626649356085,-1.9983912060864508,2.0152271117793688,0.8848624711047874,2.2310986593655335,-1.569246037312225,-1.3467882023765425,-1.5450470011758455,0.9793384745631754,2.310743957678521,1.5311429855617384,-3.2097622379331066,-0.7040609036786816,-1.0859474266872706,3.2393780808856114,2.0196125530884004,1.909750510659973,-1.9832059976105187,-3.1418480750877285,-0.5826879270962004,1.1443526539858337,0.15108152371183312,-3.1714396748048554,-4.28053951451554,1.9669702426033093,0.02111932580693803,0.5061438234046266,0.295317825732241,2.782547045296823,3.228693480720122,3.4162084813068936,-0.15604698589285476,0.7939993572139917,2.927052571950026,-0.8508689161505127,-1.405408030339924,0.1865827239930425,1.2600544696772393,-1.6468124139450442,0.3876035472763631,0.5509356559444555,-1.324331468721542,-0.8918655688174222,0.5441326462976193,0.35779844671219935,-1.0837951217327166,-0.5413777719671456,-2.7703489627201074,0.1962792431416274,-0.3527894610922646,-0.5835279017298239,-1.7516414100078177,-1.5730494653659108,-0.2633486810269925,-1.5804149498447095,2.4663900615054963,-0.36087505677032533,-1.9913305207621523,-3.378113311534879,1.526156800771753,-0.7395817645200341,2.138078131159412,-1.9673100872592117,-0.3665782159057415,-1.0750713703820587,2.864369045222351,0.588169675971538,-0.19287733814574667,-0.5591893730619555,-0.9985814460060903,-3.4435987290719905,-0.4117547862786187,1.052224902939603,-1.3853562585040773,-0.7894140169196256,-0.6709846509144122,-0.6258482324034723,-1.3959811837360339,0.9286045610163589,-0.751435816652144,-2.608090738110813,3.3816342535241404,1.4164957294123572,-1.084425610500382,1.8709263625797008,-2.3273526344490483,-1.0849731239971971,-0.495742235345724,1.304862549874622,-2.267104864811219,-1.372392633493443,-1.2613359843478915,3.017812520824438,1.892085807203374,-1.5302169527542375,0.5948495761691703,-4.256002068000497,-2.5767501537631254,3.5761577255792543,-2.3837644845299066,1.2632648861057723,-2.8039162588047684,0.9913341629292,1.9087931911335059,-1.2140208698883663,-1.8068250975924522,-2.5888131275396935,2.389871629220283,2.9982331178758304,1.078170649677542,-1.6410891178387066,-0.25570802204162446,1.8382252591929382,-0.8812952675824491,-0.07000454468527856,0.03807542687681519,-2.688244495699747,0.5142935914822615,-1.8381933847318275,0.44199427086722143,0.06540089073809877,-0.23219496851925298,0.6142436139347525,2.4189416094315823,-1.1668427261152812,3.6898999521345477,-2.462965706148877,-1.3797458367636044,1.2284761099597201,-4.463021649662168,1.2647764004629354,0.7700511355450411,2.5026102146003932,-0.11805299079037375,0.5181746264495016,-1.4648844795263833,1.2275143332637755,-1.3631537485008427,0.2523424904967823,2.998429287389884,-2.3694934611761433,2.333215504405895,-2.1465316222599578,-3.5768453898806847,-1.4952229635099186,-0.18878960840384348,0.708594638166539,0.7022124148769578,3.8220235068378416,-2.3026332821153415,-1.7216010674428708,-0.11715615051375074,-2.313454609599053,-1.3619711877699885,-1.2720845757489856,1.5439186092204162,-0.33044467945488476,-0.7100165723189348,0.9789553525049637,1.266895336848173,-1.9613943400951914,-2.6762959023907484,-0.7465852555921169,-1.5456929548546516,1.306133660843644,2.1568408753730153,3.1648619765799273,2.2013447417239393,-1.3094719968468835,2.4287269887191556,0.42464954673083366,-0.37798313265923333,2.530140962664623,-0.8391079799892058,-1.888086124547254,2.8453564719306157,0.05694946524252176,1.3770111004705525,0.728103771120209,1.573165962109515,-1.8990794521210048,-1.0982221508649286,-1.3235870429854182,-1.3671873091830475,0.15343725685691872,-0.02326763804856309,-0.5679012585742588,0.6869941862300551,1.7220130511545475,1.0055687301058225,-1.5284404317284255,1.479206731073914,4.686802692873317,-0.14695910289985098,0.6575652841208215,0.22129758295715862,-2.377120271062298,-0.6877798663229638,-0.5589365563593007,-2.3611231571074898,-0.055461563479245406,0.4262328342672086,2.261304982846706,-1.0427545145067958,-0.046161687196848065,0.8863412609549097,-0.7266894809588192,-2.017115103662374,-1.9010473786892834,-3.004936719022416,1.2394216563642413,-2.388471056354157,1.078658883583805,-1.1640845915263782,1.2191208085441552,0.7856875769157254,0.42434710334182135,2.2379480890455246,-1.6907651022525387,-0.36926888360968163,-0.5885840405283743,-1.344484248102888,-2.4205361230029956,2.5219701974567252,0.7066281604905771,0.7910717362364642,-1.9700970872245611,1.0462934945654458,0.7321526257876064,1.7165768522038047,0.02500078674303581,-1.1842214031724612,1.7165286602921324,1.5636125898310325,-0.12522654561341573,-2.6294650768257837,1.0183962667634479,-0.9778315450976535,-0.23658808986697258,-0.7374589353979083,3.2371864734532463,0.029163420613789286,1.0964797750495197,-0.03787839396723222,1.9982753037399952,-1.5392815266365127,-1.2371259094139182,-1.9637912611715882,2.732083121180172,-0.2739963022618352,1.2663530047801568,2.34381917423563,-2.371952995430702,0.4586756612457168,0.11735825991329894,0.6538899164187653,-1.873122513286922,2.1023950195400976,0.5787806474229625,-1.6593514705814185,-1.384053292134964,-2.5111592302163843,-0.0018139960699041274,-0.49357863747998676,-1.6891054028380654,2.8572093872159794,-1.5601766883258563,-0.33274157360833756,0.9802231523349711,1.0512911908695453,-2.1564102174462874,-1.912823541466377,-0.7357579453670909,1.4734832507179205,1.2006536236650354,-1.6288936369904714,-2.3505592977806704,2.7005508140994827,3.09846310352617,-0.5933779111750115,0.1981204739556505,0.7901544637410594,1.5714053677852258,-0.8429888315202803,-2.7240354069561987,-1.3755247514454416,-2.5295767704885668,-3.583744593996629,2.200633358373539,4.133923753154716,-0.4842196836072937,1.3386954365919117,0.4410721324478745,2.946285950280146,2.5007193235439735,-0.4591816334908713,2.1753841923441177,1.2509073444760128,-0.6884062096160125,0.12288593720887761,1.2808907236545908,-0.6491643196860807,0.2623923226561092,-1.3026013351771706,3.4933436695379463,1.3540282168705091,2.1856298008255313,0.8843229795568142,-0.9293456044636728,-1.0810723074930848,1.635934402715277,0.6208563030958975,3.5156087508094327,3.0877617728118025,-2.843522167826045,2.726809131494031,-0.4222917677777381,1.8924018445799031,3.2302641686128166,0.5101249194517827,-2.866060559535674,-2.5249845713834183,1.3619344771346635,1.7775080648377162,0.25367538649689286,2.3214590245804665,0.6795145052959347,0.5017342158159591,0.23648103447779978,-1.6643700918654594,-1.567399177955894,1.905035799439308,0.009824263540124844,0.7667869230567217,-0.2023901931901428,-1.3690827186822516,-0.5948882937507468,-2.6037039333118903,-0.4292756146954686,-3.6314768490608347,0.021760810476043266,2.3516992432069874,-0.8933680527506574,0.8149247154008686,0.023182155104509744,-2.429701387734847,-0.7987346093108281,-0.8724971677548367,-0.4794119358706261,-2.696331437010548,-2.6291805956998435,-3.4648027614150414,0.8206457770608879,0.34390609681649903,1.5993751334434074,2.797991478048527,-2.0680421682573327,1.2420132929025793,1.5675853317136244,0.9110637123135515,-0.7540163788454171,2.2112688029241503,-0.3194725731798313,-2.7341866172662392,0.5559474272929095,-0.11832368174969272,0.9713734628286682,1.5113698134196907,-2.3684517541991212,-1.0814200182898568,-0.9012317792313412,-0.722100772005347,-2.185628421336354,-0.5281997930578544,-0.19326443802322268,-1.59841403618971,-0.3492673362484297,0.5134045987119342,-1.8048946173386469,-3.2613010150855333,-1.4257463077728996,2.0413110912647476,-0.2042757237955979,-0.1684740020969419,1.9145040682129149,0.26131232458988546,1.1344840442067259,1.6252642549551501,-0.2819683802770104,1.8771958641007251,0.0832504510531531,-1.1339802700607555,-0.9857097308000786,-0.8598992750231896,-0.16952576965282537,0.8972960094387611,2.7911699278748894,-0.7631800281243678,2.5024057177192747,2.4959438648447994,-1.2329190138130373,-0.5268862103609232,-0.8056715993384024,-0.2041115595175009,-0.17257661968760732,0.4387039324112051,-0.834201208760312,1.1493617100602922,-1.2965612757046483,-1.1676991618419745,-0.14839210685842366,2.3330807398384223,2.6533298785046826,-2.7368807380412017,0.5060878912179418,-2.411898687539793,-0.27044751576096826,-1.2010606740176353,-0.062107681253592756,-0.32507975355430124,0.956902153004581,-2.869781928857182,2.8836597350232926,-0.7815880340963209,0.848833900559547,2.0196447094196635,-2.713438125947006,-0.7028282586679165,-0.7006944538067466,-0.5882163431973177,1.4234406193140836,-0.5335306237734386,-0.9319186882971412,-3.2949549491059904,-0.48606206372357785,1.078357899296485,0.8132776956505336,-2.6292739892993504,-0.7843840015848937,-0.2816383524947788,-0.7632655198101674,2.88374893706432,-1.2284739710695363,-0.4164530423271025,-2.6220468428659593,1.620459378858812,1.5769036984825173,-2.4599732290641008,-1.8199126272056485,0.4163931741000264,-1.098120337028537,0.8155949110890647,1.2249529187480608,2.07986290027186,1.4533980919214422,0.38958114946748257,1.0759142649361515,-0.6244987755025025,-1.25209500695348,-0.0805432931390394,-3.1546568605285863,-4.256478510102632,1.1777492127597347,-0.5114210713552247,-0.6400685842947774,0.30903971967631105,1.208263240066741,-2.323523163956151,2.31553918825954,1.2832479333357825,2.0070839438178303,-1.3263501904425699,0.8849150593506123,-0.8305573492821765,-2.663474686740527,-2.546655707465321,-0.2540842580199457,1.6985570083846722,-3.399441008203573,-0.11701680036064283,2.197701514189342,4.729321335126188,1.9638013079663108,1.187894765946446,0.6466831580008163,1.54279260025026,3.119712612722695,2.0359781876233765,0.4959581996639003,0.8231748844678075,-2.413777869343443,-0.4889442381619498,-0.7406729639845645,-2.4217311115025324,0.019661761234655795,-1.952162692752981,3.9834942003309037,0.40540975596212203,-0.7462049297524677,0.7802090536322889,-1.9767205598314845,-0.0674167259361793,1.5410576810952057,1.9925198162596691,-2.256328487514171,1.1845580786788894,1.1182930934250142,1.5344375840211808,2.3446198260705966,1.8828019635255118,-0.9845398841243902,-2.1536811668590468,2.1529755313499157,0.2354574005295116,-0.22866609476167923,0.565676779079419,-0.33116125766453425,0.5035532894155726,-3.1014564316639515,1.016515838423656,1.3012659573741319,-2.438226134777331,0.9989396637273196,3.1204896881925217,2.2474767332846786,1.3638095163421047,-2.2487553219180243,-0.960792851215789,-0.7599841364590659,-0.5806279844559706,2.1588935034680508,-1.8755952898295158,-1.8114302724898501,-0.3827598795891517,3.081244901812963,-1.2260866686228846,2.9258724504175118,3.3245556488787398,-0.4538476029754218,1.714863428796928,-1.235284007052248,-1.4454478454811097,-0.6233660613950169,-1.4165444796084368,-0.7183860298104086,0.5436145367075864,-1.7175852624620467,0.3918082543367537,-1.5862371392435988,-2.6361116395371544,-1.9222301620090219,-2.4404199486616704,-0.3988359400069927,0.1539355133923516,1.6565560586777879,-2.1200623239885856,-2.338816640987286,0.26754356821570013,-2.31776883424122,0.3343429068307633,2.754056978991336,0.18972316251377574,-0.1815640655116749,0.7618887969652753,-0.05847143271259384,0.7789525788054817,-1.0595734062170954,1.8906717585312673,0.2083431301504849,0.2879376060182663,1.9921716829487761,-1.7033447360269212,-1.3177954318281255,1.9742288461670445,1.8265765738774509,-3.1157532522079117,-1.1015877240306133,-1.5075054086945716,-2.092704387803742,-2.4761436688107614,-0.8702200032365838,1.2985298784964678,1.1686788261089676,-3.41525650227434,-1.3289481745375038,0.9257384207600405,0.5744967505138204,-0.2734919557860939,0.2092509634709982,4.2429126717333,-1.9371299187898072,-1.8583462197188574,-1.0573368566979906,-1.3350707356930798,2.318658388440813,-3.0241355039075533,0.19169495914768037,1.219118141829403,2.6063350961670837,-0.9326416558443575,0.8766915634918384,0.23840407766027416,-1.043864519455615,1.6367316923232993,-1.1725044610698954,-2.457821295848944,1.1969408750925885,0.7905564244422566,-0.4282781165653683,2.0367239485463444,-0.12922460449745324,0.38891765942764384,-3.2133923798569226,1.6907197686402504,-0.581404505283115,1.4838665696433742,-2.753394797703178,-1.624286736038789,1.491424823822885,-0.2792447415671907,0.34081726482198865,-0.7719731224315876,-1.5531857253265384,1.063273449125004,-0.5769798640046268,-1.260159109008009,1.1989485951286305,-2.8073419351443016,0.043016514079981215,-0.9499947345009033,0.52291697631842,0.8137920270653598,2.5269077306353416,-0.5949487063552976,2.7620291186324226,-1.905663769753922,-1.0741098003366265,1.5424073246676766,-1.7154666616363223,0.8094570082799535,-1.671598061650018,0.8714552276868053,-1.1396451609402036,1.3836808822458606,-2.5017661212409803,-1.3915919597846493,-2.053973660250338,-0.2649506370800901,1.7252446966492225,2.830275688283168,3.058014348783587,-2.6544631075484175,-1.385123317010093,-2.7423160490736103,-2.396793928484949,-0.5203670300928978,1.9744554623463184,3.2966971399274367,2.8094138481268374,-1.9825646709282851,0.2389726182612094,-0.373106610780419,-2.922398542307851,0.9357839806643722,-0.38966788962734267,2.52162062263469,-0.4783322310379767,-3.490803695413618,1.864832889565618,-0.5183806093328859,1.7308133001163428,-1.4179988893562239,0.34659792776863996,-0.5186123469299697,3.020156109516337,-0.5587812733323658,2.440943800561498,0.6933248487887108,2.67756640387307,-2.4667535482903813,2.0885269094625953,1.5339799826264193,-2.7841446093136364,1.6864083585749727,0.1778440813183154,0.8862047007923706,-1.3166670805871437,-2.0610668413484663,0.3743288085943818,1.2033871146068664,1.1165508966667397,3.2941493692437644,-2.063168762340394,1.1635310152466154,-2.206882363591943,0.6982367836315012,-2.635194973591374,-0.17768548663959186,-0.8086355973919381,3.1086663377439065,-1.9335416284684894,0.3432870448194846,-1.339806972693214,-1.0119910151124005,-1.4034225931797304,2.825002053636759,-2.5650412051078546,1.5454052456025045,0.5148509869584018,0.05244944700733756,2.174775878115278,1.7043130172154966,1.1997920678473728,-3.440266454861454,1.644427874999641,0.6963619683806698,1.0962766739775855,1.9022404188357807,-0.44370677598279357,-2.8787345098189285,2.055733413813119,3.4991504060928302,1.8308571670020302,-1.690175998613433,0.3096847070086315,1.030480993184412,0.7479629688140571,0.6416308464258001,-1.1073318329729611,0.4941372657263769,1.9453387524359975,0.3581525041119349,1.1947830400336539,-0.6430370358907297,-1.9024379706795163,-1.3625570913946303,-2.17982134424415,-0.20523332163675323,1.2447907628656587,1.8503991714413177,-1.3762050981883232,-0.908832108438692,2.4135900645092554,2.560781736356495,0.04362682746298666,2.3583038352777317,1.454129035156393,-1.887618998299695,0.45117771883954705,0.34299390782435607,-3.115425871487166,-1.9737408548679898,2.878254856580494,1.747938911537295,1.1260207284870742,0.7609112033546303,3.239192118752658,0.5318499940951215,-2.3128369119136116,0.2288699487904155,-1.5401380481956835,-3.44062172121891,-2.4699756501763224,1.676708519830933,1.7236083671073164,2.263849474603364,-0.29722251985990444,3.145596344642657,1.3694633856061225,-1.7628578240028947,0.9652848198449132,-1.6861989670088524,-2.0596666008931472,2.1259596133443153,2.9539889178254723,-0.37109892829839763,3.8983610666747572,-1.7299479998907146,0.5478726708073817,1.435592456963602,0.7920477589758276,0.5589671630326126,-1.843090016802856,-3.3407352891324966,2.5213710733929218,1.9489198884255023,1.2308194850803478,-0.978809747360216,0.7767797400784145,-0.6484109858090484,1.671344888616601,0.32874477741980773,2.6868540675223893,0.27970384073523197,2.572246746217799,-2.762601717520479,1.181180446945973,-2.1082661242008593,-0.7456822132019006,-1.9412692087910928,-1.9311510406593935,-2.932592244543051,0.6150842624153714,-1.0539842554295364,-0.25667978005112313,-3.1858876984816433,-0.4181738527754368,1.523649367826735,-1.9429665978465431,2.4807351363289354,0.04863036328815503,-1.880953596921581,0.49586750331845125,-1.9939503632613071,2.399434015946311,-0.17226610532524209,0.10826559933157158,-0.17840414820621112,1.269523664877411,-0.9263329055351504,-0.8559101770517262,1.2255135537561135,0.5803178595148395,-1.1052470106840202,0.6648529002957739,-1.541976363216359,0.35604803126301726,-1.2495494653695431,0.27241283227222035,-0.3249552419111149,-2.072039597178068,-1.0866865470829825,-0.8764386697531994,1.178672938680857,-1.4937438876863127,-2.221245552340141,-1.2864169354841735,1.6863941239239042,1.1000156645927723,2.026133727842201,0.38794639626335164,-1.9246136823293014,-1.339910298643555,0.31129971605638096,-1.8255088617761581,1.2368072349699208,-1.796218365504175,-1.409341786920672,1.966440826415524,2.2237566085536282,2.520037577112174,0.9569613404859294,-1.1347517108279124,1.7156301986619407,-1.5079887641503031,0.46290631905046703,-1.3330421299623278,2.4479144165338482,-1.920936478885625,-1.7965203835461443,0.5130367708815537,2.958111060216606,2.4567951765223435,2.1171947659108734,-0.4581924086598386,1.2414513339507736,-1.0010962810070627,-2.891013339248919,2.24560293823387,-0.5327024521914009,-0.28206394288656156,-0.5011841373583619,-0.8160044245339813,3.4494118698741785,-0.6645968350281852,3.959965493883759,-1.3451867918273013,-1.2816304411473245,-0.5215891635637419,0.12300014478897267,-2.75776989899922,-0.20321823572659845,-3.347803772328214,-1.7096720304799502,0.10762441348643353,1.5580152676028298,0.22644528547567083,-1.6250530795864127,-1.3234961856718068,-2.65521147387012,-1.4595807466432722,-0.46890436684082903,-1.2330461529151167,-1.574547474509959,2.098661341428669,1.841881977673713,1.8907488777803267,-1.3339674810048987,3.4833560118116673,0.4885302404467607,-2.910441480405204,0.6208288900835034,-3.048771805639784,-0.7299599777141424,0.3573518820593028,-1.4073741245770526,-0.0009550160779545371,0.5230013216122007,-1.708643606360005,1.522369058798285,-0.2578790054004337,0.6266349156976277,-2.772386961105862,1.0102665467086385,0.8463970838228428,-2.4906581233345193,2.433803575849683,0.7933662459018656,2.455999368558595,-2.6816580711294975,1.5960827896266474,0.14834366269482732,-1.502658010790351,-1.5242475992351456,2.0116345616486586,0.5180566993531497,2.177590333177879,0.843209118152337,0.5633697479671426,0.30082844415102244,-2.0338722635443194,-0.8373282982647449,1.5496278875882126,-0.7808598806898457,2.31149407750323,2.630337233144237,-2.933497311526508,0.48575214008439555,-1.2273607783614988,0.3555499223542057,3.2701742091968624,1.087815973115816,2.0566630648451123,1.506043557888735,-1.0406728068617863,-0.9372587010096696,3.072416798533133,1.8530945063675588,0.8744616493744831,-0.9502540933823643,0.9883526655963694,2.5652948955084893,-2.5225261039226092,-0.770969449237493,-0.34694270302797486,-2.259334239295649,-1.7831022023195437,2.9447021469968404,-0.4727229161566029,0.8509596210269673,-3.14556362912175,0.9093365577915945,-0.9249587782337745,1.858485674775152,2.1104376387717,-1.5008804003863567,1.835517257374381,-0.42517696572545605,-0.6322065250145656,-2.1196771038437237,-0.916306590621793,-0.05027462876159486,-0.6569566266094592,1.16243279774501,2.063191879103014,-1.009286809074011,-0.07240520866745107,-1.9282112993115348,-0.7494412943927533,-0.7734720530644097,0.12378508943705262,1.329948873467788,-1.4592120889498374,-0.24581728547039708,-0.4994524092743943,3.1514489326659056,0.9204858394245474,2.1355207581373783,0.0540523790759353,1.1678919158427683,-1.1317165169309376,2.800365314592176,-1.0094340913766784,1.0392291820845228,-1.5997975967951004,-0.2913474421142947,0.5530530396534836,0.9302509977843493,-0.8048557424248369,0.7415863612751397,-1.0433126758460471,1.5046144995258335,-0.11770327039600768,-0.06925902260332094,-0.5368261198690131,-0.792351363680022,-1.2986688445749581,0.5090057684350526,-1.4856047566951105,1.258077430792956,0.7124894644655801,0.5780866386913401,0.571938010865606,0.010703472965512805,-0.8914201164424869,1.6931869856809028,-1.1379023033882862,-2.931851661968517,-1.8849869584358887,-0.93783516227717,-1.4868991231947317,-2.8481480882107153,-2.354749183606872,0.16188676797859122,0.9874029642447313,1.2102104164608336,-1.7084557628271928,-0.8032633397674068,-0.7084187721940073,1.834672507372823,0.5318344135681609,-0.22956044987051027,-3.338626199235391,1.6497682336136341,-2.3815342594879763,-0.13672674470183718,-0.15887167798993135,0.9981574541735001,1.7446544367166132,0.07734535138555267,-0.1440743984740094,2.7540093983637304,0.7355246741168179,-1.7252752588516274,-1.3308695692201593,-0.6607671885870043,-0.21867071481090877,0.9913008088863137,1.6231513099582817,0.44949185116869717,0.8484203659378258,-1.6976440525963843,-3.2205333062373316,-1.7201324829225824,1.3468589846075303,-2.977419338133038,1.082382724498659,1.715421450280853,1.5289086403798577,-1.0358098276864676,2.548779916775603,2.9819449855925724,-0.6186446145198999,-0.9255577732525807,1.7657559571801038,-3.079443401644236,-1.1344332997536144,-0.7199685600911669,1.6345084053627514,-0.030026890405602503,1.0523746639469576,-1.0531174694259877,0.13952372025465204,-1.0828871160307618,-1.8190833349389797,0.4194245475318824,-1.9937032586496428,-0.8430106331451889,-0.2140929132749643,0.9975825839511362,-0.6755227575667415,0.9000118656902587,2.0479780507899727,-1.8140785638487864,-2.582587531333173,-1.5166664089872472,1.9540057743671415,2.1719804372243003,-2.0488685622435514,0.6375452243664367,-0.2816864441007934,-0.8902911033394305,1.645678241521392,-1.8536024825140336,0.2646678175850949,2.278405573038696,-1.9228021362519452,0.43774065013074953,1.109142247623854,-0.3046564180015286,-1.327126597184515,0.4224170980728013,1.5157462584900647,0.10139893380154834,-1.4866155237336531,0.8314899038878852,-1.7944862117429108,2.7769373325966336,1.254725596537098,-1.9138096182177051,-1.1487125291350737,0.7565277740322426,-0.6236561341945934,3.71763445543446,0.8451188208133411,-0.5406748840356581,-0.8259025084461484,-1.5083686142523571,1.8793681547861207,-2.6870878952692316,-1.1861206832230622,-2.0217926814454454,-0.873678736900357,-1.0553113713239195,3.365238541049883,2.326719238124247,-1.3244706833696906,0.49381884619950045,-2.4577191636461504,0.8597858709213088,-0.09897398867050718,-3.276051593954492,0.3623904434939959,0.5704043465784241,-1.3110428904228864,1.5356645763181942,2.053449286030018,0.1174096071894223,-1.7015009915125077,-1.6691134183221434,3.3237245036212464,-1.355820963003711,-0.519563296230818,0.05729542903209886,0.10589412365159713,-0.6876830941338259,-1.0143048617600527,0.22694977810404407,2.161585136354247,-2.5654271676496885,0.08722236463826434,1.4540615338787626,-1.6534351227135706,3.0618405514637095,2.1930505872355184,1.3138062870867528,-1.1616939971851643,-0.5391233026469261,-0.16325891813770377,1.1311302054807817,0.7396314230474007,-1.0882290643817798,-3.4986744795116844,0.7438813629778402,-1.5078218984213891,1.441170853601724,1.851726198500165,2.813033371897088,-2.511335020185272,0.003987328744492012,-1.4568758788842002,-0.9015677309838889,0.22309512183673794,-1.043052587910484,0.6743981404444804,-1.0969861976602715,3.289558678666521,1.8323825881385585,-0.02584624804757913,1.2707888282854007,-1.1742784829504953,-0.7673463367299802,-1.688958620124512,0.32174245208071045,-2.995071459814403,2.558087739963328,1.2213143408645584,0.6683342911374335,-1.962392263981321,3.0783427450416747,3.2249735763749836,-2.421276090183918,-0.9024655747859383,-1.7438021561540042,-1.2765971949335613,1.3853815940332375,0.1427979824360882,3.0015354176577347,-3.1523671236935042,1.17512897895608,-2.7162965092264164,2.210335401404958,0.8213128605968447,1.210331052991462,1.1854650170490788,0.8097585868262722,1.3921519972823935,-2.6750521101963556,-1.0551427492416656,1.1327871627388877,-0.36058833978978305,-0.9912781147526317,3.0046555614253267,2.8010024768261776,-0.9601307588536423,-0.5606774450311849,1.9993480396923404,-0.9534562247814299,2.953749937107427,-1.3272869607841349,0.5840435976240241,0.340804824913534,1.6312096249331138,1.4092311840050467,-0.2648733739476549,-0.6532008855195613,1.2594917515091708,3.4270053968530676,0.651077538781646,0.49271431172800134,-2.0293051354036375,1.095238206216526,-2.0519641000671673,1.179348585959626,-0.21153929519626194,0.09701618655060981,-2.1835420656996227,-0.23052781009976606,1.7772855082648327,3.602966575623028,-1.993979177784025,0.6280713840545192,-2.2312672511684783,3.485139624600452,-3.7477639433675205,-0.5735081176011413,1.3404059456598612,3.3793670568141545,-0.052823293711526435,-1.553640158560423,-2.2091478049263644,1.1938966481927522,-3.3638753610425205,0.42234085331450116,-0.871072449532117,3.2757412699033,0.9420564629607563,-1.3071756688834815,2.637464064566943,-1.204108598441788,-2.4388310406335743,-1.9514312872331776,-0.6644199673943266,-0.22556314712215791,-2.8261016692264307,2.922598442657681,1.4116805732570388,0.6452913096895564,-1.0291479034878213,0.3428867633874836,-3.877950666602677,-1.3068583213372436,-2.7192378658735357,2.5644586265321316,-0.5555260669733092,-0.6415729080909819,0.7371082440436522,0.3291748317097864,0.1804766894438442,-0.46612651313145764,1.8824074277299845,-3.206781841193755,0.8960538688810803,-0.7779894226235888,1.1036896244048966,-0.21667397059499793,-0.43341300671942523,2.826595016737446,0.009363395818685755,-1.5166914887343341,-4.086632130598143,3.0733588278791566,0.33487725802331914,-1.5831994294762837,-0.565676629313397,2.8936567580307426,2.9433285997453282,0.8504439418833828,2.104824924757502,-0.9452272658873307,1.3263080154241702,-3.382338637164555,0.0797855550724294,3.147864477529454,-0.3306083867313734,1.8380499057785233,-0.2601419722144065,-0.14456077028128972,0.03183639044351749,0.1670980409911011,0.6617482464521878,1.0055860038427173,-0.9031464478873396,-2.9464601174715335,-2.3229225044399193,1.3770429282816847,0.8086405258249425,-2.9827594215771467,1.591273953942158,3.1653021093706935,0.900405625654069,-1.477614711080964,-0.8503327141652856,0.14282288150028533,0.5561813451618558,-0.1432902925339198,4.0992940050408535,0.5132408481657933,0.03505153587750271,-3.149641034105858,-0.8752934085813825,-0.5034942398548727,1.0520120231771193,0.5135384551879808,0.1898307309662084,-1.4663900272588004,-1.2866137329281955,0.36058576304110845,1.2993495146535239,1.1733339359301223,-1.2144423676165612,1.5629072595325948,2.7206018908666287,1.3012318138761902,-2.3796697395358524,0.6895193062994811,-2.2696099202683846,1.6381479248895694,-2.2078965728982953,1.6430700131684024,0.21225098517334381,-0.33667695639792167,-0.021983431905838128,0.2838988197787908,2.519036156055608,-1.9804107337133035,0.3607584440972908,0.31024070106652435,0.8972342107029049,0.8986939014372715,1.798529171727009,-1.6999339559606705,0.6927450300053163,-1.0892083021606067,0.8926279777629206,0.9929504494269719,2.205323533565171,0.9501599659270942,2.446621564928641,-1.9699289065944159,0.6911356032337128,1.3587982767079012,-1.2982569666857526,-0.8544368770177215,2.7960258635312,1.5539628377869958,-3.22680841856517,0.5216167604679366,3.739557227224741,1.2656564555172958,2.7792976033634926,0.8137794967754338,0.32962540820834796,-0.10882654271987711,3.0096441115956165,-3.5458601358112767,-1.2847529913399436,1.0256574359382584,-1.9817633767884621,-0.5762918701722076,0.3271295970005209,-2.1615468479892095,-2.954798353638047,-2.508420945890468,1.1798883004773388,-2.030548393946274,-3.6630891958236904,0.677526321416706,0.21462651426733703,3.062558795109791,-0.13850879550973885,-0.8518045253373936,-2.312245576499105,0.8181730170562498,-0.25852305920755525,0.07719254611541614,1.9437577118634384,0.34419337898399877,0.6868103434009585,0.07955564951475061,-1.687730105498036,0.24453122983167913,2.838213764351291,-0.21702823567198756,1.2215007042852766,-1.963492026101715,-1.0327169559543536,-2.0786664521009808,-1.0257635694755656,-1.499970807408674,1.7910637615624956,-3.0146595402359204,-1.3047620557642556,3.0766902448355204,-1.46548125684471,-1.7392462725582647,-0.3649240686720314,-2.9039461442574392,1.7592320513060689,0.6890406045968946,-2.1235684910392885,1.0436500403425775,0.9812926798674486,0.578070067697269,2.4085299885472593,-0.30980565894373735,-1.3481244437944728,-2.0209935070314384,3.338211229566188,-0.9924153651987043,-1.1394050928016435,-2.6366066323219117,-2.494286201636544,2.897464138630167,2.009399967035425,-0.25359835577986395,-0.8987166089353443,-3.328294208308018,-1.7498217560355156,2.6015016896739445,0.7006591750728364,-2.2557236372623475,-2.413247874751284,1.3117097225986896,-0.8500991784883536,1.3621163889669758,-0.21478588821878059,0.4828048501568566,-0.3257146816529438,-0.41391455992467,-0.7588318319186953,1.8541228762970559,-2.6692098652829883,1.683560201824329,-2.454694211822447,1.0958584809548562,0.7828221362751124,-0.166453858793613,0.19604553103502367,-0.8984049018116228,1.0572236762196796,1.7507901523573526,-1.0178917887198649,1.0158572693258556,-0.5423936463840079,0.21628966686845275,-0.17594644975137455,2.1530391631764245,-1.4155649158029124,-3.3666864589811296,-2.618328676468754,0.1792256630130364,-2.321171627773286,1.1188657644054867,-0.8868919363914782,2.2979398667704913,-1.110056241727821,0.3118224550303297,-0.47965181394890966,0.5358377172465117,-2.862616043271152,-0.5999803233401629,1.3197853776913846,1.0035929395907088,-1.8158483656556377,1.152423071286931,-0.9902324503826525,1.7430627419989775,-1.9148577984019373,-0.9664786837682594,4.1430510286589115,0.09360345548649408,1.5511015505126016,1.7261299376685262,-2.580465244002378,1.217016158352293,0.08165304209229442,-0.44670568731747634,2.913029383889864,3.2729713317935185,0.19073062093326276,3.2679934140525826,0.8910071033101918,1.59593342493617,-2.412138684533342,-1.5880388274270127,1.8558809885369654,-1.6471064602308843,-1.8067423537376615,-1.1560342339662704,-2.710378278152062,2.376788696197745,1.9022430819369733,0.7526342670461429,-2.747233567066457,1.5054773298593898,0.5757038063125733,0.2962928870139381,-3.314181486245396,0.05731779961065265,3.128621747122081,-0.17060275535431585,1.5317647897138646,0.7380239100507169,2.8691587894315616,2.8765670467367586,2.548946874681314,-0.19995037796641596,-0.8237689253662552,-1.5962194807828247,0.7516514546561948,0.4811098113195747,3.2132026426142035,2.293691292318111,-0.8583409920653494,0.10461605537101672,0.581003359698575,1.0667919455662738,2.0173607240486735,0.47170578069761143,1.8394637281154758,1.6021870216405638,1.0272813919739276,0.35817443534505516,-0.731131122211095,1.7740113095545962,1.7800526549635327,-2.4757289209000968,-3.437704748655859,0.7439514472619853,2.3077109197025765,3.722009574449107,1.6424001030500084,-0.8979346469771,1.75739030862277,-0.2968209630900034,-1.8919575319914728,0.3545906294539699,-0.7512950509789784,-0.8165160692069475,-1.8444803712135498,-2.787743459468706,-0.7306840473246619,1.1688191550328424,2.1213542616306413,1.4421433229174019,0.3426219464662269,-1.474589988728085,0.0468681819437108,-1.0528369824583532,-2.660645740243748,1.7596350495424278,-0.32135216497174074,-0.1787562051807797,3.4460050006595395,-0.5123519571471112,-0.7776735439995522,-2.5287029529662024,0.39158315343066297,0.018096387436249667,-1.3664411501537677,2.13907254310969,-1.1425930168517493,-2.675942664884873,-1.4446224414228337,-0.40464248145665505,2.011571990206506,-1.1213787778935054,1.6489528569569871,-0.2065433317186112,-1.484346868010908,3.035989280357506,1.196301936496125,0.31388497875720595,1.6593358329821704,-3.0655989956827128,-1.7057285491499667,-0.42356855335120586,-1.3778703663234706,-1.650578641304846,-0.662663486756321,0.8980427982763324,2.3570337891063637,-2.487474953768703,1.9633292843573071,3.125976383352167,0.7439700588264525,1.0813279338375756,2.1551477533553354,-3.28491780012611,-2.1222918569187343,0.13697769067773252,-2.6980612014180587,1.5192717193829683,0.8163125755310774,0.8722438166413097,-3.0923080307263184,3.721432516415398,-0.9411350875308382,-0.34455174659039295,-1.8848616334818487,-1.6009431556852083,1.666131674017586,-1.0610236083286742,0.18631598562642765,2.324638845757445,0.8255884935686492,0.44233182751801275,0.08815819791664416,3.113179320069027,-1.1380141720029622,-1.641015109516715,3.332932811461385,-1.3581457572040108,0.36338428747483026,-0.7183237190353094,-0.607165569063258,1.4705884072047424,-2.5700417939154168,0.10401503520435428,0.6737055970383391,-3.0073794730117847,-4.151042743146881,-1.0839727555088454,1.5074234631614627,-1.8198328862885482,-1.4877600301783518,1.3531099937663198,0.19796285294893456,-2.8296307061400614,1.0598793335926904,0.018218882878941235,4.035613486982167,-0.843116178549697,2.1460848335295424,1.2219016294386285,-1.3960721346351832,2.139212362887889,2.490270026714571,-1.3733281634259993,-1.6489670439790076,2.3900126821829644,-0.9102560058249194,2.6112307669886143,-0.7782009516731934,1.515119157006029,-1.2948252692553424,-1.496539278491125,0.19106327881777185,2.62150241067012,1.1903041458087225,-1.384134558790385,-3.4637809340009635,-2.8381072320179697,0.03757607268131201,-0.5237521049656323,-1.9779905813845116,-0.5252877101387154,0.7939802261659636,-1.323350583169278,-1.8505685975120902,-0.9272678200426059,2.0471996963626267,-0.7590879595612908,0.011687053104158088,-0.8512706646012284,-0.5304784390941004,2.0089016606655017,2.097674157772647,0.6744921084280947,-0.019695166616126383,-0.6910551054542762,-2.6332058913357783,-1.3676663944991891,0.3189944934362277,1.8964731638847474,1.447649246550935,-2.6698998430389307,0.6342716461412385,-0.9629068002655797,0.971333611923107,-0.14424022860057079,2.037385795827992,1.7432708932618974,-1.1342282514788025,-1.568170379241443,0.04023110863490517,0.6303485290987976,-2.2157320061849046,-3.572853285772157,1.6510993355912962,2.140110985904501,-1.5424235774261665,-2.256363039598578,-3.3339986831665596,3.245208403629107,0.10211401282913794,1.8549057289007436,2.220431922407248,-0.10282686732509566,-0.7280639316571796,1.096628974583396,1.4720169970128498,0.597391490503852,1.9913971025602968,-0.7979083476754366,1.6103351205227552,0.967848964229492,1.0175564951339418,-1.3191204599649544,-0.4538042990167118,2.2090313708937486,0.18330798212864458,-3.078714953633725,-0.3136005867697841,-2.747006615201823,-3.1543870044902635,2.625323092052005,3.388017102749214,-1.7393664740065922,0.7082077156143533,-1.3728714310956456,-1.3897149019378736,-0.6997269973233329,-1.2076132976022018,-1.1235311646740416,-0.5637835109938056,-1.29966041750108,3.8265871838096452,-0.7455690073688953,-1.5548667980311976,0.4197232278451731,-0.5987843345688295,1.8332861745294626,0.7138994702679488,-1.9354879885248761,-2.889091458758708,-0.20413004093887646,-3.9681528749261057,-0.7424427458074954,2.03980947460565,2.471834891315226,-0.2696521531970616,0.515275778681781,0.3705852079134918,1.1093875404186484,-0.8773740795584032,0.09271920123675033,0.2993975328610332,-1.6506201083858905,-1.3500208103397375,2.5882236535888645,2.1789063942799896,-3.253382138866703,1.6983752921359714,1.4660848077624367,-0.06054783710475439,-0.11546370055899177,1.6010680310951457,-1.2015733546446061,0.398770961869582,-0.48082098674690193,1.4057777116900407,0.6682835169875542,-1.703040120287123,-0.7427121824756346,-2.615010958167256,1.0068086489894643,-1.574433919048949,-0.6272055574450485,-2.3518252867661262,-0.9358446165161918,-0.049923752509649595,2.172098211165032,2.9474626543737354,2.7963852924991452,1.8269802938964022,0.9292068990799264,-1.6373280230850535,-0.3496000840396734,1.9447353640531468,1.9864788531943207,1.2498468389933695,1.8974310258145497,0.5935452724096917,-0.248140014230654,0.2826676184329295,-3.635299184837525,-0.2388194558028508,-1.9196206465889583,3.4564438752814883,-1.1358181335592559,-0.2690794652830498,-0.06004234723593658,-2.1497363673501098,0.7625892998966276,1.0922777965069008,-1.0682712703048487,-0.14838185087795858,-0.18025906081396112,-0.4363881346938612,3.0200235222063605,-0.19858238968005168,-1.8496937024062057,-2.23557461689314,2.519366857636614,-2.9758923578793697,2.1736771585268966,-0.005495684364201864,-1.0511308689738605,-1.7374257796503547,0.4829874162898083,0.7674995095854223,-0.7674351073408043,-0.3917438908039874,1.4839128091954144,3.733245631897118,-1.0293819393783272,0.0013181798197108836,0.1671802980477075,-0.9001669307619332,-3.100911325363861,3.2137089918550856,0.5517769499452165,0.04306813838588489,1.0350295638707707,-1.4374748443302332,-2.9924086332915594,2.2265278394851546,-1.6122575099162462,-0.9369779388179978,1.049473831693104,-0.38584852154566257,1.437752018201345,-2.767398732637211,-2.1867555844953066,1.8009122972315479,0.185632575915154,2.0246344329148864,0.09550408168404712,-0.8035586013619507,0.2582127453250699,-1.382276508107208,-1.4618978473718003,1.4103225979177068,1.1198806285655645,-0.48269520016148987,-0.21059716484244917,2.3634622952204922,1.16558195459257,-1.5060261299040931,-0.6778827039444222,0.33082633820011054,0.853860515705978,0.11584909811094585,-0.7031282601360728,0.26250043917794785,0.6294265696977555,-2.24582927051786,-0.36820770990985685,0.6959230828267323,-3.5467302791574413,-0.8985017173152388,-1.984063794058303,-1.0533627986768814,0.7560922993457515,-2.201925281499935,-2.6413280634802483,-1.3236444716887792,-0.8747839749484391,-1.5712458597016754,0.883024741625251,-1.0305604915844393,-0.4198110855696247,-0.3482737611155327,0.2379423104827188,-0.6600738446276831,0.6784532115279316,3.158473526501793,-1.1386972083859508,0.8402199760666518,-2.1422632491045643,0.736625834691183,-0.7037234013236392,-2.0928001165792915,-0.2946134642816475,-1.8085254865614542,2.296675703612983,-2.6646316404446697,-0.8968763445711239,0.9349872675705247,-3.3787712534179106,-0.2641882823901845,-0.0004872476283547454,0.09493671033075561,1.6597663741061457,2.2174439132296655,-1.1630487875912747,1.5191613975588876,2.158012672089457,2.152472775695138,0.9823545228636846,0.841121268708799,2.948984169015638,0.47208802271550937,2.346924023808782,1.5913998792262993,-2.028239126261786,-2.665937950798521,-0.336197207515414,-2.841451257515868,2.5540511672971986,-2.734335815753445,-1.239883745054873,-0.20443692430931587,1.6531901691443631,3.024025506861528,2.369668969094459,1.814729275215449,2.9791350739311993,1.3051346128451302,0.7267176543340356,0.6424196775433868,2.517793344280833,-1.2668483862303495,-1.6674624437303445,-0.5693636409993164,-2.106853476119736,0.7233260528053557,-2.7575616463977575,1.372877373712501,0.21562136367629137,-2.1532949576194005,-0.6270303269059766,-0.14518607178558976,-0.719658700010401,3.2059034704560228,0.7656435956300675,0.18681832416682814,-1.7987353115979756,-1.7062681629372016,0.159696973233547,0.3832543190106533,1.3622647789083795,1.5001544437970513,-0.24517192923774758,-1.6990782257931942,2.253513925479219,-1.0509605704637384,-1.8293904904488418,-2.4111339980181263,-0.14282954142621002,0.34322612328646596,-2.2949115147541868,-0.8816615607663483,-0.4257883200266628,0.5052723883600272,1.9774122164874213,0.8005468050544684,0.08170212400193658,2.228191467049478,0.45649767449431283,2.2624813625120224,-0.3581008247909179,2.553037451439362,0.5514010609004335,1.3299585066281703,-0.34471196188672876,-1.9683659289857238,1.776385125683959,-3.1754197548882694,2.9585799716389833,-1.7667051424379763,1.6556635658067667,0.6427599206750233,-0.10471115226260966,0.9325548486605884,-3.2788966063002962,-2.0354837371130268,-0.28990613206598403,0.946439438313737,-3.3152712634882935,-0.05704358919845059,-2.3197030548207214,0.3347126366601156,3.5483441669384503,1.7849696626141103,-0.22457852646896437,-2.2493323150202817,-0.93020959315124,-0.3891791864442323,1.1684744332684236,-0.09592105832261297,0.221525245086162,0.3316058178115612,-1.1561211701719094,-1.8696401795412834,2.060485205846115,-0.10762197367441664,-0.4577919654044361,0.7178327167244867,1.0893569341654388,-2.3547153743590767,0.6171744225326455,-1.1108821749298463,0.7195021134199084,0.09897285035089105,-0.8464036263724237,1.567673088732183,0.08202118325602424,1.0330885155536516,-3.210662438714759,-1.582687799345186,-0.1526795715640909,-0.21589183710915452,0.36931699232056336,0.5354067086863362,0.44764741088476945,1.0241080113056988,-0.8349029005580529,1.4456601269650244,-0.734457719054718,1.541580191989199,-2.224765858282844,2.0937269383527677,-1.7678154762248808,2.630035356704213,-0.1507628716367255,-1.5001033258194036,-0.9141354965381204,-1.2403422982338488,0.17291724916261791,-0.12118574963057802,1.3019962100823488,3.671392138033025,-0.03498137469597927,0.8555696193399379,0.9239590808202389,-0.6424608804385766,2.2141790114265554,1.0366847296636768,0.9953677290048136,-2.902899566245842,3.544956019312777,3.3599469329120413,-2.3183957318904,2.2250573118430914,-2.6511417473935235,0.04512632419677882,-2.433883686626038,0.6111359722606792,-1.5731838056059666,0.5738775904636875,1.2835541574727423,-1.1154588307708633,0.526090596815762,2.5953454889660104,-1.0521104004675161,-1.8251576405410344,-2.300905887781425,-0.8202962260281703,1.9411150402721906,-0.8728261021014537,-1.3194965999591608,0.6647171433674848,2.4806104586398647,-0.09032430309752333,-2.399376832182141,2.2144701154697524,1.2030249555574661,3.4495008757439765,-2.5586750761801778,0.6899483487690736,-1.439487059786236,-0.673488837561923,2.065132705041082,2.1754793820612663,-0.17426876184819212,0.10732118236191728,0.008801992708298212,-2.1393297690577913,-1.6922878202983536,-1.0555939768872795,1.5680022665915307,-3.36068335414698,-2.546188329933398,0.5157308270727136,0.7644274220493896,-0.4808767556511582,-1.752008408534525,1.4716143385471505,1.4708685036963463,-2.3399912494294113,-0.1119387861573575,0.5847956834550044,-1.7612393570881386,-2.1764887172780503,1.3332369019863572,-0.2558495693568154,-1.3478763973459165,-1.0073912342036992,-3.7232741384686117,2.3398574059128405,-2.02547196763107,-0.7373297404244306,-1.961605096786324,-0.3428498212477696,3.891933514277347,0.0068966447389457895,3.5406659219635803,-2.52845721137904,1.0663422702292886,0.33209642554322233,-0.28802549821123,-2.534713505512616,2.944608908913237,0.3367116383769682,-3.333353176782877,-2.2290277882177505,1.0404649947056528,0.9492403715576463,0.4592468633570585,0.6884865119174516,-3.090309405025903,-0.5880147239104278,-0.27043693301741445,-2.0539389910593178,1.410164165012514,1.7508465893660656,-0.5356698329948989,-2.8690232035541623,0.9239042104230989,0.7105616477974858,-1.548015517333414,1.3569673456599107,2.944367102191025,-1.2523059085180355,1.5247702390960374,-3.3943556953770364,0.6549629514113664,-0.6182106552575729,1.2382615032702335,2.469696609177787,-1.2129852792071718,2.67583070397622,-0.16603043315248517,0.43882503798442857,-1.9857188828057537,2.9926481960792026,1.310246517192983,-3.2718784039762285,1.1291279571285178,-1.2089144754609433,-0.20580728824949115,1.2949336753693295,-1.574921934196688,1.9029138656205886,-0.7555379556072258,-2.1806955464758713,-0.3192242369464603,-1.5056250407096845,-2.5624165420359226,-0.6380433975912864,-1.971794669417059,-1.3651197303828266,2.1141928849897242,2.2965857153715796,1.5861755999187293,1.7850018978415796,-1.9905755170125208,-1.1121246766620485,-1.3673496973342045,-1.7795069663131746,1.737072993659402,0.070591820401707,2.350528666384971,-0.42927684595283017,-1.0148567756949345,-1.0584288914973652,0.012213509049199944,1.445192982705208,-0.027439456466700457,-2.4998753138200414,0.5281324939431623,2.184227076436416,-2.9688503650499247,-1.4704756460216215,-1.1968156426852405,3.9507306368205453,-0.5340740645743496,-2.685136772135518,1.5988717153749257,-0.8076230837563684,1.0225842159841227,2.0592543768725013,-3.05546175069335,-1.8965294970060338,-0.058221485779146785,0.028821162628982776,-0.6144218758524306,-0.9157740557033688,-1.7607860215518687,1.4704809279242743,-2.9086302850434054,-0.5698604266885414,-2.120188417852684,-0.8185443711264662,-0.3831787089274295,-0.9002759280140774,-1.877539144133404,0.5531169652996458,0.1610128504833337,0.13779458640182532,-0.877859554393945,1.0324972109443886,0.7370287289548204,0.3019105847788619,-2.291287684211276,-0.5638311296391099,-2.008998868429788,0.7319669841667401,-0.6877833848916988,1.1657464475113732,-3.015257384737672,1.7543516039756712,-1.4340672535368009,-1.1996802218201776,0.609354254615944,-1.8982345790000388,-1.315393312737006,-3.485055712818911,0.8478317092520816,-0.4382171395632029,-0.08564155931526699,-1.52941372313436,-1.057781711239417,-2.511724037175574,-0.26629770109366796,-3.1215106191405835,-1.6493145951152361,-1.6893557639322654,0.9966108796697443,-0.12236741789732884,-0.9326362079109916,-2.0053536122139737,2.017782072114284,-0.409043344225077,-0.3666668037990549,-0.30737900491472947,-2.6022114309324915,4.028756326307362,0.9348453659035194,1.5090474295129634,-1.979125282839012,1.9825465486200091,-0.23455070148191165,-0.9150340080309886,-1.1712085024666339,-0.5855925554529595,0.9231497919502805,0.2761177083883327,-0.0969345901773196,0.9751859863934301,-1.1860003015745235,1.0165526803233533,-0.49663347690506504,0.09502171270295451,1.0436034644360768,1.0799401704791833,1.8461214968126862,-0.04485372717730338,-2.964249558893007,-2.239724649464478,-0.13257311866850205,-0.31503429889347906,-0.9630368081227059,-0.2800808652342278,0.32266595878298276,1.8527578463292476,0.735811343543573,-1.7404717936489342,1.2254799277879116,2.1322326715399296,2.245287240428821,0.2315920261463611,0.09525280939246472,1.9848045581052887,-0.08971121828943461,-0.22408261905142413,1.5437701227434424,-3.8954950334931326,-0.813767736529223,0.8058882767109207,0.8532995195547722,-3.9695622499963474,-0.8428666447895103,-0.2431186968502961,0.1641611605600476,-2.5662782810551996,-0.11193414162688374,-1.3273005464762677,1.639141408943228,0.08059110979365021,1.1082987079561721,-0.42324007155855436,2.283446208309549,-1.5020504944345607,2.5196175038465247,-1.43302457759277,0.5950156908609595,-0.25530674564824923,2.5947730568638807,1.3007899973700041,2.4573633563409687,-1.26997418580132,2.0184294625345083,-2.448516510399373,3.2431269484021508,1.0965135474105,-2.4248619593971283,2.8943277383357757,-0.6458600567619714,-1.4226777961631896,2.941296502052162,0.8917416902491621,1.2543126368245088,-2.1724280218281478,1.6776667840963495,0.4023974355653036,3.1877277414765284,-3.5991658680967387,1.561474184581548,0.31597490795589656,1.224894727804387,2.8530611523776916,-0.7803222637822815,-4.527952625810481,2.032003510455642,0.7895236779703794,-0.43118093454864387,0.3059212009343025,0.644085456107314,-0.39912406929461064,2.3490041454090878,1.1164487952444173,0.6097298730341019,-1.3066028227386848,-0.6878255585394799,-0.6474885136797773,2.283151607321127,-0.4807939585483173,1.7400325371682666,-1.5448926445576532,1.3932579513513557,-0.3563469221921241,0.45658739003492477,-0.602707109331918,1.7707701215940481,1.103137278559218,1.6485552548355207,0.7691675064526298,-1.3770682273623442,-0.3374608619388758,-1.3014456756391128,0.69701098086175,-3.7032906616390484,1.882939623521322,-0.7948688917542359,0.618185007274949,-2.1409123395782528,-0.1697360975001826,2.034178042389355,3.115663028002555,3.0890662976599117,4.139288862506463,1.158036214987941,0.5708131094660894,-1.9873255657227842,3.594441267159421,0.8621135452357567,-0.3549163126426586,-0.3659357445905751,-1.5574571706098512,-2.1367438249575126,-0.37865377053218957,1.5463508097674186,-0.3432749744753493,-1.3131090795014309,2.7075369992881173,2.0964854992122324,1.447259209987829,-1.3285955977572017,-1.6166562274544538,-1.4100738405262767,-1.438614617163518,-1.759607950941308,-1.0490054331209018,1.4968062730052336,-1.0634512738189363,-0.8168746121252559,-2.934406650019995,0.4339171855766011,-0.2560041091523471,-1.5594644962217752,1.559249155360756,1.3550089712582878,0.20161891663645126,2.243890209623982,2.5704755830280246,-1.4696645619499018,-0.23726756417625605,1.4131846191629158,1.0645966741413029,-1.5296247617536436,1.4381947919358335,0.5998016512643392,1.1307524493352128,2.3036850276467993,3.071759721221822,-1.9945886649088707,-0.463761261491138,-1.2634303795002078,1.1750399066779287,2.7960159500816593,-0.6630699785620803,3.247642052135757,-2.0530213715851247,2.004580225686415,-0.16931716411481776,2.602104462792016,1.7175673469354544,0.3740633591176221,-0.7277117762806696,1.4202586750032242,2.3826013672984057,2.126388544515807,-0.9224562014965774,1.7699620418644384,1.4331361682866446,-1.4483220522967417,1.1815370858277952,0.17946333334892492,-0.3064256759918394,-0.15448253032350465,-3.4239899505895535,0.4336255357023598,0.44524508589759937,-1.350486457356913,-0.667773056361111,1.1456002817222921,1.8086963924432902,-0.7885113646227554,-1.4711655819648164,-0.28746312369273624,0.14425256091211888,-0.1586169770930716,-0.254050445681662,-2.8056450409879226,-0.4766761073725618,-1.6201102177314537,1.9482819348498446,1.8679745266743797,0.6552813982815268,-0.9074591344646249,-1.8736411484377828,-0.7980750993031984,-1.7010994374832018,2.6973512537977866,3.0127967319813878,0.2061487787780612,-1.03684124610934,2.7820853350104353,0.9742177513396841,0.7413807232824787,1.5945467516992065,-2.2284531377498284,-2.3165102337990064,0.4484984046927288,-3.0554968215941396,0.6288150852339365,-3.3088977056156827,-1.7019675388943145,-2.395556628723549,-2.135757393689437,1.3728782023489798,-0.6018378540898357,-0.953652705317048,1.467555572977057,-2.173233066950036,2.7185627056360144,0.23292852899121796,-1.7745561579755553,-1.642754324994374,-1.722023600121687,2.4439963723665423,-1.0480966138037688,1.3008030193082176,-1.597845963887035,1.5734410831882317,-1.7020161593697345,4.244001367896704,-1.5276120794841481,-0.5112942868341106,-2.140780251557194,-0.3215096563579295,-1.2155831408460096,2.6323449552949736,-3.769455473399939,2.790567980468178,2.0163826900361816,-1.8917950641087855,0.9165110023062175,-1.5290307355946031,0.7152079715220017,1.0305786424623202,-2.0288992784969153,3.4483030456439936,-2.951000408662454,0.557525557470921,0.9044526047935614,-2.520363750508294,1.1186311826548936,3.0109567429093955,-1.4378151420390228,2.370009517564699,-2.9492235037872816,4.056202871831984,-1.2104485657527289,0.967162391484377,0.5354469152317751,-0.9980211932571939,0.6508988555382709,3.7734625722850135,-0.9259962152950793,-1.0170737148717717,3.037246307477291,-0.6713186217978534,-2.422182503472565,-0.2237429816349037,0.39051532304468106,0.3204172121424794,2.460485641330529,1.4490144494833601,1.7217252239795957,-3.404008229122672,0.32416352826245537,-3.512988035770493,0.1866548718572789,-2.4605687729698418,-2.751628261608444,-2.17590594834591,0.14186727836600851,1.7636905924469988,-0.03273073844836734,-1.9749852873604083,-1.6213304091094554,-3.5431580398155207,-3.3421944365423673,0.3108313372512293,-2.8577110528738756,0.7979761811360288,-3.4468832757299896,-0.6030221075607463,0.732744415445801,2.0385766536092067,-1.4169265789694092,1.9404131015242532,1.2678713543614406,3.2326027550515004,-0.15424766475272522,1.8018357792517905,-0.5131127859275496,-1.4877962769164623,-0.9320397675532407,-0.2867625408917081,0.5053270019959463,-1.8564665276976862,2.1782304620040818,2.434488647204497,2.943579232773648,-2.8745223407310947,0.649027015373028,-0.3010241165054253,0.26626522191321095,0.9332398196416183,2.2528598998945895,-1.478660574638151,1.4630284538882303,0.6094345060278283,2.280980827766961,-0.5570118932761093,1.1645374975966722,-1.1283960921571627,2.535051408968115,-1.2378217720690086,0.05669918369244503,-2.7143617102384203,-2.2005783557537084,0.968690953229617,-1.4800211775755523,-1.4986316427475093,0.905078639385687,0.8080964439162707,1.412027099039156,1.8932542029140573,-2.9950065022397925,1.9441593051816883,-1.0035529565862333,-1.241423487443622,1.316220936534382,-0.8934480375330066,-1.7515933578061706,0.49447794852475485,-1.2134430537311172,-1.8739332017981978,-1.5459349316307238,-0.9782794785168654,-0.25977730884485106,-2.6315574438417895,-2.660864238743059,0.9456316528569317,0.8136806997027246,3.130287035645367,-2.4259230882296157,0.7093295240833759,-0.5477247879470518,-2.142663735854797,0.7170630995017915,2.38173360246542,0.0993090237873391,1.0923085692715402,0.4168375528661356,0.31237231558131445,1.5156270134345855,-0.12238255707844685,-3.6410514376326693,2.012109115174635,-3.4520831987313674,-1.4255849665751075,-1.5304170326880688,-1.1273253142266821,0.636783499682565,1.8650204513342326,-3.598359767101439,-2.917464354452613,-0.41310646416081237,-3.1537405958405653,1.316144819190635,-0.4809372473243696,-2.7485260846386024,-1.0978013875437314,-0.8378229619305034,2.1388047944247743,0.5498213857075235,1.224716427864727,2.074991761543989,-0.6298107235776347,3.50079831336551,-1.5397363615930193,-1.1912618658265532,-3.282035997112061,-1.2153451913375215,-1.7864427226604422,2.3899495622484133,1.1002769278366689,0.5940055798317915,-3.0613814442365284,-1.4604687904262188,0.7573613555956369,1.0833617093113732,-2.4002479472996305,1.890317362943325,1.015531677394519,-1.2120969495450709,-0.48199056935116263,1.2883972350704682,-0.2379270298419968,1.5413066749618867,1.4465628511074564,0.43469201158894116,-0.6543364186549121,1.1592241893880297,-0.14225868322573226,-0.7383807592811006,0.06724777195171627,-1.801295637468057,-2.645637434097758,-2.58204155633068,-2.7139007454109176,0.6081751196335871,-0.4655381876613226,0.44444974322768205,2.3546381499083897,1.5051015336923081,2.052668171047655,1.1910887450115442,-2.3224508318695363,2.712731241748109,-0.36003827724946497,0.003783685024991021,2.2285942722309984,-0.01278638601740161,-2.5929098013159417,-0.9290430898681822,0.9664563814660143,2.399527886783726,-0.5956321714681624,-1.032750802912729,-2.8365068733361474,-0.7945513513229927,0.22825725695633825,-0.2205864872204034,-2.854349084193125,0.6035858578345019,1.5654131158463955,0.9675085896667441,1.2127194548384161,2.50705308364169,1.0983365073651592,-0.8693260010109378,2.07034731291695,-2.371538985380161,1.837181638622305,-0.8797184713439681,-1.2366322629509932,0.023567678559731534,1.996715913291551,-1.1029779706496703,2.957791430865863,-0.7823282249885244,-0.8013717420389627,1.5043728756043468,0.09621927885450467,-0.3942122102900214,2.5529065294525064,1.1062265135814087,-2.779255710671776,-0.015082224383222712,1.443784496919987,0.40216117722949773,-2.233192431384364,-0.18762877000700087,-0.2805294178838187,2.1365168824787144,-1.4408995769731103,1.0265624243628173,0.4042554262609402,0.9448955940200982,-0.291705551487209,0.16916410585280398,-1.044742339110504,0.20626583598402526,2.390501486568243,2.3064799386932857,-0.11796242894256181,0.936580351044298,-0.15454098401624647,1.2966475585917432,-3.8199520508860476,1.9719451846967941,0.7128501076819347,0.8979950825363429,-2.13100686489718,3.193961048252544,-0.7908396794774921,1.9254440119050598,2.3979711953716976,-0.9716947605058868,-2.5171768734450852,-0.321415312438086,2.4940975040671445,2.896915436141632,0.8381032752916581,1.1471069541804528,-3.1987875139181545,-0.58451922263593,0.2992644609383839,0.36175183460861315,0.891948373874222,2.7118519247433723,-2.426126576315688,0.5007618059743955,-2.187380638680876,2.841301744410397,3.590384960081622,-2.0581830315568563,0.19138619999981124,-1.4317085873468132,-0.5667596262552038,-2.349594383121368,-0.32241732674050466,1.5615823530625357,-1.1615204890710138,1.9540440356958095,-1.4266409491083305,1.5848871786738996,-1.2857475934176301,1.52937638532825,0.6165272564879406,-2.9600246029587196,1.2978184993150974,-0.7757586813318337,-1.0208178964337642,0.7649758189293205,-2.1864157560626056,1.6147575444099427,0.6337322641213119,-0.9911170816358985,3.0099770217654096,-0.9095498222524417,-2.0719231724545586,-0.483155476818059,4.017316807261699,-0.7681805742377529,-1.8984065142129218,2.024717963123661,0.09732716165802398,2.5001316071692505,1.780470840676412,0.33676292816098313,-1.7197381702393244,-0.06030919587715919,-0.5368019869324537,1.175786773797767,1.3180031040020892,1.6909618743305996,0.22637706912787262,0.715369291839052,-0.13568887303798888,0.29460741886542785,2.866165604843316,1.9112053974416359,1.268294228182133,0.1495884115624994,-0.3315747629637313,0.01002745039431779,-2.3087368185196167,-0.15405165486711034,-0.4221244716645781,0.08088436588709282,-2.9580205755383755,-0.981833310061169,-0.78349730878373,-3.3056390316542266,0.14984523335467537,1.9971111323923154,3.654929262086124,-2.789946275300368,-3.7200518878430224,2.651073805758056,1.7663555933284303,-0.371493585746188,0.09404478982121477,1.5365204540758979,-2.4749296389181294,2.5128701017686454,-0.5948940986849596,-2.811046025019118,0.13099900757869773,-0.290496210831702,3.9336287482595096,-0.22150299966977938,-2.2388352671530547,2.24319285283371,-0.27994772620663066,-0.8123692950017216,0.6369099912750856,-1.1112172482984053,-0.2983500305926416,-1.1353040000472834,-2.303009211645608,0.011168741465371727,0.7929856864861603,2.6425717864272813,-0.9460816030614227,-0.7813463146761299,-2.890726694317135,1.8850087448580428,1.807156106930887,-2.967305479487121,-2.8029394577639546,-2.0210804919561025,3.1208058351050063,-0.09014361359327122,-0.9791843145473491,-0.9749086421111645,-0.625322673654102,1.2688526044414064,-0.05141037675719454,2.693579652619598,-0.728158811311653,-0.14624957229745877,-0.8037110820530706,2.452444830049232,-2.33854951632009,-1.3279382373117279,-2.4067617849695493,3.0076420161229622,2.4213356511224684,2.1665014288922926,-1.9083580238031128,2.870769041048902,-0.045479079551459,-2.2838456668185363,1.2776085447286905,2.396190510164648,1.4766513260807428,-3.760083407052268,-0.6377907883536966,-2.8154798673498793,-0.7043285801772551,-0.12702346276872586,-1.7832008594969102,-1.233182733027688,0.6262278090757578,2.8053884690530237,-1.715275736727039,-2.8226229289273763,-1.8269767755862298,2.098857615272896,-0.8467013302183439,-1.2836487225165707,0.6225790011945285,-2.844034336901405,-1.0348068168081053,1.1935480276096324,-1.5159639309615855,3.9064384423848035,-0.37131179028009215,1.9643692970598066,-0.495673212212957,1.6744874604530509,-1.1697561234505,-3.416848894985639,-1.8453990063491628,-2.5853839359700457,-3.328903889962159,-2.996307038223505,-0.4279647104520565,2.008938604342896,0.01285041471334055,1.7790540104188652,2.955327149522982,0.701998006787572,2.652029655044967,0.28339896216299243,0.11339837147954242,-2.347968957649058,1.1216833012593355,-3.2094600390312626,1.7191458227957641,0.34793485374590016,1.4053619881850035,-0.09826237378649114,-0.7441191219891905,0.421721654423871,-0.06211144869337922,-0.2564161612529831,-0.8792087540460696,1.2093107753332233,2.467108815101826,-0.9132292889807886,-0.702986010461772,2.239084052900545,1.1073519669655303,0.4905232321282733,-1.6271403360826124,-1.1618678092370434,1.041629530425616,-1.8677888862141687,-1.412522421494481,0.17481395703677732,0.014986999445049625,-0.356188021244865,-2.3170223884699865,-3.592142411620808,0.8417835253209888,-3.8280202685736926,0.009852030782170113,-0.9355890501346494,1.6922202268382176,-2.0115303244666727,0.7251845507357174,-0.6577689564399902,-2.5318109378080984,0.1551165567804506,-0.6541668019707152,-0.30996120935503213,0.7617421643690399,-0.1247358934299908,-3.457217305348988,-0.5850379241152074,-1.2794998757090135,-1.7158309694439544,-2.4821218309230937,1.1720410634959277,0.14268253776076043,0.353469674654706,0.5830498488292049,1.006446246139936,-1.4651525514130144,0.04904152503423295,-1.889785355874794,-1.4837932472399482,-0.8497874536361723,0.5851583157921116,-0.9694383888897177,1.2711446305892076,-0.8911298808733057,-1.4766415041996035,-0.5186489454623265,0.4716717416652185,-1.551470767030308,0.22344717693338106,1.0992408623589496,-1.3389400873116917,1.3383101642408937,-2.4202466299979832,-0.5975798426117952,-0.6800402124428453,-1.6509566098067265,0.7373735196971252,-0.9168244283372521,-1.4378796901997848,2.7381356621057122,0.771507240277378,-0.5158552617259705,-2.4169161725615433,-2.312760259616657,1.4998699317109794,1.5769035693536249,1.3359651741881005,-3.5936557012804555,-1.4818444531907096,1.2827701615700133,-1.8452955624910847,1.1228170782917024,-2.050798068484681,-2.3643262701503427,-1.245181602977957,1.4358449766205676,-0.36319946090260014,2.6169900710439453,-1.5439973837341405,0.596886110556216,-2.1544407934732046,0.8505737878061597,2.3138165765607885,-1.534740109274485,2.2315182000978195,-0.15880671393700968,0.8199751951308611,4.085351714276373,-1.2478727083635068,-2.226347614014906,0.700558760540918,-2.381585121355439,-3.841850069618727,3.272489110728917,0.9815160329598687,0.13062329269389675,-1.2716933701693298,2.8146118095279884,2.1030914772065006,-1.9272422344619005,1.7954978840800007,2.110450395078177,-2.177573345065434,-0.984478078331167,1.281125960154245,2.243250844497755,-0.291464288625172,-0.1879812806163498,-1.3310195842099268,-0.4163124883773766,-0.26600805253805115,-2.046734149806229,-1.3256978421029175,-1.8987329467748533,1.6764362343234924,2.3885206909740915,-1.5252273163383212,-0.050355778198272756,0.22110051799455266,0.5034667952198149,-1.0392116077536167,-0.04485235885068032,-1.062995714889016,0.15081969799884234,-0.87953866520153,1.4552907062600628,-2.2810653083559767,0.8123882342426564,1.6378508312454205,3.990048975326168,-0.9369051831982069,1.4000745865510893,-1.7585613232125654,-0.599342795808211,-0.33929272744251066,1.3464044352202278,1.6113836935115586,1.4088798741596957,-2.580716104481046,0.30770932172329707,2.2002110133100605,1.9759712432650745,0.41954548908178735,0.6479971233450597,-0.20225414838862435,0.01799108302758207,0.7153290993985254,-2.7406219833379875,0.12558204322349775,-0.1892984609139445,-0.4486392354208517,-1.006872891050983,3.0660963108688715,0.47973972266877957,-2.16721991232211,1.5035236733403403,1.2788299947329342,-0.43538608644696075,0.704237361269893,-0.2855814695739442,-1.6594659669021345,-1.7845253940867258,2.5156043879230654,-1.3794152703383524,-2.0435225083270807,-0.7312012682898619,1.0591117917324684,-1.4043895608742865,-2.5225761563550386,0.4461262512654563,0.4416730657371596,-1.1930978796318834,0.8950033690294719,0.49855045034834605,0.35782690329782313,0.023167200429039407,0.4196697774036139,1.588214091996079,-2.142968330968835,-3.0299945778701023,-1.2678903947083515,-0.6984290264116189,-2.4738659001217305,-1.4094041177376377,-3.417221200775954,-1.1902519365468267,2.721550978763966,-2.265702454529681,-2.017496987941324,-2.6399681454508372,-2.524225283697685,-0.8989429059347595,-2.784475036210787,0.2664156599057522,-1.1980562819325833,-2.4037054534776705,-1.6985542672437208,2.585125092406228,3.5412481457433134,1.12960849140138,-0.6296460873748554,1.1933612527129165,2.3388355340997786,-2.4482609962987985,1.1312588821227083,0.289768931740851,-1.4656602066968105,-1.576842454589862,-0.6574610883067957,2.07922197165385,-0.7563509818515315,-1.3893972994381216,0.6049384704777232,0.7918033384797664,-1.0455434002358814,-0.7782473913470846,-0.8179626698493037,1.101965106990863,0.6042691183281181,3.342511945915027,-0.47901051114052906,0.48235022579222603,2.7017059372753196,-0.5587380540982834,-2.1502300009254176,-2.594294448378928,0.13889044387267724,3.266656829908068,-1.3451422027061528,-2.557364534620698,2.0829986614968194,-0.17296377330014231,-3.097127999889073,-2.3209932365938175,-0.37558034555559416,-0.7703291999531681,0.8505592813363615,2.3763495021143863,0.7477798510615895,0.6322809573412557,-3.663772908541162,-1.4797522977682465,-2.1976334826206037,-2.2344506075455413,-0.21712030965905013,1.9742414657845415,2.7414602385984654,2.054811086063248,-0.764117982559084,-0.2518443092926858,0.3562156973338727,-0.9622519862750574,3.0093349003728416,-3.372194317245941,0.5937518204127937,1.4807594801733333,1.2923363348101045,2.0236712093714235,-1.4496405984357859,0.02881699602345286,2.596813868352192,2.3077602192217963,1.9951195088550011,1.94355226997649,-3.073954622035712,-0.5794413818471138,2.4434809575644545,-1.5739472501626046,0.44145188254592105,-1.1943028216902811,-2.5582925617259686,1.9446930523793593,-1.8297734360421618,-0.7656936949437144,0.5416801748046579,0.33178592579307997,-2.506281982865022,2.891527393427971,1.0348895378454737,-1.9281797453852507,2.0104250179334837,2.8591049516278737,2.5425940848943185,-1.5425925993669334,-0.6308221713685935,0.7022843506358059,-2.0937561803806597,1.870143151864182,2.6125100415815337,-2.439236777459511,1.226794366991898,-2.507749139468479,0.029408435413957365,0.5926130909380439,-3.256444929649535,0.8457298226655277,0.2649482981044144,0.7235274377278976,1.6238555791083982,-1.2719547177527202,0.08836288605344615,-2.9558796276872132,3.16656596375745,-0.08517956947733285,0.9311007019241501,-1.5263851288584636,-1.2866097731578237,-2.2826404371427627,0.6382651696426703,2.6508896577145524,-0.7818205672653041,-1.9757189504818147,-0.6749703039711243,0.8313916790050314,2.1604354274470343,0.7657821162373718,-0.4495037499453368,1.6244014547846515,1.8336503887158064,1.2563673860781686,-4.328638071350933,1.2711464730302748,-1.880206832410202,1.781288867886385,1.72438832160746,1.6196309130107545,-2.1397116969858008,-1.049423751214285,-0.964846263747808,-1.0796508890797605,1.7418252226985411,0.6463016502253233,-0.6878925113134475,-0.7737675400865193,-2.361137733224216,2.0603297128780946,0.4816741060354016,3.20860293855349,-1.6664389378277757,-1.6637234288496758,0.7499560513481437,1.2005476307852236,2.387158484603501,-1.0184781084196635,-1.1618474541910748,-0.10877476911450713,0.5148904779717417,1.3436249885897211,2.5266437736116605,0.743570464299284,1.0287093870498636,-2.7876072682956563,-1.3617211677649737,-3.6142173494712804,-0.6261653554530221,-2.6865249248516525,0.3690714340936108,-2.2142747965481795,2.531580334673849,0.9984540243749477,-0.18461057847299228,1.594239354434735,0.6307116049222294,0.225188019274462,-1.09556689423612,-1.4507211606607595,1.8923779555440103,0.6643103130654893,2.215865676920642,-1.1698269952303526,2.8114119477677875,-0.12602839357577728,2.2084596082040964,-3.0236572493917526,1.4249064793247208,2.389296948578881,-0.10987679904713585,0.6784420783464936,-2.4296388366773263,1.2580729368190073,3.1641869812845687,-2.2727032237094877,-0.38415079951732406,0.022859514450280897,-1.376714621351304,-0.9257312279323919,0.9098451280454598,3.1068192619936026,-0.3886832402294,0.3960761368232588,0.2726443017710738,-3.4684509540368125,0.11563937225932308,-0.19408377223710913,-0.6233212872980075,0.41868397984131905,-3.759207965624148,-0.08805156068060238,2.2316166167573495,-1.044387920148099,3.5136695367844397,-0.832992128499272,0.20063110630993228,0.0894199575029195,-1.1692386697322033,-0.1657119769952691,2.0780497263955793,0.6189671371382983,0.5753068849545703,0.589856596207566,1.422163396421815,3.17220151175011,-2.64109588447462,-1.3026951693925868,2.5474774503733,1.9975975123119025,2.7741479952310804,-0.3629735955619893,2.631390423905191,-0.69464779867406,0.3789561609997377,-1.2134144627902799,0.008781161066497947,-1.4791422645546062,0.6469664206903639,1.3617777385558196,2.1536873071475533,3.561808540425838,0.5527416078819972,-0.6373716613431852,-1.4847420079892193,-1.1272164293776592,-0.1576316859625481,0.8783819871141675,3.7221056752216763,2.022097100888558,-0.04824551467259669,-2.4935644240031336,-1.934289561737215,-1.1026289235848596,-1.6965984246252772,1.6362149768748273,0.2441536910724431,-0.40087883385937984,1.7169412700478155,-2.0060406995033557,1.3406173691529764,-0.4480422323642826,0.1568813027609073,2.0152574831897705,1.8921497999775836,-0.034640260172684366,3.2994130422529575,0.6679999228807687,0.8205396349566821,-1.783798032844731,-4.080497055730219,-0.7840101595062557,-1.331411750746829,-0.34355963915971305,-0.43446485633242155,0.7337078079894,-2.266025774583912,1.1822977591932298,-2.3738613215693922,0.2526568833577877,0.7803294892693133,-2.0924887882143683,-2.7881803143208566,0.8464421359659191,-2.14049277422133,0.4757789654821837,-0.4180211676216384,2.29822045690603,-0.8591233307541359,2.8149334251944866,-2.121616526488992,2.080232792817457,0.5689796076051465,0.3519232468988233,1.8072632094317904,2.9052490146840406,-0.5330836028870721,0.3986939048879597,-1.9326036629367844,-2.814550311734334,-0.5221034026598684,1.0963925579253417,0.2022041680229888,0.5595611293182988,0.9973172831799711,0.8677069183696825,-1.2739559681830197,2.785586569723574,2.6785991237954043,3.365191895853093,-2.482839672182449,-0.05798988154394597,-2.773085287575144,1.669143763392844,0.8891190440785098,0.7487808853274887,1.4396595198539164,-0.20600336211946457,-0.4279308350102634,0.7253478313609839,1.5780329953260555,-2.0336354690288476,0.37419750667127355,0.340171694001494,-0.2678160506749019,-1.2024267082807707,-1.0752233967250344,-2.040717157659107,1.2826092900688244,1.9114973640948336,-3.221803880732165,-2.6262439389845387,-1.320923145045773,-2.778073684211429,-0.10492473018756483,-1.9013683521709233,1.4307558967734748,-2.0950604984981203,-1.7591005891642693,-0.5027253674860609,1.2454150130051957,-1.8141366920100923,2.456834921637858,-0.9341833184641194,1.160107003721079,-1.5267977925594416,1.6454938816592155,-0.0779707614106851,0.7451473531307655,0.16584392427481173,1.9578937424738387,-3.4197962767413337,1.346526594071782,3.341952073858353,-1.3318570350233778,0.9004705205008743,-3.4293330172163623,2.1853167172769976,0.053905828889312765,-0.5449449000212948,0.7658153690487922,-0.9697897405462887,-3.4245408912924082,-1.2370166007316779,2.781260960070872,1.9676538471377012,-1.2583525242315925,-2.1784526967839852,1.991070011837907,2.3663580844740837,-0.045446533115824674,-1.0893511069535353,1.4730544342206144,0.06724848347077006,-1.6132357033467233,-0.9138944001111737,1.7959221675516936,0.9261205262579507,0.881935594607937,-1.4404275633844414,2.8251042956002803,1.4402074238792255,2.169149826840758,1.358427958233681,2.390644491102298,-0.8367458400722431,0.5754453888140102,-3.337718988530017,2.3815978243826126,-1.292886044190372,0.14701651810963035,0.511254251031975,-2.60468667091453,-0.1647452863444127,-1.9596877357712288,-2.994506651730411,-3.33918709611923,1.5430917534320587,-0.7579461779905826,0.6842731644792702,0.29455570109335893,-0.23810004321323072,-1.9994611996621936,2.725001723632905,2.1246478580344412,-1.0320577620479028,1.7121936726633755,2.826903199471094,2.667637399196066,-2.110420537102394,0.5727693801564385,1.8470722568979825,-0.7145885323538819,0.6158947881343737,-1.0598831330078333,-0.7955424923483857,-2.0896382063705525,0.06239003734462702,-4.332742540011243,0.7877389769334746,-0.7712008336148742,-1.881460272216378,-3.013163698376792,0.9741052083573323,-0.4032471525107167,0.7223013003644396,2.7460757579723563,-1.056700290576463,-1.0979961729215293,3.5802100414018048,-1.001459969710646,1.842656721587585,2.6448219050243127,1.3311016088986751,0.6861116969015792,3.3397559770240544,0.007979341126176586,0.6342045981483873,-1.459342421096221,-0.037238416222919325,1.810047625058491,2.1289454817129747,-1.471549377996025,0.1600215281741565,-0.6938223226102431,-0.6797729125465398,1.007232722969592,2.136324648135682,1.2450545418547438,-0.03545490649081491,-0.21035127509105947,2.158274303945679,1.4305686157261235,2.3208945242942476,2.698506905921237,1.830118111673627,1.5329297997944313,3.02502968917142,2.1495663969312933,1.6399495993495605,-1.3675068209408208,0.7535268566371393,0.432387912931496,2.920343822731834,0.16205438467598088,0.6731445659318859,2.0620554042340777,0.16695662084383178,-3.186539051787268,-0.8227707834697353,-2.7053686635818783,-1.1002466812187355,1.188099493947381,0.7612669585169546,1.485237829434537,-3.3362967517988213,-1.8563181090814806,0.446969002627069,-1.1252991220930715,-2.4113867962457998,0.38365720883440696,-2.1620536812932114,1.3619562703137702,1.1940462925836206,-0.6744448669929926,-1.8371000171431056,1.3765722611117788,0.8369399580029272,-0.3480643474664729,-1.0947937011419502,1.6931727241855876,0.032300412411585926,1.1204781024856918,-2.204780579386044,0.03596816380351289,-1.3939063846738542,-0.7656977927901117,-1.309493266958284,1.0589153568599623,1.1467948057492867,1.0498094019429933,-1.2226224835534318,2.377613915600503,2.5619787810603136,-1.3865272638212185,-1.2823690797473275,-0.700605754910942,-3.737403306508452,-0.12194763556611407,1.9374786168946865,-1.555456690164216,-2.9822299768989637,5.148540812140829,1.8750413555987964,-2.9958265884511115,-1.289051730285939,0.4138457908704727,-1.0125770363821112,-1.1373477611815344,-1.3281050763801525,-1.0951648336222723,-0.9545204169944664,-2.505091548442688,2.9949915204268343,-2.447330077977312,0.272730129713033,1.6767744595343421,-1.0709636745205855,2.004720481607473,2.8717174382029014,-0.12134880514723921,1.9283972882790832,-0.377158814247454,1.6232489713936418,-0.12024299894509678,-3.081306047635733,1.843361756961016,2.928151751711799,-0.16551695638472638,0.8617675768858263,-1.3429396349500022,1.0197817033292647,3.6952170024002644,-2.7118896950789355,-0.050078224669203655,-0.9923212586448886,-1.3082929304643838,-0.6215223926277699,-1.2606826014114216,0.06706038299373443,0.34599984650962134,2.0687499241981078,-3.0634973945822983,-0.11011981250944824,0.47261506657947083,3.0653437271881683,-1.0599092603934808,0.1475387321810514,-1.4469844443827777,0.5349705936740308,0.04990061763437838,1.5787806112580158,-2.6173902628092,-0.9477132460852569,0.6431960208339045,-2.2639495725547465,-1.9012426469299566,2.0052576969794935,1.7444674960058248,-1.2521126296459781,-0.5837615715747756,0.8861415447356534,-3.449063082665602,0.5096321863872151,-3.98201643688125,-1.7035644549813151,-0.21365057185500755,-0.6629303711601403,0.35540936961529335,0.9754040156415572,-1.6914036590183361,-0.49569822185626977,-2.7011103579487674,2.6810523415479217,-2.4552286812378368,-2.978578191008444,-0.26711597672402,2.7300259267568854,2.211547323352629,-0.3354060156763659,-0.6193237971807132,-0.4755681111603799,3.1252639292766164,0.45807804261354634,1.7520274869334949,0.2680381020438446,0.2779292316668203,-0.6072683359881901,-1.767694977622989,-2.5861754504846144,-0.8672247372230416,-1.9720906973057966,2.854597536786208,1.1929657454061964,-4.128245025406564,-1.2229242721360398,1.1003949524468761,1.4325428150304003,1.1118641681639485,0.9497235494388218,0.051576156676011405,1.6790345641619342,0.25363300148619444,-2.193609394753104,-0.7891753519431002,-1.4163208193768932,0.48359256727110356,-2.1547002260054597,-0.2934133067061513,1.0250252616816828,1.2776991426136721,1.0128446262036532,1.515134341788569,0.3462895024103642,-0.9432943358829348,0.051750983378726224,-0.7123653228528128,-3.6289902322892345,0.466040952751755,0.20471331038116491,-0.38729218959875283,-0.7699886434782259,0.9887489949013903,-1.1662459408777774,-0.29046261896398284,-1.7564573984920147,0.6097020269090392,-1.940866814070343,0.30966832387793103,0.8018515155275239,1.4841304900441634,-1.98028326846586,-3.0207185561554573,-0.5311336280361528,-1.8881871263280376,1.0097722807206928,-0.5176622330533225,-0.07606831707482392,-1.9373438785996404,-0.20821124463295113,-1.1108668602657252,-0.6875183438595365,-0.7841617619340461,-0.45955041432601285,0.4050657581819648,0.19521418082961892,-3.5546236848219257,1.090690000348651,-0.18948871416465224,-0.8964146542534076,2.2002199440689596,0.35485630967068177,-0.9551997899325714,-0.5351420282310024,-0.5278062942413099,-1.8729460657386956,2.32274415319085,2.6264992699849747,-1.664574992557336,0.5438214893937356,-1.0702228811307068,-1.547247104150244,-1.4575930934750825,3.031546799049279,3.7722717280978952,0.35800103971119296,1.463711702948432,1.3465515099617973,-0.37899164774049016,2.259452588391866,-1.4738564443809774,2.734908048685717,-1.7487330146990834,-2.149980621935956,2.1341248452428965,2.7918627634062148,-1.2133085008164592,1.0221004152213704,0.15023471707402422,-1.0752055145198876,2.9708063114093424,1.035989738425983,-3.202019265452497,1.2763538674575443,-0.31619716949564586,-0.2953635374876391,-0.43585474070432323,-2.597876451188659,0.02380255917242255,0.9870853577616262,-1.0844219169971014,-0.9239359389982752,-1.3250825346729922,-0.6900993985252962,-0.6580506353340185,-2.7955361007947404,-1.9959050478315024,1.5268022330772109,-0.8325869077050656,0.8310964818270071,-0.37981081024071583,1.5351895253541754,2.226114764329668,0.8438234453227758,-1.3911442368521296,2.2325972004120938,0.5626138921323919,-2.2530002383842422,0.09110761797730284,0.2769434336162938,-1.8943915375141342,0.49325165406553445,-0.5836143672379055,0.23287374692764617,-1.5188952877592286,1.5823413943710722,1.3438305743212857,0.5603346866858759,-0.3020664456007176,0.7845284301748711,2.126992032527608,-0.09931017610027677,1.8100378701914714,0.9546406568270999,-1.9918120392017178,-0.2913792267127192,-2.4579351595864214,-1.746186295461758,-1.7966986548840382,-2.2595772850434765,-3.3106739329110773,3.475219891428266,-1.3866819455144603,2.757842497782025,0.23664768038372228,3.344352089998853,0.8378793426139961,-1.393497914197239,2.2535864924456734,-1.2777785869013956,0.2607085511398113,-2.037628987650641,0.9488514188893462,-0.6953902094815976,-3.7955471149027065,-2.621122739261678,-1.4555762054290138,-0.14544935197131467,-0.9588944695855822,2.1870360224835856,0.8966507932633413,1.0546460542178615,0.5253095677108232,-0.13743883717936892,1.0283564034611803,-2.3327729188456545,0.7104701682262533,-1.5090509376188015,-3.169033880029509,-1.6593528409168306,1.540488572481886,-1.322805919796844,0.06938994537012019,2.8410706064055105,-2.531172004578617,-1.1228245809355504,0.7791608421144066,0.439665301327509,0.31152483531892716,-0.5640494462748576,-1.9477974268314249,-2.4453710487075435,2.0368290089934344,0.2418527649129668,-0.9766356060086613,-0.8157412396252005,-1.4399084871207088,-1.859671818482329,3.1805321808146805,1.1938116149639972,-1.8604133661679192,0.21491282815431062,-2.6496273427097656,-2.7907040031454478,-0.5454922318318455,2.6273911816574436,1.7074095718587707,2.6376095789712335,0.34226595791408215,2.8752695799033123,-0.6167711302044032,-1.2437282036112578,3.403039520144557,1.8963672029225733,-2.65677149090595,-2.187439697296176,-0.9297756113945941,-0.3720724029232531,0.06388440922171548,-0.9754110054072721,-0.691860587769005,-3.220724256728538,-0.8255664260524929,0.2830992378054564,1.579817515082952,1.2372605524855125,1.88207307721412,-2.571037872316638,0.6869931858634761,-0.21813748638567865,2.9706253342703244,-1.0854974626887335,1.6589841744390743,-2.7156840334270087,2.29760127047831,-2.5408984611754803,0.8150472107118558,-3.3597706220231265,-0.3615501304842049,-0.5318398187939911,3.9598919858575967,1.4997979631604477,-0.284811458438289,1.095899526584843,1.0554428751154983,0.4380190341771202,0.985981186796233,-0.017292122988806464,-0.2625308677828257,1.9606082733260621,1.2701866459701763,-1.5951081320542537,-2.3371925285968547,-1.680455869350717,1.0295999941042338,-2.0466221483062528,-1.2340694638984624,2.2944865040418345,1.325682497416167,1.1944851453764525,-0.245335774254006,-0.4546878220206938,1.37251943641913,-0.7518222935117153,-1.0650486207872318,2.120559124712689,1.2381392482595024,0.9649690079517589,-2.2474892587804187,2.3246112276179622,-2.00014733049839,-2.9162636721622666,-0.34223902651860605,-1.8201734263855551,-2.5009910046783874,-0.5956283870878157,-1.0326980687683418,-1.6356686498808202,2.573862508708305,-2.2151778694606716,-0.3853284923130592,-2.267922397406996,0.6196242903360984,-0.06149814554237737,0.6684331600320278,0.6582520189083776,-1.2566343973245582,-0.7742546561465108,0.5775667067885816,-2.633616265830887,0.13143416810722167,-3.920355788753067,-1.3966995993525322,0.45358584319432443,0.06093933648289559,0.43113390731458434,1.398052780626494,-2.9906509879883623,-0.19251665687775016,-0.38767785258385457,-2.6625232478191854,1.353915591239349,2.6307630445407204,0.8650924472366519,0.9303443994541882,-0.7408471262639401,1.111344144280142,-2.3777995758835626,-0.3863079570297954,1.6788201566451122,1.7597894141169743,2.0326432067134172,-0.7709575454992736,1.721131381714594,2.7345206791785235,0.4490616494536313,-0.9152841940175916,-3.6746779780509984,2.4448265892721532,0.34900850782573106,-1.6696341137856479,-1.712275903945973,2.3332940249865066,-2.3564187207698466,0.07306828538711652,-0.04553899771510024,-3.948822772500228,0.04885119023512859,0.18870991367626563,0.967850309391949,-2.9204132514030947,-1.6751761173512083,2.6774379316274493,1.6909268517570675,-2.056262960783356,-0.9136884640275977,-0.97074683862027,2.020148397282347,2.2786453137466984,-1.3660850139704601,-2.359697828196716,2.1222157451810655,0.32928392248126676,1.0923765868416593,2.918748323860302,-3.5640198389574422,-2.345274718294799,2.6209447169031943,-0.43970100192528977,0.8800415181644248,2.2985297799322852,-2.4366199057898963,1.5593835007809982,1.325821309867862,1.9975982993510333,-0.37532216305160726,-0.7391495915845427,1.8279297910552934,0.5274748636127226,0.2834569001261511,3.760040485965496,-1.9535130098414255,-0.9958546829515841,-2.0466676605548493,0.6918872497546297,0.2605478186342196,0.33457783686260356,3.3020871568554053,-1.5674658080761332,-2.135888619890787,0.18744795842848117,0.7006509208487885,0.48108250621600923,0.7228959345949663,-2.290236331026378,1.1740822093512393,-1.2471114507794303,-2.3494953237846223,0.2500729302062458,-1.0614840786029578,-2.450877028459327,-1.6989946040375266,-0.7962818628834422,1.6249109971756834,-2.3558719263127097,0.7130526176397731,-1.993713264169092,-0.2806518481006319,0.01353883914960844,1.7091705312536372,0.6033654785792805,1.6471324029002452,-0.7222656505363294,0.347690350863426,1.697255906827168,-0.5287504274610825,2.752728640909995,-1.309619790626339,-0.8875769157023656,-0.33343375012281523,-0.5309760299019259,2.871594901353123,-0.3736105558226068,-1.4209012426122258,-0.9462297654946094,-1.7989558713963174,0.8531738607070684,0.8698705848718244,-2.5955581599239563,2.276020514318338,0.8143138002740606,0.5664049613095754,0.35476956386956326,0.8661731968757203,-2.587826485421808,2.175212616663582,2.1311577149094645,-2.885583950813557,3.073592484653796,-1.522349897233093,1.1478059781317713,-0.1956655925135717,-1.3797235015902265,0.19422186319113988,0.12051201060910947,-0.6157823907687154,-0.6897847954900791,1.942253064057308,-2.1200066614402266,-1.6200214469660899,-1.384911317123183,1.7410248171526095,0.8069505184225017,2.3149653183292607,-1.6473381287051991,1.7406686453837377,1.261546451471382,-0.443048378826213,-0.016737893001882484,-0.8482931218812744,-0.5337871648045751,2.7492067118745345,-2.1597441241691713,0.7225142174834441,-1.7708618045818483,0.538414053584779,-0.23651043900435006,0.5646623668424073,0.11520492213470497,-0.6616453225632413,-0.7429137584297989,-1.55053602058617,-3.2845935028290447,-1.578036048014221,-1.5493427750313578,2.3017827587572772,-1.4205001474162742,-2.8166267106997194,2.1203808029401263,-0.3714922414170052,-0.24481075154954923,-2.0987049914894764,3.130101083576146,1.7990522936151299,-0.6493777431983168,0.3555609339093977,-0.4716122453998831,1.5585095492590095,-0.049004963260145126,-0.046933039657708085,1.9276739581172706,2.021160871976222,-1.421276817560951,-1.6748479327485837,3.689426562832035,-0.7118571895964858,-2.748927048763474,-1.2235130681845365,-3.58393428500711,2.077484074358828,0.921640554947279,1.0739961261530773,-3.117893826396606,-0.22008175785268594,1.7021585530414904,-2.876040773800723,-0.7348187361483656,0.8750512094219175,-0.3461944348716091,0.6189246474565332,-0.7157492670161208,-0.8784634825874366,-0.406298261176342,1.3420965915985608,2.9534430388821837,-0.7141951895751987,2.4878816895167186,-3.337713952862682,1.158009103389407,1.4385882415058437,-0.07745065246870582,-2.5283162760118403,4.4938914583661465,1.3921448111893948,0.943605641326823,-1.4367935394371365,0.45623362070964474,-1.3749568880387777,0.1016022970477816,0.5028495173144879,-0.33002932772579824,0.8920434629946447,0.871745991183683,-0.9603249887873971,-2.3388005776233793,-1.3629008035088068,4.037318561617601,-0.07341312330234231,3.5982571830333683,2.807903038871313,-1.0680689840711948,0.3838769811192796,-0.07761069989502944,-0.7558580048773977,-0.2766041329296364,1.856429808164127,1.3438255539671,0.07121877212717138,-0.7687927931953105,-1.728227821941256,-1.2638087546337244,3.4120310766246034,-2.800242409135122,-2.197562028261248,-0.8277818446329792,2.2965207635762113,1.6959358726121883],"xaxis":"x","y":[-2.117835625260438,2.6374805183797396,2.0817574592069352,0.8678323846564828,-0.8256415666525501,-2.20287480159703,-0.8838863867014464,1.7133258741336959,-0.5367420111513815,-0.5763263932459859,-0.14100296742507842,-0.8210677708536513,0.24489100850257506,0.7774746894278042,-2.0715021589436207,-0.6506285891895603,-0.10148318527909463,1.054370891087344,-1.3403937497309408,-1.2979893167316714,1.8708110197953576,1.4322157670865694,-0.1079590594355877,0.03207376416106723,0.9665889562770991,0.10378077810615298,-0.5159975569176383,-0.24547143943465993,1.7392302284063392,-0.6642829313353015,-2.019897805804091,-0.9511096641681739,4.449492113259569,-1.9218397476507605,0.10094808595624193,-0.7230263376567945,-0.4197841983717859,-1.0776062243139928,-0.8636829601493927,-1.6479299609022648,2.3688612473985984,-0.6644085878203236,1.2281981606909373,1.4359311295008097,-1.9077225429528535,-2.744774787679297,-0.07760524456470781,-1.6339909561700905,0.08486891940680714,-0.9691180006889315,-1.9318122716801438,0.23890256633704485,-0.2958214062767771,1.232893630107401,0.8877398876668859,1.5813667069798611,-1.6182768052637764,0.45573465066493174,0.3785487230692459,2.6353198580282253,-2.1406429185010927,-1.6551481278756524,-0.8245585544944057,-0.9425320612972801,-1.1653755490341144,-0.12022967447415314,-1.4423738424267172,2.021816571185515,-0.08411674612562445,1.0353538572229648,-1.2668273090173916,-0.19570847749484224,-1.3568193655078948,1.302990992652035,1.4068296466988968,1.424218671515576,0.3387461016407343,-2.1769679959490555,-1.8685399965880836,0.6121377217639826,-0.9635176489094083,-1.4446926715699864,-0.4462202934871046,0.21807436277563835,-1.0312818384276914,1.3668596715603418,-1.4318791551921202,-0.5410640165724439,-0.12230463594547364,0.07182136331054448,-0.22365757243988296,3.0894497074089666,-0.6880283846329974,-1.248039212094468,-0.38748377836168074,0.8382028570500037,-2.0748696384568985,1.2798640830184924,-1.0283472506738498,-1.0092010998537275,-0.2581287682095728,1.007572465288201,-1.7280608349469857,-1.1562678843521106,2.350451784079628,2.008466187797164,-2.060652460063816,0.7670308985998948,-0.9018300584633877,0.3400603578033474,2.335895596956087,0.39026974879255116,-1.5025920991927382,0.0448141630282786,-0.48238476869394226,-0.4351038966653088,-0.30753882293523405,1.2938718318547662,-0.7903096294491259,-2.6692266985602764,0.2729217454557427,0.2800001958279856,3.121556319622172,-0.6262654861139335,-0.21321384805043728,0.4704921904413123,-1.9637784324684235,-2.6427927621301652,2.3853420099502607,-1.0744780708555282,0.1534393728969211,2.2274722677710663,1.5733428511114742,2.0447105605278852,-0.10754718407923637,0.8348413508964652,-1.245288605326924,-1.4261508320467995,0.10511072156837209,2.107191496817899,0.9191364354691401,-0.1757846091186338,1.0218834788695283,0.6177607636269731,2.965802642561533,-1.3403960649793483,-1.2298007073904194,-0.6031836090050422,-2.9592288103402393,-0.9354532250419851,0.45267637526279497,-1.1083228886846108,1.9063266595389519,1.1900530626296766,2.372871103579495,0.06950656910043636,-1.6178119960752084,-1.1319569865947674,-0.050917013984156705,0.5235439297820894,-1.0538230500908548,-2.4802269343073347,1.771684499755396,0.5470624623368416,-0.9044442568020339,-1.525444875177304,1.2680097474197078,2.9041005196100356,-1.439337565277926,4.202082250419878,-1.2546688186806492,-1.2589151697023873,-2.118366617860451,1.1435164965262588,0.765552137635661,-0.5028170932468592,-1.9026579603134428,-1.8587012643214469,-0.6619354968564498,-0.00234942221511876,-1.1174247497615168,0.40316948686505616,-1.5710921122763746,0.7849113809648949,1.632190473547032,0.7566612308097803,-1.6036404382188205,0.8583401676252791,-0.07281293934761734,1.6905026894306576,-0.37474067477157785,0.40187246677320787,-0.19534346488054166,1.2863457004572634,0.8674054030399435,-0.8167774868702419,2.612235359468484,1.5537556094361658,0.5739860098701961,-0.2851602056563896,1.0771416954514026,1.6735349268526973,0.07604974351851257,-0.4377640574404969,-1.6951488304931022,0.1697979932760533,2.056958939666807,0.14149611911376198,-2.2339137990910793,-1.1190727851572166,-0.4578502200593289,0.6956033652703174,-2.498790307007362,0.3164511983187423,-0.5936981750227848,-1.7162549286830908,-0.038727795378912626,-1.03902748383711,-0.8575290204923892,-1.3496217625044482,-0.17141838847917437,2.2668437285916525,-0.4426832979569784,2.179827008453039,-0.8423333977326316,-0.9878896446122959,0.4149792679427418,-1.246626789986064,-0.5055973828476006,2.3018085974755067,1.6719077086522924,-1.2830666656129763,-1.5077250322450153,-1.143339518683885,-0.28120173998199505,-1.1912681012627913,0.019807690429932735,-0.30617302717231926,2.057486558076172,0.8932525104268858,1.1608691118729833,-0.23417726911224737,-0.6734829924435949,-1.6811151714306127,1.6717684373259745,2.2960770581137715,-1.347353791149597,0.0974894744797041,0.8589088907956691,-1.22091335850713,1.0449048048992278,-0.9408368923492079,1.5557391263956466,-0.3184704086627701,-0.2764690071610984,-2.348831037006406,4.187354369673707,-1.0256433512667118,1.7866978922203904,-0.434235338550515,-0.9478133968412221,-1.6633610517225597,0.1623482006922865,-0.4796849774694916,-1.0504209680118923,0.1685351505312036,1.3372974176349945,-1.9698268297268835,0.7780396170939742,-0.5241419363597722,-0.4931687079827439,-1.8597035560946213,-3.4949479683729163,-0.288130623851488,-0.9311659953125075,0.30431194986715093,1.9335206637418798,1.7603502875302246,1.7812626597254815,-1.3430788656299664,-1.3385238823128545,2.0140628855996705,2.4161822414521295,-1.4054583863970942,2.291828569534892,-0.3418048200236992,-0.22718499279124268,1.6526142758868232,1.324547224694638,-0.3822294644502806,-1.0659541166478714,0.022108107469022656,0.061714110409459275,0.561167666968606,-0.9405994054147031,-2.2541666978322716,-1.2528720233766124,-0.8757452348201664,-1.3582836949457873,-0.5176658909419131,0.04045661192885362,-0.3845889331417863,2.064522904220503,-1.885647000605997,-2.1806519778284597,-2.132130022554577,-1.7558526701614645,0.12668471104156895,1.574789056253467,-0.9905150316658609,0.8694229671884592,-0.0003488202539584455,2.18229977692409,-0.7888184766374479,0.8877071306347667,-2.151277737904922,-1.8595899742196047,2.1303999352278753,-0.964466765530166,-0.4536616031347079,0.9498744468460978,-1.2899163890113012,1.0613367400849387,0.14834514341394842,4.525761588587115,1.2397939273040248,0.23469160105550663,-0.7553302397481605,-0.45206474260607094,0.5233293505329164,0.5739324061824883,-1.5274693587397514,-1.1903615053111387,-0.35695930204207155,1.714975621502243,1.1492166141171438,-2.3904080763014672,1.5157029737806156,-0.876142947331511,1.988760981024132,-1.9411695649518341,0.23596109683391156,-1.1555745238564212,-0.6103762860115666,2.147377670095176,2.1409179542183043,-0.6061550095392365,2.764744033948395,-0.2457740793657135,-1.750588412428913,1.776369706071532,-1.3827675074684702,0.4822038005799782,-0.7728448682728588,0.20958461396412376,0.3093775773198242,0.18338082760556287,-1.2758804936714476,-0.7637032074868028,-1.5651438244371954,-0.8946438551459469,0.23777201352509728,1.021953071220754,-0.726236292388496,0.2685724965407178,-1.21816885137361,1.828339322424374,-1.0711545548844617,1.6903174862993722,-0.5272097410873802,0.7642423827766315,-1.842110880216659,-0.8314569347004664,0.48148117111948036,1.3971530954805629,-0.8489551365765103,-1.5808113818085585,-0.5895536885985684,0.6732148173400642,-2.6572714117431104,0.7995420670582136,-0.7386187521623693,1.0633074923761539,-1.259863003017986,-1.0414479837514699,1.9634941296212505,0.641321187897695,-1.293651723667102,-0.5043244675980856,-3.3216340467907095,-0.9554055399111953,-0.887894952306651,-0.27580301447355077,-1.0211090265558929,1.3592591620501369,1.1958178005557623,-0.5409040032497568,-2.1806641138397658,0.8694218614260275,2.0657987292198094,-1.9067455673351899,0.5668877227002771,0.6924090055854205,0.15351766339732978,0.21635698498977388,-3.160972708426993,3.5519043844501708,3.230593616764628,1.1096120409478225,-0.3086274025485534,1.368768967987457,-1.0305385507873104,1.7482437184842505,-2.3164338691383715,-1.069047815368947,-1.7838688771316766,-1.0667905536668172,-1.1984955697967936,0.7524298394786534,1.1445189855417988,-0.6221553826592829,1.1141028065570846,-0.6556176381414399,0.05363340974402907,-0.6945975998276553,-0.17954012584026466,-1.811625438774276,-1.5834626328755597,-1.3506329460661908,-1.6124553792946246,-2.6456075461769637,-1.3263184816866913,0.18784197566238287,-1.220984724457,-1.353998517170821,-0.38027529663133997,-1.0800909704011623,0.43502777037541907,1.9456713314441818,1.9782384198287841,0.22009980527925066,-1.598609688306898,1.213668081731636,-0.8976162406807747,-1.2556467026071942,-0.5985827620350066,0.11776329915749487,-0.49234744654949003,2.1211315672354845,0.8498422149201206,-1.2063338529163206,1.8371325746998317,3.9063837716000407,-1.352919500724829,0.30268880069812276,4.330118826227449,0.7363579191241926,0.976052576403606,2.897406571786866,-0.07973256858041108,-1.2015309744643865,0.8876183023821471,1.183505883169008,0.2707113972978949,2.7807567599507674,-0.6493708621471276,1.4317996872439473,0.5702738159135924,-1.0511206300171918,0.243752465662483,-0.8341242719533964,-0.4908899493421,-0.7773171084945221,1.2883649426811965,-0.9451944096610726,-1.3897603305310005,-0.3650144932033071,0.8307611123961803,0.47497169168707065,0.8296733613880043,-1.6698188557163927,1.5271135111903786,0.6984442798859091,1.675820540386129,0.1379522567444665,-1.465996115217891,0.2476308007527944,1.495116629771836,-0.43524236075823475,1.1982416601522736,2.4622605596525102,-0.763054182409572,0.9381961841922255,0.2107161323862152,-1.0274938696774345,-0.5208190799129554,1.8059759593994889,1.0263419223750143,-1.902220087432681,-1.59073228681364,-0.2740599123512587,0.3036284132899506,-0.6304611896264082,0.8695144699302556,1.9260950358535176,2.1532732630868328,-1.5362586566568883,0.3919288366047056,2.02815597234729,-1.1162373989308478,-0.8470075617024724,-1.3851012365182909,-2.442738581604336,-0.2983314058558699,0.6539085850245144,-0.7008449140466323,2.145077634587815,0.5203276318781456,0.5032401663797993,1.9649972490615137,2.0367846142194144,0.8631192738500363,-1.6618557276970871,-1.200584717440549,0.008513643087835199,-1.1906085894778293,0.243312605859416,-0.8210843503390873,3.2331767730913152,-0.36443968684934847,-1.487324122771366,1.537028743368544,1.2760483022792934,2.0032022845739808,1.0303500131086998,0.01120200325304964,-0.8500339921461725,-0.3066885433362064,0.12336822291197477,0.1828252442843292,-1.1248654758181054,-0.5478730078425162,1.201545501400197,-1.299798438179693,-0.5356117637445081,-2.342751057278589,-0.23313215552838293,-1.1174608359344735,-1.7754056221200734,1.2369740875286173,-1.4855815016050036,1.7962557989592107,0.9497540414432133,-2.4591248120680915,-1.355945175915325,0.8491385641972633,0.7816102906153737,0.04924577252188687,1.210713567825676,0.2852974234007016,1.1939824663921161,0.1443441305079198,-0.5727570901504395,0.024218269996689055,0.1349746308009003,1.9810051169553007,1.507645230307703,-0.4627915043443885,-1.2540039323706662,-0.8922859775190759,-1.281938979050085,2.292156318518584,-1.8753463320480583,-1.5017719386138806,-0.8766962424305081,1.444734915150366,-0.05338749294596788,-2.685447182730932,-0.7714607456771123,-0.758710371959265,-2.4705300416008,1.0070030398451422,-1.471183110873517,-0.5072090021110467,0.838242491538492,-2.5371224876635976,1.0323396819236055,-2.909215720750511,-1.8603262387063364,0.801635688751099,-1.9406754611437218,-1.7158740471010625,-0.431293186866553,1.0398322365032318,1.496938228313773,1.9387367724357043,0.6718390375656049,2.120643631866122,-1.7238434921502477,0.12190391704060981,1.7634149424288792,-1.7807360770488625,-1.611983080669735,-1.562510892424452,-0.21029767604940325,-2.2188277204315687,-1.9097652639418259,-1.2072222987551864,0.307471014895838,0.8065946297137303,-0.4559478994541088,-0.5720602188394238,-2.139079094104243,-1.398587618605471,-2.3134309093587957,-1.2481042976486747,0.1882304543005129,0.6178653148755918,-1.119070399147508,-1.64003965155176,0.711638049694797,-1.9945097627167527,-1.5704838849340343,-0.700726555457581,-1.7301483365392354,1.1043285190755687,1.6622860160904556,-0.9113872949253264,3.4555078604243734,-0.4450963577612342,1.4908229717942652,1.1387963362351334,1.0331585151750324,2.1939426727737024,-1.4761379726907418,-0.21875631812312218,-0.4852594454243618,-2.083661400333977,2.039536553579617,0.3135211496923264,1.543959972246428,-0.7065641421808734,-1.5220880370752412,-1.9189325097175147,3.3853563844044468,1.0195539877544264,-1.0396975958254815,-0.7753990499620538,-0.20190140867778827,-0.19693826901646277,0.20481844519118791,1.303649846187072,2.4554879376027654,2.1741310454941574,-1.0872386528506817,3.122362231181462,-0.10429258343936956,1.4305493502005149,-1.1525261852374842,-0.760458072285229,1.230587421291551,0.16722324811428865,2.857473764849384,0.43634239085121335,0.5045469427409874,-1.0619058606072393,-0.24496031557127063,1.5107646976494176,-1.7436614824537915,-1.664938314986278,-1.3093226587032814,-0.5799258859458686,1.7323106222595066,-1.0255423515864897,0.19288420224187078,-0.14724840324981558,-2.314926559068753,1.6622709557216055,-1.6649715933214688,-1.5963492608976786,0.24077834157114678,1.5675125856194927,-0.5916405570535946,-1.0529221083361977,0.5774996969938594,2.221029261112678,-1.7613919698137857,0.4517145357853192,-2.2434766883198836,0.8279857504169648,0.31315193021211307,1.7480365528445974,-0.6952256473968244,1.3952897861823919,0.16209645007329068,2.883103060872843,-0.31619188845443447,1.379048264499083,-0.029313274309159032,2.8308783314724697,-0.9184663830377975,0.05703088693861601,1.3680359196656733,0.5161530543576224,-0.7637534242076511,1.4633146940983381,1.0431744551073352,0.214786951300305,-1.194152948102524,-2.0080920650709353,-1.4730658719196599,2.273630297425304,0.9945613473794357,-0.8155337657948466,-0.9076652430451583,-1.1563006967933427,-0.13774392411528155,-2.5307154737509845,-1.8011656325761283,1.5561067556184038,-1.8111730126897916,-0.9448840129456206,-0.018684198174386293,0.538283995496587,0.7158052446447496,3.5869699273308226,-0.7515741358613031,1.9012666712566384,-1.7490122934020513,-1.0621962667653526,0.9704375688162775,2.589323637120773,-1.7200225584874609,0.031089460507960114,-0.3330077555588193,0.45454500086381333,-0.4870446563830747,2.92439143904811,-0.1993979850563512,0.18558379051368665,-0.6815056668723738,1.8725721501261248,1.0718491378937227,0.5998481468618133,0.4530237859677773,-0.30861551546205124,-0.7723863242734756,0.9128288267640086,-0.11417085861496105,1.4845858813563464,-1.4991457763018332,-1.1812801718439492,-0.022845499038471245,-0.716977752055634,2.0055435832156037,1.2635078176028056,1.8931587611336678,-1.0883237847820457,1.720210400522081,1.213073030509218,-1.1304744383137066,2.4387255905696086,-0.8501448024409213,-0.8786010513527511,0.15969077997103706,-0.7435984262503856,-1.31312841868062,1.5270737211913983,-0.8538693816829859,2.027855421397647,2.110484837394495,1.5036926643792448,-1.1697152652481666,0.30784281298285854,-0.8223438456299015,-0.3521354125680498,-0.37469258551329526,-1.6156179385388532,1.4756724091839661,2.623681142027416,0.1897584602952288,-1.1329318601878997,-0.9859313902331223,2.4216292664719195,-0.7082577189133107,0.6079455786432799,0.1444433830601703,1.8457313470521644,0.8136451391679229,0.4046129870143698,-0.6389265329634992,2.5808052788829454,-1.001953171574409,0.8654569329531947,-1.4352999099227937,-1.3850128465347382,-0.7606698557476083,3.567138616608511,-0.6887428125662531,1.4276143031212791,-0.336195801412836,0.2678716238960692,0.05654717021429223,1.1128334359911543,-1.275491860111236,-0.779447474047569,0.29974842421741404,0.7672461765200931,1.3474738492305258,-1.875070072632716,-0.5079827581850995,-0.3020084732372274,0.12044486046192023,-1.6631062543545123,-1.0911443098819655,2.202761873011088,-2.3259421935297095,0.7166383223562751,3.9193180625238178,-1.8053895720318671,-1.1666668828077948,-0.31335835366258574,0.062469924093753904,0.10420986484414854,-0.21296446610056524,0.0548692235583374,-0.40448712116318497,0.012748488767483662,0.38452360800837787,-3.1669190417018287,1.2094975230998943,-1.759213591352879,-1.002723402576828,-0.32950265087483793,1.8888392126379738,-0.6463569727449433,0.7630827225627498,0.6137571133740851,-1.2273460045367255,1.0388532799346324,-0.6569031532031289,-0.5220625330937537,2.1200110937160086,-0.5131387509563737,1.9055222073604299,1.214211087573769,2.8197649807040164,-0.7236446394817019,-1.5890893672777826,-1.9656835185809036,2.9969848792475653,-0.36409410296397793,3.6608595599178892,0.7816552844076344,0.00655535822884778,-0.5348703915651837,-0.731147225963179,-1.9714500213533455,-0.9008826774460342,-0.2691153218367468,-1.822002105493286,1.825319590077738,2.638777118141122,-1.6697559957430597,-2.1130051976012543,0.2407793056046274,-1.1691787685829362,0.5371572089492038,1.2271199951075071,-1.058382654115111,-1.3620565785460923,-1.2950377711480068,-0.43581365323401505,-1.0946154320434724,-1.2950464074760095,-0.6614101899972386,2.23812390427438,-1.9925061515169584,-1.5397576930371155,0.661323798601665,1.309220654361387,-0.9809928534065149,0.21161505084639104,1.3447342995999345,1.3730602679832007,-1.301531126452004,2.4681447684529076,-2.0190383951690944,-0.9376142254566685,1.2511363606479684,-0.5129924086334301,1.235807211553304,0.566277275687536,-1.2542008219005378,1.057415237406642,0.3589803359315168,-2.383193247621251,2.1269736096991965,2.093089240808521,-0.5342165466326667,1.62860503915078,-0.028210763606046338,-1.4599012422009578,-1.0957967922440142,-0.8838237553180212,4.12656120099208,-1.8341134083400987,-1.3526861865013124,-1.4495629060266604,1.6398002522452217,-0.18611647840732026,-0.21528490341890352,2.3280672877732647,0.26742586701297855,1.3838849558415314,-0.18445267448079009,2.9747516959601414,0.5241026629422519,-1.9941532847085492,-1.2459025469535525,1.0254859350297532,-0.5331987823498663,-0.8581748741050619,0.8750350873353581,-1.2667738987368542,0.6702710062912948,-1.1009323333187009,2.1053765968682603,-1.7184749349991735,-1.4813913854832974,0.642768698447891,-0.5498873213337203,-2.24693875669081,0.7100363186327703,0.6700321073659622,0.2745926203569521,-0.45068953565487724,-0.8318815347786304,-1.4801614432758883,4.768744723313961,-1.7698102459225158,-0.5989621686522908,-0.7715764575804316,1.4189072452064715,-0.6446390631171539,-2.6378008082280817,0.8598738860113019,0.13857861019300913,-1.1594556665143054,0.2141496116336467,-0.36222624834447803,0.06532764686138968,-1.0735164840460294,-1.0928755423175147,-0.1600920508844549,1.217309739288217,-0.6116880246267588,-0.8711860046462981,0.35879068271116804,-0.25563918844920375,-1.1827685928538392,0.34870272059021756,0.0386624520130643,-1.4552886276846768,-1.8584455029555726,3.768520061393291,1.4377714922086873,2.869579003446029,2.911406218825272,-0.3188454291613816,1.6026686195249384,1.0778669146467832,-1.2539692498952664,-1.610380533818446,-0.6387169049171084,-1.6576343976599153,-1.4142363014572055,0.9975428936243593,-0.9362777547765573,-1.0531323031314326,-0.5760274338926173,0.05985653072774525,-1.195473490981511,1.803053293677166,-0.6356269304450354,-0.08314637553673176,-1.2623533489752627,2.858231838888545,-0.3309433025297372,0.23294421178280908,-2.8598250985889795,-0.6296307586505483,-0.4340204495296142,-0.7166913828003241,0.16101095331403056,-0.757666003563072,-2.2258835394402126,1.5004256948731929,0.3330105720609508,0.43232079506554355,0.41336866609316814,-1.4715706404546032,-0.6782002193562479,-1.0511622555337288,0.2590010566669064,-2.0218732949799914,-0.15317544649305806,-0.1683383790577253,-0.22621786061950125,1.6321415825864407,1.8309797724513464,-1.0590709093418615,1.86010999237547,-0.3260512268387553,1.412726775612529,0.5387932511127768,0.30586942555356955,-0.2809034434398052,0.3008599326375695,0.3125692476610866,-1.3093743098242074,-0.2024189233944412,-0.4064562143680327,-1.591059262030089,1.7068284885954423,-1.9593311816777552,-1.3280479894507855,2.3112515054601395,3.646121186156972,1.0786286163562493,0.28865787132614074,0.03500180137915209,-0.2197413103576229,-2.481973550772538,-1.394539854575165,-0.0783348602832767,-0.14483854310732802,-0.21829937263151136,0.2801054834920921,1.3879079176762088,0.027882648886764436,-1.3344618863231232,0.08400822594851595,1.946081989050276,0.3891675581143313,-1.726588638919184,0.7167811307896554,1.225437873541601,-2.2632934898185577,2.7648555504441235,0.27135743298542425,0.20664954220976564,-0.528600575400762,1.9239882574911737,2.0242927918242213,1.389668690656788,-1.4790249478812312,-0.06749707871861592,0.7078006831172122,-0.4689880566545906,0.9425521856511185,1.5228152727112196,-0.4625732415860072,0.10417880354824621,-1.8742821720177416,-0.927168987480325,-0.006494017247021361,2.115105062593304,-1.459263645477565,-1.9691306925105,1.0365468929209456,0.00935085982055384,1.1737453874095933,0.1405819566110058,-1.909325674539483,1.2377739336038416,-0.33771659610784194,-0.29544930232299266,2.2577189663644837,1.2410881266066884,-0.4723790154919305,-1.582809814959937,-0.8678869108098457,1.470792786875525,-0.79948948385218,-0.9220306587994256,2.906664849528179,0.29011373159219367,-0.7788107302031593,0.6089558367633581,0.15145579654843658,-0.6761871571627375,-0.6386035605603557,1.2687312993662951,1.3140193794710548,2.5285902451522597,1.6191412720062432,-0.07528209829198317,-0.09226857496249158,-0.3419784671290925,1.5002276595322594,-0.21835818982279045,2.0059521221200707,-0.1813811714559597,-1.8974744936013384,0.8447626887665107,1.8145163978488723,-0.632997540391212,-0.6183285723342321,-0.23698501707547892,-0.12870826341323532,0.2236146285525569,-1.237124286695704,0.9112294307869901,1.8540718919160033,1.6276027438901357,2.6670680550466703,-1.2873599776006879,0.2817801371531442,-1.8357680016146578,0.8421382481680514,-0.7525853264672924,-1.632629953916456,0.6472720623781343,0.10473605203199991,-0.06632609137087823,1.8875529101913302,1.05070487319947,-1.0930399764492953,-2.122786993137899,0.39328558950204745,-1.21414283389765,-1.1374613606393233,-0.9607997631110565,-1.7866758134100191,1.8079376047255606,-1.1931171953250397,-1.1586705559477757,-0.45191297559555915,-0.19361571151574503,1.2631980748482607,0.9949731733843129,-0.3104627158732724,-0.42379525583764305,-1.2809745492607962,3.0404278156722104,-1.1173875628543257,-0.457932449764548,1.339621597153129,-1.9710484124589376,-1.617787734881379,3.3989317817781064,0.5569294046063596,-1.4488939314070939,-1.2534134515857176,-0.06883297753908284,1.131567849410191,1.15284797527107,-0.8282449900489574,2.411350753348577,0.6650232494509025,-0.6894203139050336,3.0911908431698953,-1.0833835640228489,0.11243574338596923,-1.161010091464826,0.5080620385065039,2.824631845538465,-0.8441859565487857,1.407260759517706,-1.1152805135258277,-0.8018942464668263,0.39675476711488705,2.900566934557042,2.467098074619604,0.7992650986762762,-1.26028390622059,-0.8436137669965003,0.6868322745390388,1.0212285291174525,-1.9892450588983883,1.7275615257895884,-0.2036370412789848,-1.326077728701055,2.111731563055973,-1.1151018275861109,-1.921112193800125,-0.4205098850015448,-0.030072381240439938,-0.22829828656544418,2.9352387183121516,-1.3883126231581366,1.1713461585388516,1.2443719732238236,-1.4610976543771037,0.6014616578104306,-1.4108878442373536,-0.08412302448882582,0.4351553602349951,-0.4312199492204957,0.8503142984787149,-1.708649093174888,-0.6765210593541627,-0.16091228458582002,-2.2334634907013053,-1.9373054995327996,0.7152999797126857,1.1810827296426154,-1.373485557253415,-0.542394141074515,2.552840378748967,-0.5801642969926959,-0.025395276762824385,1.1842957506080425,0.6015247426889485,-1.5242148487092624,3.152547601220999,-2.416331351976384,-0.6988355116666718,0.28034884388028003,0.5102335939140838,-0.6716536173517443,-0.5114298555438296,0.7687854231535758,-0.8488834756050185,-0.032270168643030075,-2.03655200679283,-1.1231340023359697,0.46622097366393717,-0.2798528280560586,-1.6122260304431877,-0.025710153969699903,-1.1498257414507027,-0.10496896654803392,-0.03950587212826747,-1.1695831580020795,-0.3149865467373594,-1.0781554659558035,-0.1347085537119132,0.9901733630594207,1.6402173884441351,0.1011944549971584,-0.1759142496475069,0.6352176677140486,0.013832368481528955,-0.4393738497656427,-0.4461807729733309,-0.6745914519176613,-0.0377202385725699,-0.5128535292153116,-1.0628754691360394,-2.0093849732198996,0.6992424059348754,1.0968451987534806,-0.5551328881574196,1.1235444629560078,-1.1435900754393276,-1.9828594929689312,1.1714350128983528,1.9902335985938742,0.7280729302938753,-0.9575257403801969,-0.9889716316822844,-1.6087988252336896,-0.3048241692361642,-2.4319448290173975,1.7416723040742828,0.46670299190004894,2.8391140346863715,-0.3094530947355256,2.847250292558223,3.8958415095842898,1.4278125911929436,0.15473167257746392,-0.47113328478416755,-1.222362262166138,-0.49558816687129176,1.2415152467853532,1.11417605742487,0.8955848771591981,-0.9388530343452524,-0.41967952161941163,-1.157908539910058,1.0625965694091015,0.04128498497560163,4.225623074923661,0.15762027559738515,0.10645075215856838,0.9027343194017505,-1.5587210628260708,3.065857854663051,-0.6571954235661083,-0.5820876336744017,0.5509126343637921,1.4207962983472238,1.5376893458738594,-1.5059915258785979,-2.2281528832417155,3.2650546006473977,2.547021841520236,-0.9636387388804044,-2.069122220232448,1.6397656318071792,-0.3179128476853032,0.6540840584499326,-0.336086825948816,-0.2514297026381026,-2.1461913804599737,2.076381246758824,-1.2031466438336647,1.4263684434526855,-1.5945106102547684,-2.0267034841527436,0.8803263830881609,0.560268467483579,2.0902064748278715,2.326002022574886,1.2590004730117925,-0.705689787400527,-0.6554713033464401,0.5308110458450392,-1.659037503818453,2.039174279013851,-1.1131742239534526,-0.6107032447321136,-1.0225677648434004,2.291000387454473,1.0921560152235563,-0.4640581155049391,-0.4353789285366641,-1.1813505655392116,-0.3128483104697993,0.41118330602598735,1.4376407782180542,0.5207164445393766,0.7857650741141683,4.186119769873715,-0.6089463305633378,-1.0269415332003766,1.6728610508447024,-0.47733761684960796,-0.22815095402962696,1.2880578114371128,1.5752330998723834,0.21154696126298628,-1.1704609014231078,-1.4041521002464414,-1.6400054098631345,-1.1923448470386053,-0.9899106480729448,-0.5230104251727826,-2.0974816355593084,0.5087421826207692,-1.4241914323485991,-0.24233119459030078,0.03122460320208919,0.6016646451413651,0.9896671405090105,0.3780106254948021,0.007244718124502495,-0.74999129789934,-0.4300120836330484,-1.0966252894354576,-0.6144784620548748,1.3474594938261935,0.9775384423983858,1.2366524928610656,-0.4176497899208194,-2.180896185730772,0.3144562268278225,-1.308192689318346,-1.6641326005662351,0.7606674101843169,-0.36348501288742485,-0.7502302636988876,-1.71034932957787,-0.06209973675356145,1.4279565178819127,2.1405363238290778,1.4318901674289877,0.1487926983443968,-1.1446755626791598,1.9888182001025188,2.035510881893084,-0.2349862390300003,-1.3790364219625706,2.227783299051391,-1.356160537910634,1.6423513364204314,1.3064561279819575,1.343927744706012,-1.3619918477199364,-0.8470795803310754,0.4984908783384673,-0.9265532331706197,-0.47320562188274357,-1.467543002392147,3.709742175572978,0.3663239895973041,0.9317741584754785,-0.48842475167875254,0.4762687014756067,-1.8022978357899757,2.9227679886663425,-0.5566758438165024,1.0894964844691144,-1.6959337564918766,-0.3724333355041613,0.8330077621312926,0.012307384652814736,0.510423876284223,0.3389150966878321,-0.3949054431037418,2.436853846829939,1.2191890577687172,3.2310605735555633,-0.7780859114187335,-0.19620848665640786,-2.069551588642284,0.8900046635003069,0.10195301710959936,0.24339837319557275,-1.5087247342532257,0.6005880258412775,1.0948678209760565,-0.27280278492257737,2.1119995224614887,-0.6542828346270578,-0.10451995134712729,2.3174878777717676,-0.6013917096751342,-0.35916630315852904,-0.8017555708924624,-2.235259895923472,1.126954343102563,2.1354478525984555,0.6817052534761779,-2.2505572705957935,-1.0492047705044238,1.8197429062193378,-1.0592909626512803,0.5471402412670868,0.22439345690040097,1.5856926923305026,-0.22108078809411316,0.19255968330867118,0.4405625664680898,2.1662524156607734,0.17260510812373342,0.5879505791160983,1.6523647111164497,-0.6035450814745782,0.5990203752485609,0.5455923943629226,1.331453540191885,1.6657968742115246,-1.2769107438167153,-2.439929076424164,-1.183315865807681,0.6581332129057622,-0.4430180976813375,-0.31673256114197745,1.0673096695924937,2.4209216538985294,-1.730304060350355,-0.8359279167625546,-0.799587879311708,0.43030167901983024,0.7218025736981071,1.3634999111537849,-3.590031500882397,1.6846728320499487,-1.2904191562646483,0.4120257945045631,0.8889933437913718,0.9661219653226905,-1.397620839914463,2.1935308442823223,-0.2657325872561234,0.635115724249141,0.9527347497921815,-0.6203938229576053,-0.4287767969738713,-0.38524778001900667,-0.8369470908160646,0.8415361773605234,-2.3235207962219584,-1.5521061963695246,-0.9509961375653613,0.6639304105424679,0.7594279465187252,1.7966756800702097,-1.024146625904265,0.7387161622922069,-0.5568615528174018,1.37950156041014,2.7591303500546926,0.5767273186096644,0.4847007737932203,0.7122654082083998,1.8374328529507888,-1.2587014399822918,1.2178476283267128,0.6449984827865314,1.1193567357509568,-0.2607165950029626,1.5068203260906825,0.6835675872999382,0.27254770521423555,-0.3631592197764675,0.9901135349167905,-1.3039522271340085,-2.630080508477253,-1.0218207832065314,-0.42824685380723604,1.47293232834496,-1.4991497851875824,0.21176076940937233,-1.947262261663488,-1.2451081444303145,-0.28599330284460345,-1.2243643631006575,0.7244752790868122,0.2267868926161639,-1.5478106627240986,-1.1653532189416078,1.597829016841052,-0.3660558648004187,0.10154200024998727,-0.21756039743023958,-1.949076131959351,-2.0296653100268376,0.2726396296216699,2.511860797362848,0.30846387696584127,-1.4612700385297384,2.057752843230327,-1.4733684509730842,3.7587796093867922,0.6469047538237883,1.648456735451219,-1.1442096229197727,-2.2597095319332277,-1.320412831100108,2.748550144938191,-1.0255810386206194,-1.3108778179601515,0.02851330155429196,-1.3817454400485993,-3.0400499141912767,-0.29232000827501436,0.3198356838101199,2.6376558279801157,-0.5747258237805263,-0.7594748845233426,4.444166444760286,-0.4258129817321744,-1.4793596583965785,-0.7401514662637428,0.04292887368997068,0.8211716859867468,-0.4975104674974961,-0.7661848228725616,-1.5626179163630889,-1.8158302885347892,0.7723225395622242,0.9143692730282285,-1.5769007453232584,-2.146192595126611,-1.119852699418588,0.31500185825290394,-0.1944314335812664,0.08086266996023236,4.942169416158298,-0.5024655781192222,-0.28966258102405446,-0.6899967182235492,0.2832152336199149,0.5679064769064306,0.5465434698324869,-0.18035198987527123,-0.02922891797608241,0.4281259229038051,1.4215484038385908,-1.2806913227120003,-1.0060856967550493,-0.8451114412307082,0.11162759001044509,-0.17518542057663744,-1.5719825721522367,2.0157164072256935,-1.503028625366547,0.6779883752859922,-0.8147368772534221,-0.6900395862681804,-1.337316027315171,-0.7984036672557503,0.1504504883849071,-1.3452509886706379,-0.2425030457412968,-1.246303104223998,-0.14738375612875956,-0.8620801197484959,0.17469379813612004,1.3200073488422783,1.0181283848792255,-0.8718504106199805,-0.14462213425246986,-0.9108834737589803,-1.4358941917222354,1.7354922888292372,0.5204829038153628,-0.18999236672496073,-0.4407283356041213,0.6803959722186851,-0.10816353277391554,1.4871626487114464,2.6414599242746304,-1.0276356054218012,-0.40811917335155007,-0.20764134529125522,-0.3296848960867526,0.1029412852936124,0.6128867233047091,-1.4818077584109752,2.0179268326546915,2.0631769509261866,-0.9342272642463174,1.3168847242501656,-1.5004460105586592,2.184569787539712,2.374263243584174,-0.20371947007551644,3.6556483205036665,-0.9278379118071419,-1.7156235856672866,-1.2535892316569568,2.2738142117801394,1.3301495657446334,0.1619903238676646,-1.8960497475312312,-0.7877875908009936,-0.3852719867166947,0.6986111005526211,0.709362755052086,-1.168516401008464,-0.7606433655677133,-0.9517843007463548,0.4142277254207778,-2.274682182657447,-1.7245348009971129,-1.3659292618641332,0.19200652159753762,-1.82701181319178,0.1839894747767836,0.40244103105120455,0.6996513665119061,-0.6241658816005727,-1.691330232466151,0.43696282759979627,0.9991370800473963,-0.39517464338941616,1.020773104486086,-0.9790142556756569,0.4052325457145545,-0.6454572054512184,-1.3562299949337397,-1.160034625465847,-2.476354897388267,0.3645738100242982,-0.2528317076272505,-0.43900096322425025,-0.724593130143927,1.8237138197377727,-1.5873109239749694,-1.480983962138938,-0.8711022013342214,-0.17465913612820355,1.8588288279550422,0.8806647065755975,-0.0037383722777237912,0.03543362608538085,-0.714259085475662,-0.5224175228774985,0.7818279811120362,-0.444298982658114,0.09424128838496594,1.3843549025015716,0.294826526534331,0.4873163266285352,-0.046961477573613984,-0.7532052125133534,-2.1915317516626907,1.8093197547562363,-1.902955243126761,-1.2969509482174548,0.24694041706789732,-0.5382408327883821,0.7846901821081136,0.1336792971893974,0.46283991276205166,-0.445208711864818,0.38639850951847937,0.40051092420756385,-0.9597343642949002,-1.9482410075711154,0.7538677383770748,-1.4301361735497364,-1.314905793083768,0.0826048920071036,-0.1509833981755424,-0.7885823348337478,-0.07341845799177969,-0.9768091529753585,2.528110063905746,-0.7493167816717928,1.2624934435715198,1.8498412502068842,2.078257120919258,2.822901359724957,-1.5602025555574943,1.3562731162134105,2.0077643767617497,2.88758773921507,-0.9981233238301143,-0.44643241217136226,0.13085224130061546,2.1238581417626214,-1.6244043228491598,-0.6714629159094279,0.31440922589572257,-0.7294577409132809,0.40203547092542746,-0.4839187282812294,-1.9370783521888795,2.7350545804872217,-1.3195350439855749,-3.5847633499838345,-0.06968394927991489,0.7872213188244047,-0.8590809861972736,-1.10853470452547,1.5327846240550895,1.5356114312102997,-1.3396437779147894,-1.9036828570961142,-0.9721131501224569,-0.0599505270796403,-1.8674226852470166,-0.6154845700198837,-1.8187260724464642,1.3831743375789243,-0.20607777398929053,0.8548236956606107,-1.057714166648116,-1.743897362895781,0.6532227886406867,-1.0359260886819615,-0.06347593626381798,1.60072306292556,-0.9318585028760279,1.3329263218290273,2.5193048723420466,-1.1254933215296674,-2.7707902936924627,-1.3741134745026278,1.148767692851021,-1.503239184778849,0.5631366619600336,-0.8618220206801589,2.7593856273819393,2.4517540571066623,0.36256260775462323,2.9461250219421156,1.7942007516745975,-0.38490819828741507,1.2783301373101892,0.23754978255217374,1.1434513721494526,-0.8901404454083813,0.2391524655078845,-2.0910367167395503,2.271781113564855,-1.0206424181100389,-0.5904693085395429,-0.41825329350865986,1.4613548300043455,1.157888271888356,2.5466211079085634,0.6922555591874976,-2.5607352130661956,-0.25163792620999975,-1.6655515188754568,1.879739439452047,3.5696970803475674,-0.9102594509119086,0.9778978198578615,0.16939071148458323,0.2111436315603561,-0.9904866876475795,-0.06918544448820027,0.19354715183730167,1.2376675088466378,0.12707620574110204,1.7604065972131344,2.000749445771851,-0.6388168273487125,-2.4871754981045067,-0.603898105901756,0.03033005206004752,-0.5222420045790089,2.962451478869562,-0.04444926340796668,-0.4833476853449577,-2.292140874047852,-0.23406123140531598,-0.07489754258714061,0.1681602071547056,-1.1380181782366579,0.5788150519912626,-0.49170119181555844,3.5525907661916234,-1.2003460019662509,-0.43758867471452173,-0.87714039418639,-1.178367582079932,-0.33609854721994825,-0.565437857736752,1.245196180584129,-2.0556433551701154,1.2363417276526425,0.36036402897111597,-2.1215919053232195,-2.174817773216305,0.3336468077156163,0.29875893759233413,-0.16961161696822058,-1.5109831915558,0.8584899791517243,0.04526099559746861,0.14148320796905028,-1.3830990007101378,-0.08143817439623098,-0.4202557356668379,1.6316694763549418,0.41708341742782334,-2.774376455067509,1.9732738377509658,3.02085743738769,-2.1585847906742677,0.8906654631904558,-0.5683514494238492,-1.3442354281236049,2.1600224216708193,-2.075834320012906,-1.082022414987726,-0.5193644753330704,-1.6511024874722233,-0.7286394458244686,1.3592896420625753,-0.12580988851577732,-0.7254086885499412,-0.3662325905198958,-0.8886077477674507,0.008602322743353638,1.276550277837892,-0.9554169428112445,-0.8333832658263427,-0.4913960894194208,-0.920592884879128,0.7489074855417732,0.2719407177218483,-0.7900434890231985,-1.7200222586348048,0.33757552804383256,0.15721770823660142,0.8838793587456332,-0.4073460767376226,1.1509403518716532,0.13429355424400785,1.1005321913532535,-1.7619972547413656,-0.21517411063247915,1.238938718584263,-0.7404475236617658,1.8786647310259974,-2.1498111081429485,-1.2887694702646393,0.6064857004247145,-1.5964734045952096,1.6363743255656165,1.7374417592113147,-1.0154258428509693,0.30292155851501884,-1.5006757822482406,-0.7584795663089301,2.082237174848848,0.6200462585105824,0.3341685168487832,-0.8828047477661575,0.5998847444166034,-1.42195368955666,1.716864695747574,-2.4460550257586138,0.8426916141232078,-1.9084038366222809,0.049370656242481346,-0.0702557299887007,-1.4822696219792098,1.0345753588850717,-3.785871895347501,-0.9596756312522137,0.6679151784507434,-0.4481367343327545,2.2196484455418823,0.378371670435478,-0.5746177524549353,0.27783469556777457,-2.3469181565260375,-0.6998771823643712,2.636227727215866,-0.1934924822429501,-0.23501617928825275,-0.5570942273910704,-2.843398631475981,-0.5914233344959766,-1.4242024020227257,3.544563121129626,-1.5562809927426855,2.361162450395398,-0.6205936459500458,3.490128131534228,-1.064347656447602,0.7509664202703015,-1.13616700992561,1.0517161854362413,0.9130123702240156,-0.6565504682006256,2.4226517448290723,-0.14291552473845537,0.02318463384587003,-1.430301606685151,1.191442146002981,-1.641817559880601,0.6751870113320886,0.9661496875500933,1.2989278922117533,0.42152116263403566,-2.038264164319585,1.4043529081184578,-0.46156918837373034,-0.9178151675850489,-1.2415090083577305,2.197014769118529,-1.954887989991717,1.4854455052765394,2.9541852111941744,-1.541049709610903,1.039538467040563,-0.11657503470707142,2.0405245986552285,2.010019390427762,1.8696461830944155,-1.02079995606084,-1.1384770417918662,0.23831481382995043,-1.272107685292546,-0.7568034177534471,0.45111939089049374,-0.5939945114874634,-0.8301175644683652,-1.3400740610200863,-1.003449374543599,-2.1112508649194943,-1.160956741435841,-0.03106311213467574,0.33121678120744075,0.14214672259425992,3.0217152461878354,1.663876379306942,2.5150736169952244,-0.6806432015908914,-0.4254925817614203,-1.396233833694586,-1.803242404886801,-1.4324009020083954,-0.5339170901119469,1.4497056060693294,-0.07521361330754896,0.26964510173252176,-0.9532338031805764,2.641117506184286,-0.8291396760526011,-1.2108211580216064,-0.29391543652668123,0.536826293293626,-0.8562845482097534,-1.0908659933998224,-2.1181595211010515,-1.4097437617326793,2.0157913124882683,0.18576689597700616,-1.0754909196676419,-0.6936949182023466,0.5187096940243804,0.018269988041681774,0.8538514346101614,0.3265702964448185,2.650885666652913,2.7778119523584346,-0.9399230680666207,-0.4096342873466787,0.028421612806823975,-1.4792899658744658,-0.9627214580247668,-0.765603047080066,-0.1320443664150117,-0.9033550776434341,-0.8606360634404977,-0.4347101788039746,-0.566274918868084,0.541743067671018,0.08802402976517852,-0.4933044809384632,1.7808678711187103,1.6944491919858047,1.913032259385945,0.687555931896191,-0.5686687200870564,-1.9006496194925948,-1.9631035511630928,0.16631519501118544,-2.007219059654391,-1.2516564549555955,-1.407240216547088,1.548650748121,0.34690090518185734,-0.17732307976356496,-0.9752031268113571,1.6439234765032649,-0.32866715556170617,-1.0260697308758977,-0.4909536648547599,-1.254064163319823,0.02325602271463286,-0.5459009759531473,0.11495686348176565,-0.890427997687921,-2.5272000323592714,-0.23311761105168982,-0.5159331125869269,0.9414959489751842,0.2826483191498283,0.3230279127091946,-1.3996111500342259,1.122156086642394,1.3068047594523569,0.44304135538585143,2.226974608782,0.9092686639344981,-2.2685384986977044,-0.7627511585515464,0.6391405965577884,-0.7372617647704086,-0.524541110514218,-0.4368782026912037,0.9926626576201956,2.2845000815103997,-0.010853526242701033,-0.17562337924378804,0.0915471202718656,-0.9417320348398796,0.1577557340810683,2.6319681843227283,-1.5351824171338917,-0.5567067548177134,0.4747276487482702,-1.2102177077830396,-1.8052847070792857,0.49397230689858035,-0.3400733467634147,2.5851332799613873,-1.8934085098374651,-0.9360605409757157,1.032315144550493,0.01941247959422148,-2.083427771001709,0.5980616255787525,0.016616528578696745,-1.6299917104458506,-1.178908153354006,-0.09537110961988773,-2.0858721262121005,-1.1455097341565779,1.4852714870696504,0.22941737828638187,-0.9449238162168353,-1.3351587720115552,-1.4647465497747514,-0.8124373525161394,-0.09076946951167882,-0.5511703078437281,3.7785271811469343,-0.3384190071589704,-1.323305609237458,1.1387221696743182,2.1717054653328494,-2.5635991462713936,-0.42071818714266035,-0.6817454025536999,-0.22474963973526216,0.03131060401139052,1.1256672828653231,-0.686831069144097,2.2320104287172775,0.8210147664041709,0.0007598918996884797,0.8919111288409028,0.843736884391437,2.3907797453855006,0.982436574908662,-0.6566377224149801,-2.014393592606006,-0.6967093792831407,-1.0778603596216771,0.17887834215414605,-0.975274368204085,0.6595460011501998,1.706591725867929,1.9003184478932598,2.3020714194241387,-0.658332778490306,-0.6356102454021751,0.03170340511789386,1.4526694227999832,0.2602869181784738,2.9513217477701033,-0.454912901164661,-2.1279866104300593,-2.5639598656495672,-0.5979489050102034,-2.0012021736458396,0.6688544885643839,-0.7553104910054318,1.4747253170412538,-1.5466895512069683,-1.5104591013914153,0.017277056508809778,0.8409022935352995,-2.202240998091578,2.5005956123724844,0.7626267271746925,-0.35464103830419336,2.2056065585789546,-0.27388655113587307,-0.92152455248377,-1.77237433383214,1.4688313785431797,-1.8509055561674042,1.4012019706928656,1.166225839148851,2.180249522290743,0.1483460520479081,-1.189304809392558,-0.5332360364000392,-0.7975613663239574,-0.9544840653852028,-1.2586298270906968,-2.9744911443925828,2.0374611079866223,-2.097877070920938,-1.2412240796567413,0.11645631502332845,-1.5328376561160038,1.597615682350231,-0.6937901806411928,1.6381159664220426,2.740183080664672,-0.8989622779126949,-1.1245330917713396,1.4062802385751134,-1.8018732751866202,-1.673044773692694,2.1187816883143333,-1.1217481351279206,2.1209446544381017,0.6562643309350455,-0.7449755414517258,0.7927180138722337,0.43283332290944265,1.283488666101936,-0.3395174765763045,-0.6762630901283285,-0.2087610032171898,-1.3712178273627937,2.1770800937649706,-0.8542693236854499,-0.33643511904600126,-0.3201041888790096,2.0130002471057975,-2.6986126910539796,3.3351512111268438,-1.8219129787646005,0.6943066005649986,-0.2917607271821467,-0.024401086052514905,-0.8513948951834749,-0.7531700223095424,0.9267998020953545,-0.4724602527155138,-0.9675141519064715,1.0740606556481136,-1.0193155284129722,-2.297186748165742,-0.24312825765220894,1.102304658283883,-3.1531900048593218,-0.5893773762443796,-0.09844048143207952,1.0532927132459964,3.0979891001133653,1.6387847428596576,2.486313383913508,-1.4112134217209078,0.9750080089035733,-0.09976435779266689,0.24778613732851354,0.1188011798743647,0.9231251992409134,-0.04048898200626541,-1.4937549267877956,1.5753127980384332,0.15276314714117056,-1.225800898238754,2.3437477596800944,1.4333971417735254,2.5572975169406225,-2.40215759585582,-1.0190083278912219,0.3052328882119359,0.23369321486222686,-1.6932944042023919,-1.4393758473130287,-1.7028391624291392,0.011730472929016805,2.1618797181051232,-0.09852770600996943,0.7158929438716382,0.929546543618851,1.0934491155349577,-1.1397649620802177,0.021492600562248113,-0.33524713221813424,0.8887885773271812,4.096083481303195,1.5087383116807365,-0.808714237195931,-1.3698680620926762,-0.38538472210659047,-1.2535842365891536,-0.8450620788150104,0.08557647248137784,-1.5798253765550896,1.3758159519936217,-0.15868088450482995,0.3139452166690376,2.448330554935469,-0.10102832286030426,2.8415978510132867,-0.8582901823729205,-0.8578628325219686,-0.6089713472736824,-2.391137006065609,-0.3890721343844776,-0.7195903672606605,0.06244545781530078,-0.465987989741356,-0.9934608816568391,-0.42527344912799175,0.25361378120618155,-0.20398332895548546,-0.3688480037539442,1.5542018080486366,1.3070812147394981,-0.5077482158709662,-1.206474353843147,-0.6466399891607327,-1.4262432388813584,-1.3923338381867771,0.18139993975221141,-0.030466383515492045,2.4471895138372153,0.055566873216943286,1.161738317806165,-0.5623824241473533,0.017076236823218776,-0.9715174769980968,0.8817116463399896,1.1259688155734067,1.988183221682864,-0.6824087873524072,0.6060872641587537,-0.31692825767455185,-0.5962781584901985,0.5803921805144593,0.24389108157884687,2.0437402930338138,2.7891137727780806,-0.4162000596400256,-0.36393570551482834,-1.2085801263637341,-1.7832249161514235,0.09396434953586831,1.1765844440446214,0.5611535212126652,0.6921023072102174,0.7999034896662873,2.180002296377383,-1.4166854454577873,-1.2966097436972555,-0.9176134751638851,-1.6441499372431296,-1.461755206635705,-1.8069907827224243,-1.8387346578490147,3.5302184801011327,-0.7537768835625165,-0.3495445283998225,0.5872636725450814,-0.2753901802525541,1.1964417891291776,1.4050657167318163,-2.1049444003489706,2.629243263100873,-0.06147230255709531,-0.5798435531629889,-0.23756957214962587,-0.2834395017639977,0.49215648898070274,-2.6989218068037806,-1.0148664389578719,1.3615567502834012,2.4893355980665,-1.2601557356083164,-0.6509321532951664,-0.4375942805389318,-2.125707957465659,2.2085641568376935,0.5986352479031074,-0.6080552447882746,-1.2767251478019335,-1.5430282453992412,1.1405618667026212,1.4758817079883633,3.4442776037423752,-1.3111099150886687,-0.3980357264476304,-0.6001123034808499,-1.6717313525270803,-0.4049074188802655,2.4109686587322297,1.8440303114862546,1.0106083315657575,2.5709838237919227,1.6337678337970925,2.20323207015729,0.43477294374429665,-1.0472893592349353,-1.529506988492476,3.7191280677916074,2.5227446646729166,-1.5268358955575838,-1.7152841237264917,-2.6774742083025833,0.2218643266875148,-1.2626479448551389,-1.0346592993477752,-0.33211975180443476,-1.2044308254958431,-0.28535732582549056,0.07891930306385316,1.5284716375238887,-1.4281986435192786,-0.4507127279444827,2.3452120071402116,0.5233259944548688,-0.5250786969628106,-1.3469890550441448,-1.2272378800928119,-1.9901886356476788,-0.5475310765811509,-1.6338827211118898,-2.095077315839062,-0.31051454342616397,-1.9930973463620385,1.3865534569060136,0.23336915820795393,1.8017383379896188,3.1284782509848497,3.7166609054237822,1.3552955057457208,-1.5525037904777657,-0.6840015336134562,-1.6628979236958268,-2.0151738117039106,-0.6507576621786572,-0.6396170679903115,-2.239075277186854,0.21360932840664504,-0.5173140157226426,-0.5035726048130885,-1.0216308344746865,-1.216229680630076,1.4162911179786093,-1.0077231773477793,-0.24636909619451736,-0.18895606389756994,0.6383975583541429,2.138669958815643,-1.4208728619318747,-0.8511048974007608,1.474028719311512,-2.5399820506949213,2.027315328334453,0.0026393310655468,-0.0023751459069829024,-0.5983914169309488,1.2438065273192431,-0.7766792253584011,1.2030040618639728,-1.3396390054592278,-0.6110553190401737,-1.1659162241554706,1.445885858829231,-0.5504700482328042,0.6903977077479123,-0.8878520989972362,-0.5774451527509018,0.08740375346798268,0.3895802252470229,-1.5052492514004514,-1.3235539434036678,-1.224914252720071,1.0391707672948585,0.361978207474911,-2.4296401436577075,-1.0658024370908004,-1.4946379333333857,-0.5862224528265283,-1.7808900820308655,-1.2830820598567407,-0.2874473656989864,-1.0115918314402392,-1.0642605158026837,-0.6823222962160485,-0.6697867617525544,-1.4018810311388543,1.1095229690658162,2.032970161942443,-1.0355340761559952,-0.6795952092903217,-2.127871271868961,-0.01674390436122243,0.236174350279512,-0.7726944761800547,-1.5143823363222408,1.2300700164595177,-1.1826435150973649,-1.5889575277343466,-0.8438947762067681,2.5389326704529194,2.7946772895861964,-1.0020161297616588,-0.700872518003352,1.3209794287380943,-0.8581869910927928,3.27237966798738,-0.9130685109721803,-2.2385775545664357,0.960505448754297,0.47429364974564675,1.1969771881331268,3.3640880718383066,-1.650192432075704,0.22116639947545835,1.3642683621461957,2.4432585311100152,2.0341447825638643,-1.3228491812081862,-1.1952831169716724,2.6349603196909555,0.2765811531988787,1.3093672548255768,0.06213560930012435,-2.640061242747469,-0.5350714637498318,-1.4629755858287394,1.118815488375596,0.6588715910250312,0.28057722455181033,0.25554442199572747,-0.17092782275017274,-0.2974821178120657,-0.467645375283481,2.1828095940789014,1.4247951769693914,-0.4945259346632827,3.0325874830842854,1.1360736164406418,-1.5825408855237457,-1.365718182353124,-0.9466527853699647,-1.7082162544017987,-0.7643829898768763,3.1843706385604453,-1.6804952642779591,2.240700678998339,-0.7957336605083497,-0.7440547601019221,-1.3430047380027705,-0.3582214253639565,0.6866155224448768,0.06872543495467859,-0.032843647172335295,-0.02010368996842128,0.6990168852335021,-1.0763976754620384,-0.3792234088821274,-0.7409135989459799,-0.989458393497864,0.5699318128407928,1.0004736407818695,2.8534809186825836,-0.4165237883882151,-2.191741914100545,1.712898528361673,-0.8981919800087705,-0.9549131832972492,1.376608216841597,-0.07575255127100046,-1.0148397017683763,1.5146638551446512,0.9599841405943619,-1.2332243124324174,-1.9072982806426835,2.80386747869041,0.3380540082368971,0.36259504092837125,0.16051145547968704,-0.1412307926958206,-0.5476002889670062,0.8682200023878124,-2.6603219043326907,0.583489245433451,0.35608813496870717,0.409189485646499,-0.3803269834985524,-1.0341367048334245,-1.6952206057469605,2.2607346061138744,0.6772636666120961,0.1046572931621663,0.49391779278873704,2.7373719792929965,-0.6237735165440488,-0.6055982448591498,1.3219203657644218,-2.054917610206726,-0.18899241238963632,0.17610025397877477,-1.0141366577066437,3.0396170077678013,-1.4643486593910409,-1.4116526488153351,-1.335084975953164,-0.09350306610674095,0.22897064325037614,-0.06399962194638766,-0.2051591557935408,2.8993898131550844,-1.488876157358498,-0.12427722934343136,-0.3354190382248293,-0.464514319297726,-1.9325013031398746,1.237601045119601,-0.5641678482917001,0.5640644366141241,0.9319602474995968,0.2580232757533242,-1.1621569703150256,1.563234887506926,-1.1936731638122828,2.2345240655909238,-2.3971864077079044,-0.35701491372316246,0.4835088094769024,-1.2161954526968441,0.22924086187944653,-0.6657942348417043,4.041308140550084,-0.3340793070569369,-2.3873415051642115,-0.16157414379772947,-0.763318515728624,-1.050887389656625,-0.29624791118725574,-0.37494260927635037,1.7472127012750973,-0.9174107870021727,-2.2385533557798674,0.07515501520200013,0.5959810042586928,-2.2528029650760675,-0.2595987394339561,0.5782384813361635,0.26486733447090666,1.9629422570917434,-0.005332234626600415,0.6142579515752877,0.9633367456382953,0.24523790864381212,3.2273037902828436,-0.5634301737742683,-1.1962775762623064,-1.188468282709064,0.38865390426449753,-0.7120033625735694,-0.4645978274813238,0.27058399207808076,-2.275445357629799,-0.017124117610550584,0.6750683206395337,1.4887594321112128,-1.1493921007801666,-1.2648875227123715,1.603579725200178,0.36776477463416757,-1.6017241727620486,1.918366843861009,-0.9033068449456486,-0.6157959932523774,-2.7014879277666157,1.9882018486201531,-0.790321966325582,-1.7839458218674324,-1.4257582768303465,0.9756502914580605,-1.2184335146718965,0.4057042525792348,-0.9575489334348131,-1.5018506327239196,1.5955790157163854,0.15678493664003923,-0.25870871470211154,-1.0760984186377742,-0.5263707655264822,-1.3287153921230461,4.221230735193085,-0.09338054913332218,1.5839999178768018,-0.5435786910084037,-1.575829027890888,-0.4622762075550957,-0.6033876885657978,0.05967786427630181,0.22984474268757993,0.12940398359024735,1.010632613758326,-1.1173187877738315,-1.2778100480928745,0.14556294070291653,2.3826391345878237,3.15682700446683,1.7211937301998976,2.527484179555255,-1.7364051648057979,-0.6150855126285122,0.2818622068775688,-0.23815561471529756,3.067430222319127,-0.4979655478435105,-0.27538773979028497,-0.22666814436073746,-2.4871241330786944,-1.1676142214957976,-1.135244913193689,-2.294619478794382,-0.36197114273554204,-0.1786938441115481,2.5046674268857685,-0.09944171586013875,3.3355455938259637,-0.8049613013593234,-0.39603465986665953,3.5653329342763316,-0.7995738201594038,0.5034700807751382,2.4602210598462086,-1.5683290062395607,-1.4196524365582097,-0.5450957071892759,2.739919320621202,-0.959194350652472,-0.03490699987494693,-1.445936454654769,-1.1712399675108618,-1.4688448746056986,2.3144318152951198,0.02869424592416834,-1.7934284496133537,0.08856494116104557,2.3389090572007154,0.06293130100732475,-1.7423191491921612,-0.9086370844941071,0.006999069086690202,0.24237619479215375,-0.8352865219837435,-0.4918412372265357,1.4816136889704872,0.7038680895913068,2.613293971043553,0.41828454657402214,-0.42984365485719,-0.08856753006837143,-0.7635143714072326,2.0347369189052142,-1.2526818614475286,0.6025698552384776,0.5590749904591784,-1.2440137793775636,-1.3051472715488401,1.373321000324301,-1.890773092852011,-1.0444287381426856,-0.33558584636806127,-1.7775245720598578,-1.0371143576553308,-0.7852761908166012,0.06426199326451672,-2.242436882845007,-1.3853920644828148,-2.364879090499133,0.6076685476700272,-1.6182897557438116,2.2671166822953075,-2.186238898353641,0.2645976030187016,-0.3071807741088422,-0.7313807235154308,1.6272910577418518,0.7528073926765076,-2.1527180251455036,-1.4008541424342067,-2.096991410894444,0.749999141788292,-1.3086871380991265,-1.1664294555934505,2.6963351153434716,0.37619653828349675,-0.4858434964247304,-0.7875469119766683,-0.05403617876516161,-1.954769985186451,-0.1935661850266947,2.249298889362751,-1.5518330811458074,0.706949319209806,-1.1600080071264796,-0.09434662508570803,1.174095033265647,-0.7840178294061957,0.7162429921196982,-1.6572864703238577,-0.5255327859667807,1.839710124717996,-1.4124527255580164,-0.6237538724270187,0.9995136899673877,-0.42209322982640063,1.883384161999909,-0.7412409960903922,-0.8312035487159002,-0.8861864719186916,-0.4534924612561316,-1.2220595363501032,-0.5612508012399678,-0.8529404517563306,2.2567772009001983,-0.585086612918385,0.5412923281882972,-0.699563825806831,0.035503506703786006,-0.34630398313770977,-0.25480990784659946,-0.4909224783535016,-0.27416023643551424,2.120621264740646,-1.744780613352309,-0.2986691601805924,-2.9619284159908346,1.0950812433317862,0.9815691159439152,-1.074338820077726,-1.362249876890532,1.70686500258376,2.0356475417992326,1.9444581350938273,-1.776122362607898,0.3101091146750462,-0.6892102636322956,0.42916218598287104,-1.8240091862734364,0.9411087064200231,1.5266710108919899,0.4630300812969065,-0.5198899131000063,-0.6451220365138997,-0.9410308241373756,1.421906232528714,-2.637943464745719,1.1159806463442197,-0.5016843757432085,-1.1118166718656624,0.18153817195223304,0.8933945901511545,-2.3740468702630255,0.9595303861308272,-1.2571356969686283,-0.7004733042189405,2.2981709788847775,1.089547904159981,-0.46271410303220567,-0.47777042925531754,0.11538097941138574,-0.021273940626724355,2.182963419973713,-2.819977486309645,1.2859854686155694,1.3451114137677969,-0.5658127737148897,-1.664673439523986,1.150739807173839,-1.216552139331152,0.4418344291593818,-1.3616177077594342,-1.1236210483713795,0.5753401957828504,0.31839928768414655,0.6359700657612875,3.183693941224021,-1.5714502065487488,-1.4848244560900659,-1.0197113086151288,-0.6865379408662698,0.48713801245988436,-1.1845304811953328,-1.9517702275774247,-0.8221203562032342,-0.31454475792426806,2.8198459848362525,0.2562669673450691,-0.2496105858712048,0.34350212520676743,2.3071448453983354,-0.48381461129115044,0.6768404244759142,-0.1892690090856711,-1.9278937683701527,-1.3109830804945317,-0.22063695041576106,-1.4927825931452254,-0.5335595050215195,-1.6597782394396912,0.09160945530670482,-0.7100955849335204,-1.359960398537894,1.807712136358765,1.1812790751417004,-0.23571154668909114,1.5582470612024684,-1.1863789485210623,3.465466235025597,-2.00732412458041,1.9495750870808275,-1.030423288285224,3.5615774355229,-0.45594149098649017,-0.2874621511801006,1.818555713216217,-0.003639091506852559,1.6526675297518352,0.8293285190080424,-2.163289524433316,-0.3046790104793447,1.836366535374188,-0.10488158731668329,0.7681136054120323,-1.3322108500705299,0.02217015417647784,-1.1692845300299164,1.4220330972634156,-0.49316085098803103,-0.2492778287777232,-0.07663482396443236,-0.9369781778954935,0.46527002167844,-0.7926264756056672,0.49596713865603104,0.5817511835213066,-0.1945503098960451,1.308604903296087,-1.2746959865852343,1.59511794488929,0.7190643023202898,-1.6011332645170544,3.451020230259597,0.7126193365615816,1.2002472315635102,-0.24423490614205975,1.657304859797531,2.8582867745990006,-0.5707812811429706,0.40501923957178076,-0.541764012533432,0.2886394596682144,-1.3985018197916999,0.177065425785604,-1.3287738567758893,-1.3653049923124807,-0.18829387813874734,-2.102656163933134,1.582297717696908,-1.6628677598181192,1.5189577580785159,0.9966559171461817,0.6826951180409465,-1.04148345884083,-0.6836663777028293,0.6293639348462702,0.03609651080083051,-2.3733901295637962,2.0394326926154593,-0.05549610707806225,-0.5996079229046886,1.013921876231144,-1.8323693098941987,1.2531724543137215,1.4454239467443253,-1.0816015510450356,2.7463447515627335,-1.0521016827802994,-1.8335011294154853,-0.4280741630091587,-0.3047607764176735,0.4872119343053109,-0.61371467406177,3.824075964831763,-2.290881526958367,-0.745411508654098,2.097027831397833,-1.3973537635936428,0.646722713021275,0.3472081203360792,-1.1257505223239315,0.4846573421754454,-0.3317063180056787,1.8288943679226832,-1.5823544111298837,0.09660360731962517,-0.7796474385123213,0.5714361708498404,-0.4909431482010133,2.2838417434138822,-0.515561417550027,-0.19767404170089978,-0.7127483676617766,-0.9315315704678723,1.2096047054728902,-2.0869918591833487,1.5120427470479443,0.2884991928645562,-2.3364905511994523,0.6144768044843798,-0.06939031187448634,-0.3429340887544544,-0.3329250800204437,0.35155380703731054,-0.502829914445239,-2.9161115837936333,-0.44661615008998606,0.8214181392614033,-1.5191014485169696,1.0283802938420092,-0.6988757030705586,-0.3842385280759969,-1.4064627676044108,3.8566532328150727,-3.0459612960614244,-1.4010503753557078,-0.6707577161446722,2.1191715189552554,-0.6853936097919623,-0.7366891689782042,-1.5123678549254187,0.5463772550895242,-0.1882902890297887,-1.2544992969591977,-1.7091266764303479,1.5131992558876965,-0.7764024163751604,0.9163466014618893,-0.7954582294535798,-0.309729075595039,-1.3909679963791486,0.9863982166756055,-1.1542936196911138,-0.35182939086827486,0.10333267059864458,-1.6146635310498476,-1.2832581163511785,-1.0317476995142023,3.2582693168168,2.141281986174598,0.1278151250817332,1.772164958339374,-1.5972101352405286,-0.9864157389772289,-0.4206708863080615,-0.49904881633670933,-1.0127137750575528,-1.2891552583674464,0.6065447626617438,0.2761136143202809,1.6851245130166714,-1.310744042920552,2.032443116021589,-1.4306494005405055,0.814697536518933,-0.2374807573784498,1.3084519616712227,-0.6479415672716261,-0.8034218871856953,1.1915937238479661,-0.36458538404504515,0.4940099425732225,1.8559760783213595,-1.8794527132063659,-1.0561837461487886,-2.595835920371242,-2.7994228376005235,1.7124377840529756,-1.0332714164316918,-1.4825384784404603,-0.9801663572686555,0.10055525234027354,-0.8265791950859886,-0.5420206403043761,0.42507684084174624,-1.6844735068907681,0.6010049824714159,0.01172715339561231,-0.3684155239942177,-0.1751561030864895,-1.5413577000285048,-0.597155227925982,0.8487028825182191,0.637601426569947,-0.8906637296095632,-0.21586287383337935,1.2408392800953716,-0.5769100109669393,-1.4168827224974467,-1.0281109348516504,-0.841539908917079,0.28559906897560733,0.2735722716048989,-1.915625685489657,-0.9174065232535105,1.1410185347164525,-1.6161408845466738,-1.8075516419766893,-1.549588277537033,-0.11829552239727972,-1.9198070620178795,-0.7415648550803242,-0.19417440395347638,0.8628265071550533,-0.31630657759174996,2.0718341665009503,-1.6681997552369754,-0.8374195187950983,-0.6810341345687112,3.5402162875400274,0.6278689881492392,1.458275027063889,0.007391329275692831,2.2815633252538436,-0.5590338617730715,-0.5611340423875639,2.2629128355454706,-0.5929990055197824,1.5898646644044845,-0.6919968988802667,-1.9429736206197532,-1.5564667823086582,-0.47434999693088004,-1.7188976369021989,-0.18991346629458167,-0.6617774837959164,-0.6555586893676421,1.3936567313642862,-1.5140244589101206,-0.2648176978556617,2.0021889669526094,1.9230098957604862,2.9365121360934516,1.9243343296750004,1.1384652192835039,1.0945619003150477,-0.7056970188865646,0.8734491446632873,-1.92246249737192,-1.0246199957823847,0.25432624170132767,0.8058573026779908,0.010076663929574235,0.13758551474249267,-0.6665674167259232,-0.6989006601427173,-1.2410718927366462,-0.6771806941724573,2.688462860817556,2.3941520988092626,-1.0890186027551716,-1.8125744945247586,-0.9075088485428462,0.08587540420180147,0.2893853588493403,-1.1319101976069492,-0.1905026667412693,-0.2864582557381739,-0.9111268230165251,0.6325455824659891,-0.9791886954093669,0.3743267988017642,1.360813843330778,0.17001176895076237,0.6386422963396142,-1.5058313854998138,-1.9110104854380854,2.2877915402898066,-0.4545643572859258,-0.5523116965932635,0.20776750907299654,-0.6647352861229148,0.5904667700773546,0.7974387172681384,0.7123608162006011,0.6041068274608146,-1.3129718149292573,-0.9687779642998415,-0.322347750492,0.5433177499457407,0.553739630579176,-0.4681363446916351,-1.4674904671332651,-0.25195661174295325,1.7417263414234068,-1.7889541214645908,4.171872201052029,0.014189209407916353,-0.42549523836462194,-0.7018846070412897,-1.6071626574971172,0.6096361233237984,-0.8470689890341534,0.40898220970262017,-0.4229853085497928,-0.8376834812293262,-1.4824037557537946,-0.12344320533554,-0.21859586939261805,-2.0328444375489014,3.29487609898502,-0.0233204619765322,-0.712353517362301,-1.8004602278408375,1.5396691100665367,-0.9732833655046573,0.08655962007579979,0.5212177330367289,0.9581277177101681,0.19644453010102556,-0.720632666935838,-2.775691980189898,-0.9335895515516954,0.5728680057619664,-0.16636023190443666,-1.9968750007376574,-0.8306552237098249,-1.2568313939098854,1.2511287046191775,1.007328403189026,-0.06943585959909773,2.256046049105544,0.8599750953792786,3.2279311028484203,-0.7264614539496214,-1.2969620361375231,-2.0428320184613273,-0.49275816582912585,-0.4701533795484304,-0.48017788323592325,-0.984916589906315,0.3017607920969678,-0.8784382496068023,2.5444954347179625,0.12881262990750447,-1.6049734245718021,-0.14406841522943445,-2.4153048960894794,-0.6628359951023844,-1.007723070092013,3.852253706140283,1.5797127148358907,0.4799041061386705,-0.9710397988638066,-1.4868972199471382,-0.8238492943911205,0.05766099719557609,-0.410630539268695,1.961199952854087,-1.4060508641874265,0.4713469118590136,-0.5035801474879673,-1.2753485273440843,-1.1691317429452492,-1.366709534677768,1.6829968372433424,3.4492671537637767,0.2713013938068759,0.8091491896838723,-0.5651322964553346,-1.4444881414201822,-1.2175468485865437,0.6053483038560586,-0.2992980186985795,-1.7333997861859827,-2.5318848942071885,-0.7681221078291465,-1.2828981435601459,-0.6964418500129086,-0.12560779699359087,-0.19531943632427518,-1.564506211392913,0.6678551332585728,-2.1076675345937605,2.2662655184666773,2.829823681251587,-2.0249329729701055,0.4794536330828389,-0.7782217341908054,0.1872502103906669,1.2567768293003574,-1.0491345917678294,1.348478498509967,-0.37465009971093743,3.309642677375488,-0.9878544142137008,1.1218087994862904,-1.4106611212464808,-1.3879697048718254,-0.6614518052225166,0.5974921180134828,3.0043055915424404,-1.6904838409734533,0.08112790879019129,1.8960180247379983,1.53422345874658,-1.8777313838410958,-0.06581685637715327,0.5757638466866702,0.3633217489767039,1.651212140119655,-0.9375490757842706,-0.4455221060994893,0.5427193722788414,-1.4183969057138817,3.9174538602095335,-1.4084127310720391,-1.1373497514394544,1.7781498268190392,-1.969338106046551,-1.0558311977203054,-0.9979095536107381,-0.4834010332773162,0.1630656855537967,-1.4703097189913237,-1.5779784727820336,-0.5901691830436827,-1.537296323909501,1.4298613263690676,-1.9545632873632945,2.9639291053392087,0.6274582674798014,-0.09988625553339699,-1.786075807635598,0.9662591795133524,-0.03687314950155051,-0.9382893498931241,-0.9722686665943592,-1.0119992965821696,-0.3148152419149722,0.29842024129788747,1.7181087460358653,1.91116004267448,0.13509347290818327,0.6836683850645395,3.380660591931611,0.9501481464401977,0.5472028652118434,-0.6485732827943934,0.22344829253251622,0.7187703710573365,-0.849079003474846,0.8512304384937406,0.9075937474602159,-0.8057649442385519,-1.6034953544033892,0.706936035950107,-1.8831229235186786,0.5616511792621554,-0.16736864648536726,0.8478700837614411,-1.3649417492822837,-1.1957905363417318,-1.3301576020667834,1.0707650022568926,-1.0275176668890145,1.4113657068941121,-0.04687785928778358,-0.00963852625408051,-2.3168814274723397,-1.238154437241619,0.29832974615291347,-1.8896543537337633,-0.44032003839391254,2.9346018548553507,-1.7103120302278412,2.4619158494511875,1.565845646473715,-0.400618295389299,1.3375126605206078,-0.7161562122939085,-0.9504376259361211,-0.8377162480352349,-0.7060081959368406,-0.6036623617653597,-0.935059885558905,1.894733810526598,-0.22649287260455966,2.7322466501382587,-2.190449091983191,0.5096650505626973,-1.2436055194138742,-0.6158316228305006,0.18322149290208978,3.030665163255484,-0.9011805880930354,1.2731760938258492,-0.5347296161851399,1.394251225135892,0.6365309145342564,-0.508345845203602,0.6896668994371966,-0.7313043288267408,1.1948083133497847,1.1357788178078783,2.419542059282203,0.9571501956187808,-1.3038979295791453,1.4696442429430356,-0.005949225500939323,1.8374918380556295,2.8184513882680937,0.19582793156272488,1.270093937587383,-0.14369786360108996,-0.679800112596243,-0.985117483318297,-1.7380185610206138,0.07764032034048785,0.9799867475304436,-0.38247321967781994,0.498660966944941,-0.17932020272953417,0.22305023860095796,3.148776500227986,-1.5461346607555169,-0.2371943434154077,-0.9427379883127966,-1.1784958401310266,3.559776863983648,0.4660525216802205,0.5470974088163689,-2.2512852734395277,-1.699494031917865,1.9749342505434266,-0.10581278291254212,1.7967988817903757,-2.153694152574473,-1.622198829806383,-0.19674098650679755,0.6928534179211465,2.5177128712802017,-0.21780058067702696,-0.7419757975646428,0.3968055214083898,-0.16789914232448805,-1.4460095636222847,-1.0133569398023736,-1.812276710586522,-0.031573296041109,-0.7000301894617003,0.022916677880431002,-1.173184991832444,-0.10492642561979848,-1.401950240251346,-1.5164187230549082,-0.49474133792072655,0.9541137885353079,0.3947832970671384,-0.178801251795483,-0.4656709130384551,-2.3331318231062403,-0.730700541910385,-1.381426844744086,-0.4793909003266906,-1.4615135487264952,0.7818109978658772,3.192849146589201,0.4652370810508609,-0.15345921734021575,1.227719602194683,0.004744145088057364,1.2793843670552258,2.108847673681329,0.38520324143149487,2.251611111460166,-0.12091434832108805,0.26514785142836733,-1.0686468299004603,1.404648711042133,2.481507490764095,2.8188128262565546,1.0403592963992132,0.5777250570551662,-1.3198102312966289,-1.1844645327574839,-1.7741176071052458,1.4354343167353327,-0.5870334552786468,0.4907046644758899,3.971980143801027,0.7230246529983592,-1.1165662341532847,-0.373103487144301,1.505704109294754,2.302557887611866,2.202423393885159,-1.5205241237078637,-0.9406302345820815,-1.4986668130742764,0.2015566805441017,-1.8948807199316788,-0.11966847893890031,2.1351717101964485,-0.1288101859998367,-0.2363239264538284,-0.8317710065082263,-0.3544746748274804,-1.7444685535679874,1.4510063456460556,0.26625984143361364,0.18640492272177375,-1.254380633213225,-0.05976044439706214,-1.0468527978367548,-0.8567078694843628,-0.6637149729316225,1.6299994675700271,-0.21114173611741427,-0.6836547901531231,0.5950834638851603,0.4281910868395299,-1.4431022331604795,-1.9090668889811546,-0.025976040524300636,-0.9043494152298127,-0.7080106840554071,-2.0483699698009756,2.360383061151376,-0.4851716809185882,-0.19691340071378974,-0.6320080739345721,-0.9950909967177229,-0.24065929929358426,0.7767902051847877,-1.033609361457993,-0.583127698334342,-1.828933999425998,-0.36870701788703947,-1.5066239480184898,3.1485607630290544,-1.0434436668179854,-2.2762595028776067,-1.1813552967246277,1.3105131623453112,0.812101370423679,-1.586555073014012,0.2074052627428129,-1.3082095874891806,0.02078321499879277,0.7028129964952332,-0.365823596066359,1.4831554979282753,-0.21402809237184592,-1.7894814329588775,0.270332116186662,3.53843215300244,1.7845706428990271,1.377786538525628,0.6514404743917284,-1.7182482353934507,1.7188744298739107,1.8666534035258213,-1.6201520167825159,3.922304232908627,-0.04206340467314196,-0.8523403620539773,1.690670284872728,-0.7089987725420211,0.02457679687222277,-1.7575653927725794,0.7573039497126502,0.23638739043037396,-0.3682036094150854,-1.5902239589224307,1.1598485441521187,0.48334868163149153,1.7890500198014219,-0.08695785825222137,-0.8623871320129778,-0.3384074291579973,-0.39754340424146867,-0.9436038703489681,-1.4468493476372266,-0.12888294999347283,1.1134627720768966,1.239042405774443,-0.3912232444727771,1.6351397205983471,-0.6372918289552829,-1.5282827096386495,-0.5418009589938666,2.415005152982033,1.141525757017982,3.936202764901273,2.56308819687843,1.3889613851475044,-0.17385073425369282,-0.4761837326565199,-0.15559503993573376,-0.9871233972614636,-1.4064507523674208,-2.396739341685763,2.9646327645456685,-0.987585451883576,-1.211184716371605,3.135779767647687,-2.0492745107823107,-1.4595060950876506,1.3534711215880615,-0.5334546864980547,-0.6999773608138153,1.3324349819532761,0.2456177975939579,-1.358972594169119,1.338194883765121,1.3908779224365013,0.8527992685781933,-1.769465113418383,-1.7453510069884235,-0.13910155939474897,1.8820287135416693,2.4499357227180774,1.0911560733031844,-1.2969045452548615,-0.9605501936513496,-0.27435454797121234,-1.730222888120457,0.13148673441638237,-0.24294917770399221,-0.48720511968459085,1.5386522825084432,1.6398751241636789,-1.079489603406148,-1.2195449074699776,-2.0750852538969555,-0.4152936181565961,-0.16898723200098206,-0.22239005858826635,0.8871578108577716,0.45498473395174877,-2.402843763343471,-1.6376140678925137,-0.1310203312784812,-0.49897963087505154,1.7442973865219153,1.1091606296940446,-0.41411826020978815,0.8752083619069188,-0.2340306843848735,-1.4193318976051694,-1.139655372919118,0.5188569853573394,0.39735240781872416,1.9107893758691858,-2.9446393384325216,-1.236568764693691,-1.479061609049188,-1.1779105153733556,-1.9290982205890452,-1.6498311894213549,-0.45483250945029047,0.4230639637383416,-0.043401426181226076,-0.37599063742008504,0.1174848333717418,0.6552867997516911,-2.042977120539646,1.1992313514436335,0.08062669752583493,-1.5268053508822792,-1.5388865105291263,1.7262129019595993,0.2478625668443239,1.760888638666092,-0.7314081121501184,1.0585778089793132,-1.3462305475341316,1.3985968037064784,0.3913689192906295,-0.7456635116943243,-1.2752782005065308,0.8323147325697181,-2.0118385436598194,-1.8666514839919384,-0.630991928056746,-0.6058981992918595,0.14201168483423943,0.108541329878609,-2.785081818267036,2.4455388128697844,0.7552205300426673,1.2027457374991828,0.44545834649057925,0.7425051352087894,0.004499657422599262,-2.0428258835494746,-0.8161817563321877,3.0111104125218935,0.2211092978736969,-0.1050469060505389,-1.0413936606869578,1.7101748587596286,0.45287939193221144,0.5474667674162541,0.6366775604828031,-1.0046908380778532,1.5457767334452763,-1.7096681974195056,-0.8593993256803394,-0.5135764595589578,-0.8093048483009234,-1.536224832324152,1.4598694296991477,0.6558449380866236,2.2612340154918753,1.1220984917879233,-0.20500077502863093,-0.4154038810030511,1.4148392977698827,-1.8876981078314474,-0.5051044450662412,-1.7601295165243158,-0.745199551725977,-0.2849259706972409,0.04552012592359631,-0.3462198739591538,0.6170599267607391,-0.11500355113307904,3.3388821386713854,-0.6052529749147679,-0.07203002045162177,-0.9917563117149397,-1.9800984125616383,-1.3553163018140786,-0.6747526660627186,-1.5290671788707435,0.642271327777079,-0.7125874169539693,1.2155978395005103,0.9641269458645808,-0.29882145602241433,0.015029894238064338,-1.7746018316542254,-0.9825936253651075,0.6555927363745058,-2.408070092442131,0.9994519752757534,0.8379447787331188,-0.7512310073423945,-0.525851279201953,1.2565170679042912,-1.1570179373478866,1.2087803621916857,1.3964968841041048,-1.720971003394431,-1.706979933837742,-2.0714519710875545,0.3406727815809044,-1.5898510262029024,-0.46186406858160517,0.41880325975672567,2.4062408168922067,0.38955023131843186,0.5455782054594557,-0.13176561990780025,-0.9801354621819024,-0.8581188295491788,0.6977282775019691,-0.7217122968033982,-1.7239661009051006,1.3458702777477325,0.49191167994194157,1.7739009744525418,-0.562635799033081,-0.5286839700192255,0.31085853287252313,-0.08029302097145927,0.7266035791715723,-1.488875226747053,-0.11220740761747228,-2.116705373208416,0.5236473301383209,-1.9107911896605074,-0.4864946833344289,-0.37841218309557734,0.9021909808940509,-0.42412115604741824,-0.7141928840238729,0.2586855871674669,0.9601037174598305,-1.1442628614725003,0.9249531718857331,-0.12049219793174425,-3.6553885064322813,-0.29860386171777337,0.695214078621144,-0.5353568348937537,0.5023327131867985,-0.7068052916754781,1.5402756908327735,1.8148649210519405,1.5640889455283948,0.27568572579547224,-0.4297061939102177,0.7760810418402311,-0.7131498846719342,-1.2086893276619701,1.248829539179429,-0.19405475013738357,-1.4270868903037117,-0.027394096296848552,-0.6936540429986398,-0.01789593972524115,0.03753903819697093,-1.2020628859631377,-1.4529467467454655,-0.6410136962760612,2.51083027918829,1.3635445520744542,-1.1972191371241159,4.135493501989295,-0.24485028914687748,-2.132169206008056,-0.5927252776619988,-0.1786439149143842,0.09024773954885998,-0.3147619531206627,-3.0108992591563193,-0.9683149237181008,-0.9071078675580605,-0.4337709483399195,-0.7766894217612136,-0.7839547141676783,0.059498151783554214,-0.40479252189122455,1.8606651288246863,0.4359912060908412,0.7167200906931211,-0.7072526741305105,-1.7669823847816188,-0.5224659061840567,-0.22585138824753184,2.6508782912866926,-0.3687845781195078,1.7817156823732363,-0.29292810022594323,0.03275449073820352,-1.086185374334243,-0.4350144839085947,0.7698799366383702,0.035789048635318066,0.05714386086810716,-0.9068396023936851,-2.629225806334879,0.9266728988098384,-1.5790347669359897,-1.0944732056813469,0.3332041387653347,-1.5164168615609535,0.5692177355221718,-2.0011736457007823,2.205909563266263,0.43339170683536127,-0.41331182605435585,-0.5978646546124543,0.19672144677611436,-1.605451048646689,0.4371305534660803,-0.9610334470168443,0.2636071924751661,1.1720312545962606,-2.1830631763961166,-0.5391310149204195,0.45320646321309854,0.7076049596819607,0.7116174510577035,-1.7030083528353965,-0.16028957618905154,-0.1082561587778583,2.4154426441086163,-0.10581599528466719,-0.816772538335601,0.21648469263227685,-0.8587520265236325,-0.08979199132919789,2.397441909388165,0.14646007160683175,0.3836323159590416,-1.778948249983737,1.2799339851996456,-0.6701158927972346,-1.3917996178866157,1.6683905954087725,-0.7172463504104584,-1.2715858313733432,-1.222797135637126,-2.279983335718339,-1.6687845895060092,0.24460756108352008,-0.7831895804038541,0.022260470491498836,0.6880147747179439,-1.8130963016694912,0.5426913611209135,0.4747360824318666,0.8850249454249612,0.9372546246747365,2.1859860428592697,0.8462424696524673,-0.1175147676181322,2.528345416262904,-1.5749488436457215,-2.2757519766339183,-0.05544376575419101,-0.7226075861040444,1.5368947704133924,-0.7667012223183943,1.0417625375152582,0.4973233105998968,-1.9705698197068637,0.09430192506834528,-1.7427707009275542,2.552320895036154,1.5675199940442341,2.3907637338025776,0.575185961647932,-1.0772499391509385,2.4721983520934288,-0.51469975705147,0.8316024627323473,-1.1229708269706573,-2.3495149790981658,-0.6999385661798925,-1.980920560374778,0.5542654851038469,-1.7644322406294168,-0.3867714384681908,-2.126182216056464,0.7513809578670524,-2.9403850043555058,0.12903807715804236,-0.7003192918133785,-0.4224796448556215,-1.5928055302674617,-0.3948201324117288,-0.6084445249649392,0.06007600391142071,-1.6616832608719276,-0.6506800596226465,-1.4474191242854288,2.0201799688205724,0.23862259381677217,0.01575180503189457,0.523882752435918,0.33032997804373254,0.19599790688907578,-0.35854557400134357,-0.9933254491008527,-0.26658331287475623,0.663958027267565,1.0211256615771793,-0.7319296164496906,-0.597031663651694,0.9714211881085197,-0.8430046516054603,-0.8928728650336207,0.0019509717178495365,-0.6156303325518462,-1.5415918491021041,2.1905593653380384,0.7261875049573818,3.2357971366640346,0.37312597971968425,-0.14043201519970383,-1.3265654348984914,3.3530471202572336,-1.7220223581430414,-1.1736507238722698,2.6089528185816064,1.4920118378105718,-0.22063889398698489,0.8522656349881543,1.0392183310255279,-0.0063106390189990305,1.0434263803704014,-1.0089359383329914,0.9210985769425154,1.1597587766843398,0.17376209025697575,-0.39897451076429524,-0.3688787473364043,0.0031543411495632296,4.776318518538577,0.6824335408966906,-1.727625391462363,2.1490079620748292,1.4912488463924831,-1.3380782543183658,-0.1973268204225874,-0.6234375742378228,-1.6555584015163143,-1.7836215195324165,1.9469712723905221,1.6016446757198277,0.6997001739130756,0.06375407636434,-1.3212080984309595,-0.9504266451723747,-0.28436413844149583,-3.219394648596372,-2.036062614260473,4.070885269877288,-0.18202584519522927,-1.173924211246173,-0.1864656504105347,-0.11854076494395495,-0.6329133644429297,-0.7680180901649301,-0.46541106347022904,0.420092053372976,-0.16104134001213383,1.9765336635618345,1.2665204919876016,3.6840197210348546,-0.9886064840297655,1.6594705778495966,-2.4781963454076803,3.802623307588888,0.9358711361628134,-0.12541099880419085,-0.1616738583992152,0.060618262381922326,0.1285292492696764,-0.33194200467340373,-0.23626508403336027,2.081490781639219,0.6733485379921123,-0.5987988529008683,0.34681604705614627,-1.546286520193033,-0.16204790401953517,3.1757969579889367,0.3918503147063076,0.8121520449198167,-0.32293322781258316,0.6897979588638187,0.8481604590242746,-0.2723961148712942,2.144094088462445,-1.2968570487870117,-2.391240929946712,0.19048333676768234,0.36406553160605404,-1.0365748247027686,-1.267223256473568,-0.019256541320827866,-0.1610086160360679,-0.2460854763224852,0.504181988183015,-2.097762996911411,-0.0013927939417490507,-1.125899420728974,0.3342963804955677,-1.2977197934495524,-1.6323591496108292,1.1667645882437816,-0.6681946138729316,-1.2734878730120178,0.6214347529691346,-1.163994804352546,0.8126047290280195,1.0078234575090237,0.6247634677287065,2.1622897142381685,3.5948398869994853,-1.156679512339898,2.5667224192737064,0.5555358602077943,2.7050847319898566,-2.0370209045586067,3.880742833704574,-1.248619656710519,0.9857525351746697,0.7982703976190197,-1.4343141304077138,-0.3333039211050405,-2.3384535827882402,1.4576512681684397,-1.1240237815801282,-1.0132635952211908,0.7182120452119193,1.7037873304321776,-1.1026413195751839,0.14162112594086584,-0.4551967070699114,0.45210825397741483,0.6606787942040916,-0.14016272054616685,0.9132370593017946,-0.5914242084438506,1.2374381089195718,2.0813874910784773,-2.076319678536163,-0.009001093215169703,-0.3323829002752813,-0.554023070765437,0.5970952291351649,2.0589893501668013,0.31849895849494303,-0.10481356076243421,5.394313757912423,-0.8720675135668657,1.6328854267529995,2.235724540805643,-0.8567374365120999,-0.7075788899141301,1.5878817801646,-1.1188244326394288,2.899297570214183,-2.2852575896146217,1.084543511817139,0.15809121894734743,-0.2515989033173762,-0.3338056041390134,-1.5531091839926183,-0.795967077266123,-0.04129389125355791,2.67596744872452,1.2752022107977679,-1.1629159237205065,2.4811642356283383,-1.0120892730906914,-1.7325260925384887,0.6492079233271059,-1.36775990670756,-0.6963312823906597,-1.2944922528923855,-2.3088950398717425,-0.11282279200482133,0.23500278312067463,2.4164431359727603,-1.2251873055875502,-0.8529111686907331,2.041770949588851,-0.44751635190835093,-1.4555321799309024,-1.5068966923275668,-0.6302290466622131,1.0602159707417873,2.4340939673063264,-1.0886136249953853,0.21414553371385764,0.7532508319571564,2.8027907164552586,-1.1340487415587972,-1.1411708742698115,-0.27757863445301684,-0.8983657518605006,-0.2990110440612809,-1.5341629281404485,1.583776623887386,1.512134145708472,-0.5927787161979631,1.5001242407081379,1.38662271992827,-0.46632628413443117,0.6100081899163042,-1.9697743672285446,-1.5354236034973237,-0.033214169759223394,0.2917529499670812,2.344455126346812,0.051696346084391886,1.547743876608439,-0.21410082686664325,1.4002055075563522,-0.8077391349683353,1.8877050092693262,0.1716567727906145,-0.6990174531786805,1.169581135364887,3.6204362590980175,-0.9683684611927249,-1.6040257635812147,-0.07687473429228585,-0.6372250258791683,-1.2094957970915834,0.3031528235884888,-0.9576005959890146,0.6641058055946847,-1.5193332814806826,-1.621299517868886,-0.2450819267539783,0.11043682649121163,-0.8116336085901652,-1.5083865297434043,1.2280026775512447,1.4811842002953213,-0.19146059367455592,2.0640912828314466,1.4770418204648743,-0.19568544589709191,0.040919446730857616,0.7728299587186256,2.795412960267719,-2.593133273867214,-0.6829251132904971,0.3673652438266086,1.7699785658643552,-2.756152442705424,-0.556234575056051,-0.45474678138936053,0.9910026331778642,1.4791314255517314,-2.2915782915651586,3.1106091912994867,1.8896476214607623,0.7311662816737114,0.14763985976283273,-0.8120922339502876,1.0459713494133256,1.7514701267609118,-1.2163706052055223,1.4538490347750006,1.6305704848698037,3.453689800663847,-0.9691891162312346,-1.2261295806703623,-1.7135392502238396,-1.2183155660251848,-1.7137265176439553,-1.9671380519252175,3.2400787123041095,0.3992925200944903,-1.0477392737544438,2.6603570943426793,2.0759967987897054,-1.6711768010321968,-1.4711252907696168,0.13846770689435065,2.9309844734286905,2.091291847577707,1.1923977948253253,-0.3486856561033027,-1.0512378017890147,-0.19832050424808054,-1.5716290836812405,-0.06481940370513851,-0.39277336207017777,-0.021025418843007813,-2.28510727818562,0.9055370340592469,2.7397898461414134,-1.2055026997670812,-0.9185078071109896,-0.062107384696482575,-2.9731570641955956,1.5116760632968702,1.123466277517151,-0.1568071731177996,-0.6822198317349347,1.419470851455402,-0.44369034948369,-0.8204002153767316,-0.5182985038986204,0.3926363495967911,3.4012909722588915,0.3045312204744605,-0.45534036974745695,-0.21211121558022367,1.8415095651173652,-1.2123771306267428,-1.1913546015494831,-0.4441698476711502,-0.28030272835075704,-0.14405101474366627,-1.4250969698140796,-0.2928856866270687,1.4273099054780696,1.6424523235228576,-0.9185352910945419,-1.5597908288835598,1.3957112868918946,1.089888122716224,-0.6089873034630187,0.5072329627630634,-0.663713193397628,-1.92201266077289,1.1448596602767238,0.800051539180704,-1.416132281487151,-2.432099529065177,0.4603566176468194,0.6367337187963978,-1.3245182863021998,-1.426184707580973,-0.4919090609430129,-0.34494466631000914,-1.2707007065012588,0.06861110928502484,-0.9794968027036456,1.276713611641458,-0.8241925656764841,-0.7339409777334248,2.8412964851619624,-2.078990687177724,-1.2255227924775813,2.6200337396015003,-0.9132610225041831,1.1606022764698447,-0.23518201528712815,0.026968772623044607,2.2676039549342253,-1.4048604067804762,0.6922067506808766,2.7053818812771815,-0.7291316455064254,-0.8969446021508319,-1.6009912281426444,-1.1122421974306322,-2.0811006135119676,-1.3522402309310186,-0.01985388640407844,1.6553255001210787,-1.4278222562221892,-0.18472611537522185,-1.0354133345964447,-1.3176753701655726,-0.25472457646294705,-0.3123140383958108,-1.7155049223823304,-0.1482227096665121,1.1062663016522107,-0.44406213691550783,0.9229440171771455,1.2380135072212524,-1.2704225824060187,-1.0572857733415757,0.06573027801155801,-0.6704424021213257,-1.3974230002154524,-1.625592655489693,-0.20507632993316668,0.23336416791270873,-1.1137745118787374,-1.564691710558341,-1.7732224606625573,-0.5869205119455903,-0.3125339482806162,4.598259898545829,-0.44513841149648753,-0.8358506321285729,-1.1651573572127687,-1.0200408412143989,2.8094743097930492,-0.6858011446600879,0.028159187446349,0.7511119820371895,-1.969005221969937,-0.43634944079003585,-0.8791256702659,-1.716166601570573,-0.833946882022822,2.6306504489939546,-2.4132865898171043,1.369264059124785,-0.5230118432604157,-1.2063958270860837,-2.3663950541323833,1.4764066577573394,-1.3274074933081046,2.6931404661552776,-1.4615242519549196,0.4348325487414345,-0.7545260480604975,-0.420653673401657,0.16420614092570562,3.513103864015994,-1.5012256907340609,0.7499236077939362,1.9882508217392796,2.7724319321794484,-1.1230411958976125,-1.286699686824993,2.2667967111995333,3.106912266300858,-0.34410088141211254,2.172242250696709,1.976868850295961,-0.1052564545078312,1.7002585510411328,1.9681461505464466,-1.556553653649869,2.4000415342781287,-1.1308738348508327,-1.3231445990872428,-1.6569608095189565,2.170816748163128,3.564983684896884,-1.8642972028029772,0.1140522442199195,-1.2628756248075887,-1.407780214625348,2.554718718847982,-0.48721029715528824,0.5942730542911108,1.4272838268828336,1.8875455631180071,-0.6398618547793669,-0.3466164309578411,-1.3148530378874406,-1.3002471157231885,-0.7594990518945415,1.444046916254548,-3.254260888695044,2.789685275904074,1.9907826028156337,-0.8122563571127733,-0.4531731031877192,1.1641007498179305,3.2244625368088915,-0.7871376936411432,0.13752255001460975,-1.3940466989128153,-2.033080341923674,2.6990378293516133,-0.2787338336855198,0.5943542332994497,2.776329007612614,2.7335846661835883,0.9075046540182997,0.34058566997342865,1.869938659960788,-1.9891784975906828,-2.0726542113834987,-0.936085916440977,1.7110920986131235,-1.7469914961443445,-0.342616227059087,0.04368711142113649,-0.6604337724276291,-0.02478186083838085,-1.3933602067245787,-1.5260429838498129,-0.9801069167355069,0.6651281531168162,-1.7973654375795212,1.6935514196031556,1.5599304056587526,1.2516583851861283,0.42209223540412893,-1.364228716360708,0.3542842834483013,0.5713677118948917,0.8372544594842926,2.4234288208504093,-1.1779621111459506,-1.677773664596801,-0.14933774823099727,0.09084717256624357,-0.37169406081224693,0.7261601337438802,-0.041164845709150265,0.06270818412195857,0.2776308997299831,-1.2638613953082278,2.314442365343979,0.43340260569729866,1.701883722092727,-1.0163141701849532,-1.238336603060889,-1.4645487819334606,-1.6448051668312245,0.5406133246838839,-1.189066227003522,0.049009517226798555,-0.7913752519607772,-0.5523297729323242,-0.07220354075300761,-1.937979435111958,1.1761955606424201,-0.831935871292522,-0.9828915245501251,-1.504902348434576,0.23973425722952377,-0.48670088887175356,-0.649017411600687,2.212220259006928,-1.2396662436997776,0.2642725397907758,1.124602682377949,-0.9025220955037717,-1.398889680529847,-2.7158204141037583,-1.511697078161055,3.0454084816259948,2.6815541149180135,1.3958408937278028,-2.0242667668641987,0.14886935889583686,1.1736528050507047,1.2593506990809609,3.3467397753659185,0.9343486429724579,1.1891405171429013,-2.55339639969241,-1.7245357562130774,-2.078482560554768,0.5245803458263791,-0.45802339707769424,-2.5069151476212577,-1.3803950253892492,-1.4812342115141877,-1.3153106151493683,-0.9779030574065105,-0.11946771678570392,-0.379213626269135,0.9348766996243868,-0.45645905153938227,-1.5469813143381648,-1.8838635205521563,-1.9041779389134559,-0.3332422651420778,-0.0035174849819996135,0.802121042642115,-1.1201750213695671,1.0424870087447442,-0.27711239262737797,1.8763642716180067,-0.9980390412324073,0.2540131474070234,0.2725485650729434,-0.617575331776534,-1.317215953788219,-0.12368608098499298,0.32236836511208894,1.067329590325923,-1.0711164868087888,-1.8927740074621808,-0.5756365676987514,-0.9072544843046956,-0.9746581451172825,-1.3041890988850016,3.1057951891446427,1.4054630576488718,-1.6469263976380664,-0.23888930894126192,1.8185492433659718,0.7146040496234335,0.6081479999018183,0.5381998675430782,-2.5738001010521296,0.44909519609240117,-0.0830553623496872,-1.7594977748346161,-2.283251712766818,-0.15600132160715102,1.6689515593120108,1.3952532372449378,-0.717980217412107,-0.02194018432857771,-2.419890266135484,-1.0050773793305638,0.3266107252687849,0.4962018047855332,1.424710024485069,-1.205590117859056,0.0025790185470734835,1.6096607001904484,3.4745261086124395,-1.6816502081865667,0.057793367792874493,2.764082052212644,-1.6193988196710585,-1.0762594172467963,-0.6796291444078054,-0.02878965731951248,1.224395271711758,0.6897795034724353,-0.7620043712792605,2.446098314866549,-0.03682027494909522,-0.6671873305169194,-1.4826961373988612,2.1442983012746213,2.6570462460649824,0.07057790195234977,-1.3091301886445046,2.5007511740638857,1.052124287528055,0.44987411864743,0.3963012587322577,0.6150209646010848,3.6848515071759094,-1.2271569703998184,1.7794458591757945,-2.400669381484159,1.5705029337941976,1.3492145618895128,0.8626326706946253,-0.6994501511252442,-0.40879101517450234,-0.004787499567073421,0.2040396852845511,0.3630906385634765,1.6355536109115951,1.247155049970525,0.46592802085429946,-0.4266853646975292,1.4983858564582817,-1.450687084020585,1.1787314863511809,-1.1777514509839164,-2.1593996063374616,-0.34555779122956387,-1.4438310226334112,-1.3413807939134503,-1.0616078051892097,-0.7442551614690338,-1.2659924326729888,-1.0595789703997254,1.5035915769261052,2.474729695427495,-2.1989490736011463,-2.192569856631543,-0.49585114894990245,-1.3526056335180168,1.8692399482511708,-0.6747166724252845,1.71787803953336,-1.307611806286243,-1.3841776545723974,2.6406242486486304,0.4230785563932845,-2.328191514955912,-0.6216295022148914,0.13141978481452946,-2.2591619512975987,-0.37681198086427425,-1.0690408074491886,-1.0583904486293734,-0.16100205823004604,2.8822429060288988,1.2888183460262643,0.9967527264061343,-1.6337984102168404,1.024637745637362,0.304145136554076,-0.2157348597226023,-0.5565882225422999,-0.508153379150811,-0.6687186545159068,-0.0273283372321188,-0.5813652538684424,1.1620928024322703,-0.7198611472637002,-2.023216922102566,-0.03888050363613468,0.8261580927478144,-0.3521726464144109,-1.0364877437178965,-1.7937467916601826,1.1980054868703522,2.2145310807849308,1.4877647579987896,-0.09038913966949365,-0.39570786294366944,-0.6050193774682522,-0.2240465001688245,-1.0790052326847994,0.08703310411092093,-1.235197578929694,0.22066878832360073,-1.2306948085876204,0.3481254987029624,-0.28306681319000665,0.6829044940908382,0.13076923889883477,-1.6597344057122234,-0.13864792954790944,0.22635721216445603,-0.940556435264768,1.5875489951795407,1.3054937570622587,2.562758178276657,0.4984944427728914,-1.4618692197684824,-1.3252639912307966,-0.8944984302855077,2.076021770090634,-0.3809228949499099,0.04714001729134563,1.8544816986936639,-1.7946767357485904,1.783784934179213,1.2325886492135822,-1.158460686396626,-0.5685603087824781,-0.7825003339320111,0.8541147837238066,1.2069559744156184,2.239458440760419,-0.7005948825057638,-0.14918673227306198,-1.136920335480918,0.9537121496509047,1.477988013485612,2.3501594413759945,-0.3324254956425371,-0.09275886445731557,2.0060416894533133,0.7160253660124597,1.4403205351966817,-1.3562189095299162,1.2291572054076927,-1.9971553056448716,-2.140130747337401,3.2538630475742583,-0.24076948089188,0.8274819531570686,-0.861233167643994,-1.5471852892931302,2.683305672231489,-0.26484178334311964,-1.18088322246819,-0.005172623471469608,-1.308778716624459,-0.11214904256244636,-1.25424682001395,2.465925347098311,-0.7313637543134123,-0.599779958610275,0.7642671559149777,-0.36139170647184116,-0.8807279549759711,-1.2593930036119068,-0.024232150486427087,-0.016039573125666576,1.5356885184360138,1.0274417002793572,-0.6769027767371496,-1.2868113589162862,2.3126307744399797,-0.34446907844530306,0.6248347131410464,-1.586373061154956,-2.0892768559016437,3.022786813124484,-1.5172335478693824,0.32542336701163066,0.8673424476801656,3.650914806632153,1.2935748560136606,-1.677387108820879,-2.0220637363590956,-0.03220295819672855,-0.48220342298554425,0.6560711212173738,-1.8442491393314187,-0.7363831384770905,-1.1893226599446636,0.7661249354544765,1.0511542751647034,0.743692688145014,-2.0665183240439093,1.361263695233787,0.42228005092583776,1.3630154116509574,0.23174330014905203,0.45039103560866345,0.273120004654433,0.3394747941863054,-0.41912120888765264,-1.1162368173710535,-1.0862021680249139,1.3756270328930271,0.012861495397683846,-0.10125653369484502,1.519373592237918,0.8825352600150652,3.32194930660987,-1.7528741136537165,-0.9024732572970521,0.6420912135941674,-2.004077448753829,-0.7143498484108581,1.140693491612739,-0.2662424431233713,-2.158339538745341,-0.15268181543191117,-1.4692788399937753,0.3074647915106739,-1.8567360010408374,-0.28543669783053566,-2.722412685274008,-0.7654317774028323,0.27087104125922523,-0.9431061606018745,2.723920552208759,-0.6887767847096052,1.3757062467321661,-0.3928405206948152,0.0260166420528996,-1.2192559185792187,-2.2602530931813853,-1.2622383854423158,2.782719224648054,-0.9910308305528496,3.9512240079767036,0.005193911935530402,0.3248795913099867,3.4487740228033523,-0.013493026558464244,1.4465042959033054,-1.1751526376965897,0.6686989239177483,-1.209609414232075,0.820525497347064,-1.1169908286295211,-2.5117879879911045,1.3301535728406715,-0.5655344333984768,-0.21107755453385108,-0.844295529102493,1.6380231910193062,-0.4029156820656289,-0.5317542694108042,3.243467159504108,0.24680555294276152,0.4878646461319851,1.065239039937357,3.6621143093954918,1.568983898412519,1.28794274315823,-2.001768410206593,-1.5640968064324907,0.617297927618696,0.3830049861718352,3.3321806685475313,-0.4384967929643845,0.5168724732543487,1.0485625715062779,-1.271104642113827,0.983315811471956,-1.8922268986530395,0.7857414620925316,-0.1522383592192127,1.6781157331355179,1.5758644741208196,2.1258584383004613,0.6404901860609087,2.6872728868167335,3.655936755601599,-1.4456644938139158,2.180018748665795,-1.518877976264038,-1.4316549206044002,-0.877680440506169,-0.5752144790858152,2.6518732255752333,-1.3359126549015305,-1.0917660860404925,0.7040224532552782,-0.07208671416773291,1.2911949613992033,0.11122219137984235,-2.1558434383146543,-0.9426915885203291,-0.4585044248635082,-0.7718862686895506,-0.6534201167246908,-1.4124465319969115,-2.4378642754015534,-0.27705205817709044,-0.6784160311305619,-0.5022763828737177,-1.1004392041961606,-0.21787664357113698,-1.8168474468005908,0.35513494136396345,0.9255974362901186,-1.075679507740598,0.5247290044836559,0.22817713330565123,0.48846388641243416,2.436619162220422,1.9735632802269956,-2.3022418919866317,-0.10418825379675757,-1.2263513247735034,0.9460608284439662,-2.2424522314623183,0.09510910263282266,-0.7597340513123766,-0.6237847148240647,-0.004967070297611044,0.665844579298541,-1.6329198475666986,2.856078050176474,-1.3590382394732148,-1.8102962020572588,1.5250715611315657,2.7837885548786523,-1.9067055393452648,0.2864361010245168,-1.0485054741582072,-2.0458905076517535,-2.488539279182197,0.9382674027114836,0.33979735853357657,-1.0559048177691572,0.04094326043315275,2.4921382225799835,-0.21619938034095687,-0.8703621396380394,-2.023178747894741,0.04906033888663813,0.13881524329167474,0.22535807186082235,-1.7863879587918237,-0.6770658254325695,-2.2034247925305728,-2.5722486532996873,2.147922438001919,-1.7056350686401258,0.23102750945964523,-0.8532557328929501,-0.8397417516978103,3.60193507959703,0.026879137309849435,-1.749245391890806,2.7494716973449838,0.8221396459507183,0.8282602689078302,-0.20289519719591417,-1.092390284159539,1.4546931782551045,-0.015319462438584754,-1.375815282146232,0.8587708840015983,-0.94621530391209,-1.4922348319378056,0.5579107387728971,1.1782020031367377,2.932229975415978,1.5578529616697694,0.1955323918373706,1.408416659271648,-0.5740524938808761,-1.372029793214962,-0.7603095145536028,-1.9021058563866682,1.978491394556889,-0.5608462847028629,-0.389257404108465,-0.1389062670655947,0.6803932817489501,0.13951198831490602,-1.4995056290500435,-0.6906404112212113,0.5999114245139564,-1.857762312057874,1.9055784568281042,1.8386621750622973,0.008732836008201678,-2.4016545531402507,-1.210069319157733,0.6964672994789554,-2.3039675726053064,-0.6848674063376403,-1.5430921798999384,1.5565725482754575,1.7137638045799082,-0.4978067807486725,0.7583572554321993,-0.9853468095641292,-1.0105996225441896,1.859622241308064,-0.1342734597904581,-2.2080844806852498,-1.0095711308706052,-0.9787549397276489,-1.5346907777565229,0.545855408254217,0.021215529844923704,-0.014748003572498117,-1.906798525976448,-2.1155608105897783,-1.602648845457508,-0.15802304769023237,0.887862852927193,0.3994175014249432,-1.928675650811548,-0.40842666636093106,-2.677046798501516,1.9480630275678856,-1.170096179134229,0.3778815823061033,0.12127195584945068,0.9076547190347419,-2.9311654124621604,1.214968826311544,1.5459136327827894,-0.1449898767647721,1.447319835378499,-2.178287374187471,-0.6482238476686231,1.3083462593783244,0.8621528144731629,-0.5378278092845947,-0.5400414607797513,0.40896835009523336,-1.4580071013419438,1.0514160696425137,-0.2979227022166225,1.6387983761815013,0.5745149756384458,-0.7481571622711128,-1.2620298614629795,0.19452326905977507,0.7541910701276757,1.0457906821959726,-0.4162918561497845,-1.966488872772511,1.827115141198098,-1.6878293755328306,-0.36198051786730495,-0.07602656547386522,3.638776673554679,-0.7964423133089161,-2.7012531753716726,-0.7796396662765765,0.29617974677596154,-0.8331615073405473,-1.6865740425953162,0.7654381048434401,0.586875676366356,-1.6445081592713968,-2.2542702671110106,1.1921437396094923,2.0988694907422136,1.0629118917318776,0.23022338077338,-0.6804458040576428,-0.12306562127513618,-0.7022804038626189,1.212091329670181,0.8051559979143861,-1.7877653831818332,-0.812826918382926,-1.158983651691858,-0.3248746745910826,0.7349297379983911,-0.01972673826453712,-2.1491039844767337,-2.313828415313662,0.03435256345551139,-0.1366375877059884,-1.479544566193519,-0.8335276481165607,-0.3646183820792913,-0.5722625217335866,-0.794897049558547,-0.06667691856440272,-1.0926610621092776,-0.8019048384659097,-2.6092926740438505,1.164150344333874,1.585681819521173,0.6121628758863336,-0.5655563843645178,1.1118610854494853,-2.3402521615244005,-0.7229392571779697,-0.2777003616820174,2.088175572000748,-0.256664063392088,-0.39967701227211494,-0.17449185189453176,-0.6722633865627695,0.6242928106787924,0.5359053950875602,-1.3334768616367827,-0.6327676472942504,-0.04830474740705508,-0.6415698943444966,0.5428629552014996,0.4413880276272405,0.22170529112952447,0.21349441823096468,4.465931491438855,-0.44679855836800825,-0.4149517809546937,1.1794198662656583,2.2477185458343856,-0.5136099640274615,-0.15217943714200638,-1.5113805775868674,-0.7152357080261987,1.5091965200924886,-1.0153539731377537,0.13419588498690327,-1.389559113377684,-0.41094615577116583,0.6383308804656666,-1.608098609255794,0.036767577270530975,-1.2075365549428843,1.537408908263809,0.09192525510398854,-1.8411207453621075,-2.1344459367643585,2.1576323501953056,-2.3800297594029254,-1.2073059489346576,0.21267685067597572,-0.32885466652043605,-0.6035322690238942,-1.5958845524007221,-2.0137702038568537,-1.8681188497744032,2.2969109670837327,0.22695689352806048,-1.2294877964548787,0.4743938860294814,1.0670246703360078,2.36131544125183,0.6463105672724885,-0.8416609540203894,0.3517243022686815,-1.361725575597906,0.08426986292525615,-0.6840274809334889,0.875831126941929,-1.4471672194821361,1.3694714396649494,-0.4314292230828392,-0.7852661349664911,-0.04475436882529808,-1.7027466958086999,-0.8417061098552913,-0.31120556768046803,-1.463871243993395,1.2115181931686334,0.3634816151030802,-1.0750403763105443,-0.060687590967319176,-1.0752935773626568,-1.695113671754012,2.3991188356570916,-0.7683360395184261,0.45226317764456697,-2.4293309834579904,-2.1407367489081683,-0.5018302812521455,-1.3681977669040566,0.09604549436812578,0.9087631851079682,0.3616300688028035,2.084425515873054,1.3135451308439439,-1.0614029555841678,-1.7007967227679168,1.9523531331031017,0.5952237279584343,-1.7584887769814328,2.7265491382506704,1.9207777043345038,0.12111104089045716,3.2939031388074165,-0.5660999345105031,-1.2178032039908326,-0.24994507879529437,-0.9104704592159832,-0.4174400168289343,-0.2401307755264778,0.9757426214046516,-0.2832362942492319,0.10227999511846936,-2.163551670311461,-1.1522884106007016,-0.9750937544581916,-0.7158046652429092,1.876919899186213,0.662083731978842,-1.8422231472743795,-1.5292460531307754,-0.6390843055242467,-0.37904976652140937,0.6482386792448146,0.013954125401275939,-0.1882060038452127,-2.3814687570347037,4.806609075761546,0.4774124266249351,-0.5969318247878552,-1.4959301204547975,-0.024610631559345036,-0.9217564778648186,-0.9586390429671369,2.579018473809329,3.62967098810429,0.8727953909739782,-1.322191078938863,-1.5881476226586326,-0.6755349376382013,0.17512617962013202,-1.0623059892895579,-0.25289082706777793,-1.9387899499091026,1.3886334960828266,0.6660147322623243,-0.9646019572572408,0.642114660380489,1.851245902692292,-1.9724089352818517,1.466289050810794,-1.2529710554237632,-0.7425101658502216,0.325257528056202,-1.1508331101510758,-2.2833150214290208,1.7936567572720727,0.8928753782838003,3.537347379368738,1.8135931410522454,0.08395819605625891,-2.519110524251308,-0.09700660560518846,-0.8307622863998436,1.7927774404896872,-1.8127113586079362,0.42856210561496494,-0.5595452468432388,-1.5647568940794139,-2.2682574564220714,3.413487526138286,-0.3155798209540222,1.521742764673014,-1.2894476598969231,2.314815559714026,0.6199779229096646,0.13803254914601917,1.5325210181454694,0.5468262137636333,-2.24059586819515,-0.21778992412738718,0.6632753864076679,0.43502536215572823,-0.7983071193437699,-0.6331795536901886,0.8631665302303253,1.5263790030031268,-0.974662247325424,0.6409023321856446,0.30796692278799265,0.42010962105576033,-1.8792736701201673,-0.4784183043830488,2.2580811621716435,-1.2131141726997992,2.3960510104879993,1.1517935688190701,1.1766397980395653,-1.0610046471312051,-0.031501766924018716,1.523448194178652,3.298100437183556,0.7082239010603592,3.788649364807279,-1.5091450541476332,-1.5380651230222202,-2.3142261464417335,1.8502940332348916,-0.8473171224602908,3.145912996456608,-1.2816330185795395,1.0458893295948826,0.25246986776725316,-1.3531405409289494,3.567200915521161,-1.4104547161665506,-1.676616732994385,-1.2772646275986697,0.01279219658067037,0.7432993872137402,3.4589017536289224,-0.12596287343540305,-1.8360535276265129,-0.7786249438138306,0.3991988312299216,1.5012131251449818,0.4358937592835125,0.6908588190773064,-0.9114309096885123,-1.005998868661746,-0.13448161660879618,1.1258658569838378,-0.7925926336418776,-0.6178506521622502,2.200146732610845,-1.9228124592559785,0.9895482276460824,0.3335037222336216,0.24642509322001624,-0.8128418783554074,0.3840365033050947,-0.9349773400688947,-0.8100208138421021,3.031634985161579,-1.2450073806531956,-1.4178080425333806,2.328503130268886,-0.41467192600414676,0.7762675879807543,0.7880271776252924,-1.6720404547597632,-1.7128921747935153,0.17779017527824265,-0.7492478138306402,3.646228124981095,-2.256271576784092,-1.2763691284807606,0.04077904271367545,1.273769684769125,-0.47163409392587496,-1.4296457358897945,-1.6286857736584683,-0.953812677410945,-1.2586192863442822,1.846885661812606,1.9563176020280237,0.7171522128723269,-1.81438925933539,1.672016023391115,-0.39411463582002265,1.7593303977628534,2.9721548986114223,1.3522299972216139,-1.5838607263323667,0.6378540303819674,-1.9398043408290335,-1.7299274961743267,-1.0072047949996643,-1.0885023669405693,2.041276747352312,0.6442234949998333,2.1272333413863502,0.07369350924968904,2.328014363901932,-2.446984792546612,-1.4332992137611618,-0.7853901886247774,-0.379147217022086,-0.7128771511909257,1.8074177185225755,0.3553335228238608,-2.5925238255900886,-0.9665894491790129,-0.44738639318354195,3.3408730861803497,3.3417565345485065,2.1101398781005334,0.6585748085252514,-1.126592610450046,2.6335992306095077,2.11146959591051,-0.003731187483747493,-1.783937565221496,-0.8848402736865953,0.6148188013296703,3.85363824887137,-2.3300987860590956,1.518207783971574,-0.6442679380560138,-1.4734872722925665,-1.1764959404391602,-0.421265966001578,-1.4641908002855413,1.535359122322483,0.19561958250773848,-0.09335814398461899,-0.9848884124302637,-1.558244347340067,0.8402201222955683,-0.2997728055295857,1.2629767914199341,-0.4434061184887715,-2.0193636981000065,-0.8556493663994533,0.6532562213245625,0.7060031989246904,3.6165517667282026,0.329615993697166,0.7375027611055228,-1.239545970887546,-2.8832469282023245,0.43509321021058567,2.3164928929065476,-1.9491655717673606,-0.6804002215923577,-0.7118118119472395,-2.054851133597611,1.3335220664455132,-0.6664112040101362,-0.5896386363938848,3.078284652604095,-0.4114162849864929,0.7517166729114408,-1.228538150478177,1.7781660783222528,0.0537172241176003,-1.754127282042167,0.5429287422421345,-1.9454045620514462,-1.5435839887580645,0.7127969676681024,1.5890777482980785,-0.3636569019916771,-0.6222086579904137,-1.1968249861691258,-0.7174747352134254,-1.1410843874393277,1.858496792984792,0.3577907507646467,-1.32917209801,3.6387881529786594,-1.0872173643541747,-0.15986777078616837,-0.46410800286228016,-0.8081641414384327,1.9797585573846659,0.8617766539178323,-0.06953591376686923,0.6907348721269142,-0.840621441313581,-1.7815220189564505,0.4609800772480698,0.10898857640732577,1.167730035917539,-1.0749641265859609,-0.797438976588309,0.9944678274101761,2.3434064813439592,-0.6812558830503938,-1.0811215679710455,-1.8958894153724988,-0.034731652372340646,-0.4410354203246716,-1.4817807742758657,-0.9508497629399268,1.637544260423668,0.5518482809359199,-0.721847809646098,-1.9036185462797344,1.514610647217305,1.21838371586154,1.7289446006979143,-1.0148561416223465,-1.60716855266177,2.5232180687507197,3.6351815764316977,0.29815475646374967,-1.583861219126481,2.0829783530962387,0.25174879533876177,2.8559887955734924,-0.8073029314144566,0.32121370658435133,-0.6652085163545289,2.173038859781836,-0.9387417335739879,-0.7992363785967286,-0.6800208156572781,2.6061508420862465,0.10332496132043613,0.6516062080661308,-0.5168311979507101,-0.1074422519105616,-0.3411344545849312,1.470165524431983,-0.12229859538396776,-3.0139239477558584,-1.536222300929453,2.6401102182219067,0.911465971466316,3.038868023634863,-0.9150172553600551,-1.9869898584852426,-0.9546975844444509,-2.17342773497056,-0.5581527015843637,0.21511987918541256,2.1288228919302075,-2.456369098178053,0.11781263922269893,-0.9755179544701859,0.12171327137055185,-1.6766792547932448,0.03168807051885974,-1.9408180620970372,1.024092166829041,3.1217996950197406,-0.8546785099913359,3.4291414837617706,-1.594308021858168,-0.33888764197258436,-1.949346971138613,-1.6861247917004367,1.1667794624438235,-1.5885647373869565,4.331049215018382,-1.3141125709483417,1.6206162477996242,0.9324416503949228,-1.788801540672842,2.5188813679600894,2.388003791875114,0.2692250071355527,1.3923456337519693,-1.0554425443119804,0.8194264596759078,-0.8991274727314014,1.2388219237749534,-0.2947451182640826,0.16913463932464068,2.0688236086381058,0.09026789765504736,-0.2289466833491559,-0.24801289134830365,-1.2591299122351434,-0.19954713480162262,2.71794201862365,-1.1953033389664856,-1.0137278693103862,-1.4522773100353292,0.13916442982032076,-2.2817331231962577,-2.324579604467901,-0.03272298931801701,-1.7163180325277148,0.7379360749780914,1.866278600879487,0.02978730668243741,-1.0656998929420276,1.318539027891488,-0.6000125678959247,-2.349815122543528,1.1635255075800364,0.7611967161598407,-0.8685795597838731,0.7449389413759503,0.04984920391191929,-1.1972537000513694,1.964673433861731,-0.5156044292688434,0.9398470307535927,-1.6176967108040277,0.28716598307391045,-0.17538493761296364,-0.040757293660153016,0.6313732478558864,0.4315722643406047,-0.8082454974743173,-0.8814201475040038,-2.639427826373559,0.15725369510754295,0.19966111590500552,2.349444906231713,0.30202279425305834,-1.024474031117704,0.19857287612115515,0.332116957190713,1.6821844127135261,-1.614982086333229,-0.37084701212479393,2.7848517546177987,-0.752490176354159,0.6445609421395342,-1.4200422075802612,2.302166039866968,-0.5475106353083561,0.5513676353151056,-1.0311640896121057,2.4941485559824157,2.919124644423796,-2.6905639377493293,-1.3781272340330153,1.2100395924001504,1.9956484740748932,-2.0648975033830856,-2.560461856885857,-2.142231966801309,3.594043914276518,-1.7117754551075326,-1.4655915680605285,0.6893764825520342,-0.6641147667632736,-0.5895088894169442,-0.02502837541041851,-1.019780975629114,-1.783600525818317,-2.1114550313029556,-1.5386249845048618,1.9902268717338287,-0.5006435695228065,-1.4220818081457094,0.933815159083139,-1.967874509192194,3.3808924417077297,-1.6162888446805677,-1.1549960994129478,-0.27585579008276195,-0.39070033256629977,-1.280805082369858,-1.701543135935839,-0.19582276900460008,-1.1967869337837276,-1.3790470877929408,0.8944441422649831,-0.6053050243733502,3.7504176411575143,-0.9013838796779324,-1.6125222297209663,1.1069081475909286,0.051878277609874124,-1.7885149009452896,-0.36129017021866305,2.087659450970155,-0.8216941723309938,0.9731439802470679,2.9111178759735803,-0.1448080419695327,1.6666763551154617,-0.50524261355205,4.472193116127663,-1.332911699549332,-0.54464899823307,1.9872953193580232,-1.326397153499117,-0.7058364208656539,0.015959619674244585,-2.3031327506083414,3.6920017962006706,3.989829223700453,-0.6096233369112856,2.1379729700041694,-1.5435553428215933,-2.0495746029793076,0.8758541911581036,-1.2597670458732562,2.667715224426683,-0.6381273276097512,-0.6142869402057124,-0.23679745964943152,0.19530701386903304,-1.0041032450409,0.015335720112162526,0.11426360401659272,-0.491217249764136,1.053438642935726,2.967297876503743,-0.4484655069405229,-1.434223761240396,-0.4708671821471548,-1.5160607437043885,2.0199371677457205,2.428355730248342,0.2262450614911544,0.9436561312558739,0.9677106442771227,0.002223272607624276,3.198970927810877,-0.2681284622608667,1.6648786084513296,1.4503837810075775,-0.758168312312059,1.5006773264454045,-0.5225059941336101,-0.7674905893725811,0.014532899560782816,1.8457334991225576,-1.6279549281295447,0.049704915081589036,-1.9912468568154915,1.8622799826156076,-1.7491324718413699,-1.4218209183869661,-0.5727380313727877,0.49514698493541737,3.499323612212168,-0.11540578123358594,-0.17907807774406234,1.6385855273017098,1.7188384964475902,-0.830537913310751,2.655957050903348,0.6025479904830023,-1.5823214476953897,-0.5489404180427097,0.6085914870620445,2.5264797206417358,-1.1048221281649402,1.2756510252957105,0.16017921994108994,-0.4458450245093169,-1.2199217965238,-0.6666387551512098,-0.7150032894442151,-0.4798683174335713,0.48830348143673963,0.9044478317008396,-1.639216777470221,0.269129366396014,-1.9148183887025334,-2.352843700869564,0.4537613514294749,-1.4732041981427,0.19825959326497133,-1.858454920360007,0.2066347774395853,-1.3733645194289024,-1.794308192870203,-1.5186745697805715,1.1394765236596054,1.599193851151981,-2.1537236328091556,3.2470942777592766,0.008398121809714035,-1.0116378743695553,-0.7767058080369562,-0.8931695607630892,-0.9453215729858369,2.553106027247353,1.9098161350219067,1.0584320981392608,-1.1977870499648677,1.4161929441757692,1.5967031498082223,-0.15078809046573616,0.772919591183257,-1.3234533672588027,-0.08622935636858461,1.6357714085865218,2.554078049085005,0.36420050535818016,-1.738079917283795,1.4706050779646096,0.4953724576122842,0.6090354831948857,-1.8208086603476474,-0.16475761068335906,-2.830321682863297,-1.6046284872327274,0.27851736166849816,1.5745755816919964,0.5187084118498251,-0.23568644781466022,0.7154244007287642,-0.21232986873209633,-1.061763340460694,-1.41069361197907,0.5671515012275278,-1.6144072348703327,1.9975450986090013,-0.24980279184555537,-0.1083070035545265,-0.7904015207654217,-2.656230188049566,3.3112386205311615,0.6929459057413756,-1.6831438187333672,-1.9119871637076127,-0.018151299933485444,0.827392555789698,-2.269528659942417,0.6121282027282404,-0.11604833462880558,-0.6177738710752426,0.23114332729814804,-1.4646109024625218,-0.6162762648836002,-0.9577347179182039,0.7582356469777698,0.004200400993460851,1.1487750549241824,-0.9534825733202351,0.6773531341510319,-0.7241713835568866,1.1715158655505058,2.4139847453754264,-1.3066066408966504,-0.9702582676341646,1.8059295895725878,-0.2656417970272879,-2.242020829141328,0.4929596238740109,-1.2357328290898077,-0.5123797428767398,0.9849557033247167,-0.9725652765459321,-0.9660427393535643,-1.3108729956655094,-1.4091063411872304,-0.20482256550730432,0.7256349885259702,1.967217377860027,1.4621950839139828,0.30339813472477345,3.5058894810001395,-0.9685121744698543,-0.26962242841919914,1.4729940717086856,0.6381854411636457,-3.0582555826486346,-1.057488296694251,0.08044651108391486,-1.806068189470549,-0.1248131695165682,-1.2279810898894679,-0.13466747467297321,-0.20561355059776099,0.4345330332082755,0.40801709632833505,-2.9527253630244212,1.041933380595365,-0.2807526615951901,3.8493206754567537,1.5606865235454372,-0.6650752440927407,-0.9020709704450811,0.28162334281866536,0.7179111451955094,0.3919320815177769,-3.014487677183568,1.004472313877767,0.5444297818509412,4.00094926537573,0.0756222766304804,4.560087266898598,-1.0932424708667703,-0.6422288355775563,1.0912882807539097,1.0985964286865957,-0.2262215482176802,1.5661718639864706,0.5392850297843346,0.6147205263595422,2.724661468281869,-2.200241842155439,0.8683097333234091,-1.845537545128333,-0.008082703670774903,1.8523094907474573,-0.18375947456868938,-1.0985712855136647,1.4490323078790908,-0.6634937943743998,-1.212499424561879,-2.4704436187389374,1.547333875342871,1.2390793720435458,-1.6367508289046901,-1.8170235929082748,-1.1523974207423398,0.19179527989394463,-1.5775561139247314,-2.8226451415464284,-1.52626263708372,-1.2235758301635309,0.955160536715795,-0.5208637072516324,0.3716580681497645,0.5679975313693396,-1.0422617395722173,-0.34552106722565484,0.07459244200763254,-0.08134997672719253,-0.21615316284599456,0.18809956762926192,0.619350672618303,-1.1197783478959995,0.22937918911521335,-1.5250009047631936,2.4867182419346654,-1.245474038096904,-0.9014055390731213,1.5029169590874871,0.8246422841144949,-2.730426038733815,-0.9707899307786607,1.3039658423548726,0.44316157458426175,1.3043776773981033,-1.7428094380340626,-0.12586446055446313,0.12886716203160234,-0.7274427314505334,-1.1298050408945453,-2.0447766880419573,-0.8470036922782014,0.29614806833132745,0.36662873125220635,-0.654669683291144,-0.24063529150747764,-0.5935431625087042,-0.9656367592382548,0.07253448511846747,0.9143377803655279,-0.5346775134003157,0.11636273275594157,-1.111664538005323,2.215079274209419,-1.8518440905448443,-0.4850102055040915,-0.4665077510801445,0.4732875530648054,0.4227790060356896,-0.662169533117698,-0.15322715336016451,-0.8241990813528696,0.7118679616280241,-1.5495493579982798,3.0910067395910823,1.98848896473345,-0.3066634119291105,-2.0948178710719323,1.1156848925077965,0.9521969433534706,-1.6205636949509972,-2.6288639856649842,0.4582056633558389,-0.5643596143530606,-1.9188507618078463,-0.7746175927478158,-0.5213980494282038,0.7249032842212181,2.2380689502412214,-0.5087529216126638,-1.5271368009205522,0.7667128800755858,-0.4366872445294218,-1.4036760560345753,-1.788609591386212,-1.7605078145819313,-1.7349981948786843,2.016403073032453,-1.8642389331595721,0.5207327889471844,-2.1312567926436587,-1.5128642936458678,-1.6138122607990435,-0.09073433983973178,0.09928726754566804,-1.9230630484921722,0.2381533378817613,-0.9607043670369146,1.0481583698353862,2.7387389344707826,-0.5318169800899545,-0.8612756768053895,-1.3544508950752099,-0.19313274117751672,-2.1736371355030433,-0.12437931407001522,-0.340960884001322,3.126072200175165,-1.5394015550478393,1.3452263954974666,-1.5364210325496988,1.3876452990824546,0.7295588915745546,-0.5874564879000751,-0.5224592020414691,-0.5392507267812437,-1.4767373932429597,0.26459395111274314,1.152899300836918,2.510382051935408,-0.9306343205026995,-1.2745730724574345,0.054150987045923436,-0.5362217296755808,0.135830583482664,-1.4156888008626594,-0.7373666594081713,1.176095945436637,1.461051064199827,-1.8477636365673338,2.9098402521762035,-0.4898409968700506,-1.2903071111154405,1.2989751439519104,-0.19486259444105716,-1.1785499577513838,-0.6170660307133436,-0.7275695621865208,0.9159356606554306,1.7241807018160065,2.1600221960910773,-0.17383976859778386,-0.5936320089338326,0.41576747962923793,-0.5972633228552781,0.7263048470726328,3.0852053385562934,-0.1862172656852669,1.7412597064350774,2.342421540067705,0.11394249435140975,0.8601597911659606,1.9774333752639208,-1.1985593565280168,0.37372241527737465,-1.1799723839466725,-0.9642773364273559,-1.2151256876651044,1.0326998274923627,2.5280339335145197,-0.519292249751794,1.2929186455399855,1.5039016468196953,-1.3583974747167946,-2.2320877795434235,-0.8442761176915059,0.9073839794544681,0.2098414878061108,-0.4788226502704668,1.1880769123112442,0.3316624981848547,1.6458356249855963,-0.19514207756081534,-0.7290280046759484,-1.2707988811877673,0.132336777974083,-0.36294605980816386,-0.12513373846114825,-0.43304006356840014,-2.1709334768422037,-1.2091153805750336,-0.7161874146631281,-1.9091037933383286,0.13276761295582515,0.6710588147785679,-0.5044980474363246,-0.8453781464493527,-3.2886921383595067,-0.8760974444672388,0.3241079899709338,-1.0038623154951334,-0.5050796501356929,-0.9985988290486709,-2.140910254138731,-1.921119438350474,-1.5122835501396752,0.7086354985220539,-1.393631998604604,-1.3356895725232496,-0.10225769740392075,-1.2806314270229613,-0.015210453444832735,0.7057974363654326,1.2193571233003406,-0.6381794747931804,-0.27480350313456364,-1.6335297151706987,-1.1256663550978825,-1.472735217726622,-1.506602990103814,-0.8944357058928509,-1.1217372357945827,1.8803739417762386,-1.3635323056291533,0.40558883451160443,2.333813325447537,0.5107262755342646,1.1214237549623103,-1.7016802651119947,-1.46476565563511,1.2266020644345736,0.6213045623826926,-1.802342514827172,-0.664980121656916,1.3483664469993677,1.3604864211732675,-1.2023118282071377,-0.332884528294214,-0.8151315366222899,-0.7798708099034248,-1.4221682344510334,-1.7699584147156862,0.026474233283712983,-0.6700734610037209,2.113297733791422,0.57712092627699,1.16959306087846,0.4050394398841498,-2.759456363290499,-0.30773186859971774,0.044934649533769314,-1.6638063770733342,-0.6563400511678825,0.29312710943910636,2.0438339398623158,-1.3935244523937347,2.51197407348255,0.16875032969797904,-0.9492456242556495,-1.5634675640693148,-1.6573682603892539,-2.906983200044027,-1.343835337508498,0.09431526657630505,0.5805777313540318,-1.922857720649787,-2.2847180113968366,-0.4529253879085104,0.9563573791621905,1.4083953942331047,-0.6684833787861905,-1.0988944739896347,-0.38347234253895124,2.9616753638995084,2.7289984352352525,-0.4490203859761107,2.5390221531069006,2.44722707375253,1.4894466196620433,-1.1524585124593651,1.6310395507300481,1.8492628808343927,-1.1678646442126397,4.104211090954688,0.9678997611802594,-1.2424341612772687,1.4812452113757002,-1.1778140335587495,-1.215612892973568,-0.01901536194258601,-1.171712095412738,-0.030203056261931382,-0.8345917855928092,-0.6015037811922804,1.3879787418052314,-1.6210493602200515,-0.9598265012385037,-0.6711153784613992,-0.5380131661542524,-0.7931096410344889,0.2815130611967324,0.43753788221729617,0.5625262068378373,0.8273010164708211,1.488241976473903,0.3194041698574986,2.07651614359997,1.081358543069515,0.4860321415795858,-1.8820473071850632,-2.0035026381066663,0.38365660659762146,0.5538946349197303,-0.9177021588817136,-0.06529251470380519,-0.03547741673118687,-0.2527358904027569,-2.269605748479245,-0.7254281560429297,1.1949087308915072,0.4614675652392555,-2.2586475255624525,0.02748934469358436,-1.459305777806638,0.10251242112129212,-0.7795089716925645,-0.649253484187694,0.6796052035345216,-0.5753015401513321,0.5721977194002635,0.4642711905537378,1.655836385068114,-0.9762484395215281,-1.0960264386541951,1.1179517148045326,-0.6430391683677991,-3.2848461466337278,5.249051831855911,-0.39379493477025945,0.21309375161584118,-0.40213952275323034,1.6613544403295084,-0.5361258838693354,-2.0365889342729244,-0.9287562061755389,-0.8617188658438434,0.003069868823106463,-0.08703329993762618,1.0511137301429596,0.40806478160000575,0.06618190461016289,0.6035211696836915,1.6495098217244306,1.1087095696393,-0.9007983724229702,-0.5732566930058295,0.9454099533841182,-0.32373403166362946,-0.621660740565057,0.48614390767462845,-0.36861775378800254,-0.25389707668207495,2.755781378231044,-2.2482989024671576,-0.5004292811751686,-0.9742595658204964,-0.13392584668286464,1.5192892937798712,-1.176397989148285,-0.4395560571689994,-2.0921370526817893,-0.41520764895380796,-0.42834266956570033,0.18334031741460793,0.8166048018779363,0.2855187355277173,1.2908935163188315,-1.5667301776654738,0.23088805159184528,-0.8418300098834709,-2.890251885275732,-2.985166177512401,3.954724249939686,1.5716186148777695,0.3312398204415543,0.3365085448316529,1.3275684891776993,1.1155876964714875,-0.2394923539322307,-1.3783234389922807,-1.5745145238878095,0.007490060241910727,-1.3864940752306558,0.15623679802513563,-0.7922522212490165,2.4083277992456846,-1.7651401695175826,0.8393456562790326,0.052045158470818094,-0.5135075785959181,0.9283971154367752,5.209948555457672,-1.4867035939633835,-1.2225060370344443,2.4605994043714357,-4.102061945817265,1.3820410034615473,-1.4366130105182195,-0.21114911778690718,-0.36445916086410546,0.2514870031447049,1.046645841295895,-0.4557168509061459,1.4328311698301344,0.22829824518673397,-1.6167451440307832,0.734564333334976,-0.4993230882752582,-1.483800370562468,1.8600294116314509,1.3397774238227553,-1.1423042306258,3.0286302848024094,0.7009282084628988,1.0213162756226193,-0.2334199546826444,0.6944872318305411,-1.4758674706778119,-0.3240816175407426,0.5424766516364182,1.5034406447606712,-0.7238878441756533,2.878539616827497,2.098959476415891,-2.2734535180912445,1.1740401034379562,0.5658999516380859,3.1680919412403568,0.8210198600461366,-0.7378175244544295,-0.6606546725687291,1.0509584845909479,1.0715413169755155,-1.2600877996628423,-1.0169999418017637,0.32143683729245004,5.0078893094969725,0.22360583231395761,-2.571789110882589,-1.5884559943773855,0.7323798888351944,-1.093877654879016,-1.6841490534218502,0.7773505415604601,-0.8219958560175121,0.49833534045595357,-0.6314518201399981,3.4938705360053453,1.989312345721528,0.12550780421146274,1.1799658660856471,-1.0769457773816897,2.31822649683311,0.5063340130362406,-1.7546892312771691,-2.2965821703610896,0.3625859023402709,1.2629473348823608,3.4367459197799706,-0.6958531844276746,2.041015774448422,-1.3694233609872526,0.479468903403696,1.972975477479485,2.8860006696793326,-1.2476670889163348,2.2856391825058617,-0.979473867232938,0.049908848522313004,-0.6609210085698428,1.3916260087639198,-2.3684450077453665,0.7908774267529853,-1.4494348953638918,-1.1503200569391343,3.373708522223823,-2.050233818591581,-1.1981796066546273,-0.6269878512703212,-0.8718645966609021,0.750422614426184,0.5405043490985488,1.4600849399987075,-0.5894975537003795,-1.2282436388853228,-1.2761141991443894,-0.7045015503735937,-1.9161865435216148,1.5542356474782288,-0.5085512122861489,0.04497727894682845,0.13814960167868162,-0.6443411411985571,-1.39711961645059,-0.40266994811392126,-0.45320505392026184,1.7024406019436062,-0.7126920125937156,4.023505754131608,0.5650129848225968,1.8247371032302449,0.4182586460493336,1.2849712534431674,3.127264588740332,-0.2784647357802057,1.2508543349508896,1.167906934271851,0.5392674240025472,1.414677968149839,-0.9233753637957579,-0.32721438111224727,-1.4352442973105985,0.4673797253667365,-1.4657309111541312,0.7170404280373395,0.2771830653677589,0.7661496896868077,-1.4980316265878855,-2.0889771586472423,2.113901808030339,1.2481357780355429,1.3223326013123522,0.4294030864790557,-1.4103022186452459,-0.02357285512101329,1.5295827157402968,-1.4498725700301527,-0.8729546499459743,2.023585050686906,3.657776345844365,0.47793590517397255,2.4534638733932006,-0.3472023999468393,0.7775958918850976,1.6275537320442937,-0.08613320273743078,-0.3725648874473002,4.10299694256418,-0.4145639357860022,0.3616977468270927,-1.571595508637564,-0.9697728038492864,0.8588870863641932,0.5290849743795669,1.2891112060122047,1.8731195384490666,-0.8615346205580833,-0.47837457034539194,-0.8992268040458344,-1.4204423357475624,1.3810995915666662,0.777924333237571,-0.7709667763586194,2.2127287582256367,0.8582240028741063,0.42535398966088933,0.4928216992357688,-1.4580245280394364,-0.7184038140972844,-0.34461227583092613,-1.729381434564158,-1.264014723270047,-2.602849212047736,-0.1960768206652659,1.28424645976806,-0.5220663290735088,-0.03137905077544222,0.7626557492031932,0.6892516493417665,-0.7900757842034545,-0.2648039533493401,0.11828626323930157,-1.0640206226784135,1.952833154443879,1.0540655273066635,-0.21599016720365757,-0.13585164502137975,1.098693230119741,1.1690939692239133,2.0069842045566357,-0.42498350476445934,-0.37611047240157447,1.5740430430292645,-0.8548506646332575,-1.308792575417866,-1.0716769738216654,0.009544044362234248,-0.3340508348779487,0.6102646652696545,1.0625877759135096,0.8360623613915555,0.21721120950312883,0.018381256670081288,-0.12744104811705856,-1.3748905889189325,0.5066486265640203,-0.9234309027783203,0.1321627111025386,1.1074350553777574,-1.0301488305806117,0.8495313223979293,0.13422899383475548,-0.697816374074396,-2.435324488506714,-1.2031229830259964,-1.1413439154616727,-0.13766594900113605,-0.24515142460814573,-1.6662778103890983,0.12613685301468025,-1.6571146332631328,-0.5524602888385571,1.2533944239837986,-1.2313595034395846,0.7504088339221066,1.2457564882993304,-1.3543643809289136,-0.3856231990076747,1.2443042161165048,-0.2636328702717862,2.421521638136377,-0.14079771653124779,0.357465468079402,-1.453787592026334,1.1508581672440779,0.6357094409923948,1.8127367734559823,-1.127509245159001,-0.23536105960785197,-1.8443431842400015,-0.23380210808781896,-1.075531488168358,0.729161728361601,-0.958379821187721,-1.5520101663621206,2.095993222597033,-1.8269201168874534,-0.33718724904237196,-0.8234990704235081,0.8613896391211634,-0.05113326613879151,-0.289330909234713,-1.106466862017931,0.13804514881305274,-2.034422555972048,-2.165950695950828,1.388542962269439,0.5240157783994986,-0.15960257104450706,-0.23651814283253203,-0.8922879576246968,1.2400839121767473,-0.5795417001355778,-0.4479252527355195,-1.497449275993609,0.8357053902794179,-0.12194837417024938,1.8514051125821902,-1.827890964171797,-0.20771604731848578,0.30319465516822336,-1.2653867773762142,-1.383148477887847,-1.2214866534417619,-0.3493748863025228,0.06342803228543265,0.6833193919641799,0.1546154415442938,-0.13866569696196115,1.4737532332436676,0.3099062989555411,1.6804821802649117,-0.07507269680945869,2.244437974314928,-0.28957214647126234,-0.5015065709009524,-1.2168401181709305,0.07692156362259821,1.157368424377742,3.121758713639804,-2.1363996234131353,0.005523734890620686,-1.0421354133755936,2.630874942075223,-0.6394294391478,0.302075375204319,-1.5945686349128614,1.050021969752435,0.446047285237539,-1.4182424337114645,-0.42515011112469114,0.37935658231589114,2.0629462703948005,-0.0959035006750484,-2.1270357092095353,-0.7652562684025784,-0.6899776074391333,-2.4813378845920697,-0.05910058749851134,1.1652313220937967,1.7597814664487614,-0.6946168882497556,-0.6228403497433799,-0.35502842487015956,0.20383392959789876,-1.4634371120785714,0.6133653512163579,-1.7917397354701552,-0.25471144840414783,5.979120366334962,0.7169378166291875,0.038618606349130884,-0.5783561562754462,-0.556104400359451,0.8287733320364583,2.2484396438573744,2.854875314637848,-0.1634274619383806,1.3703159821331297,-0.5944710289236512,0.3560674139833786,1.0675477599319414,-0.38528650616896304,0.928576891749532,2.721641842823537,-0.442200111328497,1.4476846402954775,-2.1945433617267303,0.0709565795754725,0.668520718234823,-0.4262956648232702,-0.2384801869776239,-0.7761057336918232,-0.6967220538882972,-1.1263872581861178,3.913258132315607,3.851955190906506,-1.2764093859386958,2.0035560590679373,0.7431889572037029,-0.5144361424057075,-0.5394328176699723,0.45956339968690246,-0.08942451862586347,-2.323417982754981,-1.4223117040886673,-0.9620740047766263,2.4265952197849336,0.3235498048975344,0.11300966061585827,-1.055800359205611,-0.8626002257521936,-0.2675596229085779,-0.6627254943396438,-1.1391835702158517,0.9973829579188235,1.4844800973334364,0.06077187116537012,-1.6015215768599134,0.051619044453381044,-0.7534100622086012,0.32541274776474316,-0.8858837757811296,0.16211904284874648,-0.4352779433833285,1.2773205151886389,-0.6286314202733108,-0.5784406902081908,-0.9857960674471546,-0.08813723106266627,3.73891098101946,-1.580091057170416,-1.5287381478561517,-1.289636798566229,1.1295479554210712,0.038213128526061665,0.39347417013837693,1.6593617550705202,-1.3876054902445722,1.2328123515366092,1.6285688643683307,2.240390238140407,-1.2362585005619025,1.9899826038806845,-1.3133150258428872,-0.2773111956211949,-1.1575658877800752,-1.002785506683241,1.6658141774619097,-0.7448544246500203,-0.24054436882192853,1.6969488014913394,1.9347218815288367,-1.1290040607527936,-1.1245184377720192,-0.9805046199665876,0.5942662972235115,-1.010796875883988,0.4287742063133378,-1.9448740942357756,-1.2426158687643545,0.4290180898773839,2.6123554236452557,-2.175704747342194,-0.02031819016808162,-0.8061859310398354,0.12956550286432275,3.257759078469162,-1.1477860013958754,-0.5660126864167233,1.8771760072922592,1.543302061495578,-0.24002175451410204,-0.46920519505840524,-1.5650456579195928,-1.1952039679585307,-1.3917301506905324,0.8955343375680207,2.2810540962117387,3.1547216144764465,1.1357434379155227,0.6441181860882266,-0.702449016350275,2.015277138392906,0.2536503938890275,-1.6447931988446844,-2.2888485905070652,1.7693924913493722,-0.4651732074441219,-0.6707689798496987,-0.6325091497325671,-1.3327316177278221,-0.03688503221785137,1.1479257508947396,0.08844436479434269,-0.6261871260918702,-0.5263686764863682,-1.310861658813097,-2.0683857029563666,0.76548290738962,-0.4195509383193468,-1.5289563352064024,-0.8319422327206807,-0.36502052331102197,1.1630775157865365,0.13776905256249192,-0.6210711710799653,1.34079413595649,-1.0461376331764358,0.6799935894058036,0.5613591271150623,-1.8131606675971546,-0.6961191014627567,-0.4655521802691454,1.0857616372222731,1.6980999708047144,-1.1528733842727543,-1.2070790371054132,-0.7372667173918438,-1.673272454028598,-0.10862055568356187,0.10259169913231336,0.18056173574590087,-0.42231145462407654,2.5527608957486434,-0.20695935877266863,0.0749989023536175,-0.5766533001916215,1.5538474440158492,1.7604776834396592,0.23081628837014723,-0.0052828124305028335,0.35180629799761465,1.5508401921647335,-0.34559344275860104,-0.37899788393233913,-0.6317865348775604,0.8896484267905093,1.6678266667169739,-1.2266408979699115,-0.8865734487901301,-0.038442703397382566,0.7939400284750087,-0.8460640078459544,-0.5083250186935113,0.24224034583801818,-1.9912690099502763,0.3937819952524974,-2.5999121233820874,-0.5540850050382655,1.2716101047235835,3.2171560609386733,3.5457672317510087,3.163380393397512,-0.14216452689230782,0.0636480641345674,-0.608895045347775,-2.1252923595026023,-1.015631753423985,-1.814628191684081,1.7909980226210054,0.6373333779580816,-1.484097804555801,-1.1469507105552061,-1.3012224043905072,0.39786496674679395,0.7355631311076744,-0.130795793644665,0.43428277747113836,1.7703983280244888,2.431065458520422,2.4854026112064744,0.2673318378464251,1.6981350622773619,-1.561179603017394,-1.4975889617541898,1.2985181547595872,2.2265732474266913,0.409872657554227,0.5287307966491235,-0.5505146363359071,1.190211614235288,-0.1888239612881009,-1.372592427235631,0.7027694544989816,1.2237915376023254,-0.6773072596316211,-0.17202075034335274,-0.7263486284747827,-1.9193260018171947,0.24757682622901664,-0.17171820399220164,0.17731986962795365,-0.01843904451713954,2.4576632410392105,0.5524988029764417,0.9876502367416058,-1.193232105646756,-0.6527348844514904,-0.47408552803275295,-0.06401782188329841,-1.0208169652496317,0.06437960250072647,-2.447534765706701,1.6352503329898525,0.9498800560760439,0.5061992146160886,-1.4709431041805328,-2.020013049110631,1.0008267595425289,3.3479079541027024,-1.3852726044185544,0.7330792200983995,-0.9387688488448923,-1.65268964435068,2.24350872956701,-0.49575921924100486,0.4261126417364283,0.20470919204004107,-1.6272372529528645,0.4673555213317978,1.4682182493627152,1.5367143749753331,-0.6021867666204684,0.7475568736823591,-0.3159437893087358,-1.073275637035673,2.6428630319765345,1.9627951063117979,4.883660854616968,0.19568491501910334,-0.43551298947477435,-1.9722792138075935,2.768181159101299,1.4693704651087731,-0.12500258066384567,-0.8178169192483383,0.7192212532355252,-1.047571018797479,0.6609817120027295,-0.9586389112665251,-1.001324753221499,-0.06536046540717182,-1.6223884500809513,-1.6512549326410075,1.3583897524052169,1.590159483852637,1.1264948345469328,1.887870394896748,0.9852564066223599,-1.3898906724451436,0.6801253408410882,2.3765994689254564,0.3798415090848227,3.9120691739307922,0.6005511557049137,-1.612667461374956,1.4844942936238432,1.3383687046440396,1.9648338799533975,-1.6295125911789254,0.20072077934541332,-0.5742359419544306,-1.092818182113661,-0.8093547319902228,1.62061123580896,-0.12353599293802134,0.8473499308372561,1.820410441296297,0.35634064879816785,-2.1434445491848058,2.297821664274981,-0.022118464475535678,-0.23658107691124275,0.21884340528165688,0.812141929333349,-0.3229253508804634,0.25284411589206246,-0.4883662272422522,-1.261228510715048,-0.4316737055631797,0.15375698845941602,-1.5960820793420645,-2.7388027544295457,0.3476478256148322,0.14617942712061255,1.4360978403338465,-0.6361286113983315,1.2412689847407488,-1.8003997610375426,-0.5990149874620934,-0.7142068175616807,-2.340776684857255,1.2635457584511693,1.057757119310813,1.8190542254661402,1.248765597263791,-0.26712411915205186,0.5820730625788866,-0.2643314607893201,-0.6592482880680882,-1.1257634845578302,0.7644509697791502,-0.031593428804163694,2.8457629775883864,-0.8713455772677555,1.0425076344302804,-1.967647921432829,0.5803100970409526,-1.770239581196894,-0.19874301934059285,-0.6471126822349365,1.850434793946017,1.2832258465573685,-2.3568814595885184,-0.33217237814455985,-1.5828204664798573,-1.135826136690308,-0.6768258243305675,-0.6995448802732304,0.041650835577108365,-0.5830444668193957,-0.26828859768689256,2.888193927974671,0.5210591943646461,1.6345436855884392,-0.6038001874921994,4.614583891865457,1.0627090639886554,-1.7193196001045483,0.6731779435955519,0.48429211615838447,0.30469627831486534,-2.043329875815757,0.49815054177430634,-0.28105712655032106,1.7333246520699104,-0.7282179138293533,3.3987346179322424,-1.1744861458000662,-0.6858569690123308,0.3866875173651403,4.2737114738828055,-1.4198332360290364,0.6011398485445667,0.8036669117539195,1.0744041204478485,-0.903547797181181,0.5587149871459517,-1.5118132546341,-0.3412911223757945,-1.3678605738651797,-1.4696969302278142,-0.2934422708868777,-2.2776538997118387,-0.33744020377144074,-1.2443524925585874,-0.07842979845381805,-1.1061049977835105,0.21854952352753548,2.551329439376349,-0.22038583483587032,0.18438654932463158,-2.8807983965114263,0.24088192931744698,1.171897728323186,3.098855047127667,0.5832725711964134,0.3849115074959204,2.3007399680017504,-1.459047926792176,-0.868391989130225,0.928348749069184,-3.1669151553231343,1.6937891720777267,3.8685092515250625,-0.8342836515044915,0.6337903861301472,-0.838854791872478,-0.27678145475675514,-1.1558862999051984,-0.02235108316511043,0.009995404362824943,-1.5278745556639373,-0.16708414445457057,1.1809999080513836,0.6195933583558624,0.4522598881711755,-0.050414409784364886,0.9207230815973597,-0.6684128054608648,1.1242661041868987,0.7856186737653136,-0.23213408199119834,1.8441629685555414,-1.6281280014029302,0.3332227971202488,-2.0592045819308376,-0.12339155298174957,-1.1636930575877613,1.5425686843120172,-0.896371090890128,-0.6622086840772226,-0.8386896164788085,-1.6158748315651514,-0.6629662691668903,-0.6823528991393105,0.8145097410561062,0.44253319767310606,0.08809049205876775,1.807228642620535,-0.814854622017068,1.7088861391391488,0.4126817978466329,-1.091992040519813,-0.08307231155988051,1.8776516047031437,-0.8013759802337139,-1.4813559697249339,0.3885251392894634,-0.37856975281037025,-0.8010565005380382,-1.2308319455298662,2.7886979305101764,1.093106517898192,-1.314732008464304,0.09470629930219102,-0.545049524128411,-1.0515310355195824,0.1570400191826191,0.8723938147233001,-0.4205096420055493,0.4869037452615253,2.7032734393535396,-0.9441416201988568,-1.5922922677871012,-1.8366509330487122,0.0038819325446237942,0.5964119919904967,2.660197130384038,-0.43868856255953886,-1.125679009815349,2.374948580069381,-2.075312857071275,-1.886804243538546,-0.06350339765071114,-1.2863935664747372,-1.4094343390815531,0.38637636344399284,-1.659564735889577,0.27446794146868286,0.23506076465744796,-0.5469193429738782,-0.2233697002876423,-1.4206700142474518,0.3883011468031713,0.37457201812901936,-2.8076134461605116,-0.4936933552398959,-1.0937091131958503,0.8486188351495285,-2.6235709826622906,-1.6776212406441924,3.277501672477978,-0.2979428787042617,0.01773589375500154,0.6146037622905554,-0.4143329364289173,0.13177298431556284,-1.3492985134184838,-1.2288496649449254,-1.4013793107452657,-0.639802427266362,0.9440306539515009,-0.37693424812359033,2.0304105754297233,0.7532497865682943,-0.17166128550959125,-0.9898049915835254,-1.7163935843660656,-0.06035934214928074,-1.09916693733029,1.409249262012169,-0.08693083576585733,0.7776242740365803,1.4374645821379473,-0.18198262220242478,0.4588982630368581,-0.46925111742504816,2.2620931736033287,-0.8728961377587734,-0.08754128871965722,-0.5239515543734047,-2.636889335303195,1.66709212051658,-0.5173265926903594,2.0515036855554403,-1.3096653741746582,0.6690409192199614,0.2989178024119516,-2.282397950131478,-0.47217811245066615,-0.6840304840179365,2.8239232940980625,-0.5360463736785428,-1.3058451243503195,1.36485568018082,2.3441123722053403,2.543955498885223,-0.38516351719730624,4.712241734820967,-0.7060641840274042,1.5417816335764505,-0.1252657471189069,-1.135337556522727,2.5786536543540373,2.452797991064032,3.737523799516192,-1.4691775945069774,0.8428003065758596,-0.36460532019328473,0.6577222016838702,0.2613010945176816,-0.1705198825678155,-1.3905415943618498,1.358027993148344,1.8099102316120987,-0.8843571364226562,-1.9636224073964863,-1.1630976190747087,0.4920150810416125,3.4242708648891056,-1.8769706088270108,-1.5173744419390267,-2.092558311609702,-0.2380754027987145,0.7440849125054676,1.6707588177710455,-0.10259544498930961,0.9999609806244295,0.44665956756020414,-0.6141201514820662,-0.8441331948924922,0.016756897590066898,-1.3309160560034348,0.4233778609506554,-0.5764205823017061,0.597165447848763,0.3568662016484754,-1.165233049754258,1.6676334475247172,0.8584381809352704,1.2937284538836449,0.879535567944901,0.8004662746375766,4.040782133901496,1.485002735443958,2.5601129473093893,-1.9272305311865991,1.1732513918818004,1.9079089152475612,3.813518678572224,-1.8330483290225323,2.714668542036313,2.2437158005543836,-1.9294516463735696,-0.0019548560998192918,2.0400562821567143,-0.9475339968671349,0.17529272035434043,-1.8870336622408563,0.2527274279054188,-1.0544848817732235,-0.7491267535854541,1.4999011379911538,0.7046093997620962,1.069449984369749,-0.6575576387004006,-1.0828959197315409,-0.7704140277245403,-2.1148716606615774,0.29018593101288137,0.7363060218857912,-1.5533662783947233,-0.8637566662860053,-1.7180503535981932,3.657756750570713,-0.9197840761708374,4.090866414605008,-1.053989565533941,-0.3821313086520267,-1.7307904114830543,-0.5847987164278936,2.048124080800454,2.0852291319731613,0.04761911748425187,0.03989040267234204,1.6257518778242859,0.27898057125190023,2.6248835510573247,-1.657355272233753,1.2979868068100635,2.0322535485285247,-0.23744427866964624,0.6463669068368565,0.6863877996540608,-0.4299633025592775,1.7439634851396912,-0.5452789423739106,-0.1314393073785237,1.2072458936006727,-0.3542327565825923,-1.3065726483468219,0.6385897855483986,-0.8801104108721098,-2.4360321739377673,0.1404667879445166,-0.46546144168576836,0.7726884545002886,-2.3762424685416392,-0.7873907275966974,-0.5037390278740184,-2.3478838616087963,1.957243668103768,-0.3605783197008804,-1.1601329780250982,-2.28057398309321,0.754208390424119,-0.9455410899181987,-0.20411782050951094,-1.818778336327536,0.09435271221101824,2.19308324636212,2.9064248431228163,0.6155546868847914,2.262341748593462,0.9502634949198385,-1.3987636241867176,-1.1301701879910235,1.4050709063030655,-0.5308615967491415,0.8588922503723907,1.7711110548948998,1.8914542702859083,0.6745457096543381,1.3770016662769813,-2.4577242079886306,-1.678614420512436,1.197504286088521,-0.45823861408876476,0.27797437684152004,-0.26791464316339236,-0.8725573135036713,0.848840261287081,-0.06813155055930697,-0.3472812769799379,0.6818243655494372,-0.7347954945856272,2.6947260564054956,1.919882163602623,2.1593911173864786,-1.6670576684955485,3.333817784901715,-1.2305768811261293,1.5731983361369701,0.9237675179129476,-0.2554579182355164,1.7170895854280328,1.955718214351178,-1.3657375671146288,-2.1301830617390505,0.730180502033165,-0.18229638752025001,-1.0031134181640118,-0.7287346229111522,-1.5759162727867957,-0.9841743619207574,1.071849412700422,-0.6641901951148527,0.25626491467910284,-1.1753393644931347,1.3463319522286707,-0.19949247286076519,-1.5139762339363887,0.40160616634148144,3.0855005708842045,1.9412797595590503,0.4777782761143452,-0.5730697011245405,-1.3905769226347875,-0.46004169224148866,-1.6525889306876027,-0.7419257648523847,1.427572272323236,-0.4507860522590015,-0.5544852462882259,-0.9676057783957412,0.7397185374351,-0.2968410109225356,0.40734409782372033,-1.5590180414016594,-0.7812528384502336,-0.30877464584715086,0.04781186370828655,-0.41979102892836856,-0.43678915598990936,0.9971740968403768,-0.3660744593604921,-1.1541597680629814,-1.1892860361196465,-1.142509605606691,-1.20532054888185,2.124761170698636,-1.213316058358307,-1.2738836550289065,-0.9705449360440846,-0.08127749892677806,0.5572285873192722,-0.5859751677192652,2.8900320891654148,1.4129526378246506,0.14993613251411014,2.254572923043665,0.4679147166219907,-0.5504253898791465,0.4768008968451025,0.5926835051515134,0.9626524339362211,1.9194907797011616,-0.1876532890740637,-1.213835284099105,0.4709896088753493,-0.6828666672904184,-2.035157631718772,2.597667926266105,2.7553982861259896,0.6185566452221469,0.35915356842712376,0.21436832480599613,2.7062038643663313,-0.5799288198929129,-0.1578360721566486,0.2922577639732538,1.0095915038816228,-0.659994645663659,1.9532758955237117,2.0867016804702434,1.7001623559309211,-0.1420207642572992,-0.2632911744342987,0.06524226017509407,0.03265301593383707,2.3764618236884227,-0.36789992689025924,0.06503048272523268,-1.4330242119244319,1.5276080211283953,0.677106806497584,1.4667919593243426,2.4970416394745185,-1.86326215153031,-0.37768771266406365,-1.8134696713116178,0.08772138128143186,2.037131039732333,-1.3691491761065289,-0.24117878033474516,0.41286603564837737,1.7782083955444532,-1.7164162106305088,-1.1274031510223204,-1.9014352284719593,-1.343702922426796,-0.04871475599115045,-0.22335174468609514,-1.5383824425915351,-0.5517891458186168,1.6039113948184773,-0.6997852130095041,0.7370966911277748,1.0317166343676039,1.067881214006174,0.3397724682185375,-0.9250325137043948,1.8442130037743438,-0.753017582690211,0.09617575627941752,0.8243901895341247,0.9608419560967383,-0.43896373232460156,-0.39340378897675693,1.5202206817783095,0.5823611297531581,1.0920689935608867,-1.0683921339445863,-0.6445447987009595,-0.9580600681246536,-1.5605628907237032,-0.26338235395314985,-1.1367476831369783,-1.7698502803184213,-1.3119776196764028,2.3067695147378644,-0.8951164536383313,1.8958094412977162,2.883655572855879,-1.5465121121726673,0.36960982961860345,1.404553551536687,1.5104611932751169,2.952501960862313,2.0661240159844647,0.1586629251285878,-0.024174516475146197,-1.8387451465684503,-0.29456121142874997,1.5373856953317477,-2.0118839836898013,-2.196820800468422,-2.1995920817633725,1.7770255086039697,0.37342333553161744,-0.3429400344230737,0.3898234479868081,-1.2159272311565419,0.6597509179383717,1.6160622156199476,0.6050980302761388,1.8278037815348234,-1.3656408822683885,-0.9505173984141112,-0.5628237150262289,2.109204607194571,-1.1804993683299716,0.08180255068525433,-1.0062782245840751,0.4024445352844104,-0.5322024130172843,1.0462194969810252,1.1108902307385091,0.8434382792587285,-0.9735741649981147,-0.8255816574487637,-0.3832159371410865,0.8403933535258221,1.279771607856652,-1.407297937142224,0.6165654974926928,-1.4421164452026998,-2.243792228659859,-1.1886856518919922,0.829197501449012,-0.19596253863933244,-0.186337731407774,0.14539731762272273,-1.7284153745351825,-0.6701149699468955,-0.5785819273907191,-0.7714536738438156,-1.9959886303664194,-1.6298961894563186,-0.7863514487587278,-0.8358185352037673,-1.0935515964657556,-1.511531252705106,0.029119198726663246,0.916714765930755,0.5385388240229342,-2.405870597669314,-0.8947972632370287,1.5867632084456398,0.9941097761409211,-0.8297799466782888,-0.3977335540502505,3.1361162161587086,-0.977972270343655,1.1539196273033594,-1.0490258158191454,-2.1798100817523056,-0.8741860586321727,2.4149259944212518,0.6736206018990969,-1.8515638258903093,3.3679561162195837,-1.681477851547692,0.6471195439890136,-1.9241053888706696,-2.594386145866229,1.203841182188579,1.7456852046880598,0.6453725466656792,-1.3909663664717298,1.7548470527374835,-0.4529575498649311,-2.344237528576741,1.1636027824443314,-1.3990770529388805,1.633781766287437,-0.865847164927189,-1.7348803821319676,1.469762037991585,0.4080629218841073,-2.024119998371235,2.427818443571504,-0.2172978171012725,-1.8552521321280513,1.9552479775018703,-1.2607997194928675,-2.4034699077590234,1.4514286704316381,0.9775295612137047,-1.5742713776491941,-1.2347598124597499,1.2101760386062081,1.5958124663107558,2.0074997652082476,-0.08681101548149738,-1.443696290680844,-0.8968104734544584,0.7176001307753173,-1.0101311264955777,-1.5395881951411796,-0.21268145392346768,-0.8464644542676749,-0.19308048632239605,0.9964950422292036,-0.09219453916118289,-2.03053491071276,2.5446696419775012,-0.7565686185326347,-0.7955679980065665,-0.39135784575472937,-2.3307205459676603,1.3329938394567138,1.17352757461712,1.2365995141053414,-0.9332286254396301,2.3272007617551123,0.939373734424778,-1.4126632172827325,-1.1143400881053505,-1.7415693031148403,-1.0955322915851136,-0.19632860260299254,-0.6834679808119292,-1.3446703282858652,-0.44782155447638483,-0.39509094601396255,-0.051659366774985775,0.552444675423818,-0.0099083503884835,0.9652236134319223,-0.6902142781407999,1.3574234797936373,0.9325861746202299,-0.33813276821939514,-1.7854420483817712,-1.6747041357658485,-0.10519868671858672,1.030714926769001,0.149086597493633,-0.5822759985914734,1.5021719520330656,3.6437291261842426,-0.9626770177029292,0.41932652011352467,1.5905534849954943,1.0002900077313832,-1.2999438847194043,-1.2818922530593624,-0.09657720190265816,-0.6302853745590311,-1.3800202876017167,-0.8100529732251428,-1.5314335673088113,-1.1145943852346387,1.041148310793963,0.289226646577115,2.184251461968237,-0.417394253317501,-1.1190776676113026,1.7147182137743886,1.630509854466468,-2.447672278582508,-0.01608080060053476,-1.270368227477365,-0.0920826035639489,0.19221884595945402,0.5417759465268677,1.2119224517103722,-1.374856522716469,-2.15311883525414,-0.495582661551923,1.768692714539185,-0.0006663622905749838,2.816711514104438,-0.03734803405194801,-0.14543209088808892,-1.0882434805118855,-2.060869593470653,-1.287572595809477,1.1111081150539894,-0.9521953833316351,1.1411136425638466,-0.7271465439909809,-0.048556363571204936,0.888865080620368,1.335554723681879,2.1197587894576206,-1.254245308479722,1.2300231855908401,-2.1996967124290667,-1.5605000788910928,-0.019465573163932267,0.03798329656571401,-0.4924183690459948,0.9288817847451531,-0.036547798255997105,0.6212145741794722,1.5310939039321045,-0.5831339981629943,1.1170278987469975,-0.5842952344686414,-1.5129724985110726,-0.6702179878369566,2.966687521048086,-0.1063618981990833,0.577141102649311,-1.956574367134373,-1.404997079559935,-2.09604922582416,-0.4440242686849648,-1.618574133710653,-0.6599795019389688,1.043848900432274,-0.8606424921323754,1.9916401971539373,-1.218176444029588,-0.6361018230635906,0.1605811154919693,0.21141728059849088,-2.2787993557115342,0.03470913319414832,0.001306853044620235,0.2739577567031282,-1.0450462133669036,-2.60887955264779,1.9544126506235346,-1.3583754150595733,0.3415662038658005,0.5662310050693435,2.8496508205719597,-1.8900715401026655,-0.9104084376218667,-1.179361244112113,1.6563818032850968,-1.7875185703194316,-1.7261799304917858,-1.336770505813231,0.12118827102307066,-0.6774308727265494,3.159612438715287,-2.4322563955213106,-1.6052937182595244,0.9595199506641828,0.2326393824126827,0.7688744965560065,-0.4531806055427208,-0.28664814105980696,-0.3117422045524469,-1.316256004204749,-0.3887098143650641,1.2545619109909183,-1.2355406055239067,-0.598183043890589,-1.1226164420354656,-1.9829982893779372,-0.5079933417729233,0.7765464877577815,-0.23246477123139428,-0.4500132484608405,-2.2000877439639743,-0.6550544740877275,-1.1434278654931405,-0.5055258311437901,-1.7906509091162317,-0.3634981089669837,-0.88312308815728,-0.5663248965929916,0.14918560779541173,2.603869870516889,1.523281177512905,-0.1468438079295654,0.485939703811552,-0.05589416116515498,0.9863014482566435,-0.42991511620298184,0.22335819605599044,3.708085584181326,-1.7788518392080712,0.1809455840497722,-0.0728138378961165,-0.5992617189797781,0.5330542745619589,-2.3622031673185075,-2.9099829740637215,0.3714227151875198,-0.3444212027643305,-0.984458141835013,-1.672247604889753,-0.6742648589885082,-0.14737709896644802,-0.5335172950044079,-2.0607309628477375,3.009509540213019,-0.09142154863531719,2.0519108438263913,-1.3334486099811156,-0.9732159776230754,-0.07351332027485519,0.4108603607319914,-1.5668397364301863,-0.6365961224855274,-1.2455807642547763,1.3693219762933944,-0.018440712145343607,-0.040026029984557975,-0.9901133204360565,-1.40757116415211,0.19214406603820294,0.665582533172815,-2.109651675472661,1.8031909329611973,-1.7686719875575392,-0.2098760151937795,0.44017604232343616,-1.1522450282398145,0.8675816384884758,0.934745764952461,2.515592860372903,1.3587561154949916,-0.9060922367949648,-0.7550257732263684,2.1737062286872506,2.1822534332229666,1.4769293245258508,0.29337182086371144,0.013973176119515363,4.330210012002173,0.03636840262493253,-2.3865993384636703,-2.4102877745196207,-0.01934831792430553,-0.1280612196790517,0.1152956185134513,-0.8617447229002408,-1.1997760800458015,-0.2741629556782458,-2.550123111011718,-0.7395823654824006,1.2723242602063027,1.3807676484868916,-0.8614716537977003,0.004589690040915175,-0.2693859251946278,1.4417573861436073,-1.6160266944644464,-1.3059873298409765,-0.03692211861196466,-0.07523814695736952,0.403711156238776,1.2618668801869126,-2.4980076241924927,1.0888060628725764,0.5941105633824587,-1.7911539735482829,1.22851912603801,-0.6704835231081777,3.438321383339947,-0.7275711074329185,1.0778069925416138,-0.10222464967136666,0.16893953240255438,-1.6587418198849817,-1.447266230461574,-0.7763782196288157,2.4606557135795217,0.09815070228448021,1.1339627965002412,-0.833057128296354,-0.7791401190129456,2.034398389670416,-0.14677114412492734,-1.6003492573758065,-0.40966847311538596,-1.827993438066452,3.652369864043067,2.2486922886173075,-0.000875193523106084,-2.115247080630068,0.10869733247792654,0.18998302033663725,0.4883568624149586,0.21507127559787853,-0.7772217389856837,1.8600303707366517,-1.0836085734312675,1.3095076033046484,-2.1177608247481174,-0.3208678851952898,0.8098498988457428,0.25789831721463075,0.10664092186161853,-1.0851540293551378,0.8557546690137462,-1.3832396345239315,0.1809240637169096,1.1965108637933437,0.8557133007653683,1.80954760046739,-1.206130187576686,-0.23509774118705848,-0.8138194115114602,1.7794635386994786,0.18636396256008197,-0.5082516782812189,0.7164717451376573,1.5351421609640472,3.5788858973881577,-0.5035468930487252,-0.28797480287501515,0.46284783352626035,0.23834333172077968,-0.29324459726464164,2.13390269151464,0.8839030787625465,-1.2504025592355223,-1.2123727612639157,-1.386636080995477,0.8647250096264208,-0.380829836920677,2.4199224524180387,0.9165513717010414,-1.4278619134111223,-0.510416965720498,-1.4633126886192795,4.56042695766853,3.1039005662272685,-2.639117474831331,0.2621893473635777,-0.08552147905699486,1.8776747342204259,-0.9203456146284373,-0.9722160610402439,0.3318263957781136,-0.29282672560258993,-0.19873666286621522,-0.09490010002393524,-1.5117801195670995,-1.1509168526053797,-0.3018384186450554,-1.0359306451863914,-0.40985978737909146,-1.79658087837734,1.0621129260655926,-1.1388882866408048,-1.3068558837263906,-2.0102956160178715,-0.7818962369277276,1.8013158479548599,-2.5052114507102674,-0.9563287869467918,1.125689739572172,-2.4835912172735606,-0.05317825386140385,-1.4552461283419953,1.9824064590195518,1.0446809851986965,1.8246889763096785,0.137674291476852,2.1779122210059594,1.545198522554532,3.1037313073940958,-1.2053260562441237,2.7869944913646507,-0.9546250051084663,2.8936754351342744,-0.8934998934830981,3.695174212757164,-0.7973712859068185,0.2820377478226479,2.653101410484277,-0.37287317668082043,-1.5674073663556114,-0.30111950551721556,1.8081998845091496,-0.18282424754113952,0.3810266183828354,-0.47784754173208016,-1.7826724225230957,0.381509379731976,-0.9725727680589111,-2.2324263539346694,0.5792677895350312,-0.11872262921346957,-1.379822789165809,-1.9759565164447628,-0.5687108277381678,-1.1245705364967478,2.194073864520684,-0.6998692536339242,-0.00441462426476382,-0.7888594913891461,-0.42501297342812366,-1.4058571101477295,3.8712282587017315,-0.12415715218688908,0.5089911614361696,-0.7668962061336174,-1.4928134614283814,-2.603306408820796,1.3872720355803168,1.434813822793188,-1.71435311375208,-0.4747491879487937,0.4760961054099086,-0.30058652359733284,-1.6635740728623583,1.1819746258355894,-1.1460744099489188,-0.5175930752121717,-1.8807636348862593,-1.5504364637540122,0.8938201939516921,-0.9971711918405651,-0.9979225229346049,-0.8085068735366615,0.22020865643117096,-0.8079277327165869,0.3240799497312276,-1.2162210588585523,1.1146339548103406,-0.6376339688204999,-1.318182512712056,1.62369086646723,-0.5205987124439666,-0.8044555951559357,-1.2956561929045707,0.33645249556392304,1.6492384823578208,1.5525013874730875,2.3744093523054377,-0.6901469731438905,1.6570089181677594,-1.837231405738012,-1.5640750815976925,-1.359034931652737,-0.48855898575424767,-1.2244596081090084,2.959634115196247,0.4033612661776619,-0.09898437459634035,-1.248258371629282,-0.466212685697262,0.4927740484176587,0.015040089750201276,-1.4116942772741496,0.7142990977169831,-1.4871115517852305,-1.0703558208570532,4.5148431168233625,-0.689278817135823,0.5901197819802385,0.0263537085969008,0.31833565328324914,-0.7992870010562027,-1.0845104100919083,-0.8993852361288466,-0.14537687044400718,1.8423349642197389,0.22665433996652545,-0.6229744489018021,-0.7174304435995128,3.4862743612841007,0.3839710785382249,1.8522049733088988,1.322611649381012,1.518899819557181,-0.2660358051673793,-0.8937897990779166,2.828345142231816,0.38530881278607715,-1.4438829797081565,-1.4390370469639722,3.360003151654353,1.0367894288521062,0.6010996265613443,0.11183263830743521,-0.07941616200603328,-2.3447860637630256,-2.301751541182981,-1.366242155598707,-0.4575047706572928,2.6827566587195606,0.7239439984142443,0.9864680507514119,2.719110628939659,-0.5523979244054673,-1.684063527597778,0.9449794087580824,1.3519648809937064,1.8530509934737711,0.11722331372801535,-1.4704046930213226,-0.2679405400024109,-0.152393813891093,-0.7600656732213174,-0.5730491211017706,-0.8477095205506042,0.027752045194141258,-0.8532924989222768,-1.6710286057840404,-0.34706254143881177,0.5160433768389419,0.9094812180514552,-1.307548597117956,-1.1913398458440327,-0.2695100343843615,-2.686312189301105,-1.0288092332608731,-0.6789934857333193,0.8228761146578448,-0.317044445528865,-1.4478039173557202,-1.144972691477122,-1.132398936771318,-1.012409607072261,0.7616555357152025,-1.3845693283567972,0.8318966456409895,1.115482620043231,-0.7695088962773311,-1.861373964065852,-2.201154263140542,-0.4581525500391384,-0.5253007032769252,-0.3442623725696666,0.7956184957431667,2.421658327616669,0.673953467537398,2.0075413086096,1.6827729648198664,0.4862699197755387,-0.02578821851819592,-2.0171423762979037,0.9434291198224761,-2.2431557455374445,-0.23529700589290256,-0.17143726358561692,-0.12701150977064068,0.07473307088922468,-0.5081001094751617,-2.0947591660453395,-0.32056577037269807,1.0557779130637543,-1.0878969483157646,-0.6391094871261972,3.3842349638979936,-1.0263187744973976,-1.1314834661170876,0.42369279654466163,-1.0206986381574805,2.264525553091905,-1.8296448308189506,-2.130438424049959,1.085369096396578,0.7045166390405639,2.1571859320423,-0.7702036673678928,-0.506281989811323,1.336925446745211,-2.3503651960428957,0.0847481137724174,0.3761059677541482,0.03156149605433052,-2.468282882039733,-0.9962698543622525,0.6089949749094008,1.6437358958548,-0.4294954199701526,0.8526732518985775,-2.6489879748796037,0.9078350853270736,1.2115666862335561,-1.2367009195421097,1.0508015800482013,1.6408952068362743,0.6475832187487883,0.0980023476758376,-0.6466124585920165,1.728306226718959,0.6910615152787309,-1.3492080108296163,1.2915913830842105,-0.8069741172301013,-1.91017186499031,1.5849328614332294,0.5333041283056591,-1.7470565833425904,-0.6352497261368808,-1.6895820310170238,-1.1979188291375704,0.37760805304262746,4.280434581312646,1.7239284925943654,-0.7992755948167731,-1.4516347756923464,-1.6929518047336016,-1.0630844643861577,-1.0626250135305433,-0.015557417783172468,-0.03398692265714454,-0.5967927492248528,0.6361669716722222,-1.5317823368990438,1.7163234597413048,-0.5825242839644199,1.239489883506301,-2.692250408131047,0.9644305488426416,0.7385606612547524,1.1214997760685155,0.313051544579939,0.08082422004060665,1.729840738613022,2.9303726769152423,-0.6064294674114016,1.485326839798875,-2.0733678153010597,0.5850858879855304,-2.0563642793855643,-0.977653710423151,0.0973610649700611,-0.3804637973154221,-1.0516246121751986,-1.631279673084593,2.319759128920738,-1.9714880301571986,1.1319543104357004,2.929241606049762,-0.7390399246739122,1.0566685097086388,-0.6465308457696163,-0.7024373720930605,1.9085856352071138,2.1578715051597697,-0.07341736922497288,0.5100898939712215,-0.7043556850982582,0.5212794465761439,-1.0209668463096886,-0.6943935874209514,0.7457202079709535,-0.01079356567246193,-1.107667964030708,1.365716475878492,0.6497936912304068,3.5548766936943452,-1.6909049871871755,1.0418335733355946,-1.1288182396964574,-1.072471439695515,-0.5800785744007441,-1.3995263144716659,-1.3853959273385075,-1.6797630892474762,-2.052799693017138,-1.3187070502047373,3.4496464741226047,-1.2967731137397855,-1.927761604144804,-0.22137312805225387,3.7400097419549043,-1.363390776677832,0.5066893938998428,1.6063432138062923,-0.1904407536593247,2.4619698748185326,2.5929359057779693,3.8620106924258044,-1.1935024231989626,-1.9328139493919645,-0.7487488960547789,2.3699708707008904,-0.9569362921578451,4.700147653631252,2.609516534700756,4.102116889798358,-0.7391465379480043,1.1389030886867184,0.8573000428663531,-1.9118762474174764,3.908070874619521,-0.5498292257234001,-0.4044399088471433,-0.23937794215551556,1.5664035894729849,-1.4765724766492234,0.8291673410587761,-0.8963566092326251,-1.166650391367881,-1.5276424724097286,-0.2601379431919262,2.065410026665807,-2.3522712015317286,3.4581877662004032,4.013122493850661,-0.2566866509302683,0.9430097778663505,1.5707176422491345,4.1911308748146485,0.10307259907509828,1.4044498317113134,-1.2145773186929236,-0.9482777199297218,-1.403373572334891,-0.06953454340786812,1.0952419258913768,-0.02748376656224118,-0.8595556929135204,-1.2548725340044091,1.839754367218374,0.20484812654874884,-0.8117323848090436,-1.2415930070168828,1.5520144107895442,0.11386596220273365,0.964391379074687,1.9065593700767969,0.6731611118803744,-1.153294220953229,-0.27294965635334567,1.1378092451145474,0.6089438285557196,-2.3501689831244135,1.6592666442719841,2.205944433564682,-1.0262198530607198,0.40204478713925457,2.4784969616395514,-1.5147356326776278,0.8988216996025181,1.947151284292714,-0.9941644735407912,-0.8868201826897326,0.5183354313032829,-1.3709804282264622,-1.6463047049971526,3.0325938261689633,-1.0311900685379356,-0.5798150491651856,0.42623755704036465,2.1365734477454104,-0.6106179171308366,-0.8891177510011844,0.7626800012873152,0.70968898318387,-0.7650933005372228,0.549568099158935,-1.1737010865491686,0.4568621135467183,2.0965195915891237,-0.5423948857584445,-0.6410881041299334,-1.6349757095543973,-1.228318092187291,0.5876506286169543,-1.1878139876247136,0.3401289798849878,-1.3277157948509133,-1.4668824953324686,0.9622993856846062,1.0461030073546698,1.2945916193611193,-1.067395088634928,-0.8107518713875481,2.0550437547476372,2.5000117841006424,-1.4869522409628595,-1.211682192797997,-1.0937535736691184,1.8321784327748,-0.3433235411868394,-1.077419921287024,1.6038499977615222,-0.9175089417782472,-0.5589260797566583,-1.4504592298087184,-1.5922926757266591,-1.064505573571523,-0.9576877505110404,1.4955277093230297,2.3431197305014635,-0.31742456627818966,1.6668316016656732,0.3909296825297394,3.4669165020264034,-1.036230341310919,-1.7431115952385088,0.1283234643992752,1.7947661050812946,0.45638812872897727,-1.5164163467341734,-1.4166122454009427,-0.3595220510899783,-0.9991484615261073,-2.1197853147416623,-0.2957555178909385,1.3034600158737095,-1.2342252609657582,-3.022435062365961,-0.307838300347482,-1.7633713245061466,-2.6725425862093046,0.6592417086984413,-0.7308075466122449,1.6473384000645117,-0.3894382660246657,-0.2518057282153401,-0.2292323745393857,0.9793742311128831,2.478786803093794,-1.287325565852225,0.9034551096667504,-0.43084574950662,0.258857742947604,-0.184553804288948,-2.066740538058535,0.49367034647717517,-1.1283483472687457,-0.49173046588960656,-1.9701738484646834,0.4995793934612558,2.4244370204015886,-0.31587477561295213,0.17463941696826663,0.6726031470782885,-0.08866403485527075,-0.6538329671941767,0.2529577173802784,-0.6747140888136172,0.2811657937149439,-1.3820485139367624,0.7497941398792133,0.508610668133507,1.4576900751893553,0.4811554399779625,-0.42709115659109825,1.1942810838167968,-0.11810605284553166,-1.376306525674257,1.1464631033539192,-1.5586955243638319,-2.306746009738205,0.3569487687010565,1.7504092948418166,1.0946200264732415,-2.128991667812655,-1.5074966102767888,0.6627182421057721,0.13606307509464557,-0.6101932542073369,1.48497793843218,-1.5516430986025271,3.414723787539265,-2.258275582790827,0.5204140915816792,0.2283141193625846,-0.3330270908931464,-0.4445712056223846,0.35623293126619493,1.9880660087142064,1.1485675502957982,2.755562561835328,0.8362530995318143,2.876974873371548,1.0897663274083207,-1.3319864565110353,-0.24823210698300507,1.1885405856335722,-1.9157843931000755,2.6029410483601176,0.5986272082415636,-2.323031238821407,1.5201149154777307,-0.8458264852768488,-0.7522161136506993,1.4037579647001324,-0.7284912250719574,-0.19613535597319706,0.13837274686220974,-0.20133242186893296,2.3916565544003436,-2.19144354905646,0.3061141800926894,-2.754292195764199,-1.501316239088886,0.9758627096584069,1.6593009613196177,-0.4021059993256384,-0.544430809303114,-1.9188950506203304,-0.4581939930035598,-0.3510204129593235,-1.2113126915666927,-0.6308340776507773,-0.4994226147113517,-0.5183708145635146,1.6191358847266581,2.2061866621964343,-1.4501577598311632,-0.7591278418877024,-0.8200711468024063,2.653016005610712,0.8214304291872317,-0.878593711103485,-0.1369626693722091,1.1377642128276946,0.977110806594185,0.18896183349179138,-0.6160342589038177,2.891124788192839,-1.280761715151321,0.8084583347884587,0.6259935672335551,-0.10416394758546718,0.7716983318857602,-0.7243251873461141,1.1167843189755216,2.5601620514797454,-1.607472505128082,0.36647276656914457,-2.4272876850824976,0.8431750588383855,0.22720383426128352,-1.965014708813528,1.3689089674984807,-0.9278319346475962,1.8335395829752124,0.12907515371165654,1.937855290456099,-0.28338963737302536,-0.8821862100430632,-1.7164904093087718,-1.077389719445047,-0.12477936921589253,-0.17746447852724165,0.21320641239044527,-0.09929207953179015,1.148523177505346,-0.01890742088021568,-1.2923657027240447,-1.5157117849353456,-1.0522588556627848,-0.11858934095167524,1.861658517608151,0.2953222045542099,-0.1204968519482853,-0.28587058403590915,-1.729510106872501,-1.3759196080197587,-1.1360969636851725,-1.3752142700798304,-1.8855093687506388,-0.3353621489822998,-0.9804439674402452,1.3508578776213098,-1.9332556330407242,0.3320317608260569,-0.44191003823787295,1.159454413382234,-1.1902264640420925,0.8247106046145154,0.7380716117495809,1.510553934049797,-2.1331120074718584,0.25550566181186557,-1.031955097523157,-0.8007165536993154,-0.4968588552932251,1.689617475936092,1.587590422612131,0.6574549789904923,0.3519377132612502,0.20642808578948352,-0.7255885791556628,2.9466329964091837,0.6692444711236867,0.7072031933668791,-0.983227715684731,-0.2881425783862509,0.4375877205961688,0.3019704965325353,-1.445502504181578,-0.23072658530761994,-0.7932378812036438,-0.25926623594425846,-1.490604991022327,1.976560556103681,-0.3046430683599514,2.039781944627668,-1.2915804584924053,0.5445837247142601,-0.11883095049450944,0.24443340565027483,-1.6393736176541969,-1.0422732586196737,-0.7936028423917372,0.8306617795225625,-1.1451339155777556,-0.7040399777161762,0.3323750573099358,2.239476949058005,2.2247026147340714,1.0074572424283175,1.9723569343699812,1.880869323333346,0.09539542845431472,-0.40668877662296354,-0.17849157677706073,-1.5519672616898919,0.1329758665195087,-1.3234614413919537,-2.057935508274023,-0.5863588100270638,1.2543268975777122,-0.8920597605618148,-2.224400281744606,1.1807250948210548,1.2917614797089112,-1.5173620424279641,0.9286517805809722,1.3683552153995704,0.5655481913303225,-1.9394346364188741,-1.513281164805337,-1.320907358429378,2.5042836200637506,4.086480190844491,-0.06012524524979495,0.9489695736496514,0.12304547992157977,0.5410047246846132,1.4846812804510414,-0.3114837735079555,0.39541429489526,-0.9645553862484929,-2.825422320106913,0.4747145115603096,-1.8575376963824708,1.4207936545390363,1.4456134809337982,-1.1403574961537193,0.8963606631076225,-2.1335189261167455,2.327948272357747,0.2630263430055213,-0.181770271720557,-0.04226694741786347,1.8076520401264933,-0.9404517712975121,0.2028606515036805,-0.9963543607748521,-0.3354548245877434,-2.4489132815995274,-0.3678591085793486,-0.2766623580839878,0.8512356718369245,-0.07401299090841558,-0.23519116854581162,0.10045811404742745,0.383234126387828,-0.8298723683084931,1.6461927925267632,-2.3428941049395204,0.795442431795236,-1.2613442683550926,1.0208323744109253,1.0840035878011374,-0.871298110934879,1.4722051735216062,0.41315041633998734,3.744283948420533,0.7813415801590844,2.5397988616708176,-1.1708607046466064,-0.6914938928995804,-0.8933490144391223,1.0035768973865495,-0.9544412560183962,2.6218544486706876,1.7492667938588617,0.5928081418540936,-0.3105892194916556,0.7038435471497102,1.3638295141944223,-0.4691487235109893,0.4273595050253434,-1.2735808288074055,2.4582386747654716,0.5012920842330904,1.2522257532060663,1.0421495156883644,-0.6475913522585744,-0.5521472494076722,4.226015652689863,-0.7766066913430492,-1.870947639584111,3.054574635171993,0.8100017091437817,-2.514842537679585,2.0337600891219725,1.2079446585848692,-1.0128427101993724,0.32246405432103487,0.6570118499449544,-0.9812243732609931,-0.7309247530119564,1.0166636396548232,0.43935966711470764,0.7972056993429931,-1.7842330751153812,0.17240986850213516,0.1715011646362016,0.34367269035241943,-1.6096454993952067,-0.7881005831176771,1.9760416698709342,-0.39151141745244217,0.09683690482908851,0.2190172986656646,0.09341526409476511,-2.3547499120362407,-1.8256998387088006,2.839307536917256,-1.1727138498341814,1.0562226603299192,0.21749867954798463,1.17862638914153,0.8295851929652691,2.891638027698649,2.348967214924487,-0.9762408613021788,-1.7525996067176306,-0.5728612933921149,0.7669115222572774,1.087349867715635,-0.9408874447515909,4.747540482201063,-1.4133232154447197,0.21941238165445678,-1.1742747097317274,-1.5979508935786997,-0.5940053602980266,-0.6160997465845096,0.8144667120006688,-0.9014344460958774,-0.5408833228440549,-0.8318997993968393,-1.2687182522612952,2.295591548340471,0.5457037469396878,4.657115957684526,1.5007377549121712,0.11590030552331988,1.6143720207184185,0.7475010395025186,1.201144407151712,3.2771121543963933,2.767178502813543,1.263589623775162,-2.223266425712999,-1.541112393562536,2.1362241719538835,-0.023898761222235883,1.7399830579673654,0.9786843857502917,2.4418241784403163,-0.5743693167598243,-1.0334984848801119,1.14147511444438,0.2162359903244805,-0.7899794357975295,-1.9630143715796677,1.213924753892128,0.6416618111970435,2.7675245004310436,2.850500361789338,-0.671477216833354,-0.5335119496371492,-0.5172434878269615,-0.14935201754729877,1.4362619172732196,1.0341833297034202,0.1347425881325623,0.8483142311077528,0.516918605938092,0.3688822313623952,0.5419996892920431,-0.3011799954219014,-1.8553557828906317,0.9591723779581851,1.1516907186262675,-1.4859724854206693,-0.8204397272852779,-1.1287228249838799,0.3094784951368825,2.1154598120486074,2.0717523859628555,0.95076909336125,1.1556369853356665,0.5710989154234606,-1.6401645971656793,0.42870584831286235,-0.8580421117896916,-0.2530043030979345,0.03145503405419764,-0.7537272669131407,1.911079600937069,0.309760776377027,-1.0908349218907156,0.05726032927305432,-1.157292397565103,0.2971547148629448,-0.15022410650912177,-0.6614155911555698,-1.5628646452692538,-0.9163876263643006,3.6703815080536435,-1.1508547091476349,-2.2733534719240294,-1.718507274559659,0.4419534188471378,-1.968023473560874,3.180677549857306,-1.526415698844329,0.4948965494114017,0.5445351827897732,0.14561953962642002,-0.25641996621708074,2.261339854990066,-0.4023384340996494,-2.2742954875949586,-1.4669991457884493,0.9656458285725636,1.548807846106292,0.8727857615105835,-0.5348061725320156,-1.0294462026864433,-0.6309601729567124,-1.3459402370631806,-0.12466682644361848,0.4013413658695174,-1.2163803460114064,2.3190831145912885,-0.6625110791035523,0.07126691158253091,-0.8232582816682396,2.936306079955606,0.6914078032816509,0.9164346078853703,0.46630474329170196,-0.03253504232663731,-0.45096585415125146,-0.7787386464383098,0.3707700277725765,-0.3281500940627801,3.8845500241387674,2.0315923895005756,1.195022116593703,1.920543804456783,-1.4684090798598417,1.022551822541836,3.367388436927706,-1.0859578887749737,1.13703026887848,2.469660986096528,-1.1520474825502247,-0.30474945283836224,-2.3848716652609663,-0.38579858264598793,0.9170005541032766,-1.7753420120122143,1.8358854658821961,-0.8926950987540143,0.4682919870940466,0.5329645956076418,1.2699067455920454,-1.6404902776411014,1.58636877424441,-1.1019950698026466,1.8590938975065272,0.4526450981829014,1.623664147710099,-1.6374564630055621,-2.0164453022269835,-1.3407801459270827,-0.15916312643684188,0.5414683372215431,-0.5364380986347773,-2.048701268101408,1.5260227059199831,-0.16255960385767063,-0.472202403543255,-0.8722784272828966,0.7989144174208908,-1.2069973663562648,1.0681805819020995,-0.2527326668811141,0.9423523153947609,-0.2929709403615592,0.19838631011864885,0.7058864316864466,-0.45700204931632354,0.47491550978390923,0.9986338579450051,0.2916569098215431,0.5588894945316706,-0.6973271649972004,0.9064017559846048,-0.09643286124137859,-0.9192664649541531,3.1317478775896257,0.5577294038110009,-0.12809986344186394,-1.212905796827275,1.3848769559402037,0.2618997354438785,0.3865492856944686,0.5025708070771449,1.587054741710908,-0.4536869835973477,-0.8316842578556579,1.1030666058629592,2.323977537159258,-0.85456385255571,-0.48838197088251695,-0.26804287486276457,-1.2891515741525066,-1.91760517742994,1.0903598526917209,0.7930073334115151,2.1883697141380383,0.615353004075078,-0.06304793443701678,-1.0926880515470634,-0.5827801368148569,0.9739256004771518,-0.5838836606552666,-0.5075789147049462,-1.644349362357262,-0.6959216838206919,0.6368753848466756,-0.6710134218201512,-0.7456361477963057,0.07038473142344127,2.1939010993350188,-1.1138311440226853,0.7342677193500927,1.3046258152068684,-0.5027543205912901,-1.2492608287436342,-2.395003071066965,-2.2925612121970618,0.5309378847290833,-0.6403731856942998,0.8718300151725787,0.07735994221350631,3.042907756929747,-0.3823364421487475,-0.8777231075446432,-0.6092386806217984,-1.4436855510813082,1.2559671163425858,1.0218772350491723,1.5401213327642709,2.4793033119872305,0.11563369055348188,0.963660335920602,-0.4262526157693319,-1.630785350944142,0.4579123086171763,-0.707865755505429,0.47736014665409776,-0.9538001866973186,0.0957148126894716,-0.6524433913086205,-0.5274152265703466,0.03906747370855056,1.40097927472621,-0.46494173364497954,1.6946347145362175,-0.6262797644985791,1.5572057321273907,-0.851529269536778,-0.10375684260114558,0.5374725114402084,-0.0008237440214013614,-1.0288490294186916,1.1122241630520242,-2.2142950721410672,0.0022400679685334974,-0.31521242481415856,2.832530919441685,-1.6357679243492649,-0.7451138274997606,-1.5296214489794246,1.6679332257517676,0.06382547734677331,-1.1796580177176061,-0.7107872964262563,0.4708323966850604,1.4873483157053655,-0.1091977917910971,-1.5306431310611137,2.2756765637460004,-1.8708452995944105,-0.15224756347654492,0.21457210296388216,-2.0312599837023444,0.6604207244152661,-1.307126661698301,4.1688298091078435,-1.5940371473326567,1.8863310059434912,-1.885118337652004,-0.8339155921835639,0.20446525219436032,0.6272162933293263,4.049489509326318,-0.058163262485079505,-1.659512554706554,1.981431272927544,2.2597608345956064,1.0570546220186567,-1.9124002160135527,2.623278696341774,1.0537331028655978,-1.5348567730459577,0.637163858974565,-1.637002906574684,0.3884907896224187,-0.8925077901222822,1.235049297178961,-1.2678670868483324,-2.6424445631110136,0.32347327376306456,1.0115787734126092,-1.5594237665999495,-2.135910830528477,1.219760724742898,0.13939335697883673,0.31467483694259807,-0.8339336369054664,-0.24947119272084786,-1.274731418339327,0.4119028820562872,1.7543605780156153,-1.6434169297175063,-0.26094409417986136,-1.6379532270546782,-0.5244494681005603,-1.7397394711175094,-0.35985319672208443,-0.27043901032845924,1.6390293700407315,-0.4873075236120732,2.671468886815647,-0.9670812621935836,1.4877559965685003,0.955587551012042,1.5810045591308421,0.40061631387139074,-0.8716647563157917,-1.286904075324061,-1.5556801713573238,1.5622997066447344,2.041041030474758,1.0763314457882467,2.338181321656067,-1.4891043800014174,0.8940025855795543,-1.3458463863250045,-1.047247538299717,2.9170917666177507,-0.1981531839636319,0.23667356539040724,0.7999151057670946,-2.193420027222762,-1.4198093798731057,-2.1100891656913983,-1.9129431378709785,0.7003966539597506,0.5449156101230316,-0.443903891341743,0.8224050776367067,0.5569989462796765,0.0823222056474059,1.634802325208459,0.2162629384689432,2.219911552756841,0.9155811998707735,2.9941873595834916,-2.2590195250562175,-1.3758386847538546,-1.1698107984961974,-0.16884449036310942,-1.1995328026333654,-0.41839559625646255,1.6165963878648941,0.8540642096842984,3.230319173092659,0.02948793725321164,-1.6684962867883655,0.6233740159870861,-0.7830516745731856,0.7254143603116348,-1.2281089557054916,0.5512826277229083,-0.1043501759038251,1.602369877878452,2.3622923714168844,-0.2423786541923766,-2.268201647347113,-1.7519560056226122,-0.5535363418403242,-0.04126903633751842,1.1549051168924052,0.3376033747553357,-1.1833056338200023,-1.4736987839889422,-0.02657360268172676,-2.04972819348409,-0.5337796790574785,1.303539197081212,1.2143681898737342,0.21626363568622795,-1.0767622581201834,0.27279551230201876,1.254534704778782,-0.007161798102979574,0.9591815604945424,-0.0969913978589171,1.2064330745257432,-0.5458568668225169,-1.6900514972754825,0.017215742244832275,-0.6288392082862825,-0.7827142711900203,0.7854241074607656,-0.9735531629097208,-0.7941759475395668,-1.489184105473507,1.5560011902619266,-2.222507533188822,0.8053969870450869,-1.1366335409133876,-0.3601960893698163,-0.3958155378428583,-0.71355519835962,-0.4086051460513372,-2.0106711650395384,-0.5555904900311751,-0.7094422096313163,1.8742700974867241,-1.9559945691038292,0.790704957893816,-0.8958466016728689,-0.839035702499864,-1.5127398096597096,0.7549573899124727,-1.019370165192623,-0.9900141522683927,0.7116775439408523,1.5883452714268032,1.269093177386867,-1.1340285183076468,3.1477380992103936,-0.026241513917625976,0.7848088038248914,-2.5546212927779433,2.838669567346416,-1.8180864642847658,-0.23600509712454415,-0.616831660882691,0.8629829561456158,1.0893184533620133,1.542116116234557,-0.7041112430537587,0.40619335044803845,-0.3911905171283279,-1.7739462246125477,-0.6280244530271967,-1.531579810658591,0.006116079396267735,0.7161992827188339,1.5688964436690929,0.4853830299095298,-2.172685428915574,-0.3237050521365275,-2.142319264256919,-0.986506208612857,-1.5261487276882466,1.3455723060828,0.6457374865663694,-1.1259490876999108,-1.5930466835326942,-1.7525354133524138,1.1703900694821538,-2.0781378380938094,-0.957343148978673,0.6575367911971229,2.2508626260718887,2.1097279522160424,-1.4228272252618366,0.9472034811233278,-0.39026521824612853,-0.4890697200850942,-0.2491586843132961,2.7503739288262556,-0.5063128210583054,-1.0897097191730676,0.8325956130385515,0.9086289593366395,2.652671643861074,0.18053989084919522,-0.10015168406672072,-0.5541783892384295,-0.8767944594214194,2.1096059861716987,0.6126614861925768,2.1392581110530573,1.1218261667143206,-0.4645843965250448,0.08514710237520988,1.5392872824177808,-1.0683787318382705,-1.7431745675578667,-1.4212454169458901,-1.885484163746434,4.073501221159307,-2.2114498550871393,-0.09037855692137982,-0.7136715382315105,0.7537075638418324,1.8797873261906641,-0.2193199045262011,-2.012993734453159,0.5582258553800388,0.36630770606095314,0.7775448471088433,0.43184078046918967,-0.2475386442699232,-0.9054826690413474,0.3037812418526657,-0.243820222328047,2.2047423719972974,-0.6595442648816591,0.3456729988908125,-0.9355272792306008,1.0206421730359927,1.2622641056931039,-1.1804203743802812,-1.8464485452401864,0.27232715993146867,-1.5467566099730992,-0.1215891811764361,-0.21366853549368925,-1.2890627010763964,0.45363077238668753,2.1808113650547,1.894667778232418,1.730942908238292,-3.1841076156996637,-2.304443599404914,-1.202130330865005,2.0190223750550698,-1.1103562124282365,-2.876273355737536,0.6296518832615094,-0.6127878587492522,-0.015424159061915398,0.11024893162695741,2.2104286689830537,-1.2189141242014234,0.05735931396535786,-0.17665987969376487,2.969216659946142,-0.5268971699114483,-0.48177087445255273,0.49282188923964565,0.25386822549193017,-0.5982052826719053,0.17999833553092567,-0.985366107419514,-0.3803868587372977,0.10092948107763927,-0.014612436791334112,3.009202967087466,-0.4603967571227151,0.5178899706933251,1.3740597618483188,-0.7981651550983786,0.6332155678736421,-1.4677317309925724,-1.3550740620644484,-0.7665064732131439,0.4445147400809821,-0.9412758130849145,-1.2762153991483054,0.585400213856331,2.145065294383769,0.027468077322741223,0.013764377504701414,-0.4263202889225425,1.7240145960188453,4.062947731690526,-0.8241756951413459,-0.9138171523136117,-0.2532052569298824,0.535154601625128,-0.4455144127389805,1.3225842869270323,-1.3982196890711374,-0.5534624029545518,0.626983099858655,0.09709269382461422,0.4519826892553347,0.49167757606420975,-0.5404545359336785,-1.9244511509936595,-1.166738607045453,-1.0124529339984547,-0.24521533713027588,-0.9789878684409383,0.8131565464809503,-0.7797308233047486,-0.25526952390855123,2.894901015511404,-0.1578669898787971,0.5668341771171279,-0.24029925372029243,2.2200119930735744,-0.7920585305473639,-0.6199365544198233,-1.5764847367894368,2.9257948197745938,-0.45097607182280103,-2.4426063160843428,-2.0455389287843775,-1.677769588794414,-1.475392295614313,0.06105339817594658,-1.077681098996221,1.990578014272149,1.1175965226983127,-1.7101108612214322,-0.2608996469746609,4.038543845112685,-0.17297034969310746,-0.013635090060750667,1.1244153919703181,-0.556873903550714,-1.5694166698627203,-0.28113958746221995,-0.35367195675960916,0.10412840127882995,0.8481337854547275,-0.18843389359431717,0.26123115498660293,-0.2927773543177749,-0.8178936724204855,-0.974670712870215,-0.7561149357986919,0.8559468963264562,3.067498902351798,0.03865988576420957,-0.9085569722355011,-2.3524107777651793,0.3142005008289911,1.0821618178964942,0.28134485972241124,2.033119232706616,-0.8943307682873457,2.209504833143273,0.12860517673766172,1.4217763376516175,0.5543492317819417,0.3746795865790365,1.2139436447194032,-0.732078195220558,1.091484756570624,0.5981753517454925,-1.2743321082235168,-1.2009964953685441,0.6400143538583297,-0.8760106106059715,-1.206437346269835,-1.4305563284449336,0.005358373807134215,0.42512384306025236,0.030850686547708495,0.36265559793471774,-0.45811772209119317,4.401706977567398,1.601833720573798,-0.16616192849573336,0.5570995118586284,2.2467104683956562,0.43567960298568764,2.0107799604676893,0.9484334992160972,-0.3466407020872585,-0.02825533559278533,-1.1915029178931866,4.286754561087854,-1.5387359474750535,-0.8861527837452313,0.5442162963629977,0.9196372383768548,-1.7691373901278524,-1.157237373540661,-0.0468457223979203,-1.3928503557434826,2.0964960581735346,-0.514001517866675,-0.8620390393223992,-1.698726949239315,-1.0053194774992118,-1.5889532444100047,-0.6779726032774644,-0.9680589978626163,2.3522288313635613,2.1503125106643184,-1.3302689720605974,-0.8348871246371316,-0.6637336451911122,1.9878002672032384,-1.5780154682707606,1.053016673315126,-1.241162660466308,-0.15062699887884803,-2.342851670678287,-2.3242421384755416,0.5231614979600998,-0.20258078133098636,0.4400900210616091,-2.313202971687814,1.5484396211029599,1.398361967857674,-0.32229337910519373,-1.5081045058447966,0.5481648051640061,-1.6422645479557778,0.7448979891622632,0.3111035989277112,-1.4349262493530444,-2.637473776435325,-0.1397417702425309,-1.063926507125317,1.4819715581905066,1.6178092311122556,-0.5810611150613921,-1.0440302647142856,-1.477750920655186,-1.2036550588142172,-0.9540200659350049,0.2384051501604183,0.9349706088039078,0.6919854628583415,-0.055472994257984146,-0.7502642728223634,1.219901942027427,-0.8362431306659168,0.9086631529932001,2.7616716824937417,0.745480777483885,1.0994266692062862,-0.9002493881742307,-0.2589981556542854,-1.9329662285860298,-1.3454292863475996,0.9279431226992428,-1.5862972535485391,-1.4490144677216077,-0.26764062315135617,3.070284121658918,-2.110046466417305,1.0610288884732546,-0.06610888109584973,0.5229434265322883,-1.2573700005419635,2.035955379139546,-1.065852511279137,1.7174983462756677,-0.5344635078346824,3.2116389915305885,0.2567431911244057,-1.7285564987186115,-0.8456713814351186,3.620496429415246,-2.701905240261116,0.020466688650138773,-1.5789743375843464,-0.6630420540125845,-0.3649529869860937,-0.7869560549151269,-0.8265553543330383,1.7253495917490633,1.599948857319345,-0.4186984084346446,2.011902770844439,-0.208274197392391,-2.3411966534526507,-1.3003167379654892,1.9778554440504457,0.710316146617516,0.44192334171553205,-1.891126243264111,-1.580256707265761,-0.35104132772692387,-0.9714576553285238,1.0570254845181244,-0.7649812079401381,4.366116091662106,1.6515645431233685,0.46771302192476166,0.6021030197912411,0.44542460924057076,0.6508247327513891,1.028065274185494,-2.546220125313333,-0.4133235698170724,-0.35718092147453934,0.1794769503115324,-1.6880374718055255,-0.1868854444641845,0.3663904334711349,-0.7979023265147733,-0.6366398518215628,-1.2580705915763302,3.4322418124053966,-1.237195391972076,-1.8132218094090455,-2.009706408659753,-2.128667150399925,4.436379992144817,-0.7901255164729225,-1.2354586297702177,0.5058726271486536,-0.0610289623031895,-0.6485504714486867,-0.9872165897722606,-0.8068160181418285,1.189882517684991,1.0647015494749192,-1.7798712348966104,0.092583908170971,-0.08401205529512037,0.3195275400703319,-0.7632426797232933,1.289710231251904,0.23353527692312293,1.8237464766156684,-1.516768620998158,3.1762105631019737,2.412693772366324,-1.629439709205004,0.15517694767926823,0.10097479910504673,-0.18068310612557453,1.1077614428928493,1.0338214044382115,1.7468072424507408,0.8658034789139457,0.19824878689972106,-0.06300165451235831,0.785772222320289,-0.9647855284768776,0.01175331240750913,-1.5971676226895928,-0.9520770228069025,1.3515078266482121,-2.2391780960969823,-0.13411521292624515,0.1571996288520756,1.5565165552237545,-0.9199317406435601,-0.18039086515870562,-0.20871050792987042,0.15728167686746158,-0.5854899558313224,-1.4197506855196154,2.198089318789223,1.166294951968577,-0.6679033524736504,-1.020866815639356,-0.3825487425024434,0.7452046529837315,-1.6312692827042485,-0.9838562769146707,1.7363817302270543,1.6438191433943754,-0.5082677567553054,2.0208581054888857,-1.680638020301056,-0.16496456856553765,-0.6542871272905028,-0.45344732736458204,0.10085640614295696,2.878384488862464,-2.3313591531242612,-0.8723685068959789,0.4276098888089507,0.30506167411250307,-0.4091742797696376,-0.8350108700817526,-0.5663834964611278,-1.7037690575442046,1.1650556712466806,-1.4464861860050213,-1.2242189385141886,-0.4183093574257645,-0.5566829765984447,0.436113354454403,-0.7934308445126748,-0.23025366394625754,-1.629610745505333,0.029480316392749086,0.6750690842822991,1.8572749783624294,2.5601495274778676,-1.0915736625742325,-0.7098071638181456,-0.8615858104411794,0.5118247072364743,-1.0385887728274426,-0.12166423141349605,-0.7377133088260679,1.8161348177161094,0.9532756499803895,-0.26302595433453185,-0.9212923764367934,0.29153531665271853,-1.6689335948502164,-1.4591050513464412,0.0014929920332919964,-1.7887425209932104,-0.5251565948081766,-0.19322133005081435,-1.6081201003987025,1.3662362195313362,1.064926231233077,-2.459972369588193,-0.28810845985073363,0.6601769206001007,2.2275272616066175,-1.3345980305230225,-0.22329577963513955,3.5636297226595706,0.8083484533784564,-1.4746978970071223,2.7457227476207615,0.9746832441554562,-1.1809839986615913,0.7431924941711673,-0.9572774908375898,1.0601296153920319,-0.19812585030808452,-1.2296924485033196,2.9356492737906614,-0.9543605096657274,1.0558623743638498,0.7659914714928576,-1.88638090593349,1.8933653083351396,-0.2587303380763957,-1.0042372022716575,0.1654119857618475,3.1118076805619737,1.4498359426838254,-1.7248571623012428,-1.456001089127596,-0.45330236236474736,2.062539646561399,0.504018433869531,1.2147366363143146,3.570912657112413,0.04791268393510897,-0.31025521633292,-0.424220222453756,0.47243201859895184,3.941327351333991,-0.8574745132499518,3.1437818896052083,1.3189327705646465,0.28406710685049785,-0.03897466986409395,-0.5912612315890684,-0.05920906180709063,-0.5085495936034459,2.614324381238251,1.605970405417553,1.130529243702385,1.4994638446868458,1.365812470951679,-0.29633337758007217,-0.14090865716336667,-1.1149803210307287,0.671281178851276,-0.15606255613506306,0.4187468799912569,-1.3015896005121084,-0.09971338771408668,-0.4775503794986546,-0.6374257747345015,-0.49198405059237676,0.7979944841681704,-0.8138739707737879,1.8531447250501578,0.3818864689216649,1.1667263279671203,0.8900855139491729,1.9302594923506222,2.0469507622217966,-0.6810352918953207,1.116822693825604,-0.027437322395287313,0.540466127909581,-0.7217119015828539,4.56378205856748,0.3574133395213169,-2.9247339317310783,-0.9996700529532581,-0.2932871161274815,-0.8362089756955466,-1.198544392435094,-1.433586691604126,3.214753258247568,1.9653319776811828,-0.24928173075049484,-0.08563583725543213,1.8917893400379802,-0.551631552180314,2.0535757262386864,1.8662257879950868,0.9901163654627334,2.752348069792853,1.4580207479190257,0.7490136612812202,1.0071411187949537,1.3434096488258573,-0.09366879942276699,0.6602948693352851,-0.3595698913981027,-0.547363395114777,-0.02683745201014641,-0.35517793631833294,0.6373023578572804,0.8527548706335114,0.1710464459877839,-0.19084989868685268,0.0783224087317617,-0.03078530277884953,-0.6357135444272177,-0.21388499518490353,-2.2926288069788203,-0.3098881917145483,-0.055164230268586585,-2.0295450756226474,-0.4781946226780432,1.080474501456033,-1.6158759884617215,-1.5162474841258726,0.2589272531507405,-2.178988703297217,-2.1057092273899762,1.8082876850186789,-1.8320610904148729,-1.697259496645372,0.3597805383002768,0.3697524297424219,-2.2865949644070778,2.8513902570822114,-1.6250901179704136,-0.9297629736815621,-1.5748315835374562,2.0348596882725567,-0.5421748698306825,0.26709872882505914,-1.4759946154632377,1.8795089952280482,0.8333722913292286,-1.7296213379253056,-1.438563501397731,3.771920463102791,-0.7652766422858153,0.11041473606287601,0.5473443976173434,1.5260061721784273,1.6625220058714154,1.8086405771136789,0.9451184986806197,1.8343124582277972,-0.14604311893980437,1.724065854796957,-0.4068686994192529,0.7335090136696647,0.9739893021067717,-0.32105995813935173,-1.0266492286507356,1.1913578064541732,-0.6002067384994849,0.31396061837207645,0.3617044999296606,0.7494960204311902,-0.23265181074209482,2.299746104668557,-0.05969574819573335,-1.883699538486852,1.2018854770567122,-0.9346562889292858,-2.4656255272553835,0.5292117643164894,-1.8028717526547948,-0.1852202998832209,0.26554495018331903,-1.3708295066873717,-1.9439979602753648,-0.11953788035688064,0.4280034530993118,-1.6092042910987652,-1.6108703295781124,-2.2361053802090622,-1.0872014873661509,0.27723819990373344,2.571740239129298,2.3710082995853177,0.5348692388250289,-1.0482270541990064,2.792924964868256,-0.12655101849647502,1.399346909999632,-0.09706824869417015,2.504519559542952,-1.4302295079338854,3.017638192084023,0.4117091604067051,0.5670993291777553,-2.3748037999834963,-2.5769226655445436,1.3630007914525093,-1.001462246202136,-0.746059313904123,-1.2528066536710687,-0.36781173840021103,-0.842933438919734,-0.971118085764742,-1.8737096567920317,4.404682170419603,1.3596140924544227,0.8984438302269846,-1.371009781044813,-0.02758233176401253,0.20020773663340224,-1.9897294816642246,0.664820667508579,0.07796146900740496,-0.778853049628576,-0.6641676983049756,-0.8156914167652951,1.2916431603879355,-0.28951107140592774,1.1127642194861302,0.09710127089903327,0.36408466741255147,-0.4850974768337253,-2.4673896614554915,-1.1680042496017498,0.18884342403804016,0.2599542634965923,-1.1795347880903886,-1.8334077577471068,-1.3199858495444767,-1.4939095795507744,-2.667207203877035,1.6650451838290894,-1.8398915187542537,-1.1317448848777183,1.968569819651004,1.1241042109312864,-1.0402478606127579,1.3803912424969922,0.30814935324115095,-0.4236425029089073,-0.6131395618642184,-1.6035107707415663,1.4758120482748207,-2.1404411479755185,0.09794714666261108,-1.0952631331780263,0.9407558549264942,1.342147839240505,-0.6132410235762719,0.9009126375177221,-1.0507243776270019,2.0636059665246225,1.1157819507341313,1.1835154390970266,-2.945272415445539,-0.4470169785844952,-1.165934579041971,1.0155467532861402,-0.6136897318120995,0.46273642992988673,-1.2920145488957395,-1.5981516998760212,1.1009509570098153,-0.8877231520979171,-2.0824805586797117,-0.09357369404443361,-1.0176841926562048,-2.4895140284903343,-1.3786532859387026,-0.5406204901411475,-0.6849025324236414,-1.3716258397549248,2.9580493000745456,-0.10701269200220974,-1.2405842775810945,-0.6151424890186139,-0.9416977262550917,-1.0011731594239552,4.3943584333926085,-0.7901322388092018,-0.396396682049593,0.1574621236910894,-0.1145320423073103,-0.1643393898514912,-1.4316990515575159,-1.980501391488207,0.3051262302060491,-0.21895306739699313,0.7383541276467771,3.965308665586639,-1.4529302879456096,0.3637590303354077,-0.2119617231628785,0.6741255277856177,-1.7008069083639357,-0.07890924464218584,1.8770147624913809,-1.923030480064496,-0.6353079429294308,1.742435643981402,1.208880438892189,2.487886179040138,0.18472680616306084,0.6312327733909862,-0.054791749273158746,-1.4234954844778476,-0.6995551026154769,0.5512391433711616,-2.2100537891167487,2.660016009799144,-0.294022622803727,-1.096012294004596,2.3277614334386385,2.510326196363149,0.1415485610766097,-0.8989829646371809,-1.2526917891683138,1.4852360047709243,-0.6555246377279583,1.6706263286459928,-1.6110346653325516,-0.054321652941405764,-2.1275409455604115,1.8951017337914062,-1.743891614889605,-1.2343953221440993,-1.082745859305781,-0.9031421240111375,1.155398850690273,3.675645711980009,-1.409377831642717,1.251205815672343,2.6955245920306377,-0.10873932092678852,-0.7254962356692313,1.5287691073018106,-0.8453636764660007,1.0601227655749401,-0.41218172288076516,1.3296689887047912,0.4229288346569526,-1.749933189371905,0.9135773095346427,0.2507977527148992,2.180407909624476,0.8732497689861788,0.8073434151712972,2.159943525387112,1.5213644480302777,1.9329732928761896,-0.28634529799355046,-0.9585014870116059,-0.2812349290778316,1.2994422640549645,2.758369266718421,-0.3293324180160775,0.2171526806701233,-1.1214234127648817,-1.1758620838002545,-0.7734439237845562,-1.943262140059309,-1.3549208207918408,-0.6859880360402439,0.29555446525192913,0.19737374811919506,-1.0249206330954155,-0.13089914551024148,2.1411434593159266,-0.3175197988009118,-1.1240510833097335,-2.099377965296391,-2.3186368674710147,0.17394971409302842,0.3319670810931895,-2.06815185539003,-0.3995502546030741,-2.0712441430674486,-1.2512862585972317,0.6312610715018103,-0.8310806968044234,1.2078481544431727,0.4486146449442194,-1.3729621550885198,-1.017063124547722,-0.35176136548719994,0.7961537902898707,2.248070880049734,2.5081758744864566,0.8940474928421013,-0.6675997381044466,-1.849228698403642,1.9125300323905379,1.0113624009451698,2.086809688671513,-1.5551247524654517,2.6226059419459307,1.3529456809915024,-2.0938012716591436,2.4493950663966713,-0.6049293883607212,-0.09368587000632246,0.12301445520121661,-0.6203502921352091,1.47214831776917,-0.6951200724805036,1.669698831860385,2.342855363169361,-0.9315332351865429,1.3239167340979907,0.3833993698997907,1.9074410127681307,3.179257615739517,-0.49905444679297023,-0.14368087111730968,-0.6543094410545351,1.2720805563038224,1.1417781821762745,-1.4626923028692602,0.448387018166222,-0.9767506037202004,-0.32716854620020963,1.7512464340083913,-0.600878262374497,-1.4727297990079435,-0.8922062604989204,-0.18701393146551556,0.8486558023617004,3.5545757567167215,-0.20534232347034415,1.2112674740963147,-1.482922832770479,-2.7919249765025382,-0.41749448759707114,1.007588523335607,0.713785252242225,-1.8213623569905266,-1.4793269338959836,-0.9488659288600506,-0.9816247354836941,0.37530214561933817,-2.0982914193453603,-2.443534327475109,-2.3209101113136588,3.479325688322059,-1.4196946670175152,-0.2542985233298776,-2.275946363456758,-0.3792729337962393,3.9554533640538487,0.28325550004379285,-1.6617901997043958,-1.2870611523542208,1.0525788169667685,-0.5607287868807238,-1.0619504309671555,-0.9588825670851513,-1.0770607859734973,-1.4397693889612724,0.5314513822671851,-2.4809024773398383,-0.08515815891879096,0.8726656693815984,-0.08922580028476543,-0.07831660709985734,-1.3332126849489254,-0.22869918035788064,-0.5637392051770383,-0.2386733623494932,0.7129792501838113,1.6705915570346428,2.91301600212003,2.330914517301062,3.1946415817571703,0.37442125695823986,0.07288701945274409,0.14866726508006842,0.9740305986357806,-0.3517180880234418,-0.6773833703851077,2.393901317141668,-0.9032659301217838,-1.270140380825333,-0.6880681053531321,-0.2981197413308483,-1.4207676032328556,2.423614738957198,-0.011777609769597356,-2.5214681361822238,-1.0852780992794375,-1.51604526839686,-1.4853969231649655,-2.298158620135625,-2.131833662084628,1.3767711776236327,1.4718129159614683,1.132582490449426,-0.035797821137148224,-2.1885922276220864,-0.21574932120171147,-2.6436573811436976,-0.6662237663526991,0.5298215375478165,-2.276482162122436,1.0142830709344568,1.991433916299048,-1.194907445055562,-0.4734188818800394,-1.7737227232062678,-1.7211338559638882,-1.0869250637067953,0.7525819490745242,3.1253327287460353,-1.7412398000697609,0.3087860459465468,-0.266797718757037,-0.6773653347289357,1.9504974940888846,-2.9947388441048144,0.011374522058046467,2.48454393513769,-1.186931373735797,-1.4418878430872388,-0.2915918160405919,-0.8898220534245707,-1.468242062971451,0.9669233760731162,-1.6109183882692948,3.4084018578434137,-1.2100172877624003,-2.299045094684472,-0.3785817756936203,-1.3745711317100069,-1.4528342152086455,-1.8024024443865378,0.3291835424569511,1.923450888616041,-0.5369462336443123,-0.34702674313906573,-0.8330374328970928,-1.5766687218231525,-0.4211612694249522,0.7644036034112051,-0.9161219951515192,1.1013335635396333,0.11237510022610442,0.34940520897273974,-0.008435856897754262,-0.682432262268398,0.9983251156331272,2.9364690079559375,-0.006638579453347865,0.9485022055921782,-0.9609268165753723,1.0335766768603272,-0.6201914046735854,-1.2634650601293624,-3.299994844686993,2.040594309952845,-0.49348230760796136,-1.2269160950273752,-1.0805122691980342,-1.246264922676863,0.6180408460261907,2.7960203064339164,2.978855192078716,0.43708050322327424,0.7723104741640033,-1.9106169547191834,-1.4655980065923475,0.3664609256463618,-0.689580358711236,-1.7542054554060749,-0.26651612435659733,-1.564508499038643,-1.8262740762746537,1.7988361856487662,-1.2343032344911546,-0.9732531074758878,1.6399355773701252,1.159652618428696,0.36663827668040055,-1.07702423280873,0.36104930345349673,0.7100712754392643,-1.3097603523239736,-0.9145189650511928,1.6942835350276082,3.7376099209585836,0.9489640585123899,1.4151730363554307,-0.8710442745778149,-0.7360239954307546,-2.4236753759299474,0.5565206637309665,-1.536546399067954,-1.6005370202471059,-0.0811240919867798,1.7052400024709389,-1.5575730917604111,2.1067038928344854,0.7782799159446132,-0.6670846968585817,0.28099937970932115,1.459896152396481,2.059895028323675,2.637637290638835,-0.983783905637811,1.355272764208023,1.9873326629348362,-1.1392017483734622,1.9670477544008544,0.9056868492349724,-0.27014658289236887,-1.235178579995311,-0.6999154090407841,0.10076057383566632,0.45845310101488457,-1.1702829931501015,-1.549741294107672,-1.5204563299355858,2.1021019568603556,1.417732239575558,-0.4897316014235909,-2.217990038798457,1.6222136728958025,-0.7604889360670488,-0.45546164416964324,-0.15343900688216316,2.395865693991827,-0.4546015062681643,0.06308495262314445,0.053125409497892656,1.392753852454128,1.7246133014875493,-1.5429405007242611,-0.5135991669317654,-0.5068937985739647,-1.8961436056342644,0.20879934207207534,-1.4776015891109848,-0.439258728525299,-2.401548406990131,2.0414870111401586,1.5973096695403766,-0.6710524032466477,-0.46661964090528585,1.620815844472399,0.17203964613197725,-1.114372560462601,-1.2704080257535704,-1.0257536593799532,1.7683878752765418,-1.219076840122932,-0.705605044370381,-1.1394974506315223,1.1543042361706513,0.9511687921979123,-0.6953351079753065,-1.0234305764562999,3.4811221157538554,0.576936517662675,-2.0219424153446184,0.2085880716503635,1.9718562939001913,-1.5235417607805755,-1.6721607147447077,0.6131467918733648,-0.15194931623427765,1.030688389071502,1.53731424483438,1.0955603566620387,1.7992099669289092,1.1503834557452222,-0.5221112155188331,0.06745398262412626,3.595875692685013,1.5276362604711708,0.07895579608333564,0.5237551918118031,0.3216994278537179,2.9398573220324,0.1240476735054462,-0.286420717522049,1.600067792048256,-2.272691947713721,-0.06481231162895704,-0.5748293416320011,2.9523026119365423,-0.7069736556813999,0.37470280364975794,-2.352225107690989,1.668502526863248,-1.477650828667599,1.0723572346144206,0.9864410500381345,2.353444866642508,3.3394316586778072,-0.10651674860151981,-1.3194293203303507,-1.7907825844938563,-0.33273415684725915,-0.6014616880992829,0.13841983942154634,1.2751598393312955,-0.9091519857941395,1.1391838777580185,0.10940296744445872,0.18172531713164233,-1.2114491372047167,-2.029744783063651,3.507854138013323,3.2451145953506186,-0.021688689860075577,0.6933678568823183,0.15516368735726468,-0.7133724247835808,-0.4859806175043033,-1.5966990428022136,-1.2000357132883546,1.3568682992142374,3.9118269090780644,-0.1336121529626233,-0.08641582798198807,-0.29651489365419736,0.7915162258477731,1.1013444049499128,-0.4082446592145116,-1.7174040303507236,0.37402217386820824,-1.3602549815205232,2.609918152307967,-0.986004890663556,-1.710526494355647,-1.8545647414507926,1.486621160614903,0.18834332592599656,-0.5904585425239461,-1.1422041177529123,1.005284064265644,-2.274039975969973,0.2501933053754158,-1.49847279642236,-1.5572484087853091,0.007739473551939915,-0.39603121967657584,0.33961003254026617,-0.7361168611935089,-0.3794805783272954,0.17945280452647153,1.7674351623399442,-0.49172796439466565,2.7050379208843984,0.6603764045537981,0.32457533395194393,-0.9497179378962507,-0.49278391825209206,-2.4125541168000493,-1.3502314499562884,0.5811481231204833,-2.5718463563693015,1.7640474533306418,-1.9634085847588898,-0.4039860195101309,0.41386236009946537,0.02218727208132854,1.4892149557433991,0.9868764922170735,-0.21916610944178633,-1.6469190882135272,0.22587301266771476,-1.975104457488581,-0.1746466846136808,-1.2781400947052113,-0.01779585328779619,-0.4838359081796988,-1.3605917576912399,0.7344325042028055,0.14833160391966158,-2.482224182993645,-0.7367404212791252,-1.9914971625428708,-2.3437605498159146,1.7399860577968103,-0.33396357370552254,-0.5898649004494176,2.814279568694902,3.9374965595843023,-1.721969321997483,-1.672942664253401,-0.29337448187332066,-1.0614072253067897,1.8868544311033846,-0.1767045126329936,2.6030480566279124,-0.24214281870536036,-0.4838901278385913,-0.6847266188487224,-0.2442805888708292,-1.0466421815894251,0.9582906159448953,-1.0369964972377996,0.513806480326991,-0.32510246791517283,-0.47410725375216267,0.47381360585653504,-0.47223824758765925,-1.701380885319522,-1.4376294544870678,1.907695160660915,0.5232748874355277,-0.9096238180031483,-0.3986680952194483,-1.3140742778786105,0.17989163734789204,-0.8350168887443699,0.8869450262545431,-1.6527908250495742,-3.1122949347312465,-0.4403579221346123,-1.901749704937865,2.3376177089092622,1.5097120597183644,-1.6820484746306703,-0.7673101920227114,0.6661443079149142,-2.392998118001228,0.3082166124222604,-0.14501761525070092,0.8099632806304932,1.5096616474337425,-1.6648329969974642,0.8599179769251948,-1.6116493306212225,0.8528926049365171,-0.2515184169543228,0.580898234472748,-1.3361238459518407,4.112445606983341,-1.681732698761631,0.731642525738818,-0.729957660772035,0.6589365742702807,-1.6125038342810731,1.5646620320593572,-0.8530662501072899,0.0427791601188851,3.405816385686003,-1.4536391023503485,-1.3512159698560242,1.9662157261739035,-2.2307633128869147,0.378944546610929,-1.0682988512187466,-0.24728421782607174,-2.154190274396746,-0.444599237389282,-2.171156885799948,-1.6183536550033721,0.395263096472196,-0.6456372251793536,-0.3428488605205599,1.943217212550952,2.578467659705739,-0.9481120322077436,1.7894018417578594,0.7303469923641716,-1.3328153624201748,0.1775103828798409,-0.9936839943391914,-0.8561041120168076,0.17880391735774928,-1.4381743133922984,1.3969367797122205,-0.3131787941263401,-0.10462357685610625,-0.10336745229253312,1.5893315116998465,0.20357859424705035,-1.1198189910511074,2.439990509346897,-2.065163169019745,2.955316902592407,-1.195307180809698,-2.59022109670069,1.0724061826682776,-0.010001812250788415,1.2303811834207798,-1.1340436005426713,-0.4280427067987023,-1.712086870623491,0.6465383601276781,1.1572322467358247,-1.8674640996222798,-2.060741369248059,-0.716555935571996,0.9051266718228855,-0.6727808043784028,-0.6005152734134697,-0.5281590042237339,-0.14883259373317434,0.31256078617522254,0.08130705851809684,1.275602878670744,-0.3826166005397871,-0.656125797164397,-0.7442906154880214,-0.39279904281308675,0.7791462736961279,1.4941201726957725,2.8053748697864513,1.34176803621041,0.4200190247780971,-0.21946864876784414,-0.18205629870596646,-0.17431044436680984,-2.3251609900718555,-0.28571563172606085,1.9610472437183923,3.6373718049683395,2.051808934930694,0.13982251791763903,-1.4746519296743916,1.7867072211231272,1.57311370826367,1.5270384191342703,-0.3487825553718691,-1.0138543384439205,-1.1962304910862276,-0.4806948298346847,-0.37604096167945056,-1.5529532928999512,-0.22788983259264653,1.1129617544226753,1.5617532769665328,1.1101543287272368,-1.5396448986953373,1.8656201959718488,1.0692069507446444,1.699127694781186,-0.7663159267068728,-1.2127938726612812,-2.116481367082646,3.5600077410204687,-0.8443873697975143,1.2813248301392541,3.740560855067678,0.06767032486103311,-2.510537084338661,-1.2418169372428203,-1.838791935282104,0.9182510199944677,0.5906798090681916,0.9385522692423063,-0.25597086397274627,0.22310767676216572,0.6472165316139109,-0.29472732339175456,1.7799715261914988,-1.884525308909671,2.749449131900366,0.10049634238257175,-0.8446238717594843,0.3407734035016833,1.0545001689165088,0.5674971837694983,-0.6711132740446096,0.9680449699746584,-1.4320884012866195,-1.262750790909598,-0.49260948246604225,-1.2112472269901682,1.571706160816723,-2.4239245711315602,-0.8435278934125319,-0.7492231726533752,0.2687665214695747,0.7533133616442953,0.7854717457860704,-2.1084784093444937,-1.3821443264431084,0.22826084149554723,1.6144750862739543,-0.8100403486327958,-0.7295118517415544,-2.4632033981741275,-1.2636754586625667,-0.807798625095707,1.6831684695250273,0.40754699240102626,0.5253738525282566,4.147994293556654,1.596472783762195,-1.4907080283584444,-0.4765896578607469,-3.2204946144681026,0.7640151548552745,1.372805142137519,0.27539687676613994,-1.321867755722911,-0.8353076724908294,1.5671558685745677,-1.2004437428026713,2.424460509223794,-1.8445454267990868,-1.0323878350408426,-0.6267584011700789,-0.9787813237166143,0.1506545815986575,-2.1906558794755746,-0.01547331114582779,0.24590052969959064,-0.6422477088701075,-1.652203887616409,3.0966865059411908,-1.4387476241908042,2.0414141397664203,-1.618441755682189,1.3776874067928226,-0.7531546349451218,-2.8363391178781696,-0.7877132628305105,2.4981380171453362,0.8230731973388388,0.12376026379227362,-1.5801709602616596,3.1007063294655377,-1.175487589327914,0.0942195223154613,-0.05751012728840694,-0.4372481766801674,-0.5884746088371101,1.5281318829721602,4.095498009625054,2.8253146491171326,1.0331238110183987,1.2823518377970828,-0.8967778333456626,-1.7490197786039978,0.06059955469877141,-0.8718875046405403,0.2577662495811201,-1.4026307517041754,0.4974122197882942,0.2244981221157285,-0.10007982915039002,4.032672674947881,-1.48216585612475,-0.029970319223750135,0.4043199391777731,0.6242565499268133,-1.0640686706944749,-2.1050413574729676,0.9826708422667674,-1.2323047372169715,0.9384532968284848,-1.7152021060747888,4.0901784948019655,2.406604862528835,-0.9299504345545103,3.6227002184491233,2.1980797843157185,1.3349328960307536,0.1668793427929801,-0.8682292073714417,-1.8070016423630808,1.2800310273541045,3.938257049355337,-1.3538973808902524,1.293417478164889,1.613475873565588,-1.1857735834635417,0.6355671165635058,-1.2017054437113974,0.5840067999473015,0.17232496422893268,-0.01811723691334629,-1.6422971042487398,1.1294525630160177,0.6644440101617718,-1.4663639895846785,-1.4523751290588331,-1.6650666068720916,-0.5955140608184932,-1.8791142694369996,0.8316671331823665,-1.7860768357018053,0.7587281441337298,1.3750756622452105,2.8147747764908275,0.3938240561323407,0.1025407053536236,2.1849317690249026,-1.7622685736619759,-1.1141666149825162,3.351150507521826,-0.48857698624100343,-1.6227727761651365,-1.7788467440569864,0.6083302537674373,-1.0763021747427801,-2.3017992211198233,3.1573253759167006,-0.9015076861530276,-1.434957730253062,-0.9903412442256773,1.4511726594041503,1.4325449460685766,-0.3345265618685506,-1.1763519212682374,-0.3632986795796191,-0.1266155556333743,-1.4345333282952992,-2.242426959377559,-1.1267132527633865,0.10826220994527101,-1.862556496364326,-0.0753946498138788,-1.4425956861029314,0.9698237787198833,-1.4535912415403618,-0.8734692839663114,-1.2659659164491828,0.6112769685551829,0.7594465066355036,1.839964165536034,0.3800491179300768,0.34407879443263356,-1.232058201125184,1.1289626829659283,-0.1796301031459909,-0.8089904073343935,-2.4041908675027024,0.009893090327909415,0.3711210371006048,-3.36968613559287,2.551667329885159,1.0533637179194224,3.461678772799992,-0.4144893132097382,3.166943183922268,-0.9480145086574953,-0.3674169817368127,0.8119667829394309,-0.9921091780506561,-1.6181452614576204,-0.13514797113307864,1.0939203239267412,-1.2242624077044781,1.0907010420321646,0.36774354594394437,0.1497424734713231,-0.7258654335670955,-0.24126704187342948,2.270176620593076,2.8394068797027994,0.08926418528508859,-1.3248465207921418,-0.1665473101013131,0.48268845525977516,2.0776191594392994,-0.4002333307424992,-0.6365791173038745,-1.40705626489335,1.285292788879017,-1.0604560419576767,-1.020846555013456,-1.0710290088718262,-0.31014643385075763,0.0798094611770132,-0.6558604270569902,-0.42531986803337846,0.7954238131092222,2.360312562675082,-0.13691375523363125,0.10602693871676946,-0.7423471421049506,-1.1077309681671443,-0.28210325779654394,-0.0557863869825054,-1.4351951128380513,-0.2463856568907072,-1.0876305035928324,-0.6893557116445472,2.229527549793258,-0.6855997503199375,0.69930570077642,-0.6541856102664472,-0.5949796974694946,-0.3433227320120658,1.7612983969270883,-2.2989780514816753,0.7439400858889175,-0.743323222431965,0.554621013569427,0.3181632863614214,-0.505080772526567,1.4843757944189508,0.4761460430569898,-1.132903854637399,0.9579488168544328,-0.549566065127794,2.1361008700772004,-1.3006430398276505,-0.2568878131499907,-0.6986285391733401,0.43243327130222203,-0.49275416810240885,-0.6488989321327608,-1.6875215139361517,1.0320552544650987,-0.769284168576815,0.6189147603904539,-1.2858283638521644,3.5949108222040564,0.008054650320234412,-0.5943271643671535,1.2440107275587544,-1.3908045970195104,0.6537213649552864,-0.8586960515923718,2.067053702611425,2.247835921705315,1.0497459123196409,-0.9708073778913046,-1.4817426948157455,3.305873957837925,-2.0818098997786474,0.8589241389009907,-0.6623372992219775,-1.7303427144131835,0.34936219634946103,-0.11588394654828288,0.47701418145359664,0.17339813095442538,-1.0403234700618667,-0.3843304109017422,1.1409410254467844,-0.04034591541514059,-0.1466564443279077,1.35652756932709,0.94151153406926,1.592280771719312,-0.3087938502115397,-0.5314053900268203,-1.2843676583400019,-0.18439920200184684,-0.6650935968931126,-0.1741513155714515,0.04942095106920456,0.08085458251894469,1.8593816910215424,-0.16941399057788029,1.4397268876175568,-2.1611172850669824,-0.9042585296088045,-1.104208128144093,-2.4263299009573647,-0.8325543672839643,0.10400261729347117,1.3763801169693775,1.0121490002527216,-0.824902664305516,0.44245587004220555,0.002488965446112744,1.612863549400848,-0.933016202321339,-0.41764572132585426,2.114768287830856,-0.18407572972502345,0.12241349035649234,-0.8567694618151798,0.41901116878055794,-1.1087197617579978,1.3208258448586858,2.793563141595023,0.5261189881198924,0.7895082487786247,0.6050055180566569,2.477792704497398,-1.4076452076603003,-1.2455223775742121,-1.507131901744981,1.531283423614889,-1.4173091488063148,-0.6344069969385838,-1.372799685390787,-1.008053931972135,1.4248012282881657,-1.2124810781194613,-0.43102881048708164,-0.0844445026981694,-1.5412475101455685,1.830784237135522,0.18121422464121897,0.7647998586140867,-0.7635222854603088,-0.7575656255920272,0.4488885373391754,-0.8085615976917392,-1.68197281235746,-0.6032564386858695,-1.177716908157442,0.3749771271906906,-1.100849172558792,0.997500055826763,2.6944784216150803,-1.5776528590354013,0.967432351571818,2.605498699993035,-1.6845503215571482,0.5128720831410088,0.7555763160698256,-0.08452976022124373,0.7907973225528228,-0.22686373320143796,1.7515613856894041,2.5141136019730803,1.3299737816476929,-1.5422835971851474,2.2945382078343832,-0.34545540033716166,0.4714051014586694,0.2753925829073595,1.4129976775889967,-1.0497010903451784,-2.150360939755715,-1.94064555254067,-0.9750732979051449,0.04581676672402307,2.904992150455084,0.5533858520180845,2.95700400624184,0.1962854274926884,-0.8968987614155417,0.4522545492968513,0.591123455337445,-2.3329433330252747,-1.277286590356299,1.1108961983872245,-0.28559950124625055,-0.6015151117813721,0.7045453464087205,0.5558634661817072,-1.421900257124284,-1.1995359010037214,0.2537792273208153,-1.3652284832855341,-2.5745984073554418,2.7220682718486153,0.768751499400517,1.2417009088750588,1.3892714621789084,1.05378304578807,1.3163014969081246,-0.09603417550858832,0.5556771400449093,-0.9661457783424474,-0.9097352470107178,-1.7107351803196895,0.7248246037611253,-0.9144759466753413,1.8344726881149425,0.5024046658197197,2.755870995245451,-1.6691715557832691,-0.42028105863195875,-1.0363266290897006,0.8981588104327123,-1.6892605249236408,2.0214584543797325,0.4667020072841481,-1.5370573693861134,1.5347314779871863,-0.8614934346763992,-0.5132187770452349,1.323165102019012,1.8944198525972502,0.07644180201830456,-0.045932919944396876,-0.1241702079971809,-1.8125121839490508,1.6561960591028873,-0.4547571577581888,-2.351910262537606,0.6203826130454163,-1.3266028896302142,-0.9556691185097275,-0.7745687667675851,-0.570624276075339,0.9273763468921955,-2.0835572019322854,-1.6177218732798342,-1.0585293129375914,1.0140057096916115,0.984102016740865,0.011806496404566179,0.2010454922599067,0.8148777102283576,1.0005491454946722,-0.8316005649566443,-0.9582599525587491,1.979064301619104,-0.8390995066481433,0.3301405213864442,1.0268120847953572,2.1104567697035614,-0.9102710782315845,-0.7767360002251231,-1.0096481412381386,0.3671220687157042,-1.9504872071182242,2.6636634874703624,-0.012081261330626706,1.6228066798112375,-1.5523013822499137,-0.06744373865114134,-1.5263326773957342,-1.8431290536439533,-0.6678665299699528,-1.157558254040902,0.9079588078664993,-1.7993517399838188,-2.8160510895394566,-1.794387589661483,-0.4096842702381987,0.015168699153665895,0.679538338354083,1.0167188199340562,-1.193669917323094,0.5541469943587912,0.44254478070348396,1.7524750807084588,-0.055994632184141885,1.1770121850976207,-1.6222925035722828,-0.13728430932546073,1.900397447004797,-1.4447684804315415,0.7723356554248141,-0.6722241173346154,1.3585394985923818,2.235932274908609,-1.731068303240558,-1.200388840001197,-1.6876820246568425,-1.375327009700001,-0.8579000256107452,-0.8261421965166005,1.4297440084216866,0.5311351173288117,1.9924209764087555,3.096872450465173,-0.4297083497532424,3.4344098246540207,-0.027146409604035776,-0.5613276847006817,-1.0467080916225817,0.199274614939389,-1.2536458773292485,0.312467710936481,2.146622809483608,-1.806287967133273,-2.0848469934971834,0.20561521098718472,0.9871627130383535,-0.7520883302069437,2.25895656467336,-1.493020048027022,2.019917077259324,-1.9296152860245845,1.8581124265443236,0.7342761192848368,-0.697620745358749,0.7624880492894959,1.7898009191111313,0.5314808338159053,0.0821044547250205,0.3803305114628289,0.01150454332149641,0.24072214250789875,-1.5666898359691421,-1.8684018117004546,-0.7760669527918012,-1.9490599577317018,1.280794827355187,-0.32474074167427425,2.8749993308240125,2.5701547735015975,0.8802801891037322,-0.5524999889769903,-1.4743987329618813,-1.777272579112319,-0.2846450093279209,2.44236760357517,0.34725545915747835,0.031862544825568405,0.7480756399290666,-0.2554700204062311,-0.3084251318123948,-0.6815180154994734,1.044864808311194,-2.0080053235572826,4.245224653946188,-0.7004101154927782,3.872158078694696,2.844446232025619,-1.0909498556872692,-0.8113842847196563,-1.0191970394051817,-0.43415871431793857,-1.9505781642700368,0.7117801053736158,3.009798528343949,-0.0625592930250507,0.6006543269259147,1.377403085765739,0.6336941688689628,-0.17091755651313337,-0.7697746789000174,-0.41281654345089314,-1.7388564018494548,-1.2169155833763559,-0.5049384641074502,-0.5202154702010544,0.35703052108875566,-1.3669589832535882,-0.3222318746332852,1.5560413893950509,-0.5061356634293118,0.4654783491536843,0.039471537544252824,-3.0002966280773427,-1.3084783770844832,-0.9713773842441087,-1.7244176966522757,-0.7441574978820875,3.1966304468305697,-0.3055143263225802,-0.23532280576288134,-0.05518687424504302,4.5266424959951115,3.2256269876664803,-0.7776895990463265,-1.086323853201625,-0.291626264288266,1.1773085063481583,-1.319276182631247,-2.588670340064687,-0.009447215611253399,1.7355243249012224,-1.3721507458350166,2.855794511569922,-0.34801210793632076,1.2393674382318578,-0.9189698475606369,-0.04681074545701018,-0.04513520450884726,-1.2519978846642081,-1.8605644935654175,-0.004114377689688691,-0.38270905473794564,-1.7645219521332545,0.4766983488872902,-0.29173407122362954,-1.1184627227817991,-0.2883685949888457,-1.2057470483471118,-1.7001974721771906,-1.264447901630309,0.29833638400754886,-0.24632967055106322,-0.2150096733250383,-0.33445858845841997,0.06472272369275806,-0.2314464430824501,-1.0191834960398947,-1.5840446772331538,-0.93779038536249,0.17722443569888652,-0.639245733100484,0.26223907323031553,-0.44594188070373025,-0.6135068536132418,0.9619632280800815,0.0627276802517236,-0.8749118875854915,0.7350587138021288,-1.3843679862026503,1.5179258890824927,0.5293041128232552,-1.2917260112604414,-0.8027278860991732,3.5124946366165037,3.1757716544401036,1.0206982745282365,-0.5987271056051952,0.8896005715287376,-0.016816162464751185,-1.2827698297553811,0.3214837309917813,0.2805214038754185,-2.4288887105510715,-1.197288841526746,-0.8936475645564412,-1.052587735274105,-1.337788786446574,1.6859129386090266,-1.4653810204896043,-0.1955532765723431,0.08064206866944557,-1.9748711556368912,1.3999941949441754,-2.3870596987046895,0.08977233909026436,-0.5352946134085076,1.8305297387100774,-0.7816739702415261,-1.4251397922437308,2.4909606246966867,1.2059496242052223,-0.26643076715542136,-0.8879751643612427,-0.7538775982181063,-0.07192465697441723,-3.2777015845363624,1.4090422925118515,1.2105602797638753,-0.9945771831304133,-0.04962759592869802,-0.6348139402217069,-0.7971536666876828,0.14415497581024794,0.7007274684422582,0.2584894223487724,0.4555489377145719,1.9367504360459316,2.2933956976293093,1.4584671670030296,1.2402155846453122,-1.8248659195764114,2.05629502900021,-0.388200485339726,2.2568853196757797,0.20341479536575274,-2.1522828402441156,0.17295502774218416,-0.7587592135776745,0.734154196317809,-0.10309143436474033,1.6245028003713937,1.9597826331202635,0.7757827939551901,0.6684526772311941,-0.2868247768017739,-0.2685868567726443,-0.02449604112255682,-1.5549331919073626,-0.8936182455680947,-1.0570967088267083,-1.4747720439585206,0.26770211985067205,1.2463626138091963,-1.9864809938716244,1.772080346628091,-0.27872852695313877,-1.2843700394132065,-0.08838669573175743,2.228127669777796,2.13848536571455,-0.11456144297275525,-0.8765089352661687,-0.7367892234432983,-1.733162029724947,-0.7203452728755395,0.24806618216359302,2.3608946660729413,1.9048942923988839,-1.1089181590913693,-1.2454698394825798,0.8223584144364131,2.195938165446032,2.0924914169274373,1.144521064156429,-0.2032318267835754,0.7444645773159257,0.24164370822609973,0.4171218324318414,-0.7196447139797384,3.4262469274546516,-1.277833031474568,0.16593975418778797,0.5952474250281444,-1.279562537049722,0.41776867582329064,0.8066134239579558,0.2733183372109798,-2.6550076679598726,0.2645917804082064,-0.6268300827760346,-0.6421563799527282,0.598272256616736,-0.277907909239241,4.23401508466493,0.6275364086637379,0.9773219762412375,1.0273964978180414,2.056972949093434,1.612339506996622,-0.5290775131097495,1.3461148201859323,0.7149791390408464,4.7614701048575165,0.6324085749737025,0.9983856819411085,0.9616405971442122,0.7186229846425338,0.9307556666265263,-1.236595916338984,1.5339877361095835,0.11282587610585093,-1.8583384415172806,-0.22631243825602113,-0.040645008356697016,0.6134235304976478,1.6734169105697183,0.291454939017548,-1.1151136174566596,-0.4397709314583172,-1.3905062340167258,-0.3204226608330059,1.2983428735145295,-0.46948080977509715,-0.9627535298828603,-1.3729455136893445,2.0775329096053476,1.2456179315160796,0.4689542857797081,-0.6841172672624851,1.907960441973367,-1.172098156428669,-0.8965969689616179,-1.497773039045025,3.3664293822747284,2.7229169499198425,0.9308828489438447,0.03911894833831254,-2.5893365409680658,1.6707962634481432,-0.7041724862543977,0.6162991721506693,0.5523246474125313,-1.0735928525589082,1.2932217401504946,-1.2491544989717882,-1.205877823281371,-0.6323607168376809,0.43526780514370933,-1.1562680499766957,-0.8615670072394795,-0.2513766554357793,-1.0689224295518283,0.7428452566554068,-0.20166317901080533,0.6299835999439672,0.263921874345741,0.7870940067664136,0.2564427596875382,-1.1980908707889424,1.7507075890082433,-1.8300597557605691,4.950292658645831,0.06285612839776773,-0.3580121276317578,0.17982135905815166,0.7512881362411093,0.602270573384728,1.7510625220584022,0.5641845912625149,0.0728376723068009,-0.08343334047680227,-0.27530660707674154,-0.8609515293776235,-0.43709559217547733,-1.4147908229368098,-0.3042566842920101,-0.19915485122846083,-1.1470633422174072,2.4583763468261948,0.6572924689617486,-0.27264523696375237,-1.431240012293603,2.061144564284643,-0.3080338459219941,-2.0368233583990714,4.411929193832865,-2.5622089331353344,1.5899584905769186,-0.206739254319352,-0.2283401560521145,-1.2221173125232272,-1.892591656160316,-1.55142709224953,1.2258793796648437,-1.256974758745145,-1.385106022089292,-1.2779572502761625,0.3309473585947256,0.3609826902685274,1.253539808924904,-1.0176536187557714,2.3510529812504974,-2.7632967357876446,-1.9391523829607964,-0.38646770500394856,-1.737120631209149,0.327539428560864,-1.8659575126732655,1.312612592037131,0.8606588770797564,3.2301054565556755,3.140903475614959,-3.178988383558483,-0.5222531747933561,-2.5923292972648793,-0.05333718494682831,-0.5857759507450141,-0.9357444536977946,-1.4691745558266762,0.803205156873778,1.1880615586490448,-0.6972621754794827,0.7073074836861196,-0.591954114171732,1.469809890333696,-0.023745392446957315,0.6907694038850103,-0.45259412043707553,-0.5006386488479907,-1.2187880173489147,0.03241491088456118,1.0437695606224517,2.7336858325976254,-0.49047953777850195,-2.0949512596467113,0.48408197722127727,-0.945051361273593,1.6068393600898063,1.0472145312866543,-0.5907653523896398,-0.8445868109532718,-1.1051731633745456,-0.0990278802393635,-0.12355501211673917,-0.6115911325490943,-0.3314901547821226,0.3712310992666553,-1.1584855260709745,-0.07110402272095732,-1.0298367929049836,-0.71705472414593,-0.6345457239176289,0.3013293368717911,0.5244801206125994,-0.6548569631392568,-1.1225478947227776,-0.8334925396692692,2.2203657929303886,-0.030961598076686043,1.208063379756085,1.6869799159515004,-0.8614514702135976,-1.9379366681881112,1.9934722114519694,1.215522755598754,2.115699090595551,-0.6300826625469806,-0.5232134877344569,0.43825252503885503,0.08621434283416224,-2.0943965062498733,0.9319353195931281,3.0295751853751423,-0.4199385861479992,-0.13471809986802713,3.32801113190807,-1.5701067001783546,1.0950046147204016,-0.8030632447403535,-0.2344610767567586,2.3766030991700293,-1.6371100208017102,-0.8972276483886642,0.8828127608627251,2.060292476561298,-1.045922289222838,-1.1092252506061948,-1.7758049174376038,0.5440706966235073,1.4316940589534086,-0.624560533685789,0.003601811961839805,4.103869608998407,-0.2538556944502508,-0.5321800733855551,-1.330006589449852,-1.301983415569338,-1.5770471118068647,0.2744777176265075,3.4876677236578875,0.6159452021611145,-0.47518333792453077,-1.117587910587867,0.23287297718819475,-0.9449375105858905,0.1238569385849406,-1.650360865809539,-0.22489006441640783,-0.24356025057550357,3.4536343558844056,-0.4919737790942766,-1.4730261614096452,1.9189684881841316,-0.522439520230543,0.27197100562003745,0.33404018868593105,0.4017016223259512,-0.9499745062139665,2.958826851159253,-1.8094182801588239,-2.153038015779876,-1.9446676482474174,-1.9595830820032976,-0.6263603629738167,-2.0314404595086577,-0.5021300258483746,4.188904051124925,-0.41781936883909543,-0.6312856303580123,0.10690184679868042,0.8183974786638832,1.3726344402968857,1.3491514891875191,-1.7706567816601044,-1.065117890683128,3.790175248714895,-2.544695386835339,1.1888174126195135,-0.4390687533497969,1.2302696275690856,-1.1779942701271988,-1.4077958513391866,-0.2184111658832202,-0.20239355294153286,-0.7215657015571477,-1.6584391843077444,0.1638899212430161,-1.053446143706254,-0.2771791092376704,-2.0791275612333857,0.5654650394379331,-0.414219709218021,4.498841135928458,1.146510019065848,4.043183108709476,2.1899514524453547,3.162322812727316,1.87973560272844,0.821053406700222,2.1665691247072245,-2.2310925251741813,1.665386603022078,1.8789685765533242,-2.1979874879577004,-0.4767944563744687,2.3423860640559084,4.083732337012267,4.146579492621969,-0.20951758874758739,-1.5326466823585827,-0.2939291276491421,0.5547123997245815,3.304761238989628,1.3164861185219727,-0.5641699668358151,-2.012011745474794,-2.5363476275782526,2.2084350887175037,-1.956192313488456,1.1485814032086408,-0.9158751321462323,0.9796118793560162,2.1128583929817517,-1.8568459934891484,-0.2995644474517687,-1.7975874275022394,2.040036216000019,-1.3982239954001356,0.48995424510165103,-0.7764750482224009,2.8191648297683956,-0.8269944657130814,0.3479594651824103,0.45108787386945853,-0.2747788545298999,1.2269634177082676,-0.9807567968026846,-1.5829975010399036,0.9776649333463318,-0.8279467196015343,-0.16955772320724416,2.3681867553541704,-3.561550677669107,1.2517980751437074,0.06933421286324089,-0.5604542175541236,0.8709371372753836,-0.1891118987471678,-0.613804425857276,2.0057397342481895,0.23720838460346375,-1.2050190628387814,-1.4102574291434582,0.517706956993794,2.1707779284142883,-0.9832876117493002,0.7515833157278649,-0.29570707872202867,-1.6915453772837827,-1.4597922712895162,-1.5269559361874407,-0.18084701544889553,-1.0586302140175174,-0.5099564309291504,-0.5115888810402079,-0.008529674299668359,0.21095691802983893,1.4267993117081395,0.9688569564801,-1.0475761824349419,-0.22705919278022133,2.3324514646213195,-0.43379732262361587,-1.8617119906555877,0.7528458301438252,-1.4183096673717857,-1.789134868231405,-0.7591409342737562,-1.0687131815692141,-1.680859761599005,1.402442794723997,-1.0623232509795526,1.3748345597059817,3.7485911277115416,-1.168033050626656,0.47967160204233533,-0.04790824834693775,-1.5021357082078752,0.7805208180715061,0.6694212575714176,0.7370415174615744,0.6078380531332114,-1.1998324579761375,2.262080633926953,-0.972209977960901,0.8806150392418095,0.6405172402877314,-0.5680361041963133,-1.428478473692278,-0.7904091419101835,-1.7113490037828232,1.9170618055478825,-0.7381813109876324,-1.2764056673229283,0.1760186343078606,-0.8503930054096416,0.3839568956565116,1.8430643925026644,-0.17634242172008208,-0.37850926836093596,0.007088498999861909,-1.1045123032150364,-0.9536906794941789,-0.365022752137766,0.65287853763065,0.914402579176855,-1.704198885009167,0.1335119871769165,1.360586438212369,2.933212435564763,1.7705163082179032,0.5718031275823497,3.9420507761467056,-1.7695782398054711,-0.7664448792706713,-1.8456836091356872,0.1649261923053933,-2.194573499621626,1.2709157212977522,-1.2435425409823428,1.0911519813914317,1.7946000558466582,0.1639184332353859,1.7130147013954264,-1.706511448537574,2.4236130341107214,1.0883575642041563,-1.9505650891503308,-0.03995271077754567,0.6516142216061699,0.5826857409344226,-2.297970519300406,1.3821787480128391,1.4641012892112883,-1.5546718785176952,-2.2145988479637184,-1.7604627219946523,1.4474797811555373,-0.6999499957377194,2.4232748557003543,3.233141216058603,2.10263632068241,0.36834075763979557,-0.34311302001083116,-0.12036858793206386,2.3233878123154397,-1.2828147375383276,0.7474445299979614,-2.244701608917361,-1.6656072837052898,-0.5086922286424156,-1.5973558977601472,-0.8450504700713776,-0.7946126630554095,2.8076331743345846,1.3232476283508914,0.8504020744159363,0.787749302537308,-0.551310435180877,-2.121703087566376,3.8518817023260277,1.8463780087630817,-0.22120953677805477,-0.5846482372554862,0.6838253731774031,-0.3435872596481631,-2.364857190265587,-1.0662117761680523,1.077712384854879,-1.0466518565510217,-0.20979471589219903,-0.4222665153079653,2.0809546692244294,-2.0240348348290773,-0.9301033347122278,-1.4864467707650908,1.9783850488071468,-0.7110241092225176,1.6872436120138294,-1.8696364086779997,-2.1625707750371665,-1.6874601775357152,0.3111397756034217,2.601516974745695,0.33160239083288323,-0.8115570691179717,0.4662223130380292,-1.2709935164257142,2.0360973889176193,3.1113053244448334,0.020456522582258663,2.060034392669242,-1.2349282538094954,-0.7042940923704372,-0.587380846282357,-0.22732672686548702,-1.1154766854617846,0.005997393138896434,1.4931032698525688,-1.4166137444569604,0.2392926598558266,-1.1887971270824795,3.8652704204993706,-1.6212356849780156,2.33471597664881,-2.463915181509197,2.0513670137090703,-0.12461190610254291,1.7718012810742498,0.9501118749830054,0.26260313815942915,0.16868609638732976,0.5331741899986343,-1.0284887683563138,0.3363704169064153,1.4766770807600833,-2.338813595983712,0.6894053092151928,-0.2991274741426586,-0.25877285800262007,0.26178512026404793,0.23148595549634357,-0.07342102456086869,0.7472546944803924,-0.6954381721928029,1.3040183874099545,1.0059986018145997,1.1497090544034423,3.8520222351717437,-0.6786898192489822,0.637170145494418,0.6511831084171046,-1.0077224820757846,0.34491486381924547,-2.163569628936384,0.5834111623953596,1.3053846479302886,-0.19819686355635852,-0.3636490312557382,0.7221637238943149,1.1281885600426822,-0.17044152569875204,0.5580180606985333,2.425561806429616,-0.7101921447867388,-1.1050481345820145,0.31068227357096845,-0.2825070963711516,0.6427497473889088,0.06963276684371648,2.4063490038301403,0.01964904607157099,0.11057447448236901,-0.4932319735794746,-1.650519035482956,-0.9358875074424423,2.4093425303031677,-0.9972503894692549,-2.6769959437010185,-1.6276446684668076,2.0080059426840715,1.7789356826116807,-0.3882668331007049,0.9789439745448901,0.03188029142052876,0.20885017069192954,0.9049106332694448,0.14324068183556846,1.2307313985956738,-0.8428475665213232,-0.6266399874209436,0.8426311294057907,-0.18602158328750956,0.7584147273584835,0.8704732068627813,-1.455595595646029,0.0028321925079736626,0.9245586489917232,-1.0835061539256727,-0.24701051774259303,-0.029137945221669452,-0.8020906801293176,-0.6360605587539044,-1.1845280070448585,0.7938263343888504,-0.2524197901749009,2.550714314980449,1.656735245881973,1.7393134486725121,-0.5959690923511246,0.4435479753729582,0.8517184182038616,-0.7073267552378216,-0.22129450538175371,0.7950477366965435,-1.4207541258592085,-1.1310467235245865,-2.0449448042827165,1.5851059155754381,-1.464610583928662,-1.1090965090752656,-0.934846144209378,0.6258217414498802,3.6783074522751096,-2.477064164409242,0.27985754457333173,3.7496954633514603,0.5765399942010007,1.9265409002213887,0.5666179566811259,-0.7746065169130113,-0.08353506165574562,-1.1729391601043408,-0.6068830847131367,-0.6702337592169073,0.3814236791633051,-1.3051189577830744,0.7412993377620758,-0.2561176586961268,0.14778862523899466,-0.4443649415631558,-0.7129022573205167,1.449840364344913,-0.342754773273281,-1.7342137200626906,-1.0145536483581532,-3.0412651598154836,-1.7120995199120408,-2.1024176708600613,-2.4669349786728887,1.0438305269616128,0.29085908072497685,-0.6218057289946178,-0.9138771001833955,-1.5519040157914816,-1.1760431738607446,0.2589607838523879,2.654966866765536,0.13818054910838845,1.4903781134580985,-0.6695442851162966,0.374823590450734,0.6215098257940052,-0.0755969070758358,-0.7360823627827371,0.4903185751408073,-0.0446406962276546,1.3954519292877468,4.351546714861854,-1.2098309274220227,-0.2453027889648767,-1.805569604611212,-0.22516646090594386,-2.19011720484068,1.5348746779601572,0.61103326159038,0.2502092050355543,-0.7408605486925364,-0.014140754374328511,1.7449469885315325,0.9209959959421453,-0.22687499542788556,-0.7040249783274201,0.5751338390584453,-1.541257690386529,-0.4378079511026236,-0.40554520944833183,-1.896293418909003,0.3753026943988401,-0.06382608173842227,-1.220036419241841,0.5583321678765373,4.099463955598623,-0.10088229192882096,3.7638637129007373,0.05122572181425324,-1.1368139742750007,-1.2265431656146866,0.44328268171342267,-0.8927637587375394,3.6608756279128976,1.2461694247747725,-1.3632062927689543,-0.7286939324529932,-0.3276839659524573,1.813875370562759,-1.386362127256694,-2.2260701836527326,-0.6266952644471941,-0.6624799877365116,0.9713586695234793,0.4511517181497342,1.965685818517604,-0.36575561201316054,0.3719865783957354,2.62413353021644,-0.3816442080741534,0.03192166553240195,0.5479577026915053,-0.9436447114115463,1.9583153152266197,0.1611338008339708,-0.8359247291076549,0.9464866970921084,1.938507274522259,0.40604423485018726,-0.47669696499706243,1.089779545371224,-1.0844825436701058,-1.4011985282842323,-0.6095217501666553,1.9550667580254637,-1.5517999795073047,0.04877477464764791,-1.121766933400443,-1.6481368901964508,1.068661247886793,-2.1304814942533157,-0.9230882769487675,1.6113815061142949,0.42151530290888967,-1.3970134596775543,0.1434844811682396,-2.1875885085349864,-2.1537281781476123,-1.2084675452344644,1.1090683211239045,-0.5626262269545064,-2.5594570361919544,-0.41092751321407656,-1.1872645791824719,-0.7738724454856317,0.7662941525166134,-0.11728071612110227,4.068105305355251,0.02059576949181517,-1.0199947114324013,-0.7104323564754917,-1.1930225224344637,0.880976904697867,-1.273513303463028,-1.9917715499199518,1.6233668538730688,1.6509135844447302,3.2501117614909076,0.6613595319945269,-0.6920189402627887,-0.43705790963559027,-0.7225370721851777,2.7564542144411597,-1.8191972529571223,2.021968958217987,-1.3797967712513426,2.8540385183122994,-1.2319904947211118,0.6990262666336519,0.4932919621152769,-1.3961461414567184,-0.2446362743399524,0.2144266864136979,-1.2792979629484762,-0.9730472288372843,0.6298108555712364,-1.8347493945929612,-0.8284043735075592,1.7939153541682487,-1.4633282700858954,1.8796044348121175,0.8556283196582054,-0.04881013031021481,-0.9450402051995312,-0.13910334786408396,1.0343647357011876,-2.116835822335352,-0.8535366183514318,-0.4459098702065316,0.6220913543088809,-1.0549495539081493,1.188032256401735,-0.158722588845081,-1.0035848271603798,-0.42260390012288257,1.1428399815323682,-1.3361921573256723,-1.3599470632938628,-0.6024890615225004,0.2836825917957181,-0.17001314628074918,3.5251519857508646,2.5659409429995095,-1.451894165188425,-0.6970835993212842,0.4362350218425218,1.6789824138496254,-0.7595914057052298,0.6230392307700401,2.3985558869722854,2.5690503496284367,-1.0797304233907985,-1.4565714082764063,-1.3246120375730128,0.2977531588306596,-1.4845191375769131,2.5748248639268856,-1.8527821612215454,-1.9812343305115463,0.40533756582252195,1.0706827784855055,0.2847529205091932,-0.8105483021707177,-2.366890373099432,2.812017935155986,0.6068706598670546,-0.11497786800141457,0.015033087488158882,0.7279023344451717,2.4742842130187426,-0.8010148179243024,-0.500350263561185,-1.9031877514155957,1.073706708310817,-0.005899405696014237,0.48862845702248936,0.11450215885242664,1.852893328753112,2.432262485916406,-1.2930237300798082,-0.22940199003263498,-0.030555103801840084,-0.20152978308877337,-0.06044224838459368,-1.2635431671989668,0.9079562963123171,-1.4833617460704378,-1.8992848738806893,0.6427300763975453,0.516570603706221,3.290557256444534,-0.24803386185533435,2.5039494083989333,0.08957114157842151,-1.1559392690250634,1.918743320395683,-0.9823453295735043,0.15420235584200095,-1.1474731350498446,-2.2807179181441697,-0.631109860851001,-0.2472280250330467,1.295519294704103,-1.03912707533111,1.1845342427754066,0.0930810370649304,-0.1682944536686058,-1.258251348530543,-1.2552920654134576,-0.1851159091954371,0.4298067120814427,-1.898977176719482,0.40847023849776315,2.187989115257123,-2.4347499419342857,0.8520762781314516,-0.4078123669433914,2.145913582241238,2.1040698038411967,-0.6364427922413831,-0.49556442274083706,-1.977584655472834,-1.32695154303444,0.4860297115769109,0.10816296965657198,-1.7572575291182349,0.6005872663876457,-1.2642694108583448,0.4093100292966134,0.38918667911659494,0.811043865507493,-0.556262341155913,0.9797358399848304,0.7925970285483807,1.3235909539223576,-0.3788971045614639,0.34736686033506203,-0.07125676453983473,-1.5384561021044314,1.7709413556443139,0.14734504187901604,-0.33094404373990916,-0.5167302794974982,-0.9645493903451429,1.4286102594943728,-2.3807398935060404,-0.7073837459718236,-1.343427539766354,-1.555354644216816,0.9337279889879638,-0.7698667638049083,1.0397635132501895,-0.6837786003722607,0.0859832386663928,-1.5407240505196804,-0.8596966607906001,1.1284875001166181,0.8859562314827325,-0.5605243519027486,-0.30579449763686756,-1.563653929017866,-1.0741480290240213,3.214368059518041,-1.1634862623212157,-2.4617376565377707,-0.3736926663947791,3.7189886040275804,-1.2914982877422037,-1.5323112543981643,1.2824711800680881,2.4808142651237253,0.39837316456221133,3.6187934990643966,-2.179121881505298,0.4619392931849865,-2.4326193370270492,-0.7305918722026381,0.06965072447908863,1.519241115669354,-0.7526098853932925,-0.5688470376594766,-1.4772541692977377,-0.9706576758000871,2.196929840917385,0.007880261408899861,-1.1635704374787517,1.724701390262396,1.7155398073389334,-0.6614127370762505,-0.3082651777264927,0.1605090098617074,-1.5588815923670816,-0.4338409606089403,3.0049745653055493,-0.7228006551141924,-1.7189881999522871,0.5882075961608633,0.04726018247398149,0.42714829057972487,0.16709309566885602,1.9760437684922152,-0.12463458822471454,-0.5821201577493602,1.021470735300811,-0.9704581045776463,2.4028813901750405,2.5211348477737077,-0.08113340296054507,-0.10771095677236783,-1.4529994470975984,-0.2655679824864782,4.913213467626998,-0.32184226745467803,-0.2939443524716216,4.3600573378500975,2.383608533558857,-0.9611493071177314,0.9422862343906733,-0.2569857299874944,-0.46904728869329126,2.421824069660971,-0.38441312256320304,-1.6528072076391136,-0.38300405637240265,0.3440757427621905,-0.8651358788137332,-0.43493077065364405,0.5529028433641754,-1.5968747646882866,0.9901771202884609,1.2458668227782492,-0.5698502893875537,-1.0081874416557075,-0.6998740203793079,0.5314390421357716,-0.5426139009727091,-1.46271874974087,-0.6460063620237988,-2.0915754615754905,-0.6138411217918205,1.6585123948417926,1.9006509405003864,0.37627874582553966,1.7383955618644842,-0.5094824850897632,-0.4656257498976021,-1.2214498077070417,2.1386855440458756,3.0099608648375336,0.752610770942841,-0.5241858508887542,2.9551396867257496,0.4680022673877945,2.1200190164258426,2.9622383282669476,0.02609117803749495,0.8748623399881175,-0.47326721007916694,-2.05106261144017,4.27695686822722,-1.0902939187283966,-1.7797434540935986,0.8882155248319418,0.6078940484515434,-0.11792038724681542,-0.4134753683449899,-1.5645534967180188,-0.986417808681743,1.4415108149150697,-2.1510066845941114,-0.9833167281527319,-0.488788647552874,-0.9876851333759311,1.2061690185233425,-1.690877834607387,0.005188626105510681,1.232737427331139,-0.18769037589258294,-1.9137997212964883,1.2764623534119068,-2.3827890465688886,0.8433331311652169,-1.5522759372026982,-0.5463045053455431,-1.5433989443726805,1.746870310511597,0.24714913026444577,-2.4974715748727054,-0.4279110750582377,2.122850799923204,-0.7707485451356982,0.913173826429974,-1.4331841021994582,-0.24867597175129266,1.297640040900898,-0.6370169649430777,-1.954303598630014,1.6623958126189298,1.8801625900150387,-0.11057811714540078,0.18719566760207274,0.5551949578539664,2.132010510671956,-1.6278443463021908,-1.279321445320752,0.22440702822263092,2.1281202882392627,-0.8641746126314054,1.169848902438909,1.05881766599134,2.892200982092089,-2.3604356382307397,2.3494400490912515,-0.2720884273039478,-1.1289273914449809,-0.2702911115249296,-1.2447767406480128,-2.0350520398767586,-0.7699901143651059,0.06525775817483476,4.5140689899510305,-1.36910209797551,0.6288109021367161,0.7012853412690582,-0.8948904670379438,0.5169883911603941,0.8015807994672426,3.0807065134759846,1.5563492295667816,0.04266423379747655,-1.316151089245068,0.6741225169796127,1.2644213219691194,-0.939036977714491,0.09888466409950245,-1.4080961872525746,0.3664755467467636,-0.25461083483555375,-0.1533135396192931,0.6450178986766799,-0.7900053449487066,0.9729495214616961,-0.982880324276042,0.47419659559495053,2.8754340097343096,-1.9946974545258775,1.3055625866778444,0.07385954605549369,1.939915945701307,2.1501745604962985,-2.3283023833904757,-2.2763433848943064,-1.471555754978555,-0.3126017431819956,1.9197505577495635,-0.84867278720409,-0.4192404722922081,0.01662389635641736,-1.421666624913698,-1.0541685008800785,-1.695218423084694,0.038419342839882964,-1.1873296427658473,-0.8655054089125314,0.7755808925094406,-0.9180286106259274,-1.860944785490431,1.302386861121135,0.1194178680925261,-1.2112478707898515,0.478493122829188,-0.3563306578434255,2.268982303497942,-0.6550047854295504,2.085305690923484,2.35592063636853,-2.1628537592324997,-1.3747980825346213,-0.17352100360965414,1.0127967968660978,2.7551883424247787,-0.845184317066205,0.12644106195844804,1.1443829412184283,-2.2941819401418835,-0.05582542334449986,-0.17489100901611326,-1.1404762703069171,0.14215796333853442,1.4828978193055067,2.00739859496211,-1.244226121445561,0.3207482627547578,-1.9274025104043842,1.1249318101257817,-0.7775511808724125,-1.8625708836105017,-1.234870775755083,-2.068993965226298,-0.06661676535770453,1.6662675498773987,-0.0518003712822636,-1.0238028894987248,-1.4510189830363747,-0.5303034881100823,-1.1931763468752843,-0.6602022534672857,-1.9175575607448405,-0.04192052593713702,-0.27864347256568534,-0.33765583961221524,-0.8688377364440399,-0.6303451973182723,0.5739578392278251,-3.198932280111428,-1.211901432101747,2.377144636024513,0.2736745484673909,-0.3090970348823291,2.4426945428356737,-1.5738697430713613,-0.6247682869460948,-0.9249469929965725,4.287257330661512,-0.2275440279579461,0.9467471871549769,4.969550328517416,-1.5124993818999533,-0.5727332222759088,-1.4283780223695515,-1.9760384032667584,1.272536874011108,0.40065955353252825,-1.0910310804625611,0.465945393222155,-1.3266126425561489,1.8300606539067912,-1.6483553415857324,-1.3449761327473553,0.394757296667176,2.337636468611064,2.5484197275263414,-1.160158581385422,1.5332231615424436,0.041329138160125255,-1.260074007033293,0.340971625829672,-0.3543775270487411,-2.4186729122564934,-1.335885930274689,0.562321250270665,1.1141993633673157,-0.6627108558404435,2.4738007576871985,-0.8669427150642732,1.3205101952173552,-1.07865658286288,-0.9293574256433171,-1.4725914488546408,-1.4735259517716468,-1.014130543179719,-1.369705684356949,-0.6944280620501682,0.8286040431787687,3.1144973759307804,-0.4717074326864899,-0.686959182096146,0.4119665254692591,0.4336908703315854,-1.1769438553046632,0.33756229795210696,-0.5522142654720924,-0.21704309012519002,-2.331103773579112,-0.34377539685073516,-1.5796800647352895,-0.2759794675427156,-1.5917164874810603,0.22427118380176134,-0.5460033688294423,-0.6983372260009223,-0.9971771494010825,-0.303628920992931,-0.07924693129739832,0.8826631793869768,-0.051629377030984386,2.9725496404473333,-1.8056842074282902,-1.5334877780082514,0.6240851025364155,0.5952140070880748,-0.047524304921622226,-0.25612409865731217,-1.691546834582875,-1.0912740135033347,0.30037693076898203,-0.9036374063335791,1.6491140961353412,0.7782141272186363,-0.9648521788945132,-1.307077987469989,-1.8737998225481118,-1.5569299906962626,1.2541617240344232,-1.1265566846547936,0.513249947356783,-2.0961671206963093,-1.4864963292221265,-0.4922213714700345,-0.5259069294005989,1.3560107295811585,-0.3449979603988304,1.1838906691661841,-0.6535822251363763,-1.5739365990343825,-0.3419938044106335,-1.0219295253716163,-1.9454328276475947,0.17260686949850104,-0.7539537586089834,-0.7117647136799974,-0.09985905533350128,-2.2078450790152893,0.980198617180399,-1.8917681965535298,1.5947902175628799,1.4715028454345083,-0.6367104636522742,-0.8375359946832226,0.773660087338559,1.268154373892519,-0.07441665976180968,0.6933988387620433,0.5826902731859246,-0.5781031805568273,0.8478605120914248,2.5418804508540265,-2.2572235246925945,-0.035461960921404495,-1.9009570931392306,0.280761474580925,-1.3991080949815895,-1.3290381074179372,-0.8474016528142281,1.8723315118599966,-1.1575764978622398,-1.6723391812610224,1.7852656306269414,-0.5732323667132819,-0.6791127544576362,-0.3465803013488817,-0.5835538237541523,0.6001124678958834,-1.7189468943808845,-0.7833808183173429,-1.3055607474851516,-0.4683811037379208,0.6942934609572214,-0.03345201224868735,1.3195896926900659,-0.5251916028703612,1.5162941549266349,0.919946070126449,-0.038107024978808274,-0.7204468823292266,0.6954191874506498,0.7217387000398037,-0.10942147839165808,-1.314951737593154,-0.29716304559903184,-0.68714132040323,-0.2830884579231871,1.0184081817578978,2.874465709413168,0.14426821152826128,-0.6988261760862897,1.9586858735787873,-1.7209094073504236,0.42148221868076874,0.37920297938641306,0.7413812995414109,1.1788043175485692,-1.0654054390726875,-2.8521685866111786,0.9681933462640836,2.52236142678413,-1.7980328220874526,-0.7908538111501515,0.7253920172234103,-1.411355038116869,0.5045904721233186,-1.1920363184501592,0.32144695034902504,0.5011088912678453,0.5486090103753991,0.09243336708531182,-2.3835198881356567,-1.2758430548077213,-0.16255207213956374,1.4847166363585527,0.48445392529230547,2.944578731053222,3.7585231712056917,-0.9316211559756082,-1.8925447029152074,-2.1457891101470055,0.1906744430955742,1.0695885505533613,1.427234289378214,-2.2306072528760117,0.7716002903386506,-1.5445152286176465,-1.0294019357541893,-0.934851811351448,1.2596719973388573,0.20621226357588265,-1.2495513367387805,0.28432626349529666,-0.9369631708056334,0.48242236612580136,-0.8947991408272059,1.4380825173296612,0.20428290778838457,2.228616214546984,-1.9025576200209595,-0.34577221269085295,0.37961544129158703,0.28722750496212324,-0.6169196829364653,-0.4165979047490257,2.062676635480106,-0.24280644558805156,0.39473330922132827,-0.5260270886508082,-1.4680105046932577,0.3116086035762831,-0.6122442127161154,-1.3325811486315722,-0.7327673706565812,-1.6413304976680405,0.7292034998978489,-0.4720311593696367,1.3198911721011992,-0.16174317396693935,-0.6273675246853826,0.5114613079146181,1.2867427713948265,-2.728490780123253,3.429547793312769,0.5905044475387133,0.39308905323690785,0.5790033764645003,-0.20090024283921495,-1.5584108076566354,-0.5527707001470362,-0.7513727017760485,-0.5849704714207022,-1.4475045272479103,-2.809175266745832,0.7823750796489203,-0.5057514604252121,-1.897704676313059,0.6991693485037875,0.5238260132223943,-1.9094237172977702,-0.926253628126294,-1.976522552261444,2.3050507957635578,-1.293647125537402,0.6921450470689158,1.7104560983219874,-0.23895880883237614,-0.3639771846962657,-0.5922645200855735,-0.9977414959313772,-0.03580625319555982,0.7931241057644379,1.6469611243478863,1.5126278004348264,1.7208962099831258,-0.6405495552363357,1.103989008839135,-0.4673924830301893,-0.8959357219251817,-0.9172824536313124,1.1142061262754117,0.6159671767918259,-0.7711443346861906,-1.4331906914527894,-0.11590405654013225,-1.0590871049619612,0.8474741042405409,-0.13260141699044173,1.2529290288125978,-0.9030937390610927,-1.127837384617185,-1.3433576039496458,0.8656398819657006,1.306736843381626,-1.1024172459844925,-2.240399243109405,1.7547809301142718,-0.9451115807329323,1.3023535846623233,-1.8111041571282986,0.0748045064748811,-0.6921044514536452,-2.556671795489065,0.27968607902904835,-0.9388254750345628,-0.5299934829948825,-0.7612163894284611,-0.8501907868322977,0.4694284339565115,0.4861360026983872,-0.8103753603433166,1.4140688086485105,-0.8634603117318723,-1.2586825412060778,-0.09790519437250443,-0.6442007874103909,-1.9581633016964373,-1.3163719484765197,-1.5717562213534026,-0.7141703067080563,0.9604570341005889,-1.184915335973236,0.4739626841520623,1.4675678295882653,-0.8442512956798678,-1.6071969360822576,-1.2054203514011312,0.9754192405468459,-1.4597312999594654,0.18342123680860128,1.096698994986776,2.4196922005766606,-0.9466170554416009,-1.456876878899191,2.504416967997058,0.6507167479374809,1.0214555660126443,-1.9193871650151935,-0.5621560552567281,2.2799319672948375,-1.616717373633262,-0.8146960712864523,-1.2897350733424675,-0.7873627574050298,-0.8281395994904377,-0.8130431748187736,1.3952130399511773,1.141979425371281,-1.263007741657834,2.3753090613032257,-1.4010192711413143,1.65796032803867,1.4260306489014076,1.533937787949903,1.330966830538415,-1.2638654055257716,-2.39571225161521,-0.8268967180409785,-2.222946932633858,0.8665359665552993,-1.6447242073012145,1.5870248247668552,0.18168818643897683,-0.2561803233851641,-0.2843838079416581,-0.9967232403502938,-0.7169172388487993,-2.251816473574346,0.33126791394748784,3.385221561679781,2.2278410259498376,0.6300354797667475,2.739664536748523,-1.9260637477090146,0.9465218607947339,2.375962297239916,2.353493300727712,-3.489385762642258,0.09055466810459391,-1.2913714888873689,0.8916495084689073,-0.7004644253953553,-0.7323917492987833,0.7937230288007141,-2.3416739937276185,-0.44091239529997034,-0.3299736214395371,2.8777941923917605,1.0781611499908048,-1.7907640865530254,-0.04490621508385912,-0.16759082822770544,-1.0641588538894509,-0.3966923350371332,2.2834467099830613,2.831765409544827,0.4413031635950308,2.3120369262314036,1.3553348779049281,-0.984298205466208,-1.426495467293503,-0.2957452707342486,-1.772230635077462,-0.45282866494426227,1.089346548312369,1.8206790594889086,-1.100157082068781,-0.713018920522472,-0.7055653405521328,-2.4275048638609773,1.565995206548829,0.6873119886124919,-1.575789061535444,-1.2509471122362292,-1.886209937876051,-0.22048511399720688,0.026877879627441873,1.1197588126990385,1.6863574001865567,-1.0737557699230706,-2.088408020777979,0.2429585153918962,1.2812127416795378,0.07696402669974586,2.034098257355408,-2.0108710056089647,-0.48585728252531796,-0.9477567330827757,1.693879859795354,2.545806252340351,0.3632771302281304,-0.15284372463410573,-0.48354316458999885,-1.9954596378282725,2.0046552898770638,0.1600626166733453,-0.005035923809955529,0.35382154730365106,2.0539969205463118,1.230133252856903,-0.8796422196926544,-0.9428813286803657,2.043981164833131,0.692296120656345,-1.3603333241964577,-0.43462811642362,0.32556421844665095,-0.6888433543895469,-2.3793611540561974,-1.4674774565597386,1.5317081923191043,0.09924201762813455,0.3137872322467062,-0.17565094719262767,0.2451764844886254,0.8258291700933981,1.3271998419721316,-0.3311531158896018,-0.6952060859231176,0.41383711203193124,-0.2854587117758245,0.28651082733693717,0.9238055248369319,-2.1052449006363805,-0.29123912072575997,-0.029246411545480785,1.1833855921781418,0.4386923026798551,-1.949891921914006,-1.201971582710788,-0.26956390050021295,-1.1949040600416747,-1.7725677182214754,-2.1818054642000138,-2.0027779234407483,-0.4857568544520578,-2.175043922822411,-1.0882113595352534,-2.0967038709829766,0.4354809390669549,-1.5341882832098868,0.6805668709274905,-1.1161159743313875,2.851481785537446,-1.3967712061645423,-1.6135309746830577,0.09453214819668927,-0.3340332929736355,-1.7515802282166801,1.4758174627218874,-0.752713176136748,0.7215042071069224,1.9367323764459068,-0.0761593999002008,-0.1278013172165855,-0.35302259450270007,-1.1498196085340686,1.3485253989176733,-2.163661694143567,-0.578643942742896,-0.06322503625085711,-1.1166274773144678,0.25703658216805925,-0.2811876261295313,2.226428932229838,1.5875317538971845,0.8977436560989411,1.0692438424794288,-0.021217197915929627,-0.8843299245365185,2.5806320910323706,1.7134486833128306,2.3696989125476176,2.4500106668407033,-0.3549386736536153,-0.42941452386470885,0.8586510849118056,-0.1328305327739699,-0.5119384709205799,-0.22607683990012306,-0.5471444369510627,0.13817652686460613,0.45632011231111846,0.5845810536925704,3.3409281439276565e-05,-0.8769259789999753,3.353478625265862,-0.9592516928049917,2.2346687937193845,-0.9330377746696085,0.06748636075033036,-1.0945893707195498,0.018657452318162973,-1.3547017069889757,-1.4299375652906459,2.1967322437515344,2.9094007443394,-0.439702975797837,-2.052858259876118,-1.3988589754488712,0.02062973462673419,-1.7291406397038334,-0.12090049506471241,-1.9709843766185406,-0.24747998601440743,-0.6850739837522084,0.8654035785826796,1.5048897166563966,-1.336315945991459,1.421238021082999,0.1244228206398942,2.4148697864872353,-0.1864963190522101,-0.935662498401138,-0.23475429898956174,-1.2217768301349445,-2.1892844106012674,-0.5189632125624655,-0.1350149101519436,0.018591024403074217,-0.6817258497340827,-0.797764468567999,-0.19435143946269787,3.4083107801395065,1.661847034511892,-0.632324560725902,-0.1792823937645369,-1.3633984256358245,-0.3203293604853211,-1.429111871817543,-0.323033968976138,-1.8780080738512406,0.7039027080198212,0.0008982402595265851,0.9165166557954704,-2.4665911977470647,-0.6875155351156483,2.4907001817900274,-0.4638161214719311,-1.302956586492002,-1.9970679627942352,-2.221808394459264,1.844812715516531,-1.3760658056832684,-0.6455286197626654,-1.4538151648959046,2.9458619791815237,-0.11831381268570255,1.5382081470545068,3.1548567459597217,-1.2506162419563438,-1.8303213567756564,-1.6336452893913906,0.856250309645352,-1.2491624328607296,-0.42453489753419826,-0.32806047451260517,-0.4196307614777723,-0.5124240078053721,-0.7377122249372143,1.4080479928234901,0.5245013407031418,-2.8731314397567456,0.8741730817449949,1.3305212850092427,0.988508672555155,1.0970636075070497,-0.9187345085657775,-0.17748196978616568,0.7661228728066439,-1.3107029954659934,1.0041111814725265,-0.07317191182601095,0.9910808531155333,-0.7673272351176547,-0.833652329462966,-0.6908291726413789,2.684647622818325,1.4783299919029815,2.67135997218154,-0.2705445774331393,-0.5066177258170054,0.4059682774651925,3.487072996807517,-1.7916170154664466,0.9708284692993265,-0.5156712015430235,-1.0547411664502675,-0.8855931442607546,1.6182017559124606,0.9494537483452985,0.565175762326643,-1.083295725434352,-2.424111073685718,-2.2188518613822743,-2.393047098285821,0.5393721638322934,0.1129620029044992,-0.7265109417212142,-0.512106901692647,-1.9653841636462288,-1.5870969688289067,0.09279361467777286,2.6970899740945344,-0.7660746880065549,-2.172768913344468,3.460885303227718,-2.0711759780606496,-2.3638655956818835,-1.0300276041372112,1.0115704551125433,-0.8697853870150015,0.9892591968216763,-0.3377644537743679,0.39187295917790305,-0.804609276664726,1.0099120185412234,-0.13740159807804375,-1.9950846429132258,-0.8734463223353572,-1.329839563398136,-1.725857555353665,3.9125742873272444,-1.577627110831558,1.1576877645332337,-0.18705456594916212,0.9886618410881696,-0.48306456916319723,-2.4540986168260313,0.510926040846994,0.32562558125392566,0.5122545028744743,-0.5028553314735963,-0.8831195746992925,-1.2953433864319728,-0.8385195321288308,-1.4071297971196217,-1.794624578791867,-0.41174847640494716,2.3600963377343627,-2.0026772911589563,-1.3943837315570304,1.047803932273664,-1.2569996154800898,-0.6992480289679174,0.7796706646667844,0.5377321894690562,0.2274558196883138,-1.1268074087248725,1.9365519756460727,2.302886744299866,2.3539434292117636,1.2005230116010819,-0.004889267070240016,-0.11862847320153758,-1.0393804589328095,-1.7671977760195106,-1.2926152024913702,0.7876552421220651,-1.2824405765294427,-0.5678737703821439,0.8074297509146363,-0.9774432235306209,-1.3422119567769182,-2.239964116096797,-0.247742581521172,-0.4883419395734887,-0.10454268096199076,-1.3218538940208802,-1.8539824300964807,-1.337678268230216,0.6522051079177462,-1.9340292561231296,-1.9370745657046768,0.46843656757287466,0.17521232471705128,1.0055262879412539,1.0394886017121518,-0.658301438475762,-1.0375701639155326,1.9557634927412588,-0.7031518252847991,-1.113691911252657,0.17432789515609914,-1.468028175642937,-1.6070536847522527,-2.8108505988137384,1.483377274181798,2.3727641694011057,-0.9205163598405811,-2.1040678386859737,-0.02649737374811617,-0.5503952775453691,-1.4314427918379402,-0.09044644340378746,-2.174101429965341,-1.155038514842176,-0.1714039565536745,0.34440944886990327,-0.1425745129221681,2.409953048594518,0.07768437112920978,-0.7447164065937795,-0.1949556471028388,-1.6718968161325745,-1.7550962404691273,-1.6743522882883901,1.0793848731960491,-1.088488267818076,-1.190048858954596,-1.1612504042912946,-2.1204656056475373,0.791216837765367,0.11234334171055324,-0.7905893304470908,-0.3696518823058862,-0.22869142662643316,-0.24853253425291102,-0.5268635821074348,-0.9930469817765585,0.6051891670542564,-1.4527494033712474,-1.0940449342475698,0.2553395731307415,0.9538616906016348,0.4093719753119373,-0.4376273487140562,1.0918555442396876,0.6603190538242366,-0.0723434946680768,0.5420151753688274,-1.5287212040020384,-0.1113907463064204,1.8104213098887774,1.4641052250561823,-0.4731409270877527,-0.6136938497210565,-1.6004679586001416,0.7139821414863813,1.1955382336393996,0.14850409445799978,0.0670315736905287,0.9603397917447065,-1.8633120291498158,-1.077325431998325,-0.29217251563413743,1.123588847582687,1.8979184339034578,-0.15121480115108601,-0.06238820704832501,0.990121434490783,-0.5844542927426566,-0.6308548438739191,-1.116478421726012,-1.6027585261847295,1.5703145989600975,-0.6169505518861129,-0.3008998279029822,0.31979234635515796,-1.2698512694233846,1.2627189983483633,-1.041668591899603,-0.38431834262695025,-0.9061171473978528,0.8860159674706984,-1.2424468839704834,-0.9912586488009769,1.0583380835282354,-1.8199261969711664,-2.6802898547275453,-0.6701433985772816,0.4376364913461522,-0.9716340297157676,-0.3418570700353722,-0.33095247424398794,-1.570170450202797,-0.5692588833508218,1.201516310386973,-0.6365848156395449,0.9668616786358616,-0.5407506814329159,-2.7310942595434695,-0.7227398615449562,-0.5163855820493942,-1.461985713378694,-0.7417304438383941,0.7788490682681034,2.9956065037520316,0.8451693308003946,0.3171038702463191,3.1042890217225074,0.5062270029816032,0.24471776292114314,-1.0113412555200663,-2.3248359380995374,-0.06187502289617486,0.5515198010198383,-1.3452223869778694,-1.4293582159166998,0.4608251562141372,-0.8289497219977309,2.9452436507383557,-1.6279591762221786,-0.3470868506665045,-0.30112103216378916,0.735180732249663,-2.142462939950586,1.1034333810522525,0.7546965386008309,-0.5032754120421705,2.465840870074097,-0.08456751382578077,-1.1418623998430035,-0.026077155469655507,-1.0891118822936552,-0.26211385767690903,-1.257743593305673,-0.45244895718116096,0.06576336108182156,-0.9946159282337765,3.1485247842079294,0.8858469193786689,-1.415311349644316,-1.5883682054859074,2.539418434535239,-1.1611908016002246,-0.5369085200880421,-1.6581548343911665,0.30578046504163237,-2.1882041342844043,-1.6694570528480337,0.14965328463206964,0.44551992057165485,-1.1261382468266872,-0.14221930351835985,-0.9645257210534824,2.2909398923153517,1.5706062710066133,1.8303807056546622,0.18372572910770169,-0.6944698195227424,2.22940855230764,-0.6775153342190267,-0.9979757071064436,-1.8024435505103191,-1.8892181569944684,0.8022091867074028,1.3296824427196505,-2.046699877975898,1.0785226798684986,0.2854887237089869,-2.701776761978451,1.6882260071197455,1.3829696331078474,3.709087924211424,-1.5347450007098071,-0.782508421057769,0.5431660581612416,0.9824166470930863,-0.7899444540755788,2.30589216283778,-1.8958511570970715,0.9516333744914279,-1.6152712064723889,-0.5553051923623328,-0.36438743367570564,-0.9412624953748231,0.6852487470846476,-1.2530186629003792,1.7206776363422578,-1.493822755202145,-0.742796780224693,-2.409265010419932,0.48215357181749147,-0.3343231434969297,0.2986249265258801,-2.4379078135281347,0.10857999394985823,1.77659110182563,-1.9032815051025387,0.0783713093083235,-2.0534642177387123,4.194453716658334,0.5075680470808048,-1.2381735515104146,-1.3967193665354516,0.6593208628559455,-0.9995759415251604,0.018260479045812177,-1.2026542879216677,-0.5393200490204548,-0.8486193661937023,1.5523329294336108,0.9668778269661298,0.3371720372546951,0.561434801852869,-0.8529698302519213,-0.2679473164654985,-0.6377173227587831,-0.8842529082851421,2.618477295062052,-0.003482305683171383,0.723414169801703,-2.2483387846488387,0.0005234662865490398,0.6932106503501672,-0.8219960862103219,-0.414356264598818,-0.9137312794663592,-1.535374362287491,-1.1471871518819303,0.0020285747179790287,-0.2916278615957698,0.2072173656637092,-0.8170514964688441,0.780681720098951,-2.0937507819015195,-0.16838965506961318,0.4429012590082431,0.09772116014672236,0.2920702729633013,-1.996984920772324,0.032394553048195086,0.0949851411193137,0.36678086341466615,0.3054045140665691,0.6902579633618424,-0.29107674473840445,-1.2382550862301833,3.2924852706271683,0.537874459848941,3.544211690878391,2.995086664639615,1.7769831366429298,0.7389693493377515,-1.520532904805372,-1.535004077916714,-0.7191600019555429,0.6608235597992866,-0.3860053029635113,1.4065273664510902,-0.14222789274744616,-1.9255714296143562,-0.21752357681875434,1.9253104672991719,-1.4652821143365913,-1.303838893257738,-0.537451673080359,1.6311971518366568,3.141546379950215,-1.4312324014308748,1.0303606261764882,-1.983119793825013,-0.5879592765192987,-0.5954715667669891,-0.6540836374478312,-0.9330583551640091,-0.620630779212513,-0.22750806463706347,0.10782319707087336,2.935518637118693,-1.2350390039157528,-0.1312048799361138,-0.06683944446166631,1.5176645352109897,0.9448157067722667,2.634005879535956,-1.2873717387784054,0.2713447250190502,-1.4522056285791725,2.3838450762071313,-0.28657083322483085,1.633826238298901,0.26007917963531224,1.4773229935591086,-0.43806733930414105,0.9350628849700496,1.4934901736931914,2.627191408971753,1.069916145941567,-0.30388214804464303,-0.8521684388377763,0.5237581418049801,0.3371429687188323,-0.4144766830069955,0.34625151196409676,-0.678694067904932,3.2799091840346066,1.9459266585329102,-1.0773571148653014,1.253772099032471,-1.6077894276182378,-0.5313549734984115,-1.8282727601457283,1.6224363907347399,-0.6728068249254082,1.726116319932308,1.0218036628495824,-0.019145955660809427,-1.1078388180727723,-0.4191001897209077,3.612339583719648,0.6242015040408349,0.22101383485943754,-0.7167057708139689,-1.4097422081583064,0.5814772606632294,0.33608978312262455,-0.9801172505699002,0.13229547352084117,2.2580825898372248,-0.6587827415569701,-1.5133991521672423,-0.9385500325274927,1.2176836978293326,0.7493958028280724,0.8423220604439211,1.5346640852034763,-0.3608822635662523,-1.0395837168557585,-1.186438514678312,0.32501870213529005,-0.6723013794932478,1.9013064715174792,1.3362398089945795,-0.26585841324553694,-0.046338710944713,1.0108808445787631,-0.16211618660235722,1.8830857726090187,2.3518419132611466,-0.8779293781371142,-1.1516376506995065,0.2987347505625428,0.6268562904369489,0.262980354383146,1.188259691604194,0.3147174507201312,-0.842852963061012,-2.1791757464356136,-1.8391230819676947,-0.29801558187361604,0.33446503931935,-0.8371761492213272,-1.7721899291218046,-0.48654670511745934,-0.10862003140277915,-1.412121647832371,3.245280988361777,0.6076864359487973,-2.1216612208006604,-0.6649710027604313,-1.4980215502835872,-2.253952337694717,0.14355091404710987,-0.8356535123623252,0.4005860475882422,1.7448490656535613,-0.5343469916396169,2.0061691449468415,3.256940578039749,1.0594754664345165,-0.36246931988910136,-0.546960939743203,0.09481442335330002,-0.03965529696835681,2.3061997712491635,-1.1225817230709598,0.5809328176391956,2.2649838313958006,-2.0819550998141265,0.23263476122277255,1.242535421657069,-1.2476382239475814,0.6789014187512327,-1.20291579964312,-0.2805303874376308,0.5020237939603016,0.186867665704916,-1.2484709215260996,0.5005450117749136,0.03234738437664755,2.6465078308320242,-2.151016455520481,-0.9167943683733686,-0.8263313584798382,0.23549408441490646,3.0915904818175597,-1.4824079222116537,0.000726685836321907,-1.8746769405330705,-0.9334965688125345,-0.042492364296479956,1.2784645100230623,0.0663164877638856,-0.9998717289257721,-0.21319848064186195,-2.334023508070579,-1.781903082041538,1.5004169919695096,-0.2715818401551754,1.22909326564993,-0.0966537792159265,-1.0738919102296909,0.2625198111424145,0.04060339169930979,0.3092596614196301,-0.9423977499825258,-0.7549775683153466,-2.2565095138545024,-3.2143751898842092,1.3842246893684556,1.01515272822387,1.3244666424805613,0.19804512826022472,2.197078454257956,-1.6784253289033841,0.6682033365399529,-1.1000562823234987,1.2834440894883674,-0.6793638148350246,-1.031304971028092,-0.7756133777671113,1.0687431835778551,-0.4529000287272434,2.8250038132716746,0.8833020648554757,-0.9628933704306212,2.0571579969186273,-1.6250032881217789,-1.8724531079422093,-0.3675477429920311,-0.2129850310210043,-0.938098657593926,1.0903472675800219,-1.3080922553646577,-0.41724955246326156,-1.7176265092075924,-2.40725402366942,0.7656616479392129,0.5357511747141849,-1.411481994578956,-0.5757923370082423,-1.1527618002628657,-0.399910044458535,0.7263792021925038,-0.9658843603446682,0.42478493593436245,-2.839347930957049,0.6240052875241351,2.659311581561017,3.641769726312963,-0.4268670681815514,-0.8128138842581235,0.45180273158409906,-3.5315183311094764,-0.9581928497885529,-0.6353742588289357,-0.6858377261717067,0.3050236621576998,-0.06609138307698041,1.1738857698127254,0.20158424984369588,-2.029917002215052,-0.4744075326387932,2.312607710234188,-1.7221994533243206,-1.3730091409110874,-0.8668036729953938,1.8876274717542325,-0.7904000076574955,1.8079621447980376,1.0220565507646577,-0.34119265220282624,-1.7919454531797845,-0.2023295745904923,-1.0936392250448086,-0.9807627993377243,-0.46445593623985737,0.0024047979866030278,1.8470207267719025,0.0976035982415871,-0.6319188077693274,-0.6612019826176181,-1.1514625251873607,1.0844366997029358,0.21475815748480093,3.3550506407958327,-0.7447645113917726,3.2049268060359006,2.18108083163689,-0.6357079043373568,1.5310382404721712,-1.4877004416619712,0.7419806141791521,-1.4670702307047057,1.6935383949078733,1.959532626494237,-0.39157368717477403,1.0181475883319004,1.8138215925771193,2.6433128845464737,0.2498780883262139,-0.7587481522323178,0.7694688219530623,-1.0749598984318414,2.062897030823108,-1.493451000679311,-0.9600656461681331,-2.2972719924059484,-0.8904731413506852,0.3343553283489772,-1.4135886958636241,-1.4907530977899712,0.3580961566177349,-1.2398703899375856,-1.3021059086340072,3.3512281734667697,0.24973933438273588,-0.052346874637307556,0.6931123658637653,-0.3524737818947921,1.3875711028611382,-2.251033754036156,-1.3300794362503139,-0.5607691078226694,-1.6320350260659318,1.094478427424253,-1.1769737604725108,-0.24395663378225918,-0.09259728527072815,0.47510251649129126,2.7632354634867697,-1.476683905646974,-1.0210191173146415,-1.8607553946261952,-1.2325010866948054,-1.7941120542128715,-0.7597097243964553,-1.4066313849276924,0.49369051771234296,-1.0752503955265056,1.094971406459528,-1.9409751445470713,0.24447498854405128,1.258196516575293,-1.1649456765188266,2.8949751536188306,0.2066554237273023,0.7123993494443338,-1.2679476147271231,1.6783048762174784,-0.6075594769819473,0.8268296409575359,-0.6107472211401499,1.6836792385484918,-1.0779890302699981,2.980662997246293,2.034263534538556,1.1213960278949668,2.2376393017529357,3.103572477902221,0.5004105654180964,-1.0788596128406538,0.4345958609588512,-1.2033969115423315,2.221496143816188,-0.9468528754910537,-0.3167789500421039,-0.5978049390201197,-1.3716738655762626,1.3170846574626944,-2.6733731548255966,2.0444397626689437,0.21146659522049233,-1.2534078230472605,1.9381676300625676,-0.8731093242904566,2.5474384647327635,0.6462325072055601,-1.8496480351158795,-1.767184591622682,-0.9865824325684603,-0.7025968586551836,2.163377295613388,1.627527977635711,1.687932565054325,0.13654602366289148,0.8632892776187905,-2.3545470176024628,0.5562714923448084,-1.2957167891099561,2.2573761258936833,-1.8301309669262582,2.4769473195300633,-0.9783618501912151,-0.8534015493354441,2.481690747401363,-0.8681879106406605,-1.013670233399527,-1.6215434597804037,1.57357193595138,0.22240536096580527,-2.6694181787629505,-3.035769245442675,0.25865643854488823,0.6250334215181637,0.9185807310352049,2.95005729373214,-0.19525398360826107,2.1086422388176413,1.715618242826496,1.6288645915748847,-0.045135142798904244,3.7339092200909962,-1.0594654789021336,-1.2149821104092515,-0.36150790535171967,0.7926500297764051,-1.4461508901674813,-1.2133530715223368,-0.07685761064299122,-1.4696959099144868,1.3440377404281287,0.295687720892927,-0.05435532465713213,-0.842623868177157,-2.771110986212649,-0.6576312319376092,0.7210104464093224,1.8059170359759762,1.731923875997653,-0.8048464049219146,1.0188194025494433,2.2567991876574838,1.2887863407867455,0.0757111880566552,-0.17381213977229007,1.2849748811086459,-1.3994730909907718,0.2986249466983319,1.3635154837400536,-0.14574523081817825,-0.46379448772461,-1.5580061151058686,-0.9951281246172813,-1.1267423274133972,-1.5507642558574692,-0.38997507490655997,-0.25004753214884845,0.6374917939838067,-1.3321237141645064,3.1600937035323273,2.257429293614341,1.8597753109146984,-0.6529118179849612,-1.3196396580809162,0.36288730598011143,-1.4090710653495302,-1.5708709241887426,-0.10815007807587562,1.4516566331833878,-1.5981169526914574,0.048661428191145176,0.766639848855048,-0.12353423399503156,-1.3579061732565088,1.1209629993402317,-0.6936973560028423,0.753559826935852,-0.6717317522577251,0.16186470024212438,0.9916644599440001,-0.24493464518440314,-0.3525410594941842,1.1435081002455765,-0.45272723130497755,-0.8531636708651417,0.014995029153379283,1.511765164697587,-1.0196235723064362,0.42935148266379836,-0.4551936967577421,-1.3358370776203674,0.9322949293229579,-0.04605172013969384,0.6060283782160226,3.792861759375914,-0.5209374212772413,0.5987463735639743,-1.578280582963365,0.6802556928181946,0.2255229540858843,-0.9369731754270105,-0.16893073697530614,-1.182768777348838,-0.06752651466283818,-1.6601816236342026,0.9335062833487572,0.3214632820967533,-1.8537107643966981,-0.12839609928270518,-0.24148430733639092,1.3624273045690134,2.1159032566909617,-0.4525834314722366,-1.4683834312976909,-0.22333327004591091,2.053585636127079,2.9963661766850036,1.458400718042106,-1.8467028407981108,1.4081839566115977,-1.0736022043949718,-0.4629933023313197,-1.3883523856969202,1.7490975979300583,-0.917932337093706,0.15797752436679355,0.36098535100054446,-1.1870490348881129,1.5928190385794836,-2.0717646062203268,0.8345650497099025,1.859261548723468,-1.4854041897158379,1.233065392835932,0.7394664129502405,-0.40647041794960587,1.0213864107426411,2.002822655041523,-1.3334132728146861,-0.20671384187577052,2.250832895761444,-1.2042485868285604,1.0349552027230213,0.8339782004780888,1.0288336782987557,-0.5096742066947033,0.3459801552713867,-0.7409722099774602,-1.5038842049822014,-1.2488347650369858,-0.5328233021025247,0.43372100973448013,1.5973428768401166,-1.8276263596378561,0.07237664251355398,-1.9103854546933428,2.9243876696798434,-1.1809189147649053,-1.1750888382417266,-0.7580348731404551,4.050570652994407,-0.8484837577462367,2.059263224850492,-1.2742257475711418,0.9601944031332947,-2.749582293279488,-0.27479187851322073,-2.1140103682396743,1.8015372162094567,-0.8558304022359539,1.0332695691142673,-0.4443928070244735,1.6166913305136998,-1.3034633696852151,-0.6750180221341877,0.7229436320896138,0.36030482129345287,0.5768746229917479,-1.9700254153387846,-0.703964034876572,-0.10719215723765954,-0.550651250503693,-0.9977283511575652,-0.9198245298926824,-0.7796154615668484,-1.1615690631267137,2.480489119472768,1.523882541392043,-0.913869573973508,-2.0392108861941938,-1.273986612947602,1.8252046153558574,1.1321740728415306,-2.5592829657373786,0.5193538436373477,-0.323130661208823,0.38568542746266854,2.2107979106478752,1.5221930011453997,0.11303455998485477,0.21892309872206978,0.7246920692929477,1.078773642877007,-2.256207288285679,1.5804919292050312,-0.13746690701228986,1.235850410154241,0.5184296246366877,-1.4384318345509917,0.3322495265766452,-1.2157170674898152,-0.8664558970726156,-0.2310084909150727,-1.2403572852163183,0.5708814761208534,-1.6081106771715217,-1.1293966114688316,0.44242287497721383,-1.9079409331312078,-0.47789868796195767,-0.9779855411744279,-1.1280713774784916,-0.31402781428573195,-1.8088879686461028,-0.4221749930756888,1.7508028095961619,-0.8182253777772487,-0.3685187828442746,0.39990153731377087,1.2816380563798966,1.4022452822363998,-0.781841350006532,2.4299904083086576,-1.3740420311451038,-0.6487604892299509,-0.9396431028920074,-2.1330184441930777,-0.7635631280376446,0.38419295976512785,0.05775105636977143,-1.8068676928595524,-0.8236757902331606,-0.8056007735990336,-2.39968970771735,-0.8522546516898816,0.8757433245619446,-2.1224776878050444,0.7804618024128137,1.2296052854026063,-1.4172764158197477,0.17421396671672038,1.248554021802503,-1.8552794235090297,-1.6707411855228187,-2.0771051177082045,0.01698564359203041,-0.012387363496278192,-0.7519833056691071,-0.7065507091454952,0.47981993755711766,0.16905305286402852,2.5936801834307173,2.3689675026094084,1.1533325163547214,3.742670387156274,-2.93842303512765,-0.01345944864765061,1.445006975338677,-0.24755437233901223,-1.498420568044524,2.750800228029403,-1.6373464847743877,1.188353937471091,0.3048375656358877,-2.1119896541543035,-0.03603211715507908,-2.019658094717991,-0.35159474390826234,-0.6553412089389463,1.5079583297515626,-0.3436019489641564,0.15489067582817181,-1.0019936078841152,-0.43873237158376377,4.004911426531656,-0.8405881190171917,0.9513600346613472,3.0041947757752374,1.5658187235915297,-0.4471459276678141,0.1026305780790471,0.3294033563029225,-1.5178266856810925,-1.3239015052528607,0.4017475607329261,-1.1838303638449268,0.5008432466689485,1.6457115033957879,3.119956907470165,-1.2889944393994581,1.729989750278064,0.17306044569264586,-1.4677296240266784,-1.3678336191016067,1.8615373588885284,0.7892748939453743,-0.7907346437313132,1.6018620499101153,-0.20722035521834514,2.8572606663456614,-1.0719214637385068,-2.3202485097980547,-1.6804580314795818,-0.542078563525977,1.5097201746528663,4.118911202952861,-2.025945293221285,-1.5602781971190776,-1.2007754694510164,-0.3602555435119874,2.3384806606389255,-0.008974177822002508,-0.40928639354750096,-0.3711017816212609,0.8683979328727931,-1.069761239980065,-0.31044971984727593,-2.028875592925451,0.5066437210625706,2.0859603125667388,0.30892188880019694,1.169400977092579,0.538005703145417,0.08303219362208057,-0.08004939493317678,-1.3754221165860663,0.966771005650512,0.5282772583376468,-0.5149277716407984,1.5490073286443342,4.032794003422141,-0.9636843065815136,-0.32786253252726444,0.0043753673396642314,3.0885266189646914,2.7327036969880707,1.2528139487150691,-0.6018462124681531,3.4492578584959714,-0.4953827016256672,1.0123058379440064,3.1603153884222297,-0.993862902436242,3.5775376697078034,-1.1931845342335763,1.529798464953969,-0.8725743304954331,0.7590772386126412,0.5531834616821728,-1.426600233654708,0.7013395322428452,1.2812277889727823,1.2010716615853194,-1.6052196462069483,0.07025305231061839,-0.9068310492413582,-1.2584951553735375,1.0426524237663415,-0.8383445295154948,-1.8753907397545582,-1.1005344271764514,-0.42399485939037324,0.6822671930166989,3.3822239056982535,-0.31387395808115653,-0.8784795426867311,-0.684960842031312,-0.6187092852087845,0.3336674219046477,0.1684974896349979,-0.10943851106054374,-1.0239172224076754,-0.6901385949448176,0.19626959536193603,-1.3905978041250933,-0.8157251441814183,1.2894417598595274,0.4941151509520899,2.0130564303671226,2.1940384185513553,0.4657145007757267,-1.7809396434417235,-1.7875569859925198,-0.7159349919019884,0.2561559330170107,0.841923110443766,1.772348978499201,-1.5619988317485771,0.7130738330282921,0.9824520279518624,3.7356252708891424,0.08986336970700924,0.5055699395547707,2.6722201232940104,0.536395503487028,-0.918152493863988,0.558488460612852,-0.6395029731129223,2.7715552504351293,0.26058997501061854,0.2069936816201332,-0.6550870677695128,2.625460918436417,-0.6506825312653136,0.6243206105860223,-1.2922310586663825,2.467574795394257,-0.9004358938264954,-0.8038260418627508,0.05466307540825573,1.0610308863644378,-0.7975211705548974,1.4103307932135938,-0.6703073293393931,-1.6625692235067147,-0.6704826341824404,-0.8216211004227455,1.1659074086171664,0.39838826036052044,-0.6391803076768184,1.6840775062764795,3.408955326568959,3.720503062035927,-0.3550812293685681,0.7710651714012798,1.732800337852303,-1.269146370568277,1.9614005338732223,-2.4676651266891305,0.2710122945299314,1.9814533957417708,-1.7005762753537517,-0.2437595952032768,0.5173811213306241,-1.7099253283959397,1.5848985164140843,-1.0520612071574973,1.837877276958988,-1.1833427337895936,-0.42854311572366166,-1.8546860926066675,-1.222908434988336,-0.6070785155746021,1.9655533801979221,-0.7300038334129791,-2.1191575068767237,0.48124050236682525,-1.241601654327459,-1.402418288301296,1.7277503650126682,-2.4414014768590366,0.865135530157315,0.4980597395457254,-2.2769890032203195,-1.6944072074574001,0.17624714435388153,-1.7305522011666568,0.5745650966366848,-0.5752327517122039,-0.02154709785960289,-0.895334798223222,3.4821447058505073,1.122028564828729,-1.6298399801319399,-0.6286994899128334,0.3007148697183708,-0.04004788070765842,-0.5783996051029067,0.19747890334909204,-0.8424011874156304,1.9131932380450904,0.9730159143296214,-0.5357442603721155,0.6730729538705564,-0.1107865214091032,0.39352804193986346,2.246816658310433,-1.0510481306876045,-1.1044972717940587,0.11483341844526025,-2.6530612738552857,0.31380544803000254,-0.5858716990878436,3.7513749611751046,0.9918408958057459,0.38796873823739486,-0.1988564725040987,1.1530771430055833,0.18121345563956454,-1.9981989249388659,1.3055918086023994,1.2300466831149037,2.3883999451881954,-1.0407075528231489,1.587058677757462,-0.26251583656915145,-1.0199055749025752,0.11681402784792101,-0.5404044086054803,-0.5790484002528453,0.5536421350558024,-1.6953978959674738,0.43907401342219193,0.3193436583853218,0.1363332204152951,0.6559551022230419,1.6812808029329176,-0.10754243173131464,-2.5100052631001675,0.6830553938147956,-1.8678212564633132,0.20576661283835093,-2.22003932832536,0.18352521849349918,-0.7362847258928833,-0.9661167762102234,1.060126490249329,-1.8016994802533888,0.05866378889901738,3.643479368348505,-1.3764314113825946,-1.059545020274901,-1.4006993980134739,-1.07763619978561,-2.496053212456884,0.07890167716982283,0.572244018918146,0.2253821232782492,-0.2847836609593698,-0.8109652118641609,2.195293747329487,3.3081949521303247,0.5603844981849772,-0.3020999867907151,-0.34820045455854043,-1.2995575391798,0.20151047249240686,-0.1084731620885824,-1.2637434307734952,-0.9202461586957763,1.8272852055637088,-1.823800886551976,-1.3627444420846868,2.926517364508175,0.2590903587305227,1.1327586501820766,2.4369244626748716,-2.060192128634255,-0.7883831736060568,0.4077463782979569,0.927515735167485,-0.9048262364114508,-1.672967771431389,-0.04473234572495616,-0.11994047105855496,-0.526119840502411,-0.8720499333303008,2.8380289960194407,0.06366019014233111,1.3156111025026382,1.621229370401716,0.3145442311581666,2.3577881575605084,-0.24467667132645843,-1.6954120849433205,-1.430002213646192,1.9605361479498573,2.0608980888561397,1.118673961601619,1.183349056221186,-1.8465237236523262,0.40308486013394657,-1.0097547897196515,-1.4257688779914977,0.6226168604504017,1.4190325814448221,-0.0064225628355679965,-0.3770348255773455,-1.4146457044189553,1.1138995111320331,1.5578559783988755,-0.6616420298309542,-1.4856807705907695,1.2892457046298194,0.20425034917415616,0.1685926004549177,-0.6332605872031069,3.7372591447912678,1.5063869755030113,3.1043553551774408,0.8697787515264346,0.2840597904637061,-0.8613012916854142,-2.4765035336115866,1.7884558371552401,1.433281012724795,-0.20451112343500052,-1.3872279072660878,-1.1942027281555543,-0.5583346528448437,0.6023106523145297,1.3798113768598377,-0.08200462809298663,-0.19332615911288253,1.679398285785482,-0.5681649728070558,0.5674938286851448,-1.4254156956926376,1.0370648811885672,-0.4040705527399149,-0.45987994108098956,1.1747542197011143,-1.5955674393946162,1.272804065825972,0.501116617593307,2.436669875610867,0.25250622222813607,1.5168018275031132,-1.7347224718316645,0.7248497964092006,0.11566597439708737,0.3061356297534514,1.0504119087726247,0.4980759121047658,0.1937382903670571,-0.2854579215983552,0.5203284113534998,-0.7279357689470961,-0.19755996733504408,-1.3749324160487628,-0.32271572473817145,0.5960595922281557,2.1949872353731297,-1.6549850538301425,4.166763561667277,0.9522882356421566,-0.5155400642932604,-0.7132626106595745,-1.9132896658014509,0.7492224173436758,1.244598415445586,-0.0348117086286696,-2.516247269555198,1.5927371951812324,-0.2849109767440522,-1.6259934414989523,-0.2441982753100004,0.9387023955320505,-0.6075359960740221,2.0332629399171545,-1.9425979326878022,0.6733624725871235,-0.134762746278109,0.23416137132492118,-0.7688666509439331,-1.1710174069859849,-0.6978060767178783,0.9641833963258548,-0.4656592310439213,0.27771379815334335,0.849786187457479,1.8697047051419415,-1.6433226964540182,-0.8594948412121,-0.3750690991143311,0.9305918458435458,0.6294555797890112,-0.09696875686508986,-0.5256320011302792,-1.1596469684725066,0.5455139035263978,-0.40990975774859884,3.869509895794867,-1.1200310751225053,2.2680072237766846,0.3874508844504676,-0.9401967063229326,-2.2751975339812773,-1.6208462850064496,-1.1464632214168626,-2.3018450965998074,-0.5843029480056041,-0.45812009575631873,-0.8041258315310562,0.6502835472062967,-0.8114317918039998,1.8071840024162857,2.6647995153374002,-0.6556721673389273,-1.3591215690618528,1.174297111099785,-0.00810844998207271,-2.224356078834463,-0.9497699756715721,-1.834286270351933,-0.18741902181016026,-0.04050061796197374,-0.05923202630229958,-0.6207244800085551,1.5759390641597923,0.43431668100968335,-1.4002719177601455,-0.9962337204582795,1.1429640040160391,-1.7101687418718292,-0.09681385107537242,0.9440023079506006,0.032086785029448554,-1.0862475459678123,-2.1534707065757,-0.7788698458842166,3.3232158301401062,-2.054398293292932,0.5030824159891729,-1.7729661666570784,1.2288840059222828,-1.5744541642891359,-1.9239682713358426,3.144546124842096,0.3487532810113374,-1.2756884861144193,1.0664812801259813,-0.046564662152403444,-2.036358614246393,-0.8119899556044755,-0.7649787498384872,-1.6347616715134075,-0.46076321897502887,1.2964856865252974,-0.026739826677793663,-2.0001726041396664,-0.5744143879031374,1.6331049649565201,-0.49113443741382556,-0.6278695802811751,-0.46163065973183276,-1.520809180253835,-1.8908076083032699,0.0075165827049894106,-0.3050651943690896,-1.8612479783738747,1.7902882125244541,1.2575151427404045,-0.6516087408014803,1.4377768341559525,-2.4856438125779206,-1.7776800489937825,-0.23926220279579416,1.8236372325931838,-2.6176237716381476,-0.5094542706954146,2.00351219657358,-1.0700001838571263,2.8846364617321383,-0.14364998760699302,-1.467317703733547,-0.3310444794862185,1.608766729592621,-2.1155461896490464,0.047726293174024305,1.8236531825541549,-0.006609407456591002,-1.9052388516287189,0.11168285310349554,0.7684258248724728,1.049409053519063,1.091903301458636,-0.13840250130684706,1.1406587422761603,0.8549752967671264,-1.9614006345062769,-2.087047925260399,0.28882368037486056,0.21683120251537946,-2.652164783451051,-0.00391253928386621,0.8682769662146939,0.20396218274808595,1.3334819714754926,-0.2955938028628036,-0.47229569113114855,-0.5959793116531967,-1.0448407650299458,-0.4722801115921375,2.753235031731119,-1.8546505872680747,-1.5113199219755453,-1.7897105685287635,-1.0356052592257492,-0.3907301987800819,-1.5966125760760477,2.535302318365728,-1.9742097604984832,-0.47784004803131197,-1.2581980795699597,-1.032670706715859,-0.7866124113412953,1.7427336912097604,-0.6596204844828425,-0.8293084040452583,0.9629985103053692,1.6129928027961313,0.7038595487859564,0.19633503556260426,-1.2449156917457322,2.3582791885576713,-0.0726736918598642,1.526632843764297,1.6523144957100748,0.4246673035537167,-1.435382675215207,1.3586780061492996,-2.339692942812,-1.1563242952580288,-0.41360895929898056,-0.043008937622274535,2.220074703423368,-1.4204252245911029,0.7586251459191852,1.85059853483777,0.25602544691879,-0.7761106488196179,-1.7502562960867962,-1.354251174123534,-0.6498887470319634,0.7110193211499215,-0.374779390105947,-2.3376287765448867,0.011297165814065403,-1.8174666379806983,-1.1094286967235312,0.8596230756760636,-1.189108727540073,0.5163928559916526,0.6756558056748487,2.3492863512894595,-1.5202832442405174,0.6709788044783898,-0.9613560372850454,2.6445852723101186,0.2890912290739906,1.0074859148353306,1.2950373157765507,-1.1602197345683476,-1.9334229194282122,0.22694998989420614,-0.07782652217359404,-1.7367007804692598,-0.8236036411201083,-2.0434376530627314,0.65840167679232,-0.018460285092787555,-0.6580750111648342,0.4011655134859429,-1.4465143816369475,1.479584426008438,3.1099500616383398,-0.7685500876342084,-0.1524940035583723,-1.6760328059425473,0.3920540027688779,0.3737423119176599,-0.9497458813427591,2.4563314952220745,0.5324485549138658,1.0636936563856136,1.374467174131389,0.5348533402210705,1.3016888372997362,-0.9217855487715941,0.6569347741134037,1.1178959060888713,0.8886737086175391,0.026558121867172057,-0.4175004782342412,-1.1157225621037221,-0.2650853539112495,0.17863448517331967,0.4112865169610508,1.1882346925023706,4.052830875860441,-0.435556326310171,-1.2720585122870953,-1.6842258299697603,2.6404464047319984,1.8015674143333877,-1.267570998720302,-1.0921691737895243,-0.38921942081565725,-1.3515910883028688,0.15387230379006642,-0.410750691568786,-0.1703762672846642,0.6531102807237209,-2.691283371827179,-1.219028065586017,1.8815558427530068,-1.4192916353894691,3.3259738088400073,0.9526986248444768,1.1766419865918172,-1.5988139399209333,-0.40423273617163064,-0.08513716306990089,0.7366663445077692,1.349621218121089,-1.7224452755017718,-0.27642176233381616,0.15350199995039576,3.900744670155687,-1.483092577682523,-1.1049230073531984,-1.0136780404911725,-0.054173119790491484,0.704207707137218,3.4066344865832443,-0.3136574292505953,-2.5308702451157736,1.7214328706452258,-0.11595869817198422,-0.2240140377141611,3.30131189165069,-0.9203093849502809,-2.180612073623786,-1.6685859455712542,-0.6078700524938159,4.218105255465005,-1.3310084174132677,-0.9466686212919617,-1.934884949380361,0.5726560598191621,-4.007716053359627,1.7816432693093218,0.09357565117865121,-0.689612380706898,-1.052859251127033,-1.1136648359584884,3.909135236081282,-1.2830846127071154,-1.3281065063005406,-1.5526009025518175,-1.6625108576955174,-2.1714751939709123,-0.0878396033295711,0.6031669908862948,0.5825568109802557,-0.9057748444211197,0.06760343363668118,-2.9406704621070223,1.1505351073020562,-0.11239379506493352,0.46383208351836736,-0.6380773780651454,-0.10743167365951711,0.13670881704386925,-0.5852757633799773,-0.5247108277428147,2.2905780942461123,2.6275150635427797,1.5079520035017189,-0.14308348018202818,-0.3432329031671948,1.2063680896890212,0.06838462920676826,1.9804136507372463,-0.21135747783075975,-0.7333695401676606,-1.6349689469112267,-2.333823850450385,0.32407383566354664,-1.2969454689356381,2.2167245660609525,-0.4215838544780941,-0.46671770540625884,0.44791755704856223,0.2938194091307055,2.520818562247045,0.7206309786142536,1.8151240327198042,-0.5160599234247938,-0.5443375327839285,-0.22355083201754913,-0.9875172380881941,-2.0564034493173353,-1.1571957647447009,0.8143700654006586,0.7615370743219726,-0.2565975283672861,-0.7273875072764739,1.8795673831658104,0.9605539703800428,-1.820910056426979,0.0018415138273277,-1.6672819688706586,4.208443131184384,-2.15191883428905,-1.2648803594460225,-1.2593467181234255,0.7044797230245041,0.03027774993194509,0.6288266205554721,-0.43421274320317277,-0.30989766044197975,2.7898121950167543,0.6549235473528279,0.9874406979695216,0.38886561896363536,1.5896060882312153,0.1053866133789633,-0.9716101614225372,0.6170460649059946,-2.0795629059919,-3.161860585158591,-1.8315596792267355,-2.6463065007293403,-2.414860430143243,1.615954844740402,-0.665350984421508,0.9119245171056407,1.742389041900641,-2.320802179469949,-0.864934720672281,0.5284294853512941,-0.18173584403855417,0.12199566653163398,-2.122866122143247,1.4251713574252993,-0.300518518322929,-0.7319036216757809,0.7051125078510545,1.3400042389427502,1.2778829223805936,-1.7676463580960802,0.46478686759261617,1.530083923500515,0.27391264160128076,-1.9044431334530243,1.553758373914473,-2.200125273457746,1.3620485746978865,-0.02404760942881485,1.362894783925666,0.2815967150481585,-1.0797566644957899,-1.7018258057073277,0.0727328109278516,2.0963348860211872,0.8278106782222618,1.0206300505323174,-1.9886516664443634,-1.1092330734832134,1.380830084951763,1.0321560551934947,1.8526079644922935,-2.2722428681375644,-1.0290602799375033,-1.6532482913386493,1.8762093640403696,-0.6661526255453941,-1.3033777372359767,-0.08426686939157332,-1.3893317896228543,0.3472424233671549,-1.6631211177052139,0.7204487396697165,0.974291127504689,0.5508866116619314,-0.5660975132641123,0.959399417546426,0.714718809079762,0.7148551192065963,-1.404120953855304,-1.5366371993540222,-0.12821373808156525,4.220740606505915,1.3576789259350328,-0.25327256155659716,-1.7069799877262193,1.4729250922161807,-0.35741656615495165,-0.4250141652917531,-0.4322464311329408,2.594454243070017,1.8550211859260268,-1.268265566451295,1.0321890765747295,-1.250705622777128,-0.7672048188166032,-0.6168338272557782,-0.9317711099909479,-1.082451056381913,-2.7536367859308637,-0.6598674946506138,-0.8167622044852059,3.3332331606393066,-1.5317889985454667,-2.4502512555835243,-0.7735208797473774,-0.1980878387461118,-0.28557316369184105,1.2071191891082593,0.812177149139796,-1.6206328021268688,-2.058952060203036,-0.18538375866905496,-1.306852813046276,-1.3852876205178823,-0.45200635975880105,-0.5136586537578458,-1.3087370762188504,0.9246102342758213],"yaxis":"y","type":"scattergl"},{"hovertemplate":"color=1\u003cbr\u003ePC 1 (15.99%)=%{x}\u003cbr\u003ePC 2 (10.31%)=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"1","marker":{"color":"#ffa07a","symbol":"circle"},"mode":"markers","name":"1","showlegend":true,"x":[0.5188737804926038,-1.0175350000859902,-2.6663748243051653,1.3495560153604653,-0.9801677270900144,3.9420555808668976,3.8461703129361275,-0.5237490923818527,-0.140678414681899,0.8871458053376754,-3.1396574213115698,-0.06424249616097287,-3.7960890343320868,0.47103466963337054,1.1870310629817906,-3.1938593255391807,-1.7782302357398936,-3.3791515659751656,2.346189686482158,0.0435444811283548,4.622485126808148,-2.507872082346157,2.208221407323026,1.122711238766611,-1.2322921757812704,-0.4696486882842998,-1.6846504928380706,-0.9634993688336224,-1.6216422135609447,-0.6825884017890572,0.9928583553941283,-3.12692248696506,-0.8431850540867937,3.0940393023977455,-1.0785209118063022,-2.741730382568599,-1.7148867160509136,3.7125581059448907,1.0644374439332471,-2.882313088268255,-0.16458910775441613,0.8193572903624571,4.22420626354579,-2.2567454689549673,0.23412836703813159,1.100831007702513,1.9794034161453504,-1.359896452305678,-1.5538463956072373,-2.477990363972889,-2.810410324631363,3.5013014347938975,-2.083079594334767,0.5048372918630918,3.585652783503929,0.7346980023476446,-0.6069169172691475,-0.26096349756054454,-0.012209437541721189,3.5170658143468887,-0.893239807298944,0.9507920897545722,2.8977802969445756,-1.7297618445271117,-3.072640534659597,-1.447147580287102,0.12782822286221376,-0.28034826850205885,3.127945351269917,-0.29902618792396074,-2.4904149029794005,-2.2031136291671554,-1.1399655862734652,-1.3601003030631469,-0.29808031757553943],"xaxis":"x","y":[0.37395848809341786,-0.0026426234166208947,0.9302463119849035,-2.1776478109067705,1.8451694003859185,-0.014618492294396906,-1.4761924016182268,-1.5536311431314094,-0.681898999153915,-1.9637696591967881,-0.32743148606610734,0.8701793854671727,0.1680867591366157,-1.7928906908349864,-0.6538296961874926,-0.6599742121523546,2.0082112717009264,-0.4015667523295047,-0.7415205379952704,2.027263545687416,-0.404065655350173,2.440568841642037,-0.6966019359462626,4.271309404154579,1.5388247826265609,2.4332596182511086,0.5212760315666468,-0.23999887854565366,0.14675644127112786,0.12497384482480503,2.3774633787285877,-1.2135501716765966,-1.2025622047782871,-2.208403924844172,-0.7058012038842678,-0.9304038889359121,1.8171262343680306,-0.4066771496325104,1.4891704956048246,-1.3381048764162047,-0.7472816516182051,1.3920447272524012,1.2162536062378075,-1.1580147726727401,0.547054302345445,-0.15622063052630236,-2.0540923213286577,-1.337493615112557,2.116640674034933,-2.084937663408544,-1.7735210807794712,-0.710755190089819,-0.8487972483138976,0.18556635326462556,-1.8837223308418787,-2.6914571188182306,-0.6650859608868726,-0.7739532569727732,1.7034240129815146,0.9551709399197453,-2.1397671288017754,-2.641589488189361,-1.5890870550626819,-1.9191818447581024,-1.9209758147952771,-0.5644998897798972,1.2533331192814046,-0.7286974475983438,0.33657363595207945,-0.8930236216460141,-1.5709738151094983,1.2546150694004918,-1.728459705345366,-1.2669867382201725,0.2863844814270785],"yaxis":"y","type":"scattergl"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"PC 1 (15.99%)"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"PC 2 (10.31%)"}},"legend":{"title":{"text":"color"},"tracegroupgap":0},"title":{"text":"LOF Results"},"width":800,"height":600},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d854eca2-25d4-465d-9b50-ae35e58d1be6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Untuk melihat index data yang terdeteksi anomali, kita bisa menggunakan cara berikut ini.


```python
anomaly_indices = np.where(lof_label_tune == 1)[0]
anomaly_indices
```




    array([  405,   536,   587,   826,  1033,  1412,  1447,  1785,  2367,
            2379,  2829,  2857,  2873,  3223,  3374,  3719,  3925,  3949,
            4405,  4666,  4980,  5030,  5185,  5222,  5322,  5513,  6059,
            6377,  6720,  6864,  7267,  7291,  7502,  7752,  7828,  8031,
            8088,  8183,  8327,  8506,  8902,  9090,  9161,  9198,  9332,
            9425,  9475,  9508,  9609,  9984, 10171, 10201, 10241, 10373,
           10535, 10787, 10933, 11305, 11497, 11938, 12196, 12218, 12502,
           12639, 12741, 13209, 13264, 13587, 13884, 14113, 14150, 14209,
           14393, 14438, 14898], dtype=int64)



Kita juga dapat mengambil data yang sifatnya anomali ini menggunakan index yang sudah ditemukan di atas. Dari proses ini kita dapat mentransformasi kembali data kita ke bentuk semula. 

Ingat bahwa kita sebelumnya membuat dua buah pca yaitu pca yang menyimpan seluruh informasi dan pca yang mengambil 90% informasi. Maka kita gunakan pca yang menyimpan seluruh informasi ini setelah itu kita kembalikan ke bentuk sebelum di scaling.


```python
anomaly = fraud_pca.iloc[anomaly_indices]

temp = pd.DataFrame(pca2.inverse_transform(anomaly))

anomaly_df = pd.DataFrame(scaler.inverse_transform(temp), 
                          columns=fraud_scaled.columns)

anomaly_df
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>income</th>
      <th>name_email_similarity</th>
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>velocity_4w</th>
      <th>bank_branch_count_8w</th>
      <th>...</th>
      <th>email_is_free</th>
      <th>phone_home_valid</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>session_length_in_minutes</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.227118</td>
      <td>0.496827</td>
      <td>5.207973</td>
      <td>39.428203</td>
      <td>-0.595193</td>
      <td>1355.366606</td>
      <td>9738.815777</td>
      <td>5409.589451</td>
      <td>4532.630855</td>
      <td>-427.834634</td>
      <td>...</td>
      <td>0.769808</td>
      <td>0.286610</td>
      <td>0.708494</td>
      <td>0.096594</td>
      <td>1233.055235</td>
      <td>0.004499</td>
      <td>16.930254</td>
      <td>0.959731</td>
      <td>0.919350</td>
      <td>3.971300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.567352</td>
      <td>0.389553</td>
      <td>318.545739</td>
      <td>58.054107</td>
      <td>9.087221</td>
      <td>713.405175</td>
      <td>-3051.520087</td>
      <td>1790.808796</td>
      <td>4080.489742</td>
      <td>290.771032</td>
      <td>...</td>
      <td>0.997683</td>
      <td>1.122376</td>
      <td>1.002801</td>
      <td>0.317107</td>
      <td>472.335355</td>
      <td>-0.000517</td>
      <td>-5.373364</td>
      <td>0.452061</td>
      <td>1.299960</td>
      <td>5.299492</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.204856</td>
      <td>0.381540</td>
      <td>44.546778</td>
      <td>31.717762</td>
      <td>12.733772</td>
      <td>2104.858405</td>
      <td>7525.384632</td>
      <td>4939.393571</td>
      <td>4647.240329</td>
      <td>363.218876</td>
      <td>...</td>
      <td>0.779109</td>
      <td>0.262245</td>
      <td>0.676326</td>
      <td>0.273754</td>
      <td>1343.382089</td>
      <td>0.028783</td>
      <td>14.859937</td>
      <td>0.506785</td>
      <td>0.787268</td>
      <td>3.825052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.842459</td>
      <td>0.556384</td>
      <td>-4.611537</td>
      <td>42.025048</td>
      <td>-5.133614</td>
      <td>1841.598725</td>
      <td>8943.189559</td>
      <td>6585.863030</td>
      <td>6137.794412</td>
      <td>-202.542230</td>
      <td>...</td>
      <td>0.557888</td>
      <td>0.061073</td>
      <td>0.933595</td>
      <td>-0.601063</td>
      <td>741.572467</td>
      <td>0.147760</td>
      <td>13.753201</td>
      <td>1.537824</td>
      <td>1.163752</td>
      <td>0.195591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.829767</td>
      <td>0.534507</td>
      <td>-8.808409</td>
      <td>39.105415</td>
      <td>-0.306587</td>
      <td>1117.997200</td>
      <td>2622.806566</td>
      <td>3424.057574</td>
      <td>3975.124879</td>
      <td>728.646151</td>
      <td>...</td>
      <td>0.973009</td>
      <td>1.225065</td>
      <td>0.132678</td>
      <td>0.555465</td>
      <td>761.448547</td>
      <td>-0.016640</td>
      <td>9.729460</td>
      <td>-0.306144</td>
      <td>0.653444</td>
      <td>5.315429</td>
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
      <th>70</th>
      <td>0.635328</td>
      <td>0.633775</td>
      <td>-29.095527</td>
      <td>32.810781</td>
      <td>2.269996</td>
      <td>3043.896905</td>
      <td>7195.093457</td>
      <td>3206.089170</td>
      <td>2746.709324</td>
      <td>-480.366615</td>
      <td>...</td>
      <td>0.468304</td>
      <td>0.671451</td>
      <td>0.268859</td>
      <td>-0.471170</td>
      <td>449.157502</td>
      <td>-0.000090</td>
      <td>4.690149</td>
      <td>1.792444</td>
      <td>1.276307</td>
      <td>8.425203</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.285200</td>
      <td>0.413444</td>
      <td>-12.380822</td>
      <td>40.090587</td>
      <td>0.944023</td>
      <td>1847.482068</td>
      <td>7699.610647</td>
      <td>6895.225435</td>
      <td>6637.646140</td>
      <td>104.947407</td>
      <td>...</td>
      <td>1.577818</td>
      <td>0.076020</td>
      <td>1.293933</td>
      <td>0.341120</td>
      <td>-99.359943</td>
      <td>0.106956</td>
      <td>-2.916104</td>
      <td>0.206275</td>
      <td>0.662265</td>
      <td>-1.119125</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.779166</td>
      <td>0.522992</td>
      <td>-87.332137</td>
      <td>23.503952</td>
      <td>11.581668</td>
      <td>904.927500</td>
      <td>4080.786202</td>
      <td>3435.538935</td>
      <td>3374.914866</td>
      <td>687.769748</td>
      <td>...</td>
      <td>-0.531899</td>
      <td>0.134679</td>
      <td>0.780280</td>
      <td>0.252662</td>
      <td>426.643105</td>
      <td>0.378293</td>
      <td>7.564526</td>
      <td>0.234643</td>
      <td>1.578899</td>
      <td>6.939092</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.182698</td>
      <td>0.567770</td>
      <td>267.628313</td>
      <td>25.144437</td>
      <td>4.212987</td>
      <td>479.193597</td>
      <td>16201.343393</td>
      <td>6869.295127</td>
      <td>4059.754066</td>
      <td>345.825120</td>
      <td>...</td>
      <td>0.458724</td>
      <td>0.570946</td>
      <td>0.522043</td>
      <td>0.825184</td>
      <td>770.750326</td>
      <td>-0.292251</td>
      <td>6.648122</td>
      <td>-0.015046</td>
      <td>0.918855</td>
      <td>4.878472</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1.404294</td>
      <td>0.324542</td>
      <td>-85.030334</td>
      <td>46.039431</td>
      <td>20.012943</td>
      <td>326.208995</td>
      <td>5002.865516</td>
      <td>4963.315858</td>
      <td>5522.380653</td>
      <td>182.337278</td>
      <td>...</td>
      <td>1.141355</td>
      <td>0.525966</td>
      <td>0.863478</td>
      <td>-0.071098</td>
      <td>888.615907</td>
      <td>0.431545</td>
      <td>22.633691</td>
      <td>1.015346</td>
      <td>1.626410</td>
      <td>1.779975</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 22 columns</p>
</div>




```python
fraud.iloc[anomaly_indices]
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
      <th>current_address_months_count</th>
      <th>customer_age</th>
      <th>days_since_request</th>
      <th>intended_balcon_amount</th>
      <th>payment_type</th>
      <th>zip_count_4w</th>
      <th>velocity_6h</th>
      <th>velocity_24h</th>
      <th>...</th>
      <th>phone_mobile_valid</th>
      <th>has_other_cards</th>
      <th>proposed_credit_limit</th>
      <th>foreign_request</th>
      <th>source</th>
      <th>session_length_in_minutes</th>
      <th>device_os</th>
      <th>keep_alive_session</th>
      <th>device_distinct_emails_8w</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>405</th>
      <td>0.7</td>
      <td>0.742771</td>
      <td>16.0</td>
      <td>40</td>
      <td>12.952886</td>
      <td>20.941790</td>
      <td>AA</td>
      <td>168</td>
      <td>9011.020696</td>
      <td>5194.170896</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>1.336989</td>
      <td>x11</td>
      <td>1</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>536</th>
      <td>0.1</td>
      <td>0.014002</td>
      <td>4.0</td>
      <td>30</td>
      <td>0.015723</td>
      <td>-0.761342</td>
      <td>AB</td>
      <td>491</td>
      <td>1078.512781</td>
      <td>2425.681429</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1500.0</td>
      <td>1</td>
      <td>INTERNET</td>
      <td>7.259926</td>
      <td>windows</td>
      <td>0</td>
      <td>2.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>587</th>
      <td>0.9</td>
      <td>0.202449</td>
      <td>189.0</td>
      <td>30</td>
      <td>12.543677</td>
      <td>7.465069</td>
      <td>AA</td>
      <td>701</td>
      <td>9885.425316</td>
      <td>3574.190293</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>7.902304</td>
      <td>other</td>
      <td>1</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>826</th>
      <td>0.1</td>
      <td>0.896931</td>
      <td>31.0</td>
      <td>50</td>
      <td>11.577133</td>
      <td>-1.395665</td>
      <td>AA</td>
      <td>1661</td>
      <td>10580.653626</td>
      <td>6697.931015</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>5.373218</td>
      <td>linux</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>0.9</td>
      <td>0.841805</td>
      <td>187.0</td>
      <td>10</td>
      <td>0.023406</td>
      <td>51.886755</td>
      <td>AA</td>
      <td>992</td>
      <td>1270.459414</td>
      <td>2789.514959</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>15.635054</td>
      <td>other</td>
      <td>1</td>
      <td>1.0</td>
      <td>7</td>
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
      <th>14150</th>
      <td>0.8</td>
      <td>0.487961</td>
      <td>25.0</td>
      <td>20</td>
      <td>0.013398</td>
      <td>-0.979439</td>
      <td>AC</td>
      <td>714</td>
      <td>3138.336067</td>
      <td>3612.627112</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>5.787567</td>
      <td>windows</td>
      <td>1</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14209</th>
      <td>0.6</td>
      <td>0.221974</td>
      <td>7.0</td>
      <td>20</td>
      <td>0.005390</td>
      <td>-1.696078</td>
      <td>AC</td>
      <td>829</td>
      <td>8297.226759</td>
      <td>6033.152656</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>200.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>3.335838</td>
      <td>other</td>
      <td>0</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14393</th>
      <td>0.9</td>
      <td>0.051105</td>
      <td>173.0</td>
      <td>50</td>
      <td>0.014303</td>
      <td>-0.663256</td>
      <td>AD</td>
      <td>932</td>
      <td>4355.509437</td>
      <td>5518.657925</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1500.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>1.055325</td>
      <td>windows</td>
      <td>1</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14438</th>
      <td>0.5</td>
      <td>0.035640</td>
      <td>19.0</td>
      <td>30</td>
      <td>0.001540</td>
      <td>8.864131</td>
      <td>AA</td>
      <td>2012</td>
      <td>5808.134611</td>
      <td>5909.788325</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>500.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>4.375013</td>
      <td>other</td>
      <td>0</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14898</th>
      <td>0.4</td>
      <td>0.709280</td>
      <td>186.0</td>
      <td>50</td>
      <td>0.043043</td>
      <td>15.297551</td>
      <td>AA</td>
      <td>964</td>
      <td>7648.922582</td>
      <td>3132.050823</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1500.0</td>
      <td>0</td>
      <td>INTERNET</td>
      <td>2.084672</td>
      <td>linux</td>
      <td>0</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 28 columns</p>
</div>




```python
from joblib import dump

dump(pca2, "Pca Uhuy")
```




    ['Pca Uhuy.exe']




```python
from joblib import load

halo = load("Pca Uhuy")
```


```python
halo
```




<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=0.9, svd_solver=&#x27;full&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=0.9, svd_solver=&#x27;full&#x27;)</pre></div></div></div></div></div>


