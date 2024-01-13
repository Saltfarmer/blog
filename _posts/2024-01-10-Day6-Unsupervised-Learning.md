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
