---
title: "Day 4 Algorit.ma : Regression Model"
header :
  image : /assets/images/AlgoritmaBanner.jpg
  teaser: /assets/images/Algoritma.png
comments : true
share : true
categories:
  - Data Science
tags:
  - Python
  - Algoritma
  - Pandas
  - Statistics
  - Regression

---

Day 4, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Inclass: Regression Model**
- Durasi: 7 hours
- _Last Updated_: Desember 2023

___

- Disusun dan dikurasi oleh tim produk dan instruktur [Algoritma Data Science School](https://algorit.ma).


```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from helper import linearity_test
```

# Introduction

Machine learning bertujuan untuk membuat mesin yang belajar berdasarkan data. Machine learning terbagi dua:

<img src="https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/assets/supervised_unsupervised.png" width="900">


**Supervised Learning**: 

* memiliki target variable. 
* untuk pembuatan model prediksi (y ~ x)
* ada ground truth (label aktual) sehingga ada evaluasi model

**Unsupervised Learning**: 

* tidak memiliki target variable. 
* untuk mencari pola dalam data sehingga menghasilkan informasi yang berguna/dapat diolah lebih lanjut. umumnya dipakai untuk tahap explanatory data analysis (EDA)/data pre-processing.
* tidak ada ground truth sehingga tidak ada evaluasi model 

# Regression Model

Regression model merupakan **Supervised Learning** karena data yang dibutuhkan harus memiliki target variabel (y). Target variabel dari Regression Model harus bertipe numerik, namun untuk prediktornya (x) boleh numerik/kategorik

📝**Business Problem**

Pemilihan variabel target biasanya dikaitkan dengan masalah bisnis yang ingin diselesaikan:

1. Sebuah dealer mobil berusaha membangun sebuah model untuk memprediksi harga mobil untuk digunakan sebagai patokan ketika membuka harga transaksi. Untuk itu mereka mengembangkan sebuah model dengan:

     * Variabel target: 
     * Variabel prediktor: 

2. Seorang mahasiswa pertanian diminta untuk melakukan analisis regresi untuk memprediksi produktivitas padi dari berbagai lahan di Pulau Jawa. Untuk itu ia mengembangkan sebuah model dengan:

     * Variabel target:
     * Variabel prediktor: 

# **Regression Modeling Workflow** - Predicting Property Sales Price: in Jakarta, Tangsel, and Depok Area

Sebagai Tim Data di sebuah institusi perbankan, kita diminta untuk melakukan analisis untuk mengetahui prediksi harga properti untuk acuan dasar data credit KPR. 

Keinginan untuk memiliki properti sendiri merupakan impian banyak orang. Selain bisa dijadikan tempat tinggal, memiliki properti di Jakarta dan Depok adalah salah satu aset investasi yang menguntungkan karena harganya yang cenderung naik setiap tahunnya.  

Dalam proses pencarian tempat tinggal idaman ini, beberapa orang mungkin saja mengalami hambatan, yaitu kesulitan dalam mencari tempat tinggal yang sesuai dengan spesifikasi yang diinginkan dan budget yang dimiliki. Banyak orang menemukan tempat tinggal dengan harga yang cukup mahal namun tidak sesuai dengan spesifikasi yang ditawarkan.

***Apakah terdapat sistem yang dapat memberikan referensi harga properti?”*** menjadi tujuan analisis pada pembahasan analisis kita.

## Simple Linear Regression

Pada Simple Linear Regression, kita akan membuat model regresi dengan **satu buah variabel prediktor**. Formula untuk simple linear regression adalah

Formula model simple linear regression:

$$
\hat{y}=\beta_0+\beta_1.x_1
$$

dimana:
- $\hat{y}$ : nilai prediksi target variabel
- $\beta_0$ : nilai intercept (nilai target variabel ketika kita tidak memiliki prediktor sama sekali)
- $\beta_1$ : nilai slope (nilai kemiringan garis regresi / nilai kontribusi prediktor dalam menentukan target variabel)

Bagaimanakah kita mendapatkan garis yang paling representatif terhadap data kita?

> Regresi bekerja berdasarkan konsep **Ordinary Least Square** yang mencari persamaan garis linear dengan nilai **error terkecil**. Error adalah selisih nilai prediksi/nilai pada garis dengan nilai aktual.

<!-- Note: Jika Anda membuka docstring (dokumentasi) untuk fungsi OLS() pada statsmodel, ada keterangan parameter `endog` dan `exog` yang dapat diakses pengertiannya secara lebih lanjut pada [dokumentasi berikut](https://www.statsmodels.org/stable/endog_exog.html) -->

### 1. Import data

Kita akan gunakan data `properti.csv` yang tersimpan di folder data_input. Data yang diambil adalah data hasil scrapping dari website https://www.rumah123.com/


```python
# code ini untuk mengatur pemisah angka ribuan
pd.options.display.float_format = '{:,.3f}'.format
```


```python
# Data
properti = pd.read_csv("data_input/properti_jual.csv")
```


```python
properti.head()
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>294</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Utara</td>
      <td>3500000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>78</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Selatan</td>
      <td>2500000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Timur</td>
      <td>265000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>120</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
      <td>2600000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>130</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Depok</td>
      <td>1300000000</td>
    </tr>
  </tbody>
</table>
</div>



Berikut deskripsi variabel dari data tersebut:
- `K..Mandi`: Jumlah kamar mandi pada suatu properti

- `K..Tidur`: Jumlah kamar tidur pada suatu properti

- `L..Bangunan`: Luas bangunan properti (m2)

- `Sertifikat`: Jenis sertifikat atas properti yang di jual
    - `Sertifikat Hak Pakai`
    - `Sertifikat Hak Sewa`
    - `Sertifikat lainnya (PPJB, Girik, Adat, dll)`
    - `Sertifikat PPJB`
    - `Sertifikat Hak Milik`
    
- `Tipe.Properti`: Jenis properti yang dijual
    - `Rumah`: Untuk tipe properti rumah
    - `Apartemen`: Untuk tipe properti apartemen
    
- `Kota`: Lokasi kota tempat properti di jual
    - `Depok`
    - `Jakarta Selatan`
    - `Jakarta Timur`
    - `Jakarta Utara`
    - `Jakarta Barat`
    - `Jakarta Pusat`
    - `Tangerang Selatan`
- `Price`: Nominal harga properti yang dijual

### 2. Inspect data berdasarkan tipe datanya


```python
## code here
properti.dtypes
```




    K..Mandi          int64
    K..Tidur          int64
    L..Bangunan       int64
    Sertifikat       object
    Tipe.Properti    object
    Kota             object
    Price             int64
    dtype: object



Tipe data yang belum sesuai adalah : 
- 'Sertifikat', 'Tipe.Properti', 'Kota'


```python
# Object nama kolom kategori
cat_var = ['Sertifikat', 'Tipe.Properti', 'Kota']

# Mengubah tipe data dari beberapa kolom
properti[cat_var] = properti[cat_var].astype('category')
```


```python
# Cek kembali hasil proses perubahan tipe data
properti.dtypes
```




    K..Mandi            int64
    K..Tidur            int64
    L..Bangunan         int64
    Sertifikat       category
    Tipe.Properti    category
    Kota             category
    Price               int64
    dtype: object



### 3. Cek missing value

Sangat penting untuk mengidentifikasi apakah terdapat missing value di dataset kita sebelum dilakukan pemodelan machine learning. 
> ❓ Karena missing value dapat mempengaruhi performa model secara signifikan

#### Missing value ada disebabkan karena:
1. Kesalahan koleksi data
2. Permasalahan pada saat preprocessing
3. Sesimple karena data tidak terkumpul oleh sebagian observasi

Metode yang dapat dilakukan untuk mengetahui apakah pada data yang diolah memiliki nilai *missing* dengan menggunakan fungsi `isnull().sum()`.


```python
## melihat nilai missing
properti.isnull().sum()
```




    K..Mandi         0
    K..Tidur         0
    L..Bangunan      0
    Sertifikat       0
    Tipe.Properti    0
    Kota             0
    Price            0
    dtype: int64



### 4. 🎯 Mendefinisikan Business Problem

Dari pernyataan bisnis yang diajukan, **kita ingin melakukan prediksi harga properti berdasarkan Luas Bangunan**

❓ Berdasarkan kolom-kolom data, mari kita coba tentukan kolom apa yang akan menjadi *target* dan *prediktor*?

- Variabel target: 
- Variabel prediktor: 

### 5. Exploratory Data Analysis (EDA)

* <u>**Descriptive Statistics**</u>

Analisis statistik deskriptif digunakan untuk memberikan **gambaran awal mengenai distribusi dan perilaku data** dengan melihat nilai minimum, nilai maximum, rata – rata (mean), dan standar deviasi dari masing-masing variabel independen dan variabel dependen.

Method `describe()` menampilkan 8 ringkasan statistika deskriptif. Secara default menampilkan ringkasan untuk kolom numerik.


```python
properti.describe()
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6,287.000</td>
      <td>6,287.000</td>
      <td>6,287.000</td>
      <td>6,287.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.909</td>
      <td>2.410</td>
      <td>92.866</td>
      <td>1,705,079,211.070</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.948</td>
      <td>0.975</td>
      <td>66.115</td>
      <td>1,254,170,884.604</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>205,000,000.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000</td>
      <td>2.000</td>
      <td>41.000</td>
      <td>740,000,000.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000</td>
      <td>2.000</td>
      <td>75.000</td>
      <td>1,350,000,000.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000</td>
      <td>3.000</td>
      <td>130.000</td>
      <td>2,350,000,000.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000</td>
      <td>4.000</td>
      <td>600.000</td>
      <td>5,970,000,000.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
properti[properti['L..Bangunan'] < 19]
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Timur</td>
      <td>250000000</td>
    </tr>
    <tr>
      <th>282</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>HP - Hak Pakai</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>360000000</td>
    </tr>
    <tr>
      <th>557</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>663</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
      <td>3700000000</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>260000000</td>
    </tr>
    <tr>
      <th>1128</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>1500000000</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>270000000</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>1500000000</td>
    </tr>
    <tr>
      <th>1740</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>235000000</td>
    </tr>
    <tr>
      <th>1787</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>250000000</td>
    </tr>
    <tr>
      <th>2078</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
      <td>2000000000</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>260000000</td>
    </tr>
    <tr>
      <th>2686</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>2768</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>2775</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>450000000</td>
    </tr>
    <tr>
      <th>2881</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>250000000</td>
    </tr>
    <tr>
      <th>2944</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Barat</td>
      <td>400000000</td>
    </tr>
    <tr>
      <th>3041</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>250000000</td>
    </tr>
    <tr>
      <th>3214</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>350000000</td>
    </tr>
    <tr>
      <th>3253</th>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>350000000</td>
    </tr>
    <tr>
      <th>3342</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>325000000</td>
    </tr>
    <tr>
      <th>3361</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Timur</td>
      <td>350000000</td>
    </tr>
    <tr>
      <th>3673</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>280000000</td>
    </tr>
    <tr>
      <th>3720</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>300000000</td>
    </tr>
    <tr>
      <th>3755</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>255000000</td>
    </tr>
    <tr>
      <th>3848</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>300000000</td>
    </tr>
    <tr>
      <th>3965</th>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>590000000</td>
    </tr>
    <tr>
      <th>4048</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>380000000</td>
    </tr>
    <tr>
      <th>4109</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>4191</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Rumah</td>
      <td>Tangerang Selatan</td>
      <td>1350000000</td>
    </tr>
    <tr>
      <th>4225</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>4255</th>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>450000000</td>
    </tr>
    <tr>
      <th>4282</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>4346</th>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>310000000</td>
    </tr>
    <tr>
      <th>4369</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Pusat</td>
      <td>3750000000</td>
    </tr>
    <tr>
      <th>4658</th>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>550000000</td>
    </tr>
    <tr>
      <th>4675</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Tangerang Selatan</td>
      <td>350000000</td>
    </tr>
    <tr>
      <th>4778</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Barat</td>
      <td>525000000</td>
    </tr>
    <tr>
      <th>4898</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>230000000</td>
    </tr>
    <tr>
      <th>5467</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Barat</td>
      <td>4500000000</td>
    </tr>
    <tr>
      <th>5529</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Barat</td>
      <td>580000000</td>
    </tr>
    <tr>
      <th>5579</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>5595</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>375000000</td>
    </tr>
    <tr>
      <th>5641</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>5775</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>565000000</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>6019</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>6057</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>290000000</td>
    </tr>
    <tr>
      <th>6120</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Rumah</td>
      <td>Tangerang Selatan</td>
      <td>1000000000</td>
    </tr>
    <tr>
      <th>6130</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Barat</td>
      <td>4500000000</td>
    </tr>
    <tr>
      <th>6216</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Depok</td>
      <td>550000000</td>
    </tr>
    <tr>
      <th>6222</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>350000000</td>
    </tr>
    <tr>
      <th>6225</th>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>350000000</td>
    </tr>
  </tbody>
</table>
</div>



Luas bangunan dibawah 14 $m^2$ kemungkinan besar tidak tepat atau salah ukur


```python
properti.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 6287 entries, 0 to 6286
    Data columns (total 7 columns):
     #   Column         Non-Null Count  Dtype   
    ---  ------         --------------  -----   
     0   K..Mandi       6287 non-null   int64   
     1   K..Tidur       6287 non-null   int64   
     2   L..Bangunan    6287 non-null   int64   
     3   Sertifikat     6287 non-null   category
     4   Tipe.Properti  6287 non-null   category
     5   Kota           6287 non-null   category
     6   Price          6287 non-null   int64   
    dtypes: category(3), int64(4)
    memory usage: 264.7 KB
    

❓ Apakah ada hal yang menarik dari hasil describe di atas?

* <u>**Cek korelasi antar variabel target dan prediktor**</u>

Biasanya uji korelasi ini akan sangat berhubungan dengan uji regresi yang menunjukkan apakah masing-masing variabel saling berhubungan erat. Meskipun variabel tersebut saling berhubungan erat atau berkorelasi, belum tentu variabel tersebut saling mempengaruhi. 

Dalam analisis korelasi ini, output yang dihasilkan hanya dalam rentang **-1 sampai 1**

- Bila korelasi dua variabel numerik **mendekati -1** artinya **korelasi negatif kuat**
- Bila korelasi dua variabel numerik **mendekati 1** artinya **korelasi positif kuat**
- Bila korelasi dua variabel numerik **mendekati 0** artinya **tidak berkorelasi**

![korelasi](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/assets/correlation-coef.jpg)

- Menggunakan nilai korelasi


```python
properti['Price'].corr(properti['L..Bangunan'])
```




    0.7856264787666598




```python
properti[properti['L..Bangunan'] >= 14][['Price', 'L..Bangunan']].corr()
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
      <th>Price</th>
      <th>L..Bangunan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Price</th>
      <td>1.000</td>
      <td>0.789</td>
    </tr>
    <tr>
      <th>L..Bangunan</th>
      <td>0.789</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



- Menggunakan visualisasi


```python
properti.plot.scatter(x = 'L..Bangunan', y = 'Price')
plt.show()
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_32_0.png)
    



```python
properti[(properti['Price'] < 1e9) & (properti['L..Bangunan'] > 450)]
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6206</th>
      <td>2</td>
      <td>2</td>
      <td>478</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Barat</td>
      <td>480000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
properti = properti[(properti['L..Bangunan'] >= 14) & (properti['L..Bangunan'].index != 6206)]
```


```python
properti.plot.scatter(x = 'L..Bangunan', y = 'Price')
plt.show()
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_35_0.png)
    


* **Kesimpulan dari plot:**

> 

Berdasarkan kesimpulan dan analisis yang sudah dilakukan, adakah yang perlu kita lakukan pada data kita?

`L.Bangunan` berkorelasi positif dengan `price` meskipun masih terdapat beberapa outlier

* <u>**Identifikasi Outlier**</u>

Dari hasil analisa statistik deskriptif belum diketahui apakah pada data yang kita miliki memiliki ***outlier*** atau tidak, maka dari itu mari coba kita lihat dengan menggunakan visualisasi ***Box Plot***.

Visualisasi ***Box Plot*** dapat dibuat dengan menggunakan fungsi `boxplot()` dari `library matplotlib`. 


```python
# Melihat nilai outlier
properti.boxplot(column = 'L..Bangunan')
```




    <Axes: >




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_39_1.png)
    


Masih ada outlier


```python
from matplotlib.cbook import boxplot_stats

bp_stats = boxplot_stats(properti['L..Bangunan'].values)

bp_stats[0]['fliers'].min()
```




    263



### 6. Membuat Model Simple Linear Regression 

#### **6.1 Model dengan outlier** 

Dari tahapan EDA (checking outlier) dengan boxplot, diketahui bahwa variabel `L..Bangunan` memiliki nilai outlier.

Selanjutnya jika kita ingin menjawab business problem yang kita miliki, yaitu kita ingin melakukan prediksi harga **Price** properti berdasarkan besaran **Luas Bangunan**. Kita akan memakai seluruh observasi yang ada terlebih dahulu.

<u>**Tahapan 1 - Menentukan Target dan Prediktor**</u>
   - Y   : `df['target']`
   - X   : `sm.add_constant(df['prediktor'])`. Supaya intercept tidak dianggap 0


```python
# membuat objek untuk prediktor (pilih kolom yang akan digunakan)
X_data = properti['L..Bangunan']

# menambahkan intercept/add_constant
X_data = sm.add_constant(X_data)

# membuat objek target
Y_data = properti['Price']
```

<u> **Tahapan 2 - Membuat model Prediksi**</u>

Untuk membuat model regresi linier di Python kita akan menggunakan fungsi `OLS()` dari package `statsmodels`.

Syntax: `sm.OLS(target, prediktor).fit()`


```python
# sm.OLS(properti['Price'], properti['L..Bangunan']).fit().summary()
# Nggak work kare di sm.OLS tidak ada constantnya
# https://stackoverflow.com/questions/30286095/do-i-need-to-add-a-constant-when-using-sm-ols
```


```python
# Membuat model
lm_outlier = sm.OLS(Y_data, X_data).fit()
```

<u>**Tahapan 3 - Melihat hasil / menginterpretasikan model** </u>
   - intercept dan slope: `model.params`
   - summary model      : `model.summary()`  


```python
# Summary model
print(lm_outlier.params, lm_outlier.summary())
```

    const         303,614,296.475
    L..Bangunan    15,068,066.597
    dtype: float64                             OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Price   R-squared:                       0.628
    Model:                            OLS   Adj. R-squared:                  0.627
    Method:                 Least Squares   F-statistic:                 1.057e+04
    Date:                Mon, 08 Jan 2024   Prob (F-statistic):               0.00
    Time:                        10:15:37   Log-Likelihood:            -1.3728e+05
    No. Observations:                6276   AIC:                         2.746e+05
    Df Residuals:                    6274   BIC:                         2.746e+05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        3.036e+08   1.67e+07     18.182      0.000    2.71e+08    3.36e+08
    L..Bangunan  1.507e+07   1.47e+05    102.809      0.000    1.48e+07    1.54e+07
    ==============================================================================
    Omnibus:                     1807.386   Durbin-Watson:                   2.034
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6246.081
    Skew:                           1.431   Prob(JB):                         0.00
    Kurtosis:                       6.962   Cond. No.                         197.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


📝 **Interpretasi model :**

1. **Model linier regresi** untuk kasus prediksi harga properti di daerah Jakarta, Depok, dan Tangsel adalah:

> `Price = b0 + b1*L..Bangunan`

> `Price =  ... + ...*L..Bangunan`
   - slope    : 1.507e+07 
   - intercept: 3.036e+08
   
2. **Signifikansi prediktor** (bandingkan nilai p-value dengan alpha)
   - H0: Luas Bangunan tidak mempengaruhi Harga Properti
   - H1: Luas Bangunan mempengaruhi Harga Properti
   
    **Note:** Ketika p-value < 0.05 (alpha) maka kesimpulannya adalah menolak H0 yang berarti Luas Bangunan signifikan berpengaruh terhadap Harga properti

   > p-value : 0.00
   
   > Kesimpulan : Tolak $H_0$
   
3. **Goodness of fit** (melihat nilai R-squared, dimana **0 ≤ $R^2$ ≤ 1**)
   
   Nilai R-Squared merepresentasikan % variasi dari data yang berhasil dijelaskan oleh model.
   
   - Semakin mendekati 1, mengindikasikan model semakin fit
   - Semakin mendekati 0, mengindikasikan model tidak fit
   
   > R-Squared: 0.617
   
   > Kesimpulan : Cukup fit

**[Optional] Other Information in Summary**

1. Tabel 1, sisi kiri menyimpan informasi dasar dari model
    - Dep. Variable   : Target variabel (Y)
    - Model           : Model regresi linier
    - Method          : Metode yang digunakan untuk membuat model regresi linier
    - No. Observations:	Jumlah observasi yang digunakan ketika membuat model regresi linier
    - DF Residuals    :	Degrees of freedom error/residual (**No. Observations - parameter**)
    - DF Model        :	Degrees of freedom model (**jumlah prediktor**)


2. Tabel 1, sisi kanan menyimpan informasi kebaikan model
    - **R-squared**         : Goodness of fit
    - **Adj. R-squared**    : Goodneess of fit untuk multiple linear regression
    - F-statistic       : Statistik hitung dari F-test (uji simultan)
    - Prob (F-statistic): p-value dari F-test 
        
        a. H0 --> Tidak ada prediktor yang berpengaruh signifikan terhadap target
        
        b. H1 --> Min terdapat 1 prediktor yang berpengaruh signifikan terhadap target
    - Log-likelihood    : Log dari nilai likelihood.
    - AIC               : Akaike Information Criterion (information loss)
    - BIC               : Bayesian Information Criterion (serupa dengan AIC, namun perhitungan nilainya berbeda)

3. Tabel 2 menyimpan informasi dari koefisien regresi
    - **coef**              : Estimasi koefisien
    - std err               : Estimasi selisih nilai sampel terhadap populasi
    - t                     : Statistik hitung dari t-test (uji parsial)
    - **P > |t|**               : P-value dari t-test
    - [95.0% Conf. Interval]: Confidence Interval (CI) 95%


4. Tabel 3 menyimpan hasil uji statistik error/residual
    - Omnibus	D’Angostino’s test: Statistik hitung untuk pengujian **Skewness** dan **Kurtosis**
    - Prob(Omnibus): p-value dari **Omnibus	D’Angostino’s test**
    - Skewness: Mengukur kecondongan distribusi error
    - Kurtosis:	Mengukur keruncingan distribusi error
    - Durbin-Watson: Statistik hitung pengujian autokorelasi
    - Jarque-Bera:	Serupa dengan **Omnibus	D’Angostino’s test**, namun memiliki perhitungan yang berbeda
    - Prob (JB): p-value dari **Jarque-Bera**
    - Cond. No: Pengujian multicolinearity

<u> **Tahapan 4 - Hasil Visualisasi 2 Dimensi** </u>
    
- hanya bisa untuk 1 variabel prediktor


```python
# Plot scatter
properti.plot.scatter(x='L..Bangunan', y='Price')

# Plot garis linear model
plt.plot(properti['L..Bangunan'], lm_outlier.fittedvalues, c='red')
plt.show()
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_52_0.png)
    


<u>**Tahapan 5 - Melakukan Prediksi Model**</u>

❓🏠 **Business Question**:
Terdapat properti A dengan Luas Bangunan 90m2. Kita diminta untuk memprediksi harga properti A, berapakah harga prediksinya?


```python
## code here (manual)
intercept = 3.036e+08
slope = 1.507e+07

house_price_new = intercept + slope*90
house_price_new
```




    1659900000.0



Untuk melakukan prediksi terhadap beberapa Luas bangunan, dapat menggunakan `model.predict()`

Eg : Data properti dengan Luas Bangunan terbaru


```python
new_house = pd.DataFrame({'L.B': (75, 320, 188, 60, 90)})
new_house
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
      <th>L.B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
## code here
# predict copiers datasest
X_new = sm.add_constant(new_house)

lm_outlier.predict(X_new)
```




    0   1,433,719,291.236
    1   5,125,395,607.458
    2   3,136,410,816.677
    3   1,207,698,292.284
    4   1,659,740,290.189
    dtype: float64



#### **6.2 Model tanpa outlier** 

Untuk membuat model tanpa outlier, langkah pertama yang harus dilakukan adalah melakukan filtering pada data. 

❓ Coba tinjau lagi nilai outlier pada data properti kita, kira-kira berapa threshold batas outlier yang dimiliki?


```python
# melihat nilai outlier kembali dengan boxplot
properti.boxplot('L..Bangunan')
```




    <Axes: >




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_60_1.png)
    


Dalam kasus ini dikarenakan outlier `L..Bangunan` di atas 280 m2, maka data yang digunakan adalah data dengan `L..Bangunan` < 280 m2.


```python
# remove outlier (membuang nilai outlier dengan L.B < 280m2)
properti_new = properti[(properti['L..Bangunan'] <= 263)]
properti_new
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>78</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Selatan</td>
      <td>2500000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Timur</td>
      <td>265000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>120</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
      <td>2600000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>130</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Depok</td>
      <td>1300000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>97</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Barat</td>
      <td>3200000000</td>
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
    </tr>
    <tr>
      <th>6282</th>
      <td>2</td>
      <td>3</td>
      <td>80</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Timur</td>
      <td>960000000</td>
    </tr>
    <tr>
      <th>6283</th>
      <td>2</td>
      <td>3</td>
      <td>84</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Depok</td>
      <td>1070000000</td>
    </tr>
    <tr>
      <th>6284</th>
      <td>2</td>
      <td>2</td>
      <td>132</td>
      <td>Lainnya (PPJB,Girik,Adat,dll)</td>
      <td>Apartemen</td>
      <td>Jakarta Utara</td>
      <td>1900000000</td>
    </tr>
    <tr>
      <th>6285</th>
      <td>3</td>
      <td>3</td>
      <td>83</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Depok</td>
      <td>1260000000</td>
    </tr>
    <tr>
      <th>6286</th>
      <td>4</td>
      <td>3</td>
      <td>200</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Selatan</td>
      <td>2900000000</td>
    </tr>
  </tbody>
</table>
<p>6160 rows × 7 columns</p>
</div>




```python
# melakukan modeling dengan data baru 
# define predictor variable
# X_data_no = properti_new['L..Bangunan']
X_data_no = sm.add_constant(properti_new['L..Bangunan'])

#define target variable
Y_data_no = properti_new['Price']

# build model with outlier
lm_no_outlier = sm.OLS(Y_data_no, X_data_no).fit()

lm_no_outlier.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th>  <td>   0.615</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.615</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   9838.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 08 Jan 2024</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>10:49:42</td>     <th>  Log-Likelihood:    </th> <td>-1.3459e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  6160</td>      <th>  AIC:               </th>  <td>2.692e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  6158</td>      <th>  BIC:               </th>  <td>2.692e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td> 2.044e+08</td> <td> 1.74e+07</td> <td>   11.730</td> <td> 0.000</td> <td>  1.7e+08</td> <td> 2.39e+08</td>
</tr>
<tr>
  <th>L..Bangunan</th> <td> 1.636e+07</td> <td> 1.65e+05</td> <td>   99.185</td> <td> 0.000</td> <td>  1.6e+07</td> <td> 1.67e+07</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1802.302</td> <th>  Durbin-Watson:     </th> <td>   2.036</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5875.961</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.479</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.762</td>  <th>  Cond. No.          </th> <td>    194.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



📝 **Interpretasi model:**

1. **Model linier regresi** untuk kasus prediksi harga properti di daerah Jakarta, Depok, dan Tangsel adalah:
   > `Price = b0 + b1*L..Bangunan`
   
   > `Price =  ... + ...*L..Bangunan`
   
   - slope    : 1.636e+07
   - intercept: 2.044e+08	
   
2. **Signifikansi prediktor** (bandingkan nilai p-value dengan alpha)
    - H0: Luas Bangunan tidak mempengaruhi Harga Properti
    - H1: Luas Bangunan mempengaruhi Harga Properti
    
    **Note:** Ketika p-value < 0.05 (alpha) maka kesimpulannya adalah menolak H0
    
    > p-value : 0.00
    
    > Kesimpulan : $H_0$ ditolak
   
3. **Goodness of fit** (melihat nilai R-squared, dimana **0 ≤ $R^2$ ≤ 1**)
   
   Nilai R-Squared merepresentasikan % variasi dari data yang berhasil dijelaskan oleh model. **Formula**:  
      
   $R^2=1- \frac {∑ \limits_{i=1}^n (Y_i−\hat Y)^2}{∑ \limits_{i=1}^n(Y_i−\bar Y)^2}$
   
   - Semakin mendekati 1, mengindikasikan model semakin fit
   - Semakin mendekati 0, mengindikasikan model tidak fit
   
   > R-Squared: 0.615
   
   > Kesimpulan : Cukup fit


```python
# visualize the result
fig1, ax1 = plt.subplots()
properti.plot.scatter(x='L..Bangunan', y='Price', c='yellow', ax=ax1, ylim=[0,6.5e9])
properti_new.plot.scatter(x='L..Bangunan', y='Price', c='blue',ax=ax1)
# visualisasi model no_outlier
plt.plot(properti_new['L..Bangunan'], lm_no_outlier.fittedvalues, c='black')
# visualisasi model dengan outlier
plt.plot(properti['L..Bangunan'], lm_outlier.fittedvalues, c='red')
plt.show()
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_65_0.png)
    


## **Leverage vs. Influence**

**Leverage** adalah nilai yang letaknya jauh dari letak observasi lainnya, sering disebut sebagai **outlier**. Nilai leverage dapat mempengaruhi model linier regresi atau pun tidak.

- Ketika **leverage mempengaruhi (menurunkan R-Squared)** model linier regresi: **high influence**, sebaiknya **di-exclude** -> membuat model menjadi lebih jelek
- Ketika **leverage tidak mempengaruhi (meningkatkan R-Squared)** model linier regresi: **low influence**, sebaiknya **di-include** -> membuat model menjadi lebih baik


```python
print("R-Squared model dengan outlier :", (lm_outlier.rsquared).round(2))
print("R-Squared model tanpa outlier :", (lm_no_outlier.rsquared).round(2))
```

    R-Squared model dengan outlier : 0.63
    R-Squared model tanpa outlier : 0.62
    

**Kesimpulan**: **high influence** -> membuat model menjadi lebih jelek

## Multiple Linear Regression

Linear regression dengan **lebih dari satu prediktor** bisa meningkatkan performa model karena lebih banyak informasi yang dapat menjelaskan target.

Formula multiple linear regression:

$$
\hat{y}=\beta_0+\beta_1.x_1+...+\beta_n.x_n
$$

dimana $\hat{y}$ merupakan prediksi target variabel dan $x_1,...,x_n$ prediktor lainnya. 

Workflow pada simple linear regression dan multiple linear regression adalah sama. Berikut merupakan worklownya:

### 1. Preparation Data

Kita akan membuat multiple linear regression menggunakan data properti untuk memprediksi `Price` berdasarkan keseluruhan variabel.
- y: Price
- x: K..Mandi, K..Tidur, L..Bangunan, Sertifikat, Tipe.Properti, dan Kota

#### 💡 Categorical Predictor: Dummy Variable Encoding

Di Python, data input dan output untuk model *machine learning* harus berbentuk numeric. Ini berarti, ketika data kita mempunyai nilai kategorikal, harus di-*encode* menjadi numerik terlebih dahulu.

Sebelum melakukan fitting model, kita harus mengubah **prediktor kategorik menjadi dummy variable**, dengan cara:

- Dilakukan dengan menggunakan fungsi `pd.get_dummies()`
- **One hot encoding** = mengubah kolom kategorik menjadi kolom-kolom baru dari setiap kategori yang berisi nilai 0 dan 1 
- **Dummy variable** =  mengubah kolom kategorik menjadi kolom-kolom baru yang terdiri dari **k-1 kategori**, berisi nilai 0 dan 1. Kategori yang tidak menjadi kolom, akan menjadi kondisi basis. Untuk membuat dummy variable, tambahkan parameter `drop_first=True`

![](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/assets/one_hot-dummy.png)

📌 Dalam kasus regresi wajib memakai dummy variabel. 

**💡 NOTES**: Salah satu kolom di-*drop* karena bersifat redundan (berulang). Untuk kolom yang hanya memiliki 2 kategori, tidak ada perbedaan hasil/efek baik ketika memilih ordinal ataupun dummy, akan tetapi best practicenya menggunakan dummy variabel.

Mari kita coba menerapkan **Dummy Variable Encoding** untuk kolom-kolom kategorikal.


```python
properti.head()
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>294</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Utara</td>
      <td>3500000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>78</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Jakarta Selatan</td>
      <td>2500000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Apartemen</td>
      <td>Jakarta Timur</td>
      <td>265000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>120</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
      <td>2600000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>130</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Rumah</td>
      <td>Depok</td>
      <td>1300000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Object nama kolom kategori
cat_dummy = properti.select_dtypes('category').columns

# dummy encoding
properti_enc = pd.get_dummies(data = properti,
                             columns = cat_dummy,
                             drop_first = True,
                             dtype = 'int64')

# melihat hasil encoding
properti_enc
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Price</th>
      <th>Sertifikat_HP - Hak Pakai</th>
      <th>Sertifikat_HS - Hak Sewa</th>
      <th>Sertifikat_Lainnya (PPJB,Girik,Adat,dll)</th>
      <th>Sertifikat_PPJB</th>
      <th>Sertifikat_SHM - Sertifikat Hak Milik</th>
      <th>Tipe.Properti_Rumah</th>
      <th>Kota_Jakarta Barat</th>
      <th>Kota_Jakarta Pusat</th>
      <th>Kota_Jakarta Selatan</th>
      <th>Kota_Jakarta Timur</th>
      <th>Kota_Jakarta Utara</th>
      <th>Kota_Tangerang Selatan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>294</td>
      <td>3500000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>78</td>
      <td>2500000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>265000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>120</td>
      <td>2600000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>130</td>
      <td>1300000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6282</th>
      <td>2</td>
      <td>3</td>
      <td>80</td>
      <td>960000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6283</th>
      <td>2</td>
      <td>3</td>
      <td>84</td>
      <td>1070000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6284</th>
      <td>2</td>
      <td>2</td>
      <td>132</td>
      <td>1900000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6285</th>
      <td>3</td>
      <td>3</td>
      <td>83</td>
      <td>1260000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6286</th>
      <td>4</td>
      <td>3</td>
      <td>200</td>
      <td>2900000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6276 rows × 16 columns</p>
</div>



### 2. Membuat Model Multiple Linear Regression


```python
# membuat objek prediktor dan target
Y = properti_enc['Price']
X = sm.add_constant(properti_enc.drop(columns='Price'))

# Membuat model
lm_multiple = sm.OLS(Y, X).fit()

# Melihat summary
lm_multiple.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th>  <td>   0.691</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.690</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   934.1</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 08 Jan 2024</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>11:24:54</td>     <th>  Log-Likelihood:    </th> <td>-1.3669e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  6276</td>      <th>  AIC:               </th>  <td>2.734e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  6260</td>      <th>  BIC:               </th>  <td>2.735e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    15</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                      <td></td>                        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                    <td>-6.805e+07</td> <td> 4.05e+07</td> <td>   -1.680</td> <td> 0.093</td> <td>-1.47e+08</td> <td> 1.14e+07</td>
</tr>
<tr>
  <th>K..Mandi</th>                                 <td> 8.037e+07</td> <td>  1.8e+07</td> <td>    4.457</td> <td> 0.000</td> <td>  4.5e+07</td> <td> 1.16e+08</td>
</tr>
<tr>
  <th>K..Tidur</th>                                 <td>  8.87e+07</td> <td> 1.67e+07</td> <td>    5.313</td> <td> 0.000</td> <td>  5.6e+07</td> <td> 1.21e+08</td>
</tr>
<tr>
  <th>L..Bangunan</th>                              <td> 1.417e+07</td> <td> 2.28e+05</td> <td>   62.107</td> <td> 0.000</td> <td> 1.37e+07</td> <td> 1.46e+07</td>
</tr>
<tr>
  <th>Sertifikat_HP - Hak Pakai</th>                <td>-1.917e+08</td> <td> 1.23e+08</td> <td>   -1.564</td> <td> 0.118</td> <td>-4.32e+08</td> <td> 4.86e+07</td>
</tr>
<tr>
  <th>Sertifikat_HS - Hak Sewa</th>                 <td>-2.992e+08</td> <td> 3.13e+08</td> <td>   -0.956</td> <td> 0.339</td> <td>-9.13e+08</td> <td> 3.15e+08</td>
</tr>
<tr>
  <th>Sertifikat_Lainnya (PPJB,Girik,Adat,dll)</th> <td> 2.888e+07</td> <td> 3.11e+07</td> <td>    0.929</td> <td> 0.353</td> <td> -3.2e+07</td> <td> 8.98e+07</td>
</tr>
<tr>
  <th>Sertifikat_PPJB</th>                          <td>-1.797e+08</td> <td> 4.94e+08</td> <td>   -0.364</td> <td> 0.716</td> <td>-1.15e+09</td> <td> 7.89e+08</td>
</tr>
<tr>
  <th>Sertifikat_SHM - Sertifikat Hak Milik</th>    <td>-7.895e+07</td> <td> 3.02e+07</td> <td>   -2.616</td> <td> 0.009</td> <td>-1.38e+08</td> <td>-1.98e+07</td>
</tr>
<tr>
  <th>Tipe.Properti_Rumah</th>                      <td>-3.403e+08</td> <td> 3.01e+07</td> <td>  -11.307</td> <td> 0.000</td> <td>-3.99e+08</td> <td>-2.81e+08</td>
</tr>
<tr>
  <th>Kota_Jakarta Barat</th>                       <td> 3.911e+08</td> <td> 3.24e+07</td> <td>   12.072</td> <td> 0.000</td> <td> 3.28e+08</td> <td> 4.55e+08</td>
</tr>
<tr>
  <th>Kota_Jakarta Pusat</th>                       <td> 5.054e+08</td> <td> 3.78e+07</td> <td>   13.385</td> <td> 0.000</td> <td> 4.31e+08</td> <td> 5.79e+08</td>
</tr>
<tr>
  <th>Kota_Jakarta Selatan</th>                     <td> 7.807e+08</td> <td> 3.74e+07</td> <td>   20.891</td> <td> 0.000</td> <td> 7.07e+08</td> <td> 8.54e+08</td>
</tr>
<tr>
  <th>Kota_Jakarta Timur</th>                       <td>-8.513e+06</td> <td> 3.05e+07</td> <td>   -0.279</td> <td> 0.780</td> <td>-6.83e+07</td> <td> 5.13e+07</td>
</tr>
<tr>
  <th>Kota_Jakarta Utara</th>                       <td>  3.45e+08</td> <td> 3.72e+07</td> <td>    9.287</td> <td> 0.000</td> <td> 2.72e+08</td> <td> 4.18e+08</td>
</tr>
<tr>
  <th>Kota_Tangerang Selatan</th>                   <td> 2.589e+08</td> <td> 3.21e+07</td> <td>    8.061</td> <td> 0.000</td> <td> 1.96e+08</td> <td> 3.22e+08</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1458.887</td> <th>  Durbin-Watson:     </th> <td>   2.052</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5120.034</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.143</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.789</td>  <th>  Cond. No.          </th> <td>6.40e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.4e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### 3. Interpretasi Model Multiple Linear Regression

**[Contoh Interpretasi Variabel Kategorik]**

- Terdapat category golongan darah A, AB, B, dan O
- Cara menginterpretasikan model apabila memiliki lebih dari 2 kategori :

> y = b0 + 1.96* gol.darahAB + 0.97* gol.darahB + 1.39* gol.darahO

     - slope gol.darahAB = 1.96, nilai y ketika golongan darah nya adalah AB sebesar **b0 + 1.96**
     - slope gol.darahB = 0.97, nilai y ketika golongan darah nya adalah B sebesar **b0 + 0.97**
     - slope gol.darahO = 1.39, nilai y ketika golongan darah nya adalah O sebesar **b0 + 1.39**
     - nilai y ketika golongan darah nya adalah A sebesar **b0 saja** (b0 + est. gol.darahAB * 0 + est. gol.darahB * 0 + est. gol.darahO * 0)
     - Golongan darah AB meningkatkan nilai y sebesar 1.96 poin dibandingkan golongan darah A (Basis)


```python
lm_multiple.params
```




    const                                       -68,046,193.387
    K..Mandi                                     80,369,440.910
    K..Tidur                                     88,704,257.583
    L..Bangunan                                  14,172,990.715
    Sertifikat_HP - Hak Pakai                  -191,668,410.368
    Sertifikat_HS - Hak Sewa                   -299,168,643.727
    Sertifikat_Lainnya (PPJB,Girik,Adat,dll)     28,879,971.583
    Sertifikat_PPJB                            -179,723,373.918
    Sertifikat_SHM - Sertifikat Hak Milik       -78,951,929.200
    Tipe.Properti_Rumah                        -340,252,862.068
    Kota_Jakarta Barat                          391,095,409.104
    Kota_Jakarta Pusat                          505,410,922.104
    Kota_Jakarta Selatan                        780,736,333.606
    Kota_Jakarta Timur                           -8,512,539.802
    Kota_Jakarta Utara                          345,049,019.597
    Kota_Tangerang Selatan                      258,934,983.035
    dtype: float64




```python
lm_multiple.pvalues[lm_multiple.pvalues > 0.05].index
```




    Index(['const', 'Sertifikat_HP - Hak Pakai', 'Sertifikat_HS - Hak Sewa',
           'Sertifikat_Lainnya (PPJB,Girik,Adat,dll)', 'Sertifikat_PPJB',
           'Kota_Jakarta Timur'],
          dtype='object')



📝 **Interpretasi model:**

**1. Interpretasi masing-masing variabel** 

contoh:
   - `K..Tidur`: Setiap 1 Kamar Tidur, meningkatkan harga properti sebesar 88,704,257.583
   - `L..Bangunan`: Setiap 1 $m^2$ meningkatkan harga 14,172,990.715
   - `Sertifikat_HS - Hak Milik`: Jika sertifikatnya SHM, maka properti berkurang sebesar 78,951,929.200
   - `Kota_Jakarta Selatan`: 
   
**2. Signifikansi prediktor**
   - variabel signifikan (p-value < 0.05) : 'K..Mandi', 'K..Tidur', 'L..Bangunan',
       'Sertifikat_SHM - Sertifikat Hak Milik', 'Tipe.Properti_Rumah',
       'Kota_Jakarta Barat', 'Kota_Jakarta Pusat', 'Kota_Jakarta Selatan',
       'Kota_Jakarta Utara', 'Kota_Tangerang Selatan'
   - variabel tidak signifikan (p-val>e < 0.05)'const', 'Sertifikat_HP - Hak Pakai', 'Sertifikat_HS - Hak Sewa',
       'Sertifikat_Lainnya (PPJB,Girik,Adat,dll)', 'Sertifikat_PPJB',
       'Kota_Jakarta Timur' : 

note: untuk prediktor kategorik, dianggap signifikan mempengaruhi target jika salah satu kategori signifikan

### 4. Prediksi

Setelah model terbentuk, model tidak dapat langsung digunakan sebelum melewati tahap evaluasi.

Tahapan evaluasi dapat dilakukan dengan melakukan **prediksi** terhadap data yang ada. Prediksi model dapat dilakukan dengan memanfaatkan fungsi `predict()`, berikut syntax yang dapat digunakan:

> `nama_model.predict(data_prediktor)`


```python
X_predict = pd.DataFrame({"K..Mandi": [1, 2, 3],
                         "K..Tidur": [2, 4, 5],
                         "L..Bangunan": [123, 321, 531],
                         "Sertifikat": ["SHM - Sertifikat Hak Milik", "PPJB", "HGB - Hak Guna Bangunan"],
                         "Tipe.Properti": ["Apartemen", "Rumah", "Rumah"],
                         "Kota": ["Depok", "Jakarta Selatan", "Jakarta Pusat"]})

X_predict.head()
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Sertifikat</th>
      <th>Tipe.Properti</th>
      <th>Kota</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>123</td>
      <td>SHM - Sertifikat Hak Milik</td>
      <td>Apartemen</td>
      <td>Depok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>321</td>
      <td>PPJB</td>
      <td>Rumah</td>
      <td>Jakarta Selatan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>531</td>
      <td>HGB - Hak Guna Bangunan</td>
      <td>Rumah</td>
      <td>Jakarta Pusat</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_predict_enc = pd.get_dummies(data = X_predict,
                             columns = cat_dummy,
                             drop_first = True,
                             dtype = 'int64')

X_predict_enc = X_predict_enc.reindex(columns = X.columns, fill_value=0)
X_predict_enc.shape
```




    (3, 16)




```python
res = lm_multiple.predict(X_predict_enc)
res
```




    0   1,922,103,884.819
    1   5,325,846,029.281
    2   8,375,645,740.335
    dtype: float64




```python
# membuat kolom prediksi yang berisi hasil dari prediksi model
properti_enc['Prediksi'] = lm_multiple.predict(X)
properti_enc.head()
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
      <th>K..Mandi</th>
      <th>K..Tidur</th>
      <th>L..Bangunan</th>
      <th>Price</th>
      <th>Sertifikat_HGB - Hak Guna Bangunan</th>
      <th>Sertifikat_HP - Hak Pakai</th>
      <th>Sertifikat_HS - Hak Sewa</th>
      <th>Sertifikat_Lainnya (PPJB,Girik,Adat,dll)</th>
      <th>Sertifikat_PPJB</th>
      <th>Sertifikat_SHM - Sertifikat Hak Milik</th>
      <th>Tipe.Properti_Apartemen</th>
      <th>Tipe.Properti_Rumah</th>
      <th>Kota_Depok</th>
      <th>Kota_Jakarta Barat</th>
      <th>Kota_Jakarta Pusat</th>
      <th>Kota_Jakarta Selatan</th>
      <th>Kota_Jakarta Timur</th>
      <th>Kota_Jakarta Utara</th>
      <th>Kota_Tangerang Selatan</th>
      <th>Prediksi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>294</td>
      <td>3500000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4,620,582,658.209</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>78</td>
      <td>2500000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2,246,452,582.268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>265000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>560,223,658.899</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>120</td>
      <td>2600000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2,057,066,220.233</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>130</td>
      <td>1300000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1,862,458,903.773</td>
    </tr>
  </tbody>
</table>
</div>



### 5. Goodness of Fit: R-Squared vs. Adj. R-Squared

Perbedaan R-Squared dan Adj. R-Squared:

- **R-Squared**: Seberapa baik model menjelaskan data, dengan mengukur seberapa besar informasi (variansi) dari target dapat dijelaskan oleh prediktor. Sehingga, jelas ketika **prediktor bertambah**, informasi (variansi) yang dirangkum semakin banyak atau dengan kata lain jelas nilai **R-Squared akan meningkat**.

    > Syntax: `nama_model.rsquared`

- **Adj. R- Squared**: tidak demikian pada adj. r-squred, karena disesuaikan dengan jumlah prediktor yang digunakan. Adj. r-squared akan meningkat hanya jika prediktor baru yang ditambahkan mengarah pada hasil prediksi yang lebih baik (prediktor signifikan mempengaruhi target)

    > Syntax: `nama_model.rsquared_adj`
   
Mari kita bandingan nilai R-Squared dan Adj. R-Squared antara `lm_outlier` dengan `lm_multiple`!


```python
# R-Squared
print('R-Squared Simple Linear Regression :', (lm_outlier.rsquared))
print('R-Squared Simple Linear Regression :', (lm_multiple.rsquared))
```

    R-Squared Simple Linear Regression : 0.6275163087637301
    R-Squared Simple Linear Regression : 0.6911833941775697
    


```python
# Adj R-Squared
print('Adj R-Squared Simple Linear Regression :', (lm_outlier.rsquared_adj))
print('Adj R-Squared Simple Linear Regression :', (lm_multiple.rsquared_adj))
```

    Adj R-Squared Simple Linear Regression : 0.6274569393516746
    Adj R-Squared Simple Linear Regression : 0.6904434182850239
    

📝 **Kesimpulan**: Berdasarkan nilai R-Squared dan Adj R-Squared, maka model yang terbaik adalah model `lm_multiple`

### Model Evaluation : Nilai Error

Untuk melihat apakah prediksi yang dibuat menghasilkan nilai error terkecil
  
**Error/residual adalah selisih antara hasil prediksi dengan nilai aktual.**

$$
Error/residual = actual - prediction = y - \hat y
$$

Terdapat beberapa nilai error yang ada :

1. MAE (Mean Absolute Error): Memperlakukan error dengan lebih ringan. **Formula:**
   $$
   MAE = \frac{1}{N} \sum_{i=1}^{N} \left | y_{i} - \hat{y} \right |
   $$

1. RMSE (Root Mean Square Error): Memperlakukan error dengan lebih sensitif. Ketika nilai error besar, maka nilai RMSE akan semakin besar dan sebaliknya. **Formula:**
   $$
   RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{i} - \hat{y})^{2}}
   $$
     
1. MAPE (Mean Absolute Percentage Error): Menunjukan seberapa besar penyimpangan error dalam bentuk persentase
   $$
   MAPE = \frac{1}{N} \sum_{i=1}^{N} \frac {\left | y_{i} - \hat{y} \right |} {y_{i}}
   $$
   
RMSE digunakan ketika model yang dibuat memuat observasi outlier. Sedangkan, MAE digunakan ketika model yang dibuat tidak memuat observasi outlier. MAPE adalah metrik yang baik untuk interpretasi karena mudah dipahami.

**MAE**

Fungsi `meanabs(kolom_target, kolom_prediksi)`


```python
# code here
from statsmodels.tools.eval_measures import meanabs
meanabs(properti_enc['Price'], properti_enc['Prediksi'])
```




    481478265.53622854



**RMSE**

Fungsi `rmse(kolom_target, kolom_prediksi)`


```python
# code here
from statsmodels.tools.eval_measures import rmse
rmse(properti_enc['Price'], properti_enc['Prediksi'])
```




    696545892.4274079



**MAPE**

Fungsi `mean_absolute_percentage_error()`

Pada fungsi tersebut nantinya akan kita isi dengan parameter yaitu 

- `y_true` = Parameter ini akan diisi dengan kolom target
- `y_pred` = Parameter ini akan diisi dengan kolom hasil prediksi


```python
# code here
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(properti_enc['Price'], properti_enc['Prediksi'])
```




    0.3754900268095911



# Assumption Checking

Limitasi dari pemodelan linear regresi adalah terdapat beberapa asumsi yang perlu dipenuhi agar model linear regresi dikatakan model yang baik. 
Pendekatan Ordinary Least Square/Linear Regression dikatakan BLUE (Best Linear Unbiased Estimator) ketika memenuhi beberapa uji asumsi berikut:
1. **Linearity**: antara x dan y nya ada hubungan linear  
    - Bisa dilihat dari R-squared model, jika R-squared kecil, maka kemungkinan antara prediktor dan target, tidak ada hubungan linear
2. **Normality of Residual**: Residual nya berdistribusi normal 
    - Saat berdistribusi normal, error berada di sekitar 0
3. **No-Heteroscedasticity**: Variansi residual konstan (tidak membentuk sebuah pola)  
4. **Little to No-Multicollinearity**: antar variabel prediktor nya harus independence (tidak mempunyai hubungan)  


## Linearity

Untuk menguji apakah variabel target dan prediktor memiliki hubungan linear. Dapat dilihat dengan nilai korelasi. 

Linearity artinya target variabel dengan prediktornya memiliki hubungan yang linear atau hubungannya bersifat garis lurus. Selain itu, efek atau nilai koefisien antar variabel bersifat additive. Jika linearity ini tidak terpenuhi, maka otomatis semua nilai koefisien yang kita dapatkan tidak valid karena model berasumsi bahwa pola yang akan kita buat adalah linear.


```python
# Residual vs fitted values
linearity_test(lm_multiple)
```


    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_99_0.png)
    


Bagaimana apabila ada yang tidak linear?
- Exclude variable tersebut dari model
- Apabila mayoritas variable prediktor tidak linear, maka bisa ganti model

## Normality of Residual

**Harapannya ketika membuat model linear regression**, error yang dihasilkan berdistribusi normal. Artinya error banyak berkumpul disekitar angka 0. Untuk mengecek residual menyebar normal, pengujian yang paling sering dilakukan adalah Shapiro test:
- $H_0$: Residual berdistribusi normal
- $H_1$: Residual tidak berdistribusi normal

Dalam melakukan pengujiannya kita akan dibantu library `scipy` dan memanfaatkan fungsi `shapiro()`. Untuk memanfaatkan fungsi tersebut, kita akan mengeluarkan nilai residu dari model yang sudah dibuat dengan menambahkan `.resid` pada objek model yang dibuat.

📌 **Note**: Jika asumsi normalitas tidak terpenuhi, maka hasil uji signifikansi serta nilai standard error dari intercept dan slope setiap prediktor yang dihasilkan bersifat bias atau tidak mencerminkan nilai sebenarnya. Jika residual memiliki distribusi yang tidak normal, bisa lakukan **transformasi/scaling data pada target variabel** atau **menambahkan sample data**.


```python
pd.DataFrame({
    'Prediction': lm_multiple.fittedvalues,
    'Actual': properti_enc['Price'],
    'Residual': lm_multiple.resid
}).head()
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
      <th>Prediction</th>
      <th>Actual</th>
      <th>Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4,620,582,658.209</td>
      <td>3500000000</td>
      <td>-1,120,582,658.209</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2,246,452,582.268</td>
      <td>2500000000</td>
      <td>253,547,417.732</td>
    </tr>
    <tr>
      <th>2</th>
      <td>560,223,658.899</td>
      <td>265000000</td>
      <td>-295,223,658.899</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2,057,066,220.233</td>
      <td>2600000000</td>
      <td>542,933,779.767</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1,862,458,903.773</td>
      <td>1300000000</td>
      <td>-562,458,903.773</td>
    </tr>
  </tbody>
</table>
</div>



Untuk melakukan pengujian asumsi normality of residual bisa menggunakan visualisasi histogram.


```python
lm_multiple.resid.hist()
```




    <Axes: >




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_104_1.png)
    


Selain itu bisa juga menggunakan pengujian statistik yaitu **Shapiro Test**. Dalam melakukan pengujiannya kita akan dibantu library `scipy` dan memanfaatkan fungsi `shapiro()`. Untuk memanfaatkan fungsi tersebut, kita akan mengeluarkan nilai residu dari model yang sudah dibuat dengan menambahkan `.resid` pada objek model yang dibuat.


```python
from scipy.stats import shapiro
shapiro(lm_multiple.resid)
```

    C:\Users\SaltFarmer\miniconda3\envs\algoritma\lib\site-packages\scipy\stats\_morestats.py:1816: UserWarning: p-value may not be accurate for N > 5000.
      warnings.warn("p-value may not be accurate for N > 5000.")
    




    ShapiroResult(statistic=0.9078583717346191, pvalue=0.0)



Nilai p-value yang kita harapkan pada uji shapiro test yaitu **p-value > alpha**.

Handling asumsi yang tidak terpenuhi untuk normality of residuals yaitu dengan cara:
- menambahkan data
- transformasi pada target varibel (y)

## No Heteroscedasticity (Homoscedasticity)

Homocesdasticity menunjukkan bahwa residual atau error bersifat konstan atau tidak membentuk pola tertentu. Jika error membentuk pola tertentu seperti garis linear atau mengerucut, maka kita sebut dengan `Heterocesdasticity` dan akan berpengaruh pada nilai standard error pada estimate/koefisien prediktor yan bias (terlalu sempit atau terlalu lebar).

Untuk mengecek terjadinya heteroscedasticity kita dapat menggunakan Breusch-Pagan test:
- $H_0$: residual homogen(tidak membentuk sebuah pola/acak)
- $H_1$: residual heteros (membentuk sebuah pola)

Kita bisa menggunakan method `het_breuschpagan()` dari library `statsmodels`.

Pada `lm_multiple_new` yang kita miliki, kita bisa memvisualisasikan sebaran dari residual yang ada dengan menggunakan scatter plot.


```python
plt.scatter(y = lm_multiple.resid, x = lm_multiple.fittedvalues)
plt.axhline(y = 0, color = 'r')
```




    <matplotlib.lines.Line2D at 0x1fefc3bc8e0>




    
![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/4_rm-main/output_111_1.png)
    



```python
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

name = ['Lagrange multiplier statistics', 'p-value', 'f-value', 'f p-value']

test = sms.het_breuschpagan(lm_multiple.resid, lm_multiple.model.exog)
lzip(name, test)
```




    [('Lagrange multiplier statistics', 985.8090677322887),
     ('p-value', 1.4820389736444784e-200),
     ('f-value', 77.76864569434078),
     ('f p-value', 8.904001347049246e-219)]



## No Multicolinearity

Harapannya pada model linear regression, tidak terjadi multikolinearitas. Multikolinearitas terjadi ketika antar variabel prediktor yang digunakan pada model memiliki hubungan yang kuat. Ada atau tidak multikolinearitas dapat dilihat dari nilai VIF(Variance Inflation Factor). 

VIF dibagi menjadi beberapa nilai berikut:
- 1 = tidak berkorelasi antar prediktornya
- antara 1 dan 5 = korelasinya moderate
- Lebih besar 5 = paling kuat berkorelasi antar prediktornya
- Biasanya VIF lebih besar 10 adalah yang menunjukkan variabel prediktor sangat berkorelasi kuat.

> Ketika nilai VIF > 10 maka **ada hubungan yang kuat antar prediktor**. Yang diingkan ketika membuat model, nilai VIF < 10 agar **tidak ada hubungan antar prediktor**.


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

vif = [variance_inflation_factor(X[['K..Mandi', 'K..Tidur', 'L..Bangunan']].values, i) for i in range(len(X[['K..Mandi', 'K..Tidur', 'L..Bangunan']].columns))]
pd.Series(data=vif, index = X[['K..Mandi', 'K..Tidur', 'L..Bangunan']].columns).sort_values(ascending=False)
```




    K..Mandi      16.904
    K..Tidur      14.499
    L..Bangunan    7.280
    dtype: float64



Jika terjadi multicollinearity, yang bisa dilakukan adalah:
- Membuang salah satu variabel
- Membuat variabel baru, dari rata-rata nilai kedua variabel
