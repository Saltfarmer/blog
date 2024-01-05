---
title: "Day 2 Algorit.ma : Practical Statistics"
header :
  image : /assets/images/AlgoritmaBanner.jpg
  teaser: /assets/images/Algoritma.png
comments : true
share : true
categories:
  - Data Science
tags:
  - Anaconda
  - Python
  - Algoritma
  - Pandas
  - Visualization

---

Day 3, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Coursebook: Practical Statistics**

This notebook was made based on main materials `Practical Statistics.ipynb`

Version: BRI Audit Analytics - January 2024

___


# Training Objectives

**Descriptive Statistics**

- Understanding 5 number summary
- Central tendency measure
- Measure of spread
- Variable relationship
- Z Score and Central Limit Theorem

**Inferential Statistics**

- Probability Density Function
- Data distributions
- Hypothesis test
- Error and confidence intervals

# Practical Statistics

Practical Statistics merupakan salah satu bagian penting dalam pengolahan data sehingga mendukung *busines case* yang ingin diangkat. Practical Statistics berisi kaidah statistika yang banyak diterapkan dalam praktik data science agar dapat **memahami dan mengolah data dengan tepat**. 

Secara umum, Practical Statistics terbagi 2, masing-masing membantu kita dalam hal tertentu:

* **Descriptive Statistics**: meringkas informasi dalam data agar terambil insight secara cepat. Nilai-nilai yang didapatkan merupakan rangkuman dari data, tujuannya untuk menggambarkan keadaan data secara umum.

* **Inferential Statistics**: menyimpulkan sesuatu tentang kondisi di lapangan, berdasarkan data yang kita punya (sample -> population).

![](assets/PS.png)

Untuk lebih memahami *practical statistics*, mari kita melakukan analisis menggunakan data asli.


```python
import pandas as pd
import numpy as np #perhitungan statistik
import matplotlib.pyplot as plt # untuk visualisasi
import math # perhitungan statistik
from scipy import stats #untuk perhitungan statistik
import seaborn as sns # untuk visualisasi

pd.set_option('display.float_format', '{:.2f}'.format) # suppress scientific notation
np.set_printoptions(suppress=True) # suppress scientific notation
```

# Study Case: Credit Card Balance Analysis

**1. Business Question**

Credit Card Balance Analysis, atau Analisis Saldo Kartu Kredit, dilakukan sebagai bagian dari analisis debitur dalam sebuah perusahaan kartu kredit. Hasil analisis dapat menentukan debitur mana yang memiliki risiko pembayaran kredit yang tinggi, atau bagaimana behavior debitur. Selain itu, menggabungkan data saldo kredit dengan informasi seperti limit kredit dapat membantu menghitung pemanfaatan kredit kartu, informasi yang berpengaruh pada Rating kredit seorang pemegang kartu.

Asumsi data:

- Balance dihitung sebagai jumlah semua transaksi selama periode penagihan/billing cycle. Sebagai contoh, jika seorang pemegang kartu mengeluarkan `$400`, `$500`, dan `$600` dalam 3 bulan, maka saldo rata-rata akan dicatat sebagai `$500`.

Kita sebagai tim data diminta untuk **menganalisa performa Credit Card Balance** nasabah. Data tersimpan dalam folder data_input dengan nama file `credit_card.csv`, gunakan `stringAsFactors = T` supaya kolom bernilai string berubah langsung menjadi tipe factor.

**2. Read Data**


```python
cc = pd.read_csv('data_input/CC.csv')
cc.head()
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.89</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.03</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104.59</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
    </tr>
    <tr>
      <th>3</th>
      <td>148.92</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.88</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
    </tr>
  </tbody>
</table>
</div>



**Deskripsi Kolom**

- `Income`: Besaran gaji nasabah per tahun (dalam $10000)
- `Limit` : Besaran kredit limit
- `Rating` : Skor yang diberikan kepada individu berdasarkan kelayakan kreditnya. Semakin besar maka semakin baik
- `Cards` : Jumlah banyaknya kartu kredit yang dimiliki oleh nasabah
- `Age` : Usia nasabah
- `Education` : Level/lamanya pendidikan yang ditempuh oleh nasabah
- `Gender`: Jenis kelamin nasabah
    - Male
    - Female
- `Student` : Apakah nasabah seorang pelajar atau bukan
    - Yes $\rightarrow$ Pelajar
    - No $\rightarrow$ Bukan pelajar
- `Married`: Status pernikahan
    - Yes $\rightarrow$ Sudah menikah
    - No $\rightarrow$ Belum menikah
- `Ethnicity`: Etnis nasabah
    - African American
    - Asian
    - Caucasian
- `Balance`: Rata-rata jumlah saldo kartu kredit

# Descriptive Statistics

Descriptive Statistics membantu kita **menggambarkan karakteristik** dari data, sehingga berguna dalam proses **Exploratory Data Analysis (EDA)**. Terdapat 3 hal pada descriptive statistics:

- Ukuran pemusatan data (Measure of Central Tendency)
- Ukuran penyebaran data (Measure of Spread)
- Hubungan antar data (Variable Relationship)

## Measure of Central Tendency

Ukuran pemusatan data adalah **suatu nilai yang cukup untuk mewakili seluruh nilai pada data**.

### Mean

Cara paling umum untuk membuat perkiraan nilai tunggal dari data yang banyak adalah dengan merata-ratakannya.

* Formula: $$\frac{\sum{x_i}}{n}$$
* Fungsi pada Python: `mean()`

**Contoh:**

Berapa rata-rata `Rating` atau skor yang diberikan kepada nasabah berdasarkan kelayakan kreditnya?


```python
# code here
cc['Rating'].mean()
```




    354.94



* Sifat nilai mean: **sensitif terhadap outlier**

> Outlier adalah nilai ekstrim yang jauh dari observasi lainnya. Kurang tepat apabila menggunakan nilai mean yang diketahui ada data outliernya.

**Contoh lain:**

Ada sebuah Kantor Cabang BRI di daerah Bekasi yang merekap jumlah pengunjung per bulan. 

Dengan nilai mean:



```python
# data pengunjung
pengunjung = pd.Series([55, 50, 40, 70, 60, 45, 35, 35, 60, 1000, 250, 70])
```


```python
# rata-rata pengunjung
pengunjung.mean()

```




    147.5



Apakah nilai mean di atas dapat diandalkan? _____

> Nilai mean tidak dapat diandalkan karena terdapat outlier

Masalah ini dapat diatasi oleh nilai **median**.

### Median

Median atau nilai tengah diperoleh dengan mengurutkan data terlebih dahulu kemudian mencari nilai tengah dari data.

- Baik untuk data yang memiliki **outlier** atau berdistribusi **skewed** (condong kiri/kanan)
- Fungsi pada Python: `median()`

Mari hitung ulang nilai pusat dari `pengunjung` menggunakan median:


```python
# median
pengunjung.median()
```




    57.5




```python
# bandingkan dengan mean
pengunjung.mean()
```




    147.5



Untuk nilai desimal, apabila tidak sesuai dengan konteks bisnis (Pengunjung), **chaining** hasil dengan fungsi rounding `round()`


```python
pengunjung.mean().round()
```




    148.0



### Modus (Mode)

Modus berguna untuk mencari nilai yang paling sering muncul (frekuensi tertinggi).

- Modus digunakan untuk data kategorik
- Fungsi pada Python: `mode()`

**Contoh:**

Berasal dari `Ethnicity` mana nasabah di Bank tersebut paling banyak berasal? 


```python
# code here
cc['Ethnicity'].mode()
```




    0    Caucasian
    Name: Ethnicity, dtype: object



> Modus untuk `Ethnicity` adalah ____

### Knowledge Check

Dari pernyataan berikut, jawablah benar atau salah. Apabila salah tuliskan pernyataan yang benar.

1. Median adalah pusat data yang hanya melibatkan sebagian data dalam perhitungannya.

    - [ ] Benar
    - [x] Salah

2. Mean adalah pusat data yang sensitif terhadap outlier.

    - [x] Benar
    - [ ] Salah

3. Nilai pusat data yang cocok untuk tipe data kategorik adalah modus.

    - [x] Benar
    - [ ] Salah

## Measure of Spread

Ukuran penyebaran data mewakili seberapa menyebar atau beragam data kita.

### Variance

Variance menggambarkan seberapa beragam suatu data numerik tunggal menyebar dari pusat datanya

- Formula variance:

$$var = \frac{\sum(X_i - \bar{X})^2}{n-1}$$

- Variance tidak dapat diinterpretasikan karena satuannya dalam kuadrat.
- Fungsi di Python: `var()`

![](assets/var.PNG)

**Contoh:**

Bank BRI sedang dalam rencana membuka kantor cabang baru. Bank BRI menyeleksi daerah mana yang cocok untuk cabang baru mereka. Mereka mengumpulkan informasi harga sewa bangunan di daerah A dan B sebagai berikut: 


```python
# harga dalam satuan juta
harga_A = pd.Series([400,410,420,400,410,420,400,410,420,400,410,420,400]) 
harga_B = pd.Series([130,430,650,540,460,320,380,550,650,470,330,140,270]) 
```

Bandingkan rata-rata harga bangunan kedua daerah:


```python
# code here
print(harga_A.mean())
print(harga_B.mean())
```

    409.2307692307692
    409.2307692307692
    

Mari bandingkan dari sisi lain, yaitu tingkat keberagaman data (variance). Daerah mana yang harganya lebih bervariasi?


```python
# code here
print(harga_A.var())
print(harga_B.var())
```

    74.35897435897435
    28707.692307692312
    

Daerah manakah yang lebih baik untuk dijadikan area perkantoran?

> ...

**Karakteristik Variance**

- Skala variance dari 0 sampai tak hingga. Semakin besar nilainya maka artinya semakin menyebar dari pusat datanya (mean).

- Variance memiliki satuan kuadrat, sehingga tidak dapat langsung diinterpretasikan. Biasanya digunakan untuk membandingkan dengan nilai var lain dengan satuan yang sama.

- **Nilai variansi sangat bergantung dengan skala data**. Hati-hati apabila membandingkan antar variabel yang berbeda skala.

### Standard Deviation

Standard deviation menggambarkan **seberapa jauh simpangan nilai yang dianggap umum, dihitung dari titik pusat (mean) nya.** Kita dapat menentukan apakah suatu nilai dikatakan menyimpang dari rata-rata namun masih dikatakan umum, atau sudah tidak umum. 

Karena dihitung dengan **mengakarkan variance**, satuannya sudah sesuai dengan data asli dan bisa diinterpretasikan.

* Formula standar deviasi: $$sd = \sqrt{var}$$
* Fungsi di Python: `std()`


```python
# standar deviasi harga_A & harga_B
print(harga_A.std())
print(harga_B.std())
```

    8.623164985025761
    169.43344506824002
    


```python
# tinjau nilai mean harga_A & harga_B
print(harga_A.mean())
print(harga_B.mean())
```

    409.2307692307692
    409.2307692307692
    


```python
print(harga_A.mean()-harga_A.std(), harga_A.mean()+harga_A.std())
print(harga_B.mean()-harga_B.std(), harga_B.mean()+harga_B.std())
```

    400.60760424574346 417.853934215795
    239.7973241625292 578.6642142990092
    

**Interpretasi nilai normal/wajar : mean +- sd** (karena satuan mean dan sd sama, yaitu jutaan rupiah)

- Harga sewa pada daerah A umumnya jatuh pada interval 400.60760424574346 417.853934215795
- Harga sewa pada daerah B umumnya jatuh pada interval 239.7973241625292 578.6642142990092

**Business question**

Apabila kita ditawarkan suatu bangunan di daerah B dengan harga 800, apakah harga tersebut masih wajar? Apakah sebaiknya kita membeli bangunan tersebut? Hubungkan dengan nilai mean dan standar deviasi yang diperoleh.

Hitung range "harga normal" daerah B:


```python
# code here
print(harga_B.mean()-harga_B.std(),'-', harga_B.mean()+harga_B.std())
```

    239.7973241625292 - 578.6642142990092
    

> ...

### Range using `boxplot()`

Distribusi data numerik pada umumnya divisualisasikan dengan `boxplot()`, yang meliputi komponen:

- Box: menggambarkan Q1, Q2 (median), dan Q3
  + Kuartil 1 (Q1): nilai ke 25%
  + Kuartil 2 (Q2 atau median): nilai ke 50% (nilai tengah)
  + Kuartil 3 (Q3): nilai ke 75%
  + Interquartile Range (IQR): selisih antara Q3 dan Q1
- Whisker: pagar bawah dan atas (PENTING: hati-hati, nilai ini bukan nilai minimum dan maksimum data)
- Data outliers: nilai ekstrim data yang berada di luar pagar bawah dan atas

![](assets/boxplot.PNG)

Beberapa hal yang harus diperhatikan dalam boxplot:

- Banyaknya data dari Q1 ke nilai minimum (bukan pagar bawah) adalah 25%
- Banyaknya data dari Q1 ke Q2 adalah 25%
- Banyaknya data dari Q2 ke Q3 adalah 25%
- Banyaknya data dari Q3 ke nilai maksimum (bukan pagar atas) adalah 25%

Insight yang dapat diperoleh dari boxplot:

1. Pusat data dengan median (Q2)
2. Sebaran data dengan IQR (lebar kotak)
3. Outlier, nilai ekstrim pada data
4. Bentuk distribusi data:
  + box yang berada ditengah = **distribusi normal**
  + box yang mendekati batas bawah = **distribusi skewed kanan**
  + box yang mendekati batas atas = **distribusi skewed kiri**

**Contoh:**

Visualisasikan sebaran data `Rating` dari data `cc`! Analisis informasi yang didapatkan.


```python
cc['Rating'].plot.box(vert=False)
```




    <Axes: >




    
![png](output_45_1.png)
    



```python
from matplotlib.cbook import boxplot_stats

bp_stats = boxplot_stats(cc['Rating'].values)

bp_stats
```




    [{'mean': 354.94,
      'iqr': 190.0,
      'cilo': 329.085,
      'cihi': 358.915,
      'whishi': 721,
      'whislo': 93,
      'fliers': array([949, 828, 728, 750, 805, 730, 817, 982, 754, 832, 747], dtype=int64),
      'q1': 247.25,
      'med': 344.0,
      'q3': 437.25}]




```python
cc[cc['Rating'].isin(bp_stats[0]['fliers'])]
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>186.63</td>
      <td>13414</td>
      <td>949</td>
      <td>2</td>
      <td>41</td>
      <td>14</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>African American</td>
      <td>1809</td>
    </tr>
    <tr>
      <th>85</th>
      <td>152.30</td>
      <td>12066</td>
      <td>828</td>
      <td>4</td>
      <td>41</td>
      <td>12</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>1779</td>
    </tr>
    <tr>
      <th>139</th>
      <td>107.84</td>
      <td>10384</td>
      <td>728</td>
      <td>3</td>
      <td>87</td>
      <td>7</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>African American</td>
      <td>1597</td>
    </tr>
    <tr>
      <th>174</th>
      <td>121.83</td>
      <td>10673</td>
      <td>750</td>
      <td>3</td>
      <td>54</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>African American</td>
      <td>1573</td>
    </tr>
    <tr>
      <th>184</th>
      <td>158.89</td>
      <td>11589</td>
      <td>805</td>
      <td>1</td>
      <td>62</td>
      <td>17</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>1448</td>
    </tr>
    <tr>
      <th>193</th>
      <td>130.21</td>
      <td>10088</td>
      <td>730</td>
      <td>7</td>
      <td>39</td>
      <td>19</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>1426</td>
    </tr>
    <tr>
      <th>293</th>
      <td>140.67</td>
      <td>11200</td>
      <td>817</td>
      <td>7</td>
      <td>46</td>
      <td>9</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>African American</td>
      <td>1677</td>
    </tr>
    <tr>
      <th>323</th>
      <td>182.73</td>
      <td>13913</td>
      <td>982</td>
      <td>4</td>
      <td>98</td>
      <td>17</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>1999</td>
    </tr>
    <tr>
      <th>347</th>
      <td>160.23</td>
      <td>10748</td>
      <td>754</td>
      <td>2</td>
      <td>69</td>
      <td>17</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Caucasian</td>
      <td>1192</td>
    </tr>
    <tr>
      <th>355</th>
      <td>180.68</td>
      <td>11966</td>
      <td>832</td>
      <td>2</td>
      <td>58</td>
      <td>8</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>African American</td>
      <td>1405</td>
    </tr>
    <tr>
      <th>390</th>
      <td>135.12</td>
      <td>10578</td>
      <td>747</td>
      <td>3</td>
      <td>81</td>
      <td>15</td>
      <td>Female</td>
      <td>No</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>1393</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import statistics 
  
# Plot between -10 and 10 with .001 steps. 
x_axis = cc['Rating'].sort_values()
  
# Calculating mean and standard deviation 
mean = statistics.mean(x_axis) 
sd = statistics.stdev(x_axis) 
  
plt.plot(x_axis, norm.pdf(x_axis, mean, sd)) 
plt.show() 
```


    
![png](output_48_0.png)
    


- Apakah data memiliki outlier?

> YA

- Central tendency (mean, median, modus) mana yang cocok dipakai untuk data ini?

> Median

- Bagaimana bentuk distribusi data?

> Mengekor panjang di nilai yang tinggi (Right skewed / Positive skewed)

## Variable Relationship

Karena pada data kita punya banyak kolom atau variabel, kita juga ingin tahu hubungan antar variabel dalam data kita.

Ukuran yang digunakan untuk melihat **hubungan linear** antara dua variabel numerik.

*Two variables is associated if distribution of one depends on the value of the other. We can look the association with visualization or statistic measure.*

### [Scatter Plot](https://www.data-to-viz.com/graph/scatter.html)

Sebuah scatterplot menampilkan hubungan antara 2 variabel numerik. Untuk setiap titik data, nilai variabel pertama direpresentasikan pada sumbu X, yang kedua pada sumbu Y.

Contoh: scatter plot antara data Rating dan Balance nasabah pada data `cc`:


```python
# code here
cc[['Rating', 'Balance']].plot(kind='scatter', x='Rating', y='Balance')
```




    <Axes: xlabel='Rating', ylabel='Balance'>




    
![png](output_52_1.png)
    


Tips: Untuk memutuskan variabel mana yang akan ditempatkan pada sumbu x dan mana yang akan ditempatkan pada sumbu y, tampilkan variabel yang ingin Anda jelaskan atau prediksi sepanjang sumbu y.

**Bagaimana cara mengetahui bahwa scatter plot memiliki asosiasi?**

- tes visual: bandingkan dengan scatter plot lain dengan random data (tanpa pola). jika scatter plot asli terlihat ada pola, berarti ada asosiasi.

**Bagaimana kita menentukan asosiasi/hubungannya?**

- Arah: trend polanya naik, turun, atau keduanya?
- Lengkungan: apakah polanya linier atau melengkung?
- Variasi: Apakah titik-titiknya tersusun rapat di sepanjang pola?
- Outliers/Pencilan: Apakah Anda menemukan sesuatu yang tidak terduga?


```python
# dummy random data
X = np.random.randn(100)
Y = np.random.randn(100)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(X, Y)
ax2.scatter(cc.Rating, cc.Balance);
```


    
![png](output_54_0.png)
    



### Covariance

Covariance menunjukkan bagaimana variansi 2 data (variable yang berbeda) bergerak bersamaan

* Formula Covariance: $$Cov(X, Y) = \frac{1}{n-1}\sum\limits^n_{i=1}(X_i - \mu_X)(Y_i - \mu_Y)$$
* Fungsi di Python: `cov()`

![](assets/covariance-positive-vs-negative.jpg)

- Nilai covariance positif mengindikasikan pergerakan nilai yang searah / berbanding lurus.
- Nilai covariance negatif mengindikasikan pergerakan nilai yang berbalik arah.

**Contoh:**

Hitunglah covariance antara `Income` dengan `Rating` pada data `cc` . Bagaimana hubungannya?


```python
# code here
cc[['Income', 'Rating']].cov()
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
      <th>Income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>1242.16</td>
      <td>4315.49</td>
    </tr>
    <tr>
      <th>Rating</th>
      <td>4315.49</td>
      <td>23939.56</td>
    </tr>
  </tbody>
</table>
</div>



Interpretasi nilai: Covariance 4000 dan positif

> Kelemahan: Seperti variance, covariance tidak memiliki batasan nilai untuk mengukur kekuatan hubungan antar dua variabel (-inf s.d inf), sehingga kita hanya bisa mengetahui apakah hubungannya positif atau negatif. Oleh karena itu, hadir **correlation**.

### Correlation

Correlation memampatkan nilai covariance dari -inf s.d inf menjadi **-1 s.d 1** sehingga bisa diukur kekuatan hubungan antar data (variable).

* Formula Correlation: 

$$Cor(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$$
* Fungsi di Python: `corr()`

- Nilai korelasi mengindikasikan kekuatan hubungan antara dua variable numerik sebagai berikut:
![](assets/correlation-coef.jpg)

Bila korelasi dua variable numerik mendekati:
  - -1 artinya korelasi negatif kuat
  - 0 artinya tidak berkorelasi
  - 1 artinya korelasi positif kuat

**Contoh:**

Adakah korelasi antara `Income` dengan `Rating` pada data `cc` . Bagaimana hubungan dan kekuatannya?



```python
# code here
cc[['Income', 'Rating']].corr()
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
      <th>Income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>1.00</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>Rating</th>
      <td>0.79</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



Jawaban: Korelasi positif dan cukup tinggi

Visualisasi korelasi dengan scatter plot:


```python
cc.plot.scatter(x='Income',
                y='Rating');
```


    
![png](output_64_0.png)
    


[Optional] Visualisasi korelasi dengan heatmap:


```python
sns.heatmap(cc.select_dtypes(include='number').corr(), # nilai korelasi
            annot=True,   # anotasi angka di dalam kotak heatmap
            fmt=".3f",    # format 3 angka dibelakang koma 
            cmap='Blues'); # warna heatmap
```


    
![png](output_66_0.png)
    


Ilustrasi correlation:

![](assets/correlation.png)

### Knowledge Check

Dari pernyataan berikut, jawablah benar atau salah. Apabila salah, tuliskan pernyataan yang benar.

1. Ketika korelasi variabel A dan B bernilai -1 artinya tidak ada korelasi antara nilai A dan B.

    - [ ] Benar
    - [x] Salah

2. Scatter plot dapat digunakan untuk menggambarkan hubungan antara dua variabel numerik.

    - [X] Benar
    - [ ] Salah

**Quick Summary Descriptive Statistics**

- **Central Tendency** = Mean, Median, Mode
  + Untuk data numerik: Mean, Median
    - Ketika terdapat outlier: Median
    - Ketika tidak terdapat outlier: Mean
  + Untuk data kategorik : Mode
  
- **Measure of Spread** = Variance, Standard Deviation
  + Variance Sulit diinterpretasi secara sendirinya
  + Standard deviation memiliki satuan data asli, sehingga bisa diinterpretasi
  + Range dan IQR dapat divisualisasikan dengan Boxplot
  
- **Variable Relationship**: Covariance, Correlation
  + Covariance hanya bisa dilihat nilai positif dan negatifnya. 
    - Rangenya `-inf` s.d `inf`
  + Correlation digunakan untuk mengukur kekuatan hubungan
    - Rangenya -1 s.d 1
    - semakin mendekati -1 $\rightarrow$ Korelasi berbanding terbalik
    - semakin mendekati 1 $\rightarrow$ Korelasi berbanding lurus
    - semakin mendekati 0 $\rightarrow$ Tidak ada korelasi

---

# Inferential Statistics

Inferential Statistics membantu kita **menarik kesimpulan tentang keseluruhan data (populasi) dengan menggunakan sebagian informasinya saja (sampel)**

![](assets/statistical_cycle.png)

Setiap data memiliki distribusi. Distribusi data yang spresial dan berperan dalam inferential statistics adalah **distribusi normal**

## Normal Distribution

![](assets/normal-distribution.jpg)

Karakteristik:

- Kurva membentuk lonceng simetris, artinya puncaknya adalah titik pusat (mean = median)
- Luas area dibawah kurva = 1 (menyatakan probabilitas)
- Persebaran data:
  + 68% data berada di rentang +- 1 standar deviasi dari mean
  + 95% data berada di rentang +- 2 standar deviasi dari mean
  + 99.7% data berada di rentang +- 3 standar deviasi dari mean
- **Standar normal baku** adalah distribusi normal dimana mean = 0 dan standar deviasi = 1. 

Distribusi normal banyak digunakan pada inferensial statistik karena dicetuskannya **Central Limit Theorem**.

> Semakin bertambahnya jumlah sampel yang diambil secara acak, maka **distribusi rata-rata sampel** akan mengikuti distribusi normal

Karakteristik distribusi normal inilah yang dimanfaatkan untuk penghitungan inferensial statistik:

- **Menghitung Probabilitas:**
  + Probability Mass Function $\rightarrow$ diskrit/kategorik
  + Probability Density Function $\rightarrow$ kontinu/numerik
- **Membuat Confidence Interval**
- **Uji Hipotesis**

## Probability Mass Function

* Menghitung peluang untuk data diskrit, contoh:
  + peluang hujan/tidak hujan
  + peluang produk yang terjual
  + peluang nasabah good credit/bad credit
  
* Formula: jumlah kejadian terjadi dibagi dengan jumlah kejadian total

**Contoh:**

Terdapat 100 nasabah dari sebuah Bank, 90 diantaranya merupakan nasabah dengan status good (good credit), sedangkan sisanya sebanyak 90 adalah status bad (bad credit). Berapakah peluang nasabah bad credit?



```python
# code here

```

## Probability Density Function

- Menghitung probability data **kontinu**. Data kontinu merupakan data yang memiliki nilai dalam rentang tertentu, dan bisa memiliki angka desimal atau pecahan, contohnya:
  + tinggi badan
  + rating nasabah
  + profit/revenue

- Tahapan:
  1. Hitung Z-score (ubah nilai data asli ke standar normal baku = Z-score standardization)
  2. hitung peluang berdasarkan Z-score dengan menggunakan fungsi `pnorm()`


-  Formula Z-score:

$$Z = \frac{x-\mu}{\sigma}$$

Keterangan:

- Z = Z-score
- x = titik data
- $\mu$ = mean
- $\sigma$ = standar deviasi

> Z-score merupakan sebuah nilai yang merepresentasikan **berapa standard deviasi data tersebut menyimpang dari rata-ratanya**
  

**Contoh**

Tinggi badan pria dewasa di Indonesia berdistribusi normal dengan rata-rata 165 cm dan standar deviasi 10 cm. Berapa peluang pria dewasa di Indonesia memiliki tinggi badan > 180 cm?

Diketahui:

- mean = 165
- stdev = 10
- titik data = 180cm 


```python
mean = 165
std = 10
titik_data = 180
```


```python
# code here
from scipy.stats import norm

# hitung Z-score lalu ubah jadi peluang
Z_score = (titik_data - mean) / std
Z_score
```




    1.5



Menghitung peluang dengan [`norm.cdf()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)


```python
# menghitung peluang
peluang = 1 - norm.cdf(Z_score) # Karena yang dicari peluang dibagian kanan data > 180
peluang
```




    0.06680720126885809




```python
# membuat dummy normal distribution
x = np.linspace(-3, 3, 1000000)
y = norm.pdf(x)

# Plotting
plt.ylabel('Density')
plt.xlabel('Standard Deviations')
plt.plot(x, y);
plt.axvline(x.mean(), color = "red", linestyle = "--"); #garis rata-rata
```


    
![png](output_80_0.png)
    


*Cumulative distribution function of the given Random Variable* $\rightarrow$ cdf adalah fungsi yang digunakan untuk menghitung area/luas di bawah kurva dari titik z sampai ujung kiri kurva

Ditanya: Berapa peluang pria dewasa di Indonesia memiliki tinggi badan > 180 cm?


```python
# menghitung peluang area yang ditanya
1 - norm.cdf((180-mean)/std)
```




    0.06680720126885809



Insight: Peluang pria dewasa di Indonesia memiliki tinggi badan > 180 cm > 94.52%

**Dive deeper!**

Rata-rata income wanita dewasa = 10 juta dengan standar deviasi = 3 juta.

Peluang wanita dewasa di indonesia yang rata-rata incomenya < 5 juta?



```python
print("Peluang wanita dewasa di indonesia yang rata-rata incomenya < 5 juta adalah" , norm.cdf((5000000 - 10000000) / 3000000))
```

    Peluang wanita dewasa di indonesia yang rata-rata incomenya < 5 juta adalah 0.0477903522728147
    

## Confidence Interval

Confidence interval (selang kepercayaan) berguna untuk menduga nilai mean populasi dengan sebuah interval. Menebak dengan sebuah interval akan meminimalisir error dibandingkan hanya dengan menebak satu nilai.

-  Formula: 

$$CI = \bar{x} \pm Z_{\frac{\alpha}{2}}*SE$$

- Keterangan: 
  + $\bar{x}$ = rata-rata sampel
  + $Z_{\frac{\alpha}{2}}$ = Z-score ketika alpha/2
  + $\alpha$ = tingkat error yang ditolerasi
  + tingkat kepercayaan = 1-$\alpha$
  + SE = standard error
  
SE mengukur kebaikan sampel dalam mewakilkan populasi. Semakin kecil, maka sampel semakin representatif (baik).

$$SE = \frac{\sigma}{\sqrt n}$$

- Ket: 
  + $\sigma$ = standar deviasi populasi
  + $n$ = jumlah sampel

* Tahapan:
  + hitung mean sampel
  + hitung standar deviasi sampel & SE
  + tentukan tingkat kepercayaan & $\alpha$
  + tentukan Z alpha/2
  + hitung confidence interval

**Contoh**

Dari data `cc` yang berisikan sampel **400 nasabah** suatu Bank diketahui memiliki rata-rata `Balance` kredit sebesar **520**. Semisal diketahui Bank tersebut memiliki standard deviasi populasi untuk Balance sebesar **465**.

Berapakah confidence interval untuk rata-rata Balance seluruh nasabah? Gunakan tingkat kepercayaan 95%!

1. Diketahui:

- mean sampel = 520  
- stdev populasi = 465
- jumlah sampel (n) = 400

2. Hitung nilai SE



```python
# code here
import math
from scipy import stats

# std populasi dibagi akar n
SE = 520 / (400**0.5)
SE
```




    26.0



2. Tentukan tingkat kepercayaan dan alpha

- Tingkat kepercayaan: 95%
- alpha (tingkat error): 100% - 95% = 5%, artinya kita mentoleransi error sebesar 5%, bahwa mungkin saja rata-rata Balance nasabah aslinya terletak di luar Confidence Interval


```python
alpha = 0.05

print('alpha: ', alpha)
```

    alpha:  0.05
    

3. Hitung Z alpha/2

alpha dibagi 2 karena ingin membuat batas bawah dan batas atas (dalam dunia statistika dikenal sebagai two-tailed)


```python
# code here
from scipy.stats import norm

# luas di bawah kurva - kedua bagian hijau
Z = alpha/2
```

**Notes:**

- `norm.cdf()` untuk mencari peluang (x) dari sebuah titik/nilai z di distribusi normal baku (q)
- `norm.ppf()` untuk mencari titik/nilai z di disribusi normal baku (q) dari sebuah peluang (x)

![](assets/two-tailed.png)

4. Hitung confidence interval

`CI = mean -+ (Z * SE)`


```python
lower = mean - (Z*SE)
upper = mean + (Z*SE)
print(lower, upper)
```

    164.35 165.65
    

Kesimpulan: Confidence Interval rata-rata 164.35-165.65

## Hypothesis Testing

Uji hipotesis bertujuan untuk menguji **dugaan**. Uji hipotesis sering disebut juga sebagai **uji signifikansi** yang digunakan untuk menguji apakah suatu treatment memberikan perubahan/pengaruh signifikan terhadap suatu kondisi.

Istilah-istilah:

- Hipotesis: dugaan sementara yang harus diuji
  + $H_0$ / null hypothesis: 
    * kondisi awal
    * memiliki unsur kesamaan (=, >=, <=)
  + $H_1$ / alternative hypothesis: 
    * kontradiktif dengan $H_0$
    
- $\alpha$:
  + tingkat signifikansi yaitu tingkat error yang masih bisa ditoleransi
  + umumnya 0.05
- $1-\alpha$: tingkat kepercayaan

- $p-value$:
  + hasil perhitungan statistik yang menunjukkan peluang data sampel terjadi dengan kondisi H0.

Pengambilan kesimpulan:

- Jika $p-value$ < $\alpha$, maka tolak $H_0$ -> terima h1
- Jika $p-value$ > $\alpha$, maka gagal tolak $H_0$ -> terima h0

**Contoh Hipotesis**

1. Hipotesis dua arah (!=)

- $H_0$ : Rata-rata saldo rekening tidak berbeda secara signifikan antara nasabah yang menggunakan layanan internet banking dan yang tidak menggunakan layanan tersebut. (=)
- $H_1$ : Rata-rata saldo rekening **berbeda secara signifikan** antara nasabah yang menggunakan layanan internet banking dan yang tidak menggunakan layanan tersebut. (!=)

2. Hipotesis satu arah (<)

- $H_0$ : Penambahan teller tidak memberikan perbedaan durasi pembayaran (>=)
- $H_1$ : Penambahan teller **menurunkan** durasi pembayaran (<)

3. Hipotesis satu arah (>)

* $H_0$: Penerapan diskon tidak memberikan perbedaan jumlah pembelian produk (<=)
* $H_1$: Penerapan diskon **meningkatkan** jumlah pembelian produk (>)

### Z-Test

Uji hipotesis yang menggunakan Z-test bila:

- standar deviasi populasi diketahui
- jumlah sampel banyak (n > 30)

**Contoh**

BRI merupakan salah satu Bank terbaik di Indonesia. Bila diketahui rata-rata likes dari suatu post di platform mereka sebesar **14000** likes dengan standar deviasi **5000** likes.

Demi meningkatkan likes dari tiap post, BRI memutuskan untuk menggunakan influencer sebagai brand ambassador pemasaran produk. Setelah menggunakan influencer, diambil **50** postingan acak yang ternyata memiliki rata-rata likes **17500**.

Sebagai tim marketing, lakukan analisis apakah menggunakan jasa influencer secara signifikan meningkatkan customer engagement (dari sisi rata-rata jumlah likes) atau tidak? Gunakan tingkat kepercayaan **95%**.


Jawaban:

**I. Tentukan hipotesis**

- $H_0$: Tidak meningkatkan customer engagement
- $H_1$: Meningkatkan customer engagement


**II. Hitung nilai statistik**

Diketahui deskriptif statistiknya:

- mean populasi   = 14000
- stdev populasi  = 5000
- n               = 50
- mean sampel     = 17500

Ditentukan oleh user:

- tingkat kepercayaan = 95%
- alpha               = 5%



```python
# nilai statistic descriptive
mean_populasi = 14000
std_populasi = 5000
n = 50
mean_sample = 17500

print('mean_populasi: ', mean_populasi)
print('std_populasi: ', std_populasi)
print('n: ', n)
print('mean_sample: ', mean_sample)
```

    mean_populasi:  14000
    std_populasi:  5000
    n:  50
    mean_sample:  17500
    

$$Z = \frac{\bar X-\mu}{SE}$$

Z = (rata2 sampel - rata2 populasi) / standar error

$$SE = \frac{\sigma}{\sqrt n}$$

Standar error = standar deviasi populasi / akar dari banyak sampel


```python
# menghitung nilai SE
SE = std_populasi / (n**0.5)

# menghitung nilai z
Z = (mean_sample - mean_populasi) / SE
print(SE, Z)
```

    707.1067811865476 4.949747468305833
    

SE mengukur kebaikan sampel dalam mewakilkan populasi. Semakin kecil, maka sampel semakin representatif (baik).

Z-score merupakan sebuah nilai yang merepresentasikan **berapa standard deviasi data tersebut menyimpang dari rata-ratanya**


```python
# luas keseluruhan daerah di bawah kurva - luas Z
p_value = 1 - norm.cdf(Z)
p_value
```




    3.7154918619553e-07




```python
#formatting the result
result=f'{p_value:.10f}'
print("The result is:",result)
```

    The result is: 0.0000003715
    

Daerah/area yang kita ingin cari dilihat dari tanda ketidaksamaan di hipotesis alternatif yaitu:

* $H_1$: Penggunaan influencer meningkatkan customer engagement (>)

![](assets/p-value.PNG)

**c. Bandingkan P-value dengan alpha**

Pengambilan kesimpulan:

* Jika $p-value$ < $\alpha$, maka tolak $H_0$ 
* Jika $p-value$ > $\alpha$, maka gagal tolak $H_0$

p-value = 0.0000003715 < alpha = 0.05 maka tolak $H_0$ dan terima $H_1$ terjadi peningkatan signifikan

**IV. Kesimpulan**

> Tolak null Hypthesis, terjadi peningkatan signifikan


**[Additional]: Menggunakan fungsi [`ztest()`](https://www.statsmodels.org/devel/generated/statsmodels.stats.weightstats.ztest.html)**

Gunakan fungsi `ztest()` untuk menghitung z-statistics dan p-value jika data disimpan dalam bentuk Dataframe.

```python
zstats, pval = ztest(x1=...,
                     value = ...,
                     alternative = ...)
```
parameter :
- `x1` : number of observations
- `value` : rata-rata dari x1 di $H_0$
- `alternative` :
   - jika $H_1$ tidak sama (!=) dengan nilai tertentu, isi dengan `two-sided`
   - jika $H_{1}$ lebih besar (>) dari suatu nilai, gunakan `larger`
   - jika $H_{1}$ lebih kecil (<) dari suatu nilai, gunakan `smaller`

Contoh aplikasi:

```python
from statsmodels.stats.weightstats import ztest 

mean_data = nama_dataframe['nama spesifik kolom'].mean()
z_test = ztest(x1=nama_dataframe['nama spesifik kolom'], # data kita
               value=mean_data,  #rata-rata 
               alternative='larger')

p_value = z_test[1] # ambil nilai p_value dari z_test

```


```python
from statsmodels.stats.weightstats import ztest 

mean_data = cc['Rating'].mean()
z_test = ztest(x1=cc['Rating'], # data kita
               value=mean_data,  #rata-rata 
               alternative='larger') #Pemmilihan alternative hypothesisnya

p_value = z_test[1] # ambil nilai p_value dari z_test
p_value
```




    0.5



### T-test

Uji hipotesis menggunakan T-test jika:

* standar deviasi populasi tidak diketahui atau
* jumlah sampel sedikit (n <= 30)

Bentuk t-distribution mirip dengan normal distribution, hanya saja lebih landai ketika jumlah sampel sedikit:

![](assets/t-distribution.jpg)

**Contoh Kasus**

Mari kita asumsikan Bank BRI memiliki dua kelompok nasabah bank, yaitu kelompok yang memiliki behavior scoring tinggi dan kelompok yang memiliki behavior scoring rendah. 

Diketahui data saldo rekening antara kedua kelompok sebagai berikut:


```python
behavior_score_high = pd.Series([30.4, 52.7, 70.6, 55.7, 56.3, 34.2, 59.6, 42.3, 21.1, 50.5, 12.2, 58.6, 12.0, 56.1, 49.4, 60.9, 60.0, 35.3, 15.0, 50.3])

behavior_score_low = pd.Series([6.5, 13.3, 6.8, 9.2, 10.0, 1.5, 21.7, 16.2, 5.9, 25.0, 18.4, 12.6, 22.2, 22.0, 21.6, 20.5, 19.4, 14.5, 12.6, 12.0])
```

Tujuan kita adalah menguji **apakah terdapat perbedaan signifikan dalam rata-rata saldo rekening antara kedua kelompok tersebut?**

Jawab:

**I. Tentukan hipotesis**

- $H_0$: Rata-rata saldo rekening antara kelompok dengan behavior scoring tinggi dan kelompok dengan behavior scoring rendah tidak berbeda secara signifikan. 
- $H_1$: Terdapat perbedaan signifikan dalam rata-rata saldo rekening antara kedua kelompok.

**II. Hitung P-value dengan [`ttest_ind()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)**

Gunakan fungsi `ttest_ind()` untuk menghitung t-statistics dan p-value dua independent sample

```python
t_stats, pval = ttest_ind(a= ...,
                          b= ...,
                          alternative = ...)
```

parameter:

- `a` : data atau observasi sampel berbentuk Series atau array
- `b` : data atau observasi sampel berbentuk Series atau array
- `alternative` : tergantung hypothesis alternative ($H_1$)
    - jika $H_1$ tidak sama  (!=) dengan nilai tertentu, isi dengan `two-sided`
    - jika $H_1$ kurang dari (<) dengan nilai tertentu, isi dengan `less`
    - Jika $H_1$ lebih besar (>) dengan nilai tertentu, isi dengan `greater` 


```python
# code here
from scipy import stats

t_test = stats.ttest_ind(behavior_score_high, behavior_score_low, alternative= 'two-sided')
p_value = t_test[1]
p_value
```




    3.087655217734725e-08




```python
#formatting the result
result=f'{p_value:.10f}'
print("The result is=",result)
```

    The result is= 0.0000000309
    

**III. Bandingkan P-value dengan alpha**

Dalam membuat keputusan uji statistik, kita dapat membandingan p-value dengan alpha:

- selang kepercayaan = 95%
- alpha = 5% -> 0.05

p-value = 0.0000000309 < alpha = 0.05, maka $H_0$ ditolak


**IV. Kesimpulan**

Dengan menggunakan tingkat kepercayaan 95% dapat disimpulkan penentuan $H_0$ ditolak

**Summary penggunaan hipotesis testing:**

![](assets/uji_hipotesis_mean.png)

# Further Readings

- Descriptive Statistics: https://courses.lumenlearning.com/suny-natural-resources-biometrics/chapter/chapter-1-descriptive-statistics-and-the-normal-distribution/

- Dealing with small data set: https://measuringu.com/small-n/

- t-Distribution and some case examples: https://stattrek.com/probability-distributions/t-distribution.aspx
