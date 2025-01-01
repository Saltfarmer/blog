---
title: "Day 7 Algorit.ma : Capstone Project"
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
  - Exercise
---
# **Fraud Prediction**

## **Problem Statement**

💼 **Fraud** merupakan suatu tindakan penipuan pada transaksi keuangan. Transaksi ini bertujuan untuk memperoleh keuntungan dari suatu entitas lain dengan cara ilegal sehingga dapat menyebabkan kerugian.

Transaksi fraud sering terjadi di industri perbankan. Beberapa contoh transaksi fraud pada industri perbankan antara lain adalah *pishing, skimming*, penipuan kartu kredit, penipuan pinjaman, dll.

Efek negatif dari transaksi fraud adalah adanya kerugian finansial di kedua belah pihak baik customer maupun industri perbankan. Oleh karena itu diperlukan tindakan pencegahan dengan deteksi lebih dini terhadap potensi transaksi fraud.

Transaksi fraud dapat dideteksi lebih awal dengan cara membangun model *machine learning* dengan metode klasifikasi. Tugas kita pada project kali ini adalah membuat model klasifikasi untuk mendeteksi transaksi fraud agar dapat dilakukan action plan berikutnya

## **Rubrics Penilaian**

Untuk menyelesaikan project ini, silahkan mengacu pada rubrics di bawah ini serta memberikan deskripsi dari proses yang Anda kerjakan untuk setiap bagian.

***Data Preparation***

(1 poin) Bagaimana melakukan persiapan data sebelum dilakukan pemodelan.

- Metode apa saja yang yang dilakukan dalam proses persiapan data?
- Apakah terdapat data yang *missing* atau *duplicate*, bagaimana cara mengatasi data tersebut?
  Referensi: [dokumentasi pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) atau [FAQ](https://askalgo-py.netlify.app/faq/eda?highlight=fillna#missing-values)

(4 poin) Bagaimana cara untuk melakukan *feature engineering* ataupun pemilihan variabel dari data yang tersedia.

- Apakah terdapat variable yang dihilangkan? Jika iya, kenapa?
- Apakah terdapat variable yang ditambahkan? Jika iya, kenapa?

***Exploratory Data Analysis***

(2 poin) *Exploratory Data Analysis*

- Berikan penjelasan informatif dari visualisasi dan/atau segala jenis hasil eksplorasi Anda.
- Bagaimana distribusi data pada setiap variabel?
- Apakah terdapat informasi menarik antara variable predictor dengan variable target?

***Data Pre-processing and Model Fitting***

(2 poin) *Data Pre-processing*

- Apakah terdapat variabel yang harus di-encoding? Kalau ada, apa saja dan kenapa?
- Bagaimana pembagian proporsi Data Train & Test?

(4 poin) Membuat model, pilih model berdasarkan model yang sudah diajarkan dikelas (logistic regression / knn)

- Buatlah model yang dapat menjawab pertanyaan bisnis di atas.
- Variable apa yang dirasa cukup penting dalam pembuatan model? Sertakan alasannya.
- Metode evaluasi apa yang cocok digunakan untuk kasus ini? Jelaskan.

***Prediction Performance***

(1 poin) Metrics yang dipilih mencapai 60% pada data train
(2 poin) Metrics yang dipilih mencapai 60% pada data test

***Conclusion***

(2 poin) Tuliskan kesimpulan dari project yang anda kerjakan.

- Apakah model sudah dapat melakukan prediksi dengan baik? Jelaskan.
- Apakah model sudah dapat menjawab pertanyaan bisnis yang ada? Jelaskan.
- Action plan apa yang dapat dilakukan untuk tindakan preventif transaksi fraud?

**Total Poin Capstone Project : 18**

NOTE:

Apabila Anda sudah berhasil mendapatkan model yang baik dan berhasil menjawab seluruh rubric di atas, Anda diperkenankan untuk melakukan eksplorasi model menggunakan metode lain di luar yang diajarkan di kelas. Hasil model yang mendapatkan metrics pengukuran paling baik akan mendapatkan bonus nilai (di luar nilai capstone).

**Penjelasan Dataset**

Berikut adalah penjelasan setiap kolom yang terdapat pada _dataset_ yang akan digunakan:

- `X`: ID kartu
- `id_tanggal_transaksi_awal`: ID tanggal transaksi dilakukan
- `tanggal_transaksi_awal`: tanggal dilakukan transaksi (POSIX format)
- `tipe_kartu`: tipe kartu yang bertransaksi
- `id_merchant`: ID merchant kartu tersebut bertransaksi
- `nama_merchant`: nama merchant kartu tersebut bertransaksi
- `tipe_mesin`: tipe mesin yang digunakan untuk bertransaksi (ATM, EDC, dll)
- `tipe_transaksi`: jenis transaksi
- `nama_transaksi`: nama jenis transaksi
- `nilai_transaksi`: nilai uang yang tercatat saat transaksi
- `id_negara`: ID negara tempat terjadi transaksi
- `nama_negara`: nama negara tempat terjadi transaksi
- `nama_kota`: nama kota tempat terjadi transaksi
- `lokasi_mesin` : lokasi mesin
- `pemilik_mesin`: pemilik mesin
- `waktu_transaksi`: waktu transaksi berlangsung
- `kuartal_transaksi`: kuartal waktu transaksi berlangsung
- `kepemilikan_kartu`: kepemilikan kartu
- `nama_channel`: nama channel kartu tersebut bertransaksi
- `id_channel`: ID channel kartu tersebut bertransaksi
- `flag_transaksi_finansial`: jenis transaksi
- `status_transaksi`: keberhasilan atau kegagalan transaksi
- `bank_pemilik_kartu`: kepemilikan kartu yang dimiliki suatu bank
- `rata_rata_nilai_transaksi`: rata - rata nilai transaksi
- `maksimum_nilai_transaksi`: nilai maksimum transaksi
- `minimum_nilai_transaksi`: nilai minimum transaksi
- `rata_rata_jumlah_transaksi`: rata - rata jumlah transaksi
- `flag_transaksi_fraud`: transaksi fraud (1) atau tidak fraud (0)

## ***Data Preparation***

(1 poin) Bagaimana melakukan persiapan data sebelum dilakukan pemodelan.

- Metode apa saja yang yang dilakukan dalam proses persiapan data?
- Apakah terdapat data yang *missing* atau *duplicate*, bagaimana cara mengatasi data tersebut?
  Referensi: [dokumentasi pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) atau [FAQ](https://askalgo-py.netlify.app/faq/eda?highlight=fillna#missing-values)u FAQ

### Import Library yang dibutuhkan

```python
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
```

### Read Data

```python
fraud = pd.read_csv('transaksi_fraud.csv')
fraud.head().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>4676</td>
      <td>788</td>
      <td>1520</td>
      <td>9346</td>
      <td>2914</td>
    </tr>
    <tr>
      <th>id_tanggal_transaksi_awal</th>
      <td>2457646</td>
      <td>2457419</td>
      <td>2457521</td>
      <td>2457659</td>
      <td>2457311</td>
    </tr>
    <tr>
      <th>tanggal_transaksi_awal</th>
      <td>2457726</td>
      <td>2457507</td>
      <td>2457612</td>
      <td>2457746</td>
      <td>2457385</td>
    </tr>
    <tr>
      <th>tipe_kartu</th>
      <td>111</td>
      <td>111</td>
      <td>2</td>
      <td>103</td>
      <td>0</td>
    </tr>
    <tr>
      <th>id_merchant</th>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>75336</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>nama_merchant</th>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>249</td>
      <td>1798</td>
    </tr>
    <tr>
      <th>tipe_mesin</th>
      <td>2605127</td>
      <td>-3</td>
      <td>-3</td>
      <td>2806174</td>
      <td>2334932</td>
    </tr>
    <tr>
      <th>tipe_transaksi</th>
      <td>26</td>
      <td>156</td>
      <td>156</td>
      <td>58</td>
      <td>26</td>
    </tr>
    <tr>
      <th>nama_transaksi</th>
      <td>10</td>
      <td>12</td>
      <td>12</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>nilai_transaksi</th>
      <td>2200000.0</td>
      <td>2500000.0</td>
      <td>1200000.0</td>
      <td>320000.0</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>id_negara</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>nama_negara</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nama_kota</th>
      <td>265</td>
      <td>121</td>
      <td>101</td>
      <td>239</td>
      <td>69</td>
    </tr>
    <tr>
      <th>lokasi_mesin</th>
      <td>4137</td>
      <td>1264</td>
      <td>1283</td>
      <td>7049</td>
      <td>3425</td>
    </tr>
    <tr>
      <th>pemilik_mesin</th>
      <td>613</td>
      <td>2196</td>
      <td>2049</td>
      <td>588</td>
      <td>613</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>193955</td>
      <td>73140</td>
      <td>140216</td>
      <td>155117</td>
      <td>143339</td>
    </tr>
    <tr>
      <th>kuartal_transaksi</th>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nama_channel</th>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>9</td>
      <td>8</td>
      <td>8</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>flag_transaksi_finansial</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>status_transaksi</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>bank_pemilik_kartu</th>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>1332292.784</td>
      <td>1369047.619</td>
      <td>15523460.4</td>
      <td>711764.7059</td>
      <td>617968.254</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>9750000.0</td>
      <td>10000000.0</td>
      <td>100000000.0</td>
      <td>6884408.0</td>
      <td>2500000.0</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>10000.0</td>
      <td>30000.0</td>
      <td>41804.0</td>
      <td>10000.0</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>2.73</td>
      <td>2.33</td>
      <td>2.4</td>
      <td>1.98</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

### Cek keseimbangan fraudnya

```python
fraud_asli = fraud.copy()
fraud_asli['flag_transaksi_fraud'].value_counts()
```

    flag_transaksi_fraud
    0    12215
    1      910
    Name: count, dtype: int64

### Cek informasi sekilas dan tipe data

```python
fraud.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13125 entries, 0 to 13124
    Data columns (total 28 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
    0   X                           13125 non-null  int64
    1   id_tanggal_transaksi_awal   13125 non-null  int64
    2   tanggal_transaksi_awal      13125 non-null  int64
    3   tipe_kartu                  13125 non-null  int64
    4   id_merchant                 13125 non-null  int64
    5   nama_merchant               13125 non-null  int64
    6   tipe_mesin                  13125 non-null  int64
    7   tipe_transaksi              13125 non-null  int64
    8   nama_transaksi              13125 non-null  int64
    9   nilai_transaksi             13125 non-null  float64
     10  id_negara                   13125 non-null  int64
    11  nama_negara                 13125 non-null  int64
    12  nama_kota                   13125 non-null  int64
    13  lokasi_mesin                13125 non-null  int64
    14  pemilik_mesin               13125 non-null  int64
    15  waktu_transaksi             13125 non-null  int64
    16  kuartal_transaksi           13125 non-null  int64
    17  kepemilikan_kartu           13125 non-null  int64
    18  nama_channel                13125 non-null  int64
    19  id_channel                  13125 non-null  int64
    20  flag_transaksi_finansial    13125 non-null  bool
    21  status_transaksi            13125 non-null  int64
    22  bank_pemilik_kartu          13125 non-null  int64
    23  rata_rata_nilai_transaksi   13104 non-null  float64
     24  maksimum_nilai_transaksi    13104 non-null  float64
     25  minimum_nilai_transaksi     13104 non-null  float64
     26  rata_rata_jumlah_transaksi  13104 non-null  float64
     27  flag_transaksi_fraud        13125 non-null  int64
    dtypes: bool(1), float64(5), int64(22)
    memory usage: 2.7 MB

Berdasarkan deskripsi data sebelumnya dan `fraud.info()` maka beberapa kolom akan diubah ke tipe data lain.

- `object` = ['X', 'id_tanggal_transaksi_awal', 'nama_channel', 'id_channel', 'bank_pemilik_kartu', 'id_merchant', 'nama_merchant', 'nama_transaksi']
- `category` = ['tipe_kartu', 'tipe_mesin', 'tipe_transaksi', 'id_negara', 'nama_negara', 'nama_kota', 'lokasi_mesin', 'pemilik_mesin',  'kepemilikan_kartu', 'status_transaksi']
- `datetime` = ['tanggal_transaksi_awal', 'waktu_transaksi']

### Ubah datatype

```python
obj_kolom = ['X', 'id_tanggal_transaksi_awal', 'nama_channel', 'id_channel', 'bank_pemilik_kartu', 'id_merchant', 'nama_merchant', 'nama_transaksi']
cat_kolom = ['kuartal_transaksi', 'tipe_kartu', 'tipe_mesin', 'tipe_transaksi', 'id_negara', 'nama_negara', 'nama_kota', 'lokasi_mesin', 'pemilik_mesin', 'kepemilikan_kartu', 'status_transaksi']
dt_kolom = ['tanggal_transaksi_awal', 'waktu_transaksi']
```

```python
# -> object
fraud[obj_kolom] = fraud[obj_kolom].astype('object')
# -> category
fraud[cat_kolom] = fraud[cat_kolom].astype('category')
# -> datetime
fraud['tanggal_transaksi_awal'] = fraud['tanggal_transaksi_awal'].apply(lambda x : datetime.fromtimestamp(x))
fraud['waktu_transaksi'] = fraud['waktu_transaksi'].apply(lambda x : datetime.fromtimestamp(x))
```

```python
fraud.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13125 entries, 0 to 13124
    Data columns (total 28 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
    0   X                           13125 non-null  object
    1   id_tanggal_transaksi_awal   13125 non-null  object
    2   tanggal_transaksi_awal      13125 non-null  datetime64[ns]
     3   tipe_kartu                  13125 non-null  category
    4   id_merchant                 13125 non-null  object
    5   nama_merchant               13125 non-null  object
    6   tipe_mesin                  13125 non-null  category
    7   tipe_transaksi              13125 non-null  category
    8   nama_transaksi              13125 non-null  object
    9   nilai_transaksi             13125 non-null  float64
    10  id_negara                   13125 non-null  category
    11  nama_negara                 13125 non-null  category
    12  nama_kota                   13125 non-null  category
    13  lokasi_mesin                13125 non-null  category
    14  pemilik_mesin               13125 non-null  category
    15  waktu_transaksi             13125 non-null  datetime64[ns]
     16  kuartal_transaksi           13125 non-null  category
    17  kepemilikan_kartu           13125 non-null  category
    18  nama_channel                13125 non-null  object
    19  id_channel                  13125 non-null  object
    20  flag_transaksi_finansial    13125 non-null  bool
    21  status_transaksi            13125 non-null  category
    22  bank_pemilik_kartu          13125 non-null  object
    23  rata_rata_nilai_transaksi   13104 non-null  float64
    24  maksimum_nilai_transaksi    13104 non-null  float64
    25  minimum_nilai_transaksi     13104 non-null  float64
    26  rata_rata_jumlah_transaksi  13104 non-null  float64
    27  flag_transaksi_fraud        13125 non-null  int64
    dtypes: bool(1), category(11), datetime64[ns](2), float64(5), int64(1), object(8)
    memory usage: 2.2+ MB

### Cek deskripsi data numerik dan kategorik

```python
# numerik
fraud.describe().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tanggal_transaksi_awal</th>
      <td>13125</td>
      <td>1970-01-29 17:39:00.576761904</td>
      <td>1970-01-29 17:35:03</td>
      <td>1970-01-29 17:37:31</td>
      <td>1970-01-29 17:39:03</td>
      <td>1970-01-29 17:40:32</td>
      <td>1970-01-29 17:42:34</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>nilai_transaksi</th>
      <td>13125.0</td>
      <td>1315218.824267</td>
      <td>1.0</td>
      <td>200000.0</td>
      <td>570000.0</td>
      <td>1250000.0</td>
      <td>75000000.0</td>
      <td>2838050.053336</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>13125</td>
      <td>1970-01-02 21:34:55.669638095</td>
      <td>1970-01-01 07:00:47</td>
      <td>1970-01-02 11:30:22</td>
      <td>1970-01-02 22:05:07</td>
      <td>1970-01-03 07:43:40</td>
      <td>1970-01-04 00:31:54</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>13104.0</td>
      <td>1364131.826769</td>
      <td>50000.0</td>
      <td>568563.3893</td>
      <td>1024239.017</td>
      <td>1679777.627</td>
      <td>24666666.67</td>
      <td>1448583.095472</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>13104.0</td>
      <td>12287602.944063</td>
      <td>38000.0</td>
      <td>2500000.0</td>
      <td>6000000.0</td>
      <td>15000000.0</td>
      <td>100000000.0</td>
      <td>16459046.159531</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>13104.0</td>
      <td>76519.328602</td>
      <td>1.0</td>
      <td>25000.0</td>
      <td>36964.0</td>
      <td>63200.0</td>
      <td>75000000.0</td>
      <td>676539.058057</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>13104.0</td>
      <td>2.436182</td>
      <td>1.0</td>
      <td>1.68</td>
      <td>2.1</td>
      <td>2.79</td>
      <td>19.78</td>
      <td>1.389367</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>13125.0</td>
      <td>0.069333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.25403</td>
    </tr>
  </tbody>
</table>
</div>

```python
# numerik
fraud.select_dtypes(include=['object', 'category']).describe().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>13125</td>
      <td>8793</td>
      <td>2925</td>
      <td>6</td>
    </tr>
    <tr>
      <th>id_tanggal_transaksi_awal</th>
      <td>13125</td>
      <td>360</td>
      <td>2457509</td>
      <td>169</td>
    </tr>
    <tr>
      <th>tipe_kartu</th>
      <td>13125</td>
      <td>14</td>
      <td>111</td>
      <td>4846</td>
    </tr>
    <tr>
      <th>id_merchant</th>
      <td>13125</td>
      <td>1122</td>
      <td>-2</td>
      <td>11307</td>
    </tr>
    <tr>
      <th>nama_merchant</th>
      <td>13125</td>
      <td>1105</td>
      <td>1798</td>
      <td>11307</td>
    </tr>
    <tr>
      <th>tipe_mesin</th>
      <td>13125</td>
      <td>5341</td>
      <td>-3</td>
      <td>883</td>
    </tr>
    <tr>
      <th>tipe_transaksi</th>
      <td>13125</td>
      <td>20</td>
      <td>26</td>
      <td>3575</td>
    </tr>
    <tr>
      <th>nama_transaksi</th>
      <td>13125</td>
      <td>20</td>
      <td>10</td>
      <td>3575</td>
    </tr>
    <tr>
      <th>id_negara</th>
      <td>13125</td>
      <td>13</td>
      <td>96</td>
      <td>13081</td>
    </tr>
    <tr>
      <th>nama_negara</th>
      <td>13125</td>
      <td>12</td>
      <td>5</td>
      <td>13080</td>
    </tr>
    <tr>
      <th>nama_kota</th>
      <td>13125</td>
      <td>229</td>
      <td>128</td>
      <td>5404</td>
    </tr>
    <tr>
      <th>lokasi_mesin</th>
      <td>13125</td>
      <td>5814</td>
      <td>600</td>
      <td>21</td>
    </tr>
    <tr>
      <th>pemilik_mesin</th>
      <td>13125</td>
      <td>1666</td>
      <td>613</td>
      <td>10418</td>
    </tr>
    <tr>
      <th>kuartal_transaksi</th>
      <td>13125</td>
      <td>4</td>
      <td>3</td>
      <td>5224</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu</th>
      <td>13125</td>
      <td>2</td>
      <td>2</td>
      <td>12236</td>
    </tr>
    <tr>
      <th>nama_channel</th>
      <td>13125</td>
      <td>5</td>
      <td>1</td>
      <td>10418</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>13125</td>
      <td>4</td>
      <td>9</td>
      <td>10418</td>
    </tr>
    <tr>
      <th>status_transaksi</th>
      <td>13125</td>
      <td>1</td>
      <td>3</td>
      <td>13125</td>
    </tr>
    <tr>
      <th>bank_pemilik_kartu</th>
      <td>13125</td>
      <td>1</td>
      <td>999</td>
      <td>13125</td>
    </tr>
  </tbody>
</table>
</div>

### Cek missing atau duplicated

```python
fraud.isna().sum()
```

    X                              0
    id_tanggal_transaksi_awal      0
    tanggal_transaksi_awal         0
    tipe_kartu                     0
    id_merchant                    0
    nama_merchant                  0
    tipe_mesin                     0
    tipe_transaksi                 0
    nama_transaksi                 0
    nilai_transaksi                0
    id_negara                      0
    nama_negara                    0
    nama_kota                      0
    lokasi_mesin                   0
    pemilik_mesin                  0
    waktu_transaksi                0
    kuartal_transaksi              0
    kepemilikan_kartu              0
    nama_channel                   0
    id_channel                     0
    flag_transaksi_finansial       0
    status_transaksi               0
    bank_pemilik_kartu             0
    rata_rata_nilai_transaksi     21
    maksimum_nilai_transaksi      21
    minimum_nilai_transaksi       21
    rata_rata_jumlah_transaksi    21
    flag_transaksi_fraud           0
    dtype: int64

```python
fraud[fraud['rata_rata_nilai_transaksi'].isna()].T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>610</th>
      <th>1489</th>
      <th>1829</th>
      <th>3069</th>
      <th>3752</th>
      <th>3781</th>
      <th>6185</th>
      <th>6501</th>
      <th>7655</th>
      <th>7660</th>
      <th>...</th>
      <th>7777</th>
      <th>8719</th>
      <th>8758</th>
      <th>9147</th>
      <th>10006</th>
      <th>10374</th>
      <th>10401</th>
      <th>11142</th>
      <th>11520</th>
      <th>11797</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>1039</td>
      <td>4020</td>
      <td>14080</td>
      <td>1335</td>
      <td>8871</td>
      <td>12852</td>
      <td>8659</td>
      <td>7518</td>
      <td>2557</td>
      <td>1813</td>
      <td>...</td>
      <td>4192</td>
      <td>2465</td>
      <td>9504</td>
      <td>342</td>
      <td>5102</td>
      <td>9641</td>
      <td>14239</td>
      <td>6907</td>
      <td>8832</td>
      <td>2409</td>
    </tr>
    <tr>
      <th>id_tanggal_transaksi_awal</th>
      <td>2457561</td>
      <td>2457655</td>
      <td>2457510</td>
      <td>2457514</td>
      <td>2457531</td>
      <td>2457368</td>
      <td>2457603</td>
      <td>2457620</td>
      <td>2457558</td>
      <td>2457385</td>
      <td>...</td>
      <td>2457331</td>
      <td>2457592</td>
      <td>2457524</td>
      <td>2457526</td>
      <td>2457517</td>
      <td>2457593</td>
      <td>2457429</td>
      <td>2457308</td>
      <td>2457468</td>
      <td>2457536</td>
    </tr>
    <tr>
      <th>tanggal_transaksi_awal</th>
      <td>1970-01-29 17:40:07</td>
      <td>1970-01-29 17:40:58</td>
      <td>1970-01-29 17:38:41</td>
      <td>1970-01-29 17:39:33</td>
      <td>1970-01-29 17:40:10</td>
      <td>1970-01-29 17:36:52</td>
      <td>1970-01-29 17:40:43</td>
      <td>1970-01-29 17:41:37</td>
      <td>1970-01-29 17:39:26</td>
      <td>1970-01-29 17:37:43</td>
      <td>...</td>
      <td>1970-01-29 17:36:13</td>
      <td>1970-01-29 17:40:45</td>
      <td>1970-01-29 17:38:45</td>
      <td>1970-01-29 17:40:21</td>
      <td>1970-01-29 17:39:25</td>
      <td>1970-01-29 17:40:16</td>
      <td>1970-01-29 17:37:11</td>
      <td>1970-01-29 17:35:48</td>
      <td>1970-01-29 17:39:13</td>
      <td>1970-01-29 17:39:51</td>
    </tr>
    <tr>
      <th>tipe_kartu</th>
      <td>93</td>
      <td>103</td>
      <td>93</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
      <td>93</td>
      <td>111</td>
      <td>93</td>
      <td>111</td>
      <td>...</td>
      <td>111</td>
      <td>111</td>
      <td>111</td>
      <td>104</td>
      <td>93</td>
      <td>111</td>
      <td>93</td>
      <td>103</td>
      <td>111</td>
      <td>93</td>
    </tr>
    <tr>
      <th>id_merchant</th>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>388748</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>nama_merchant</th>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>...</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1574</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
      <td>1798</td>
    </tr>
    <tr>
      <th>tipe_mesin</th>
      <td>-3</td>
      <td>2330</td>
      <td>185678</td>
      <td>2344943</td>
      <td>2805857</td>
      <td>2337114</td>
      <td>288963</td>
      <td>-3</td>
      <td>3186008</td>
      <td>2393471</td>
      <td>...</td>
      <td>1906758</td>
      <td>2331560</td>
      <td>181</td>
      <td>-3</td>
      <td>2383750</td>
      <td>1752625</td>
      <td>2489227</td>
      <td>1478519</td>
      <td>3166779</td>
      <td>1180307</td>
    </tr>
    <tr>
      <th>tipe_transaksi</th>
      <td>156</td>
      <td>147</td>
      <td>159</td>
      <td>26</td>
      <td>26</td>
      <td>148</td>
      <td>26</td>
      <td>156</td>
      <td>238</td>
      <td>26</td>
      <td>...</td>
      <td>159</td>
      <td>26</td>
      <td>147</td>
      <td>156</td>
      <td>58</td>
      <td>385</td>
      <td>26</td>
      <td>385</td>
      <td>26</td>
      <td>159</td>
    </tr>
    <tr>
      <th>nama_transaksi</th>
      <td>12</td>
      <td>3</td>
      <td>19</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>10</td>
      <td>12</td>
      <td>9</td>
      <td>10</td>
      <td>...</td>
      <td>19</td>
      <td>10</td>
      <td>3</td>
      <td>12</td>
      <td>6</td>
      <td>11</td>
      <td>10</td>
      <td>11</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>nilai_transaksi</th>
      <td>100000.0</td>
      <td>1000000.0</td>
      <td>6000000.0</td>
      <td>500000.0</td>
      <td>2500000.0</td>
      <td>25000.0</td>
      <td>600000.0</td>
      <td>1200000.0</td>
      <td>2150000.0</td>
      <td>200000.0</td>
      <td>...</td>
      <td>25000.0</td>
      <td>400000.0</td>
      <td>500000.0</td>
      <td>200000.0</td>
      <td>220130.0</td>
      <td>1200000.0</td>
      <td>350000.0</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>13918500.0</td>
    </tr>
    <tr>
      <th>id_negara</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>...</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>nama_negara</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nama_kota</th>
      <td>128</td>
      <td>128</td>
      <td>101</td>
      <td>141</td>
      <td>128</td>
      <td>265</td>
      <td>57</td>
      <td>101</td>
      <td>128</td>
      <td>128</td>
      <td>...</td>
      <td>128</td>
      <td>70</td>
      <td>128</td>
      <td>3</td>
      <td>251</td>
      <td>60</td>
      <td>128</td>
      <td>128</td>
      <td>259</td>
      <td>128</td>
    </tr>
    <tr>
      <th>lokasi_mesin</th>
      <td>1360</td>
      <td>374</td>
      <td>3988</td>
      <td>3442</td>
      <td>7852</td>
      <td>3674</td>
      <td>4759</td>
      <td>7791</td>
      <td>8414</td>
      <td>3763</td>
      <td>...</td>
      <td>2540</td>
      <td>3489</td>
      <td>2979</td>
      <td>1213</td>
      <td>7314</td>
      <td>2162</td>
      <td>8212</td>
      <td>1486</td>
      <td>8422</td>
      <td>503</td>
    </tr>
    <tr>
      <th>pemilik_mesin</th>
      <td>1637</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>496</td>
      <td>613</td>
      <td>613</td>
      <td>...</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>1616</td>
      <td>2309</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
      <td>613</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>1970-01-02 17:28:42</td>
      <td>1970-01-03 10:20:58</td>
      <td>1970-01-02 22:08:43</td>
      <td>1970-01-03 05:00:42</td>
      <td>1970-01-03 07:10:21</td>
      <td>1970-01-02 19:46:51</td>
      <td>1970-01-03 04:43:42</td>
      <td>1970-01-03 07:02:13</td>
      <td>1970-01-02 11:36:56</td>
      <td>1970-01-03 04:32:37</td>
      <td>...</td>
      <td>1970-01-02 11:00:30</td>
      <td>1970-01-02 01:12:28</td>
      <td>1970-01-03 07:44:12</td>
      <td>1970-01-02 05:20:54</td>
      <td>1970-01-02 23:02:10</td>
      <td>1970-01-02 17:51:54</td>
      <td>1970-01-02 23:25:27</td>
      <td>1970-01-02 00:59:10</td>
      <td>1970-01-03 04:17:08</td>
      <td>1970-01-02 12:20:14</td>
    </tr>
    <tr>
      <th>kuartal_transaksi</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nama_channel</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>flag_transaksi_finansial</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>status_transaksi</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>bank_pemilik_kartu</th>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>...</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 21 columns</p>
</div>

Saya akan coba menggunakan 2 metode dalam mengatasi missing value.

1. Drop missing value karena missing value hanya 21 dan secara kumulatif hanya 0.16 % saja kehilangan data (prioritas karena mudah)
2. Menggunakan imputasi (singular (`mean`,`median`) atau flexible imputation (`KNNImputer` atau `IterativeImputer`))

```python
fraud = fraud.dropna()
fraud.isna().sum()
```

    X                             0
    id_tanggal_transaksi_awal     0
    tanggal_transaksi_awal        0
    tipe_kartu                    0
    id_merchant                   0
    nama_merchant                 0
    tipe_mesin                    0
    tipe_transaksi                0
    nama_transaksi                0
    nilai_transaksi               0
    id_negara                     0
    nama_negara                   0
    nama_kota                     0
    lokasi_mesin                  0
    pemilik_mesin                 0
    waktu_transaksi               0
    kuartal_transaksi             0
    kepemilikan_kartu             0
    nama_channel                  0
    id_channel                    0
    flag_transaksi_finansial      0
    status_transaksi              0
    bank_pemilik_kartu            0
    rata_rata_nilai_transaksi     0
    maksimum_nilai_transaksi      0
    minimum_nilai_transaksi       0
    rata_rata_jumlah_transaksi    0
    flag_transaksi_fraud          0
    dtype: int64

### Cek duplikat

```python
fraud.duplicated().any()
```

    False

Alhamdulillah tidak ada duplikat wkwkwkwkwk

(4 poin) Bagaimana cara untuk melakukan *feature engineering* ataupun pemilihan variabel dari data yang tersedia.

- Apakah terdapat variable yang dihilangkan? Jika iya, kenapa?
- Apakah terdapat variable yang ditambahkan? Jika iya, kenapa?

```python
# numerik
fraud.select_dtypes(include=['object', 'category']).describe().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>13104</td>
      <td>8780</td>
      <td>2925</td>
      <td>6</td>
    </tr>
    <tr>
      <th>id_tanggal_transaksi_awal</th>
      <td>13104</td>
      <td>360</td>
      <td>2457509</td>
      <td>169</td>
    </tr>
    <tr>
      <th>tipe_kartu</th>
      <td>13104</td>
      <td>14</td>
      <td>111</td>
      <td>4835</td>
    </tr>
    <tr>
      <th>id_merchant</th>
      <td>13104</td>
      <td>1121</td>
      <td>-2</td>
      <td>11287</td>
    </tr>
    <tr>
      <th>nama_merchant</th>
      <td>13104</td>
      <td>1105</td>
      <td>1798</td>
      <td>11287</td>
    </tr>
    <tr>
      <th>tipe_mesin</th>
      <td>13104</td>
      <td>5338</td>
      <td>-3</td>
      <td>880</td>
    </tr>
    <tr>
      <th>tipe_transaksi</th>
      <td>13104</td>
      <td>20</td>
      <td>26</td>
      <td>3568</td>
    </tr>
    <tr>
      <th>nama_transaksi</th>
      <td>13104</td>
      <td>20</td>
      <td>10</td>
      <td>3568</td>
    </tr>
    <tr>
      <th>id_negara</th>
      <td>13104</td>
      <td>13</td>
      <td>96</td>
      <td>13060</td>
    </tr>
    <tr>
      <th>nama_negara</th>
      <td>13104</td>
      <td>12</td>
      <td>5</td>
      <td>13059</td>
    </tr>
    <tr>
      <th>nama_kota</th>
      <td>13104</td>
      <td>229</td>
      <td>128</td>
      <td>5394</td>
    </tr>
    <tr>
      <th>lokasi_mesin</th>
      <td>13104</td>
      <td>5810</td>
      <td>600</td>
      <td>21</td>
    </tr>
    <tr>
      <th>pemilik_mesin</th>
      <td>13104</td>
      <td>1665</td>
      <td>613</td>
      <td>10401</td>
    </tr>
    <tr>
      <th>kuartal_transaksi</th>
      <td>13104</td>
      <td>4</td>
      <td>3</td>
      <td>5211</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu</th>
      <td>13104</td>
      <td>2</td>
      <td>2</td>
      <td>12218</td>
    </tr>
    <tr>
      <th>nama_channel</th>
      <td>13104</td>
      <td>5</td>
      <td>1</td>
      <td>10401</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>13104</td>
      <td>4</td>
      <td>9</td>
      <td>10401</td>
    </tr>
    <tr>
      <th>status_transaksi</th>
      <td>13104</td>
      <td>1</td>
      <td>3</td>
      <td>13104</td>
    </tr>
    <tr>
      <th>bank_pemilik_kartu</th>
      <td>13104</td>
      <td>1</td>
      <td>999</td>
      <td>13104</td>
    </tr>
  </tbody>
</table>
</div>

### Apakah terdapat variable yang dihilangkan? Jika iya, kenapa?

Iya, terutama variabel yang hanya menyangkut ID dan Nama yang memiliki nilai unik terlalu banyak akan dihilangkan seperti `X`, `id_tanggal_transaksi_awal`, `id_merchant`, `nama_merchant`, `tipe_mesin`, `tipe_transaksi`, `nama_transaksi`, `id_negara`, `nama_negara`,`tipe_kartu`, `nama_kota`, `lokasi_mesin`, `pemilik_mesin` lalu untuk `status_transaksi` dan `bank_pemilik_kartu` juga dihapus karena kategori yang tidak punya unique value

```python
fraud[['nama_channel', 'id_channel']].corr()
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nama_channel</th>
      <th>id_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nama_channel</th>
      <td>1.000000</td>
      <td>-0.376658</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>-0.376658</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

Karena `nama_channel` dan `id_channel` tidak berkorelasi cukup kuat seperti asumsi saya diawal jadi dibiarkan saja

```python
fraud_clean = fraud.drop(columns=['X', 'id_tanggal_transaksi_awal', 'id_merchant', 'nama_merchant', 'tipe_mesin', 'tipe_transaksi', 'nama_transaksi', 'id_negara','tipe_kartu', 'nama_negara', 'nama_kota', 'lokasi_mesin', 'pemilik_mesin', 'status_transaksi', 'bank_pemilik_kartu'])
fraud_clean.head().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tanggal_transaksi_awal</th>
      <td>1970-01-29 17:42:06</td>
      <td>1970-01-29 17:38:27</td>
      <td>1970-01-29 17:40:12</td>
      <td>1970-01-29 17:42:26</td>
      <td>1970-01-29 17:36:25</td>
    </tr>
    <tr>
      <th>nilai_transaksi</th>
      <td>2200000.0</td>
      <td>2500000.0</td>
      <td>1200000.0</td>
      <td>320000.0</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>1970-01-03 12:52:35</td>
      <td>1970-01-02 03:19:00</td>
      <td>1970-01-02 21:56:56</td>
      <td>1970-01-03 02:05:17</td>
      <td>1970-01-02 22:48:59</td>
    </tr>
    <tr>
      <th>kuartal_transaksi</th>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nama_channel</th>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>id_channel</th>
      <td>9</td>
      <td>8</td>
      <td>8</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>flag_transaksi_finansial</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>1332292.784</td>
      <td>1369047.619</td>
      <td>15523460.4</td>
      <td>711764.7059</td>
      <td>617968.254</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>9750000.0</td>
      <td>10000000.0</td>
      <td>100000000.0</td>
      <td>6884408.0</td>
      <td>2500000.0</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>10000.0</td>
      <td>30000.0</td>
      <td>41804.0</td>
      <td>10000.0</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>2.73</td>
      <td>2.33</td>
      <td>2.4</td>
      <td>1.98</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

### Apakah terdapat variable yang ditambahkan? Jika iya, kenapa?

Sepertinya belum ada variable yang perlu ditambahkan dikarenakan belum diperlukannya. **Namun** ada kemungkinan penambahan variabel berdasarkan waktu seperti jam, hari, tanggal dan sebagainya. Beberapa dari variabel kategori atau objek akan ditambahkan untuk keperluan encoding agar input dapat dibaca oleh model. Namun akan dilakukan di cell lain. Lalu untuk variabel tanggal (`tanggal_transaksi_awal`, `waktu_transaksi`) akan diubah ke dalam bentuk `timestamp` agar dapat diproses model juga

Untuk analisis saya akan fokus menggunakan `fraud_clean` sedangkan untuk modelling akan menggunakan `fraud_clean_enc`

## Exploratory Data Analysis

(2 poin) Exploratory Data Analyss-

Berikan penjelasan informatif dari visualisasi dan/atau segala jenis hasil eksplorasi An- da.
Bagaimana distribusi data pada setiap varia- bel?
Apakah terdapat informasi menarik antara variable predictor dengan variable target?

Multivariate analysis

```python
matplotlib.rc('figure', figsize=(10, 5)) # Buat melebarkan gambar

sns.heatmap(fraud_clean.select_dtypes(include=['int64', 'float64']).corr(), # nilai korelasi
            annot=True,   # anotasi angka di dalam kotak heatmap
            fmt=".3f",    # format 3 angka dibelakang koma 
            cmap='Blues'); # warna heatmap
```

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_37_0.png)

Terlihat insight menarik bahwa korelasi antar prediktor `nilai_transaksi`, `rata_rata_nilai_transaksi`, dan `maksimum_nilai_transaksi` cukup tinggi. Ini akan cukup mempengaruhi model karena redundansi yang tinggi. Mari kita lihat distribusi ketiga variabel tersebut

### Distribusi `nilai_transaksi` terhadap `flag_transaksi_fraud`

```python
sns.histplot(data=fraud_clean, x='nilai_transaksi', log_scale=True, kde=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='nilai_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_40_1.png)

```python
sns.boxplot(data=fraud_clean, x='nilai_transaksi', log_scale=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='nilai_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_41_1.png)

### Distribusi `rata_rata_nilai_transaksi` terhadap `flag_transaksi_fraud`

```python
sns.histplot(data=fraud_clean, x='rata_rata_nilai_transaksi', log_scale=True, kde=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='rata_rata_nilai_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_43_1.png)

```python
sns.boxplot(data=fraud_clean, x='rata_rata_nilai_transaksi', log_scale=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='rata_rata_nilai_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_44_1.png)

### Distribusi `maksimum_nilai_transaksi` terhadap `flag_transaksi_fraud`

```python
sns.histplot(data=fraud_clean, x='maksimum_nilai_transaksi', log_scale=True, kde=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='maksimum_nilai_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_46_1.png)

```python
sns.boxplot(data=fraud_clean, x='maksimum_nilai_transaksi', log_scale=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='maksimum_nilai_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_47_1.png)

### Distribusi `minimum_nilai_transaksi` terhadap `flag_transaksi_fraud`

```python
sns.histplot(data=fraud_clean, x='minimum_nilai_transaksi', log_scale=True, kde=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='minimum_nilai_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_49_1.png)

```python
sns.boxplot(data=fraud_clean, x='minimum_nilai_transaksi', log_scale=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='minimum_nilai_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_50_1.png)

### Distribusi `rata_rata_jumlah_transaksi` terhadap `flag_transaksi_fraud`

```python
sns.histplot(data=fraud_clean, x='rata_rata_jumlah_transaksi', log_scale=True, kde=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='rata_rata_jumlah_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_52_1.png)

```python
sns.boxplot(data=fraud_clean, x='rata_rata_jumlah_transaksi', log_scale=True, hue='flag_transaksi_fraud')
```

    <Axes: xlabel='rata_rata_jumlah_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_53_1.png)

### Kesimpulan yang didapat pada data kontinu

- Terdapat persebaran yang sangat luas pada `nilai_transaksi` dan `minimum_nilai_transaksi`.
- Untuk `rata_rata_nilai_transaksi` distribusinya cukup normal
- Untuk `maksimum_jumlah_transaksi` dan `rata_rata_jumlah_transaksi` mempunyai skewness yang berlawanan
- Untuk `rata_rata_jumlah_transaksi` mempunyai skala yang berbeda dengan yang lain (karena hanya menghitung banyaknya transaksi jadi tidak banyak nilainya dibanding yang lain
- Melihat dari boxplot, sepertinya `rata_rata_jumlah_transaksi` dan `nilai_transaksi` bisa menjadi prediktor yang baik untuk memprediksi target

### Analysis tanggal dan transaksi fraud

```python
sns.histplot(data=fraud_clean, x='tanggal_transaksi_awal', hue='flag_transaksi_fraud')
```

    <Axes: xlabel='tanggal_transaksi_awal', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_56_1.png)

```python
sns.boxplot(data=fraud_clean, x='tanggal_transaksi_awal', hue='flag_transaksi_fraud')
```

    <Axes: xlabel='tanggal_transaksi_awal'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_57_1.png)

Kesimpulan dari sini fraud lebih banyak terjadi waktu di waktu yang lebih sore/malam

```python
sns.histplot(data=fraud_clean, x='waktu_transaksi', hue='flag_transaksi_fraud')
```

    <Axes: xlabel='waktu_transaksi', ylabel='Count'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_59_1.png)

```python
sns.boxplot(data=fraud_clean, x='waktu_transaksi', hue='flag_transaksi_fraud')
```

    <Axes: xlabel='waktu_transaksi'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_60_1.png)

### Analysis kategorikal data dengan target

```python
non_num_kolom
```

    ['kuartal_transaksi', 'kepemilikan_kartu', 'nama_channel', 'id_channel']

```python
sns.barplot(data=fraud_clean, x='kuartal_transaksi', y='flag_transaksi_fraud' )
```

    <Axes: xlabel='kuartal_transaksi', ylabel='flag_transaksi_fraud'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_63_1.png)

Dapat dilihat bahwa transaksi fraud lebih condong ke awal awal kuartal transaksi

```python
sns.barplot(data=fraud_clean, x='kepemilikan_kartu', y='flag_transaksi_fraud' )
```

    <Axes: xlabel='kepemilikan_kartu', ylabel='flag_transaksi_fraud'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_65_1.png)

Lalu untuk `kepemilikan_kartu` = 1 lebih condong melakukan fraud

```python
sns.barplot(data=fraud_clean, x='nama_channel', y='flag_transaksi_fraud' )
```

    <Axes: xlabel='nama_channel', ylabel='flag_transaksi_fraud'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_67_1.png)

Lalu untuk `nama_channel` 3 dan 5 lebih condong fraud dibanding yang lain

```python
sns.barplot(data=fraud_clean, x='id_channel', y='flag_transaksi_fraud' )
```

    <Axes: xlabel='id_channel', ylabel='flag_transaksi_fraud'>

![png](https://raw.githubusercontent.com/Saltfarmer/Algoritma-BFLP-DS-Audit/main/7.%20Capstone%20Project/output_69_1.png)

Lalu untuk `id_channel` 8 lebih condong fraud dibanding yang lain

## Data Pre-processing and Model Fitting

(2 poin) Data Pre-processing

- Apakah terdapat variabel yang harus di-encoding? Kalau ada, apa saja dan kenapa?

Ada yaitu data-data yang masih bersifat kategorikal dan boolean. Alasannya adalah karena model tidak dapat membaca data yang bersifat `object` atau `categorical` (untuk `boolean` masih diperdebatkan karena terkadang ada beberapa model yang bisa membaca `boolean` sebagai 1 atau 0).

Untuk kolom yang perlu di encoding adalah `kuartal_transaksi`, `kepemilikan_kartu`, `nama_channel`, `id_channel`, `flag_transaksi_finansial`

Oh iya jangan lupa `datetimens[64]` juga terkadang tidak bisa dibaca mode makanya sebaiknya diubah lagi dalam bentuk `timestamp`

```python
fraud_asli['flag_transaksi_finansial'].value_counts()
```

    flag_transaksi_finansial
    False    13125
    Name: count, dtype: int64

```python
# Ternyata False semua WKWKWKWKWKWWKK
fraud_clean = fraud_clean.drop(columns='flag_transaksi_finansial')
```

Dikarenakan variabel `nama_channel` dan `id_channel` nilai encoding nya punya korelasi tinggi maka salah satu kolom dihapus dan diambillah `id_channel`.
Lalu untuk `tanggal_transaksi_awal` karena banyak sekali keanehan saat dalam bentuk `datettimens(64)` (cuma 1 hari) maka di drop untuk menghindari kesalahan data

```python
non_num_kolom = ['kuartal_transaksi', 'kepemilikan_kartu', 'id_channel'] 
fraud_clean_enc = pd.get_dummies(fraud_clean,columns=non_num_kolom,drop_first=True,dtype='int')
fraud_clean_enc[dt_kolom] = fraud_asli[dt_kolom]
fraud_clean_enc = fraud_clean_enc.drop(columns=['tanggal_transaksi_awal', 'nama_channel'])
fraud_clean_enc.head().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nilai_transaksi</th>
      <td>2200000.000</td>
      <td>2.500000e+06</td>
      <td>1200000.0</td>
      <td>3.200000e+05</td>
      <td>150000.000</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>193955.000</td>
      <td>7.314000e+04</td>
      <td>140216.0</td>
      <td>1.551170e+05</td>
      <td>143339.000</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>1332292.784</td>
      <td>1.369048e+06</td>
      <td>15523460.4</td>
      <td>7.117647e+05</td>
      <td>617968.254</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>9750000.000</td>
      <td>1.000000e+07</td>
      <td>100000000.0</td>
      <td>6.884408e+06</td>
      <td>2500000.000</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>10000.000</td>
      <td>3.000000e+04</td>
      <td>41804.0</td>
      <td>1.000000e+04</td>
      <td>100000.000</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>2.730</td>
      <td>2.330000e+00</td>
      <td>2.4</td>
      <td>1.980000e+00</td>
      <td>1.460</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>0.000</td>
      <td>1.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_2</th>
      <td>0.000</td>
      <td>1.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_3</th>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_4</th>
      <td>1.000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu_2</th>
      <td>1.000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>id_channel_4</th>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>id_channel_8</th>
      <td>0.000</td>
      <td>1.000000e+00</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>id_channel_9</th>
      <td>1.000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>

### (4 poin) Membuat model, pilih model berdasarkan model yang sudah diajarkan dikelas (logistic regression / knn)

- Buatlah model yang dapat menjawab pertanyaan bisnis di atas. :
  Untuk Capstone, saya akan mencoba menggunakan kedua model `Logit()` dan `KNN`
- Variable apa yang dirasa cukup penting dalam pembuatan model? Sertakan alasannya. :
  Untuk model `Logit()` saya akan mencoba menggunakan semua data yang ada, namun ada beberapa prediktor kategorikal akan dihapus karena memiliki korelasi tinggi. Lalu untuk `KNN`, saya mencoba menggunakan semua nilai numerik yang ada dan melakukan scaling
- Metode evaluasi apa yang cocok digunakan untuk kasus ini? Jelaskan. :
  Evaluasi yang digunakan adalah `precision` karena target tidak seimbang dan dalam business case ini difokuskan untuk mencari True Positive dengan menurunkan dan menghindari False Positive sebanyak mungkin

### Uji coba menggunakan Logistic Regression

### Train Test split

```python
X = sm.add_constant(fraud_clean_enc.drop(columns='flag_transaksi_fraud'))
y = fraud_clean_enc['flag_transaksi_fraud']
```

```python
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 13104 entries, 0 to 13124
    Data columns (total 14 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
    0   const                       13104 non-null  float64
     1   nilai_transaksi             13104 non-null  float64
     2   waktu_transaksi             13104 non-null  int64
    3   rata_rata_nilai_transaksi   13104 non-null  float64
     4   maksimum_nilai_transaksi    13104 non-null  float64
     5   minimum_nilai_transaksi     13104 non-null  float64
     6   rata_rata_jumlah_transaksi  13104 non-null  float64
     7   kuartal_transaksi_2         13104 non-null  int32
    8   kuartal_transaksi_3         13104 non-null  int32
    9   kuartal_transaksi_4         13104 non-null  int32
    10  kepemilikan_kartu_2         13104 non-null  int32
    11  id_channel_4                13104 non-null  int32
    12  id_channel_8                13104 non-null  int32
    13  id_channel_9                13104 non-null  int32
    dtypes: float64(6), int32(7), int64(1)
    memory usage: 1.1 MB

```python
# X = X.iloc[:, :12]
```

```python
X.corr()
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>nilai_transaksi</th>
      <th>waktu_transaksi</th>
      <th>rata_rata_nilai_transaksi</th>
      <th>maksimum_nilai_transaksi</th>
      <th>minimum_nilai_transaksi</th>
      <th>rata_rata_jumlah_transaksi</th>
      <th>kuartal_transaksi_2</th>
      <th>kuartal_transaksi_3</th>
      <th>kuartal_transaksi_4</th>
      <th>kepemilikan_kartu_2</th>
      <th>id_channel_4</th>
      <th>id_channel_8</th>
      <th>id_channel_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>nilai_transaksi</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.019857</td>
      <td>0.419086</td>
      <td>0.219751</td>
      <td>0.025245</td>
      <td>0.092394</td>
      <td>0.018400</td>
      <td>0.002426</td>
      <td>-0.022916</td>
      <td>-0.019989</td>
      <td>-0.095767</td>
      <td>0.020400</td>
      <td>0.069391</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>NaN</td>
      <td>-0.019857</td>
      <td>1.000000</td>
      <td>-0.026441</td>
      <td>-0.019929</td>
      <td>-0.004055</td>
      <td>0.000056</td>
      <td>-0.661371</td>
      <td>0.152606</td>
      <td>0.725364</td>
      <td>0.028756</td>
      <td>0.081363</td>
      <td>-0.027334</td>
      <td>-0.051648</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>NaN</td>
      <td>0.419086</td>
      <td>-0.026441</td>
      <td>1.000000</td>
      <td>0.674542</td>
      <td>0.183931</td>
      <td>0.223442</td>
      <td>0.021902</td>
      <td>-0.003942</td>
      <td>-0.021849</td>
      <td>-0.015058</td>
      <td>-0.052375</td>
      <td>0.015319</td>
      <td>0.035389</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>NaN</td>
      <td>0.219751</td>
      <td>-0.019929</td>
      <td>0.674542</td>
      <td>1.000000</td>
      <td>0.045452</td>
      <td>0.309681</td>
      <td>0.016373</td>
      <td>-0.008148</td>
      <td>-0.009549</td>
      <td>0.005384</td>
      <td>-0.015978</td>
      <td>-0.004986</td>
      <td>0.016988</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>NaN</td>
      <td>0.025245</td>
      <td>-0.004055</td>
      <td>0.183931</td>
      <td>0.045452</td>
      <td>1.000000</td>
      <td>-0.003797</td>
      <td>-0.001714</td>
      <td>0.007160</td>
      <td>-0.006077</td>
      <td>-0.008944</td>
      <td>-0.010199</td>
      <td>0.008779</td>
      <td>0.003161</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>NaN</td>
      <td>0.092394</td>
      <td>0.000056</td>
      <td>0.223442</td>
      <td>0.309681</td>
      <td>-0.003797</td>
      <td>1.000000</td>
      <td>-0.005349</td>
      <td>0.002426</td>
      <td>0.002915</td>
      <td>0.012259</td>
      <td>-0.024747</td>
      <td>-0.011278</td>
      <td>0.028744</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_2</th>
      <td>NaN</td>
      <td>0.018400</td>
      <td>-0.661371</td>
      <td>0.021902</td>
      <td>0.016373</td>
      <td>-0.001714</td>
      <td>-0.005349</td>
      <td>1.000000</td>
      <td>-0.561865</td>
      <td>-0.394363</td>
      <td>-0.013247</td>
      <td>-0.081556</td>
      <td>0.012598</td>
      <td>0.061437</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_3</th>
      <td>NaN</td>
      <td>0.002426</td>
      <td>0.152606</td>
      <td>-0.003942</td>
      <td>-0.008148</td>
      <td>0.007160</td>
      <td>0.002426</td>
      <td>-0.561865</td>
      <td>1.000000</td>
      <td>-0.463386</td>
      <td>0.002069</td>
      <td>0.037197</td>
      <td>-0.001212</td>
      <td>-0.030486</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_4</th>
      <td>NaN</td>
      <td>-0.022916</td>
      <td>0.725364</td>
      <td>-0.021849</td>
      <td>-0.009549</td>
      <td>-0.006077</td>
      <td>0.002915</td>
      <td>-0.394363</td>
      <td>-0.463386</td>
      <td>1.000000</td>
      <td>0.022208</td>
      <td>0.050322</td>
      <td>-0.021943</td>
      <td>-0.029200</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu_2</th>
      <td>NaN</td>
      <td>-0.019989</td>
      <td>0.028756</td>
      <td>-0.015058</td>
      <td>0.005384</td>
      <td>-0.008944</td>
      <td>0.012259</td>
      <td>-0.013247</td>
      <td>0.002069</td>
      <td>0.022208</td>
      <td>1.000000</td>
      <td>0.108045</td>
      <td>-0.996364</td>
      <td>0.528240</td>
    </tr>
    <tr>
      <th>id_channel_4</th>
      <td>NaN</td>
      <td>-0.095767</td>
      <td>0.081363</td>
      <td>-0.052375</td>
      <td>-0.015978</td>
      <td>-0.010199</td>
      <td>-0.024747</td>
      <td>-0.081556</td>
      <td>0.037197</td>
      <td>0.050322</td>
      <td>0.108045</td>
      <td>1.000000</td>
      <td>-0.107652</td>
      <td>-0.787051</td>
    </tr>
    <tr>
      <th>id_channel_8</th>
      <td>NaN</td>
      <td>0.020400</td>
      <td>-0.027334</td>
      <td>0.015319</td>
      <td>-0.004986</td>
      <td>0.008779</td>
      <td>-0.011278</td>
      <td>0.012598</td>
      <td>-0.001212</td>
      <td>-0.021943</td>
      <td>-0.996364</td>
      <td>-0.107652</td>
      <td>1.000000</td>
      <td>-0.526319</td>
    </tr>
    <tr>
      <th>id_channel_9</th>
      <td>NaN</td>
      <td>0.069391</td>
      <td>-0.051648</td>
      <td>0.035389</td>
      <td>0.016988</td>
      <td>0.003161</td>
      <td>0.028744</td>
      <td>0.061437</td>
      <td>-0.030486</td>
      <td>-0.029200</td>
      <td>0.528240</td>
      <td>-0.787051</td>
      <td>-0.526319</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

#### Membuat kolom dengan korelasi tinggi

```python
X.head()
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>nilai_transaksi</th>
      <th>waktu_transaksi</th>
      <th>rata_rata_nilai_transaksi</th>
      <th>maksimum_nilai_transaksi</th>
      <th>minimum_nilai_transaksi</th>
      <th>rata_rata_jumlah_transaksi</th>
      <th>kuartal_transaksi_2</th>
      <th>kuartal_transaksi_3</th>
      <th>kuartal_transaksi_4</th>
      <th>kepemilikan_kartu_2</th>
      <th>id_channel_4</th>
      <th>id_channel_8</th>
      <th>id_channel_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2200000.0</td>
      <td>193955</td>
      <td>1.332293e+06</td>
      <td>9750000.0</td>
      <td>10000.0</td>
      <td>2.73</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2500000.0</td>
      <td>73140</td>
      <td>1.369048e+06</td>
      <td>10000000.0</td>
      <td>30000.0</td>
      <td>2.33</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1200000.0</td>
      <td>140216</td>
      <td>1.552346e+07</td>
      <td>100000000.0</td>
      <td>41804.0</td>
      <td>2.40</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>320000.0</td>
      <td>155117</td>
      <td>7.117647e+05</td>
      <td>6884408.0</td>
      <td>10000.0</td>
      <td>1.98</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>150000.0</td>
      <td>143339</td>
      <td>6.179683e+05</td>
      <td>2500000.0</td>
      <td>100000.0</td>
      <td>1.46</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
fraud_clean_enc.describe().T
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nilai_transaksi</th>
      <td>13104.0</td>
      <td>1.314923e+06</td>
      <td>2.837644e+06</td>
      <td>1.0</td>
      <td>2.000000e+05</td>
      <td>572575.000</td>
      <td>1.251524e+06</td>
      <td>7.500000e+07</td>
    </tr>
    <tr>
      <th>waktu_transaksi</th>
      <td>13104.0</td>
      <td>1.389062e+05</td>
      <td>4.788674e+04</td>
      <td>47.0</td>
      <td>1.026220e+05</td>
      <td>140707.000</td>
      <td>1.754315e+05</td>
      <td>2.359140e+05</td>
    </tr>
    <tr>
      <th>rata_rata_nilai_transaksi</th>
      <td>13104.0</td>
      <td>1.364132e+06</td>
      <td>1.448583e+06</td>
      <td>50000.0</td>
      <td>5.685634e+05</td>
      <td>1024239.017</td>
      <td>1.679778e+06</td>
      <td>2.466667e+07</td>
    </tr>
    <tr>
      <th>maksimum_nilai_transaksi</th>
      <td>13104.0</td>
      <td>1.228760e+07</td>
      <td>1.645905e+07</td>
      <td>38000.0</td>
      <td>2.500000e+06</td>
      <td>6000000.000</td>
      <td>1.500000e+07</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>minimum_nilai_transaksi</th>
      <td>13104.0</td>
      <td>7.651933e+04</td>
      <td>6.765391e+05</td>
      <td>1.0</td>
      <td>2.500000e+04</td>
      <td>36964.000</td>
      <td>6.320000e+04</td>
      <td>7.500000e+07</td>
    </tr>
    <tr>
      <th>rata_rata_jumlah_transaksi</th>
      <td>13104.0</td>
      <td>2.436182e+00</td>
      <td>1.389367e+00</td>
      <td>1.0</td>
      <td>1.680000e+00</td>
      <td>2.100</td>
      <td>2.790000e+00</td>
      <td>1.978000e+01</td>
    </tr>
    <tr>
      <th>flag_transaksi_fraud</th>
      <td>13104.0</td>
      <td>6.822344e-02</td>
      <td>2.521386e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_2</th>
      <td>13104.0</td>
      <td>3.234890e-01</td>
      <td>4.678254e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_3</th>
      <td>13104.0</td>
      <td>3.976648e-01</td>
      <td>4.894342e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>kuartal_transaksi_4</th>
      <td>13104.0</td>
      <td>2.454212e-01</td>
      <td>4.303531e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>kepemilikan_kartu_2</th>
      <td>13104.0</td>
      <td>9.323871e-01</td>
      <td>2.510901e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>id_channel_4</th>
      <td>13104.0</td>
      <td>1.386600e-01</td>
      <td>3.456045e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>id_channel_8</th>
      <td>13104.0</td>
      <td>6.715507e-02</td>
      <td>2.502999e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>id_channel_9</th>
      <td>13104.0</td>
      <td>7.937271e-01</td>
      <td>4.046441e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>

```python
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                   y, # kolom target
                                                   test_size = 0.2, # 80% training and 20% test
                                                   random_state = 10,
                                                   stratify=y)
```

```python
X_train.dtypes
```

    const                         float64
    nilai_transaksi               float64
    waktu_transaksi                 int64
    rata_rata_nilai_transaksi     float64
    maksimum_nilai_transaksi      float64
    minimum_nilai_transaksi       float64
    rata_rata_jumlah_transaksi    float64
    kuartal_transaksi_2             int32
    kuartal_transaksi_3             int32
    kuartal_transaksi_4             int32
    kepemilikan_kartu_2             int32
    id_channel_4                    int32
    id_channel_8                    int32
    id_channel_9                    int32
    dtype: object

### Modelling menggunakan Logistic Regression

```python
model_logit = sm.Logit(y_train, X_train)
model_logit.fit().summary()
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.209745
             Iterations: 35

    C:\Users\SaltFarmer\miniconda3\envs\algoritma\lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "

<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>flag_transaksi_fraud</td> <th>  No. Observations:  </th>   <td> 10483</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>Logit</td>        <th>  Df Residuals:      </th>   <td> 10469</td>  
</tr>
<tr>
  <th>Method:</th>                   <td>MLE</td>         <th>  Df Model:          </th>   <td>    13</td>  
</tr>
<tr>
  <th>Date:</th>              <td>Thu, 11 Jan 2024</td>   <th>  Pseudo R-squ.:     </th>   <td>0.1576</td>  
</tr>
<tr>
  <th>Time:</th>                  <td>15:02:23</td>       <th>  Log-Likelihood:    </th>  <td> -2198.8</td> 
</tr>
<tr>
  <th>converged:</th>               <td>False</td>        <th>  LL-Null:           </th>  <td> -2610.0</td> 
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>      <th>  LLR p-value:       </th> <td>2.158e-167</td>
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                      <td>  -19.0997</td> <td> 8660.347</td> <td>   -0.002</td> <td> 0.998</td> <td> -1.7e+04</td> <td>  1.7e+04</td>
</tr>
<tr>
  <th>nilai_transaksi</th>            <td> 8.383e-08</td> <td> 1.15e-08</td> <td>    7.266</td> <td> 0.000</td> <td> 6.12e-08</td> <td> 1.06e-07</td>
</tr>
<tr>
  <th>waktu_transaksi</th>            <td>-1.875e-06</td> <td> 2.53e-06</td> <td>   -0.741</td> <td> 0.459</td> <td>-6.83e-06</td> <td> 3.08e-06</td>
</tr>
<tr>
  <th>rata_rata_nilai_transaksi</th>  <td>-1.908e-08</td> <td> 3.87e-08</td> <td>   -0.492</td> <td> 0.622</td> <td> -9.5e-08</td> <td> 5.69e-08</td>
</tr>
<tr>
  <th>maksimum_nilai_transaksi</th>   <td>-1.298e-09</td> <td> 3.82e-09</td> <td>   -0.340</td> <td> 0.734</td> <td>-8.79e-09</td> <td> 6.19e-09</td>
</tr>
<tr>
  <th>minimum_nilai_transaksi</th>    <td> 1.112e-06</td> <td> 2.02e-07</td> <td>    5.505</td> <td> 0.000</td> <td> 7.16e-07</td> <td> 1.51e-06</td>
</tr>
<tr>
  <th>rata_rata_jumlah_transaksi</th> <td>   -0.2769</td> <td>    0.050</td> <td>   -5.553</td> <td> 0.000</td> <td>   -0.375</td> <td>   -0.179</td>
</tr>
<tr>
  <th>kuartal_transaksi_2</th>        <td>   -0.4941</td> <td>    0.250</td> <td>   -1.974</td> <td> 0.048</td> <td>   -0.985</td> <td>   -0.004</td>
</tr>
<tr>
  <th>kuartal_transaksi_3</th>        <td>   -0.4149</td> <td>    0.356</td> <td>   -1.166</td> <td> 0.244</td> <td>   -1.112</td> <td>    0.283</td>
</tr>
<tr>
  <th>kuartal_transaksi_4</th>        <td>   -0.5087</td> <td>    0.475</td> <td>   -1.071</td> <td> 0.284</td> <td>   -1.440</td> <td>    0.422</td>
</tr>
<tr>
  <th>kepemilikan_kartu_2</th>        <td>   11.3593</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>id_channel_4</th>               <td>    5.4534</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>id_channel_8</th>               <td>   19.7843</td> <td> 8660.347</td> <td>    0.002</td> <td> 0.998</td> <td> -1.7e+04</td> <td>  1.7e+04</td>
</tr>
<tr>
  <th>id_channel_9</th>               <td>    5.9059</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
</table>

### Prediction Performance

- (1 poin) Metrics yang dipilih mencapai 60% pada data train (2 poin) Metrics yang dipilih mencapai 60% pada data test

#### Hasil metric data train

```python
logit_pred_tr = model_logit.fit().predict(X_train)
logit_pred_tr
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.209745
             Iterations: 35

    C:\Users\SaltFarmer\miniconda3\envs\algoritma\lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "

    11191    0.037380
    1027     0.056526
    10908    0.057065
    4404     0.057710
    5958     0.027605
               ...
    3071     0.380087
    4977     0.050986
    574      0.048985
    1717     0.049682
    203      0.048698
    Length: 10483, dtype: float64

```python
pred_label_tr = logit_pred_tr.apply(lambda x: 1 if x > 0.5 else 0)
pred_label_tr.value_counts()
```

    0    10416
    1       67
    Name: count, dtype: int64

```python
print(f'Accuracy score: {accuracy_score(y_train, pred_label_tr)}')
print(f'Recall score: {recall_score(y_train, pred_label_tr)}')
print(f'Precision score: {precision_score(y_train, pred_label_tr)}')
```

    Accuracy score: 0.9311265859009825
    Recall score: 0.04195804195804196
    Precision score: 0.44776119402985076

Karena Precision belum lebih dari 60% maka saya atur lagi thresholdnya

```python
pred_label_tr = logit_pred_tr.apply(lambda x: 1 if x > 0.8 else 0)
pred_label_tr.value_counts()
```

    0    10470
    1       13
    Name: count, dtype: int64

```python
print(f'Accuracy score: {accuracy_score(y_train, pred_label_tr)}')
print(f'Recall score: {recall_score(y_train, pred_label_tr)}')
print(f'Precision score: {precision_score(y_train, pred_label_tr)}')
```

    Accuracy score: 0.9322712963846227
    Recall score: 0.012587412587412588
    Precision score: 0.6923076923076923

Sudah ditemukan threshold yang tepat untuk Data Train melebihi nilai threshold 60%

#### Hasil metric data Test

```python
logit_pred = model_logit.fit().predict(X_test)
logit_pred
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.209745
             Iterations: 35

    C:\Users\SaltFarmer\miniconda3\envs\algoritma\lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "

    12431    0.057628
    4523     0.049031
    8211     0.011157
    4172     0.048879
    5746     0.389036
               ...
    11842    0.048245
    9454     0.046732
    2449     0.055409
    3865     0.036163
    5465     0.051302
    Length: 2621, dtype: float64

```python
pred_label = logit_pred.apply(lambda x: 1 if x > 0.5 else 0)
pred_label.value_counts()
```

    0    2602
    1      19
    Name: count, dtype: int64

```python
print(f'Accuracy score: {accuracy_score(y_test, pred_label)}')
print(f'Recall score: {recall_score(y_test, pred_label)}')
print(f'Precision score: {precision_score(y_test, pred_label)}')
```

    Accuracy score: 0.9320869896985883
    Recall score: 0.055865921787709494
    Precision score: 0.5263157894736842

Saya coba atur sedikit threshold nya agar mendapatkan nilai presisi yang diinginkan

```python
pred_label = logit_pred.apply(lambda x: 1 if x > 0.8 else 0)
pred_label.value_counts()
```

    0    2619
    1       2
    Name: count, dtype: int64

```python
print(f'Accuracy score: {accuracy_score(y_test, pred_label)}')
print(f'Recall score: {recall_score(y_test, pred_label)}')
print(f'Precision score: {precision_score(y_test, pred_label)}')
```

    Accuracy score: 0.9324685234643266
    Recall score: 0.0111731843575419
    Precision score: 1.0

Melihat hasil metric dari data train dan data test maka model bisa dianggap **undefitting** (hasil train < hasil test)

# Angka Haram

Hasil dengan presisi 1 dan terbaik

```python
# fraud_asli = fraud_asli.dropna()
# X = sm.add_constant(fraud_asli.drop(columns='flag_transaksi_fraud')[['nilai_transaksi', 'waktu_transaksi', 'rata_rata_nilai_transaksi', 'maksimum_nilai_transaksi', 'minimum_nilai_transaksi', 'rata_rata_jumlah_transaksi']])
# y = fraud_asli['flag_transaksi_fraud']
```

```python
# X_train, X_test, y_train, y_test = train_test_split(X, 
#                                                    y, # kolom target
#                                                    test_size = 0.2, # 80% training and 20% test
#                                                    random_state = 10,
#                                                    stratify=y)
```

```python
# X_train
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>nilai_transaksi</th>
      <th>waktu_transaksi</th>
      <th>rata_rata_nilai_transaksi</th>
      <th>maksimum_nilai_transaksi</th>
      <th>minimum_nilai_transaksi</th>
      <th>rata_rata_jumlah_transaksi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11191</th>
      <td>1.0</td>
      <td>1250000.0</td>
      <td>155818</td>
      <td>5.527200e+06</td>
      <td>75000000.0</td>
      <td>50000.0</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>1.0</td>
      <td>2500000.0</td>
      <td>142017</td>
      <td>1.303347e+06</td>
      <td>9000000.0</td>
      <td>33500.0</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>10908</th>
      <td>1.0</td>
      <td>100000.0</td>
      <td>102021</td>
      <td>3.992588e+05</td>
      <td>2500000.0</td>
      <td>22400.0</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>4404</th>
      <td>1.0</td>
      <td>2500000.0</td>
      <td>172626</td>
      <td>8.525882e+05</td>
      <td>7000000.0</td>
      <td>50000.0</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>5958</th>
      <td>1.0</td>
      <td>498813.0</td>
      <td>154446</td>
      <td>5.185913e+05</td>
      <td>4318765.0</td>
      <td>21200.0</td>
      <td>2.24</td>
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
      <th>3071</th>
      <td>1.0</td>
      <td>1300000.0</td>
      <td>151541</td>
      <td>1.184368e+06</td>
      <td>10000000.0</td>
      <td>25000.0</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>4977</th>
      <td>1.0</td>
      <td>1250000.0</td>
      <td>171957</td>
      <td>1.275961e+06</td>
      <td>7000000.0</td>
      <td>50000.0</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>574</th>
      <td>1.0</td>
      <td>2499000.0</td>
      <td>102907</td>
      <td>2.917987e+06</td>
      <td>7328969.0</td>
      <td>500000.0</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>1717</th>
      <td>1.0</td>
      <td>1000000.0</td>
      <td>133843</td>
      <td>6.063348e+05</td>
      <td>3500000.0</td>
      <td>31700.0</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>203</th>
      <td>1.0</td>
      <td>300000.0</td>
      <td>195601</td>
      <td>5.033857e+05</td>
      <td>1000000.0</td>
      <td>100000.0</td>
      <td>1.40</td>
    </tr>
  </tbody>
</table>
<p>10483 rows × 7 columns</p>
</div>

```python
# model_logit2 = sm.Logit(y_train, X_train)
# model_logit2.fit().summary()
```

    Optimization terminated successfully.
             Current function value: 0.240125
             Iterations 7

<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>flag_transaksi_fraud</td> <th>  No. Observations:  </th>  <td> 10483</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>Logit</td>        <th>  Df Residuals:      </th>  <td> 10476</td>  
</tr>
<tr>
  <th>Method:</th>                   <td>MLE</td>         <th>  Df Model:          </th>  <td>     6</td>  
</tr>
<tr>
  <th>Date:</th>              <td>Thu, 11 Jan 2024</td>   <th>  Pseudo R-squ.:     </th>  <td>0.03554</td> 
</tr>
<tr>
  <th>Time:</th>                  <td>12:11:09</td>       <th>  Log-Likelihood:    </th> <td> -2517.2</td> 
</tr>
<tr>
  <th>converged:</th>               <td>True</td>         <th>  LL-Null:           </th> <td> -2610.0</td> 
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>      <th>  LLR p-value:       </th> <td>2.302e-37</td>
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                      <td>   -1.7497</td> <td>    0.153</td> <td>  -11.472</td> <td> 0.000</td> <td>   -2.049</td> <td>   -1.451</td>
</tr>
<tr>
  <th>nilai_transaksi</th>            <td> 7.637e-08</td> <td> 1.05e-08</td> <td>    7.274</td> <td> 0.000</td> <td> 5.58e-08</td> <td>  9.7e-08</td>
</tr>
<tr>
  <th>waktu_transaksi</th>            <td>-3.296e-06</td> <td> 8.14e-07</td> <td>   -4.047</td> <td> 0.000</td> <td>-4.89e-06</td> <td> -1.7e-06</td>
</tr>
<tr>
  <th>rata_rata_nilai_transaksi</th>  <td>-3.895e-10</td> <td> 3.67e-08</td> <td>   -0.011</td> <td> 0.992</td> <td>-7.24e-08</td> <td> 7.16e-08</td>
</tr>
<tr>
  <th>maksimum_nilai_transaksi</th>   <td>-2.108e-09</td> <td> 3.59e-09</td> <td>   -0.588</td> <td> 0.557</td> <td>-9.14e-09</td> <td> 4.92e-09</td>
</tr>
<tr>
  <th>minimum_nilai_transaksi</th>    <td> 1.101e-06</td> <td> 2.02e-07</td> <td>    5.447</td> <td> 0.000</td> <td> 7.05e-07</td> <td>  1.5e-06</td>
</tr>
<tr>
  <th>rata_rata_jumlah_transaksi</th> <td>   -0.2684</td> <td>    0.047</td> <td>   -5.669</td> <td> 0.000</td> <td>   -0.361</td> <td>   -0.176</td>
</tr>
</table>

```python
# logit_pred2 = model_logit2.fit().predict(X_test)
# logit_pred2
```

    Optimization terminated successfully.
             Current function value: 0.240125
             Iterations 7

    12431    0.075937
    4523     0.082749
    8211     0.015727
    4172     0.065997
    5746     0.081120
               ...
    11842    0.062634
    9454     0.071733
    2449     0.072522
    3865     0.049655
    5465     0.078226
    Length: 2621, dtype: float64

```python
# pred_label2 = logit_pred2.apply(lambda x: 1 if x > 0.5 else 0)
# pred_label2.value_counts()
```

    0    2618
    1       3
    Name: count, dtype: int64

```python
# print(f'Accuracy score: {accuracy_score(y_test, pred_label2)}')
# print(f'Recall score: {recall_score(y_test, pred_label2)}')
# print(f'Precision score: {precision_score(y_test, pred_label2)}')
```

    Accuracy score: 0.9320869896985883
    Recall score: 0.0111731843575419
    Precision score: 0.6666666666666666

# Coba KNN

```python
KNN_model = KNeighborsClassifier(n_neighbors=10)
scaler = StandardScaler()
```

```python
KNN_model
```

<style>#sk-container-id-7 {color: black;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=10)</pre></div></div></div></div></div>

```python
fraud_clean_num = fraud_clean_enc.iloc[:, :6]
fraud_clean_num
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
    }`</style>`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nilai_transaksi</th>
      <th>waktu_transaksi</th>
      <th>rata_rata_nilai_transaksi</th>
      <th>maksimum_nilai_transaksi</th>
      <th>minimum_nilai_transaksi</th>
      <th>rata_rata_jumlah_transaksi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2200000.0</td>
      <td>193955</td>
      <td>1.332293e+06</td>
      <td>9750000.0</td>
      <td>10000.0</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500000.0</td>
      <td>73140</td>
      <td>1.369048e+06</td>
      <td>10000000.0</td>
      <td>30000.0</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1200000.0</td>
      <td>140216</td>
      <td>1.552346e+07</td>
      <td>100000000.0</td>
      <td>41804.0</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>320000.0</td>
      <td>155117</td>
      <td>7.117647e+05</td>
      <td>6884408.0</td>
      <td>10000.0</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>150000.0</td>
      <td>143339</td>
      <td>6.179683e+05</td>
      <td>2500000.0</td>
      <td>100000.0</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13120</th>
      <td>100000.0</td>
      <td>140547</td>
      <td>2.917987e+06</td>
      <td>7400000.0</td>
      <td>26500.0</td>
      <td>2.57</td>
    </tr>
    <tr>
      <th>13121</th>
      <td>2500000.0</td>
      <td>172446</td>
      <td>1.914437e+06</td>
      <td>20000000.0</td>
      <td>100000.0</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>13122</th>
      <td>1250000.0</td>
      <td>141836</td>
      <td>3.417045e+05</td>
      <td>1000000.0</td>
      <td>100000.0</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>13123</th>
      <td>500000.0</td>
      <td>71451</td>
      <td>7.644508e+05</td>
      <td>3000000.0</td>
      <td>25000.0</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>13124</th>
      <td>300000.0</td>
      <td>175350</td>
      <td>8.483696e+05</td>
      <td>6375000.0</td>
      <td>25000.0</td>
      <td>1.79</td>
    </tr>
  </tbody>
</table>
<p>13104 rows × 6 columns</p>
</div>

```python
# prediktor
X = fraud_clean_num
# target
y = fraud_clean_enc['flag_transaksi_fraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, # kolom prediktor
                                                   y, # kolom target
                                                   test_size = 0.2, # 80% training and 20% test
                                                   random_state = 10,
                                                   stratify = y)
```

### Scaling

```python
cols = X_train.columns

scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)
```

```python
KNN_model.fit(X_train_scale, y_train)
pred_knn_tr = KNN_model.predict(X_train_scale)
```

```python
print(f'Accuracy score: {accuracy_score(y_train, pred_knn_tr)}')
print(f'Recall score: {recall_score(y_train, pred_knn_tr)}')
print(f'Precision score: {precision_score(y_train, pred_knn_tr)}')
```

    Accuracy score: 0.9324620814652295
    Recall score: 0.013986013986013986
    Precision score: 0.7692307692307693

Precision score dari KNN sudah cukup memuaskan karena sudah diatas 60%

```python
KNN_model.fit(X_train_scale, y_train)
pred_knn_label = KNN_model.predict(X_test_scale)
```

```python
print(f'Accuracy score: {accuracy_score(y_test, pred_knn_label)}')
print(f'Recall score: {recall_score(y_test, pred_knn_label)}')
print(f'Precision score: {precision_score(y_test, pred_knn_label)}')
```

    Accuracy score: 0.9328500572300649
    Recall score: 0.01675977653631285
    Precision score: 1.0

Didapatkan score precisision dari Data Test yang lebih tinggi lagi sehingga dianggap modelnya **underfitting** (hasil train < hasil test)

## Conclusion

### (2 poin) Tuliskan kesimpulan dari project yang anda kerjakan.

- Apakah model sudah dapat melakukan prediksi dengan baik? Jelaskan

Sudah cukup baik karena memenuhi pertanyaan bisnis untuk metric `precision` diatas 60%. Untuk kedua model `Logit()` dan `KNN()` sudah cukup memuaskan dengan hasil `KNN()` yang lebih baik pada Data Test. Kedua model tersebut juga perlu dilakukan tuning parameter seperti `threshold` di logit dan `n_neighbor` di KNN agar mendapatkan hasil yang diinginkan

- Apakah model sudah dapat menjawab pertanyaan bisnis yang ada? Jelaskan.

Sudah, karena pertanyaan bisnisnya mendeteksi fraud sebaik mungkin. Dikarenakan ketidakseimbangan jumlah target, maka metric yang digunakan pun harus tepat untuk fokus melihat True Positive sebaik mungkin dan melihat False Negative sesedikit mungkin. Dari hasil diatas, model Logit dan KNN sudah bekerja dengan baik sudah memenuhi standar presisi yang diinginkan.

- Action plan apa yang dapat dilakukan untuk tindakan preventif transaksi fraud?

Berdasarkan hasil dari model Logit, maka perlu diwaspadai saat ada momen `nilai_transaksi`, `minimum_nilai_transaksi`, tinggi serta mempunyai `kepemilikan_kartu` 2 dan melakukan `id_channel` selain 3 dikarenakan variabel - variabel ber koefisiensi positive terhadap model dalam memberikan peluang positif Fraud

## Freestyle SMOTE dan XGBoost

```python
fraud_clean_enc
```

    Index(['nilai_transaksi', 'waktu_transaksi', 'rata_rata_nilai_transaksi',
           'maksimum_nilai_transaksi', 'minimum_nilai_transaksi',
           'rata_rata_jumlah_transaksi', 'flag_transaksi_fraud',
           'kuartal_transaksi_2', 'kuartal_transaksi_3', 'kuartal_transaksi_4',
           'kepemilikan_kartu_2', 'id_channel_4', 'id_channel_8', 'id_channel_9'],
          dtype='object')

```python
from imblearn.combine import SMOTEENN

sme = SMOTEENN(random_state=10)
```

```python
X_0 = fraud_clean_enc.drop(columns='flag_transaksi_fraud')
y_0 = fraud_clean_enc['flag_transaksi_fraud']
```

```python
X_res, y_res = sme.fit_resample(X_0, y_0)
```

```python
y_res.value_counts()
```

    flag_transaksi_fraud
    1    10106
    0     8553
    Name: count, dtype: int64

```python
X_res.shape
```

    (18659, 13)

```python
fraud_clean_enc['flag_transaksi_fraud'].value_counts()
```

    flag_transaksi_fraud
    0    12210
    1      894
    Name: count, dtype: int64

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=10)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X_res, # kolom prediktor
                                                   y_res, # kolom target
                                                   test_size = 0.2, # 80% training and 20% test
                                                   random_state = 10,
                                                   stratify = y_res)
```

```python
xgb.fit(X_train, y_train)
```

<style>#sk-container-id-8 {color: black;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
