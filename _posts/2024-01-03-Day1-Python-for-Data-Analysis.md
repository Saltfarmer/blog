---
title: "Day 1 Algorit.ma : Python For Data Analysis"
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

---

Day 1, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Coursebook: Python for Data Analysts**

This notebook was made based on main materials `Python for Data Analysts.ipynb`

Version: BRI Audit Analytics - January 2024


___

# Training Objectives

- Introduction to the `pandas` library. 
- Introduction to `DataFrame`  
- Data Types
- Exploratory Data Analysis I
- Indexing and Subsetting

# 1. Perkenalan Jupyter Notebook 

## 1.1 Tipe Cell

### Tipe cell dalam notebook:

**1. Markdown**

Cell markdown diperuntuhkan untuk menuliskan narasi.

# Halo 1
## Halo 2

Pada bagian *markdown* terdapat beberapa hal yang dapat dilakukan, seperti membuat beberapa hal berikut ini:

- Heading

Pada bagian ini dapat ditambahkan heading dengan menambahkan hashtag `#`.

    - `#` -> Heading 1
    - `##` -> Heading 2
    - `###` -> Heading 3

- Emphasis

Ketika ingin mengatur jenis tulisan dengan memberikan karakter yang lebih tegas kita bisa memanfaatkan `*`.

    - *kata* -> Untuk mengatur tulisan menjadi Italic
    - **kata** -> Untuk mengatur tulisan menjadi Bold
    - ***kata*** -> Untuk mengatur tulisan menjadi Italic & Bold
    
- Bullets

Untuk membuat beberapa point, terdapat beberapa metode yang bisa digunakan.
    
    - Untuk membuat point dalam bentuk angka, bisa menggunakan angka 1.
    - Untuk membuat point dalam bentuk bullets, bisa menggunakan - atau *.

**2. Code** 

Cell code diperuntuhkan untuk menuliskan kode.


```python
# Ini merupakan cell untuk code
print("hello world")
```

    hello world
    

**Mini Quiz**

Q: Apakah perbedaan paling mencolok antara cell mardown dan code?

A: 

### Mode cell dalam notebook:

**1. Command mode (cell berwarna BIRU)**

- B: menambahkan cell baru di Bawah (Below)
- A: menambahkan cell baru di Atas (Above)
- DD: Delete cell
- C: Copy cell
- V: Paste cell
- Y: Mengubah ke code cell
- M: Mengubah ke markdown cell
- Enter: Mengubah command mode menjadi edit mode

**2. Edit mode (cell berwarna HIJAU)**

- Ctrl + Enter: eksekusi satu cell
- Esc: Mengubah edit mode menjadi command mode

## 1.2 Shortcut

Kumpulan shortcut: **CTRL + SHIFT + P**

# 2. Basic Python Programming

## 2.1 Variables

**Variable** adalah sebuah nama yang dipakai untuk menunjukkan sebuah nilai. Tanda `=` dipakai untuk membuat variable baru. Proses ini sering disebut sebagai **assignment**.


```python
# melakukan assignment
perusahaan = "Algoritma"

# print isi objek
perusahaan
```




    'Algoritma'



Mari kita coba buat sebuah objek, yang berisikan nama kita!


```python
# code here
nama = 'Gama Candra Tri Kartika'

print(nama)
```

    Gama Candra Tri Kartika
    

## 2.2 Case Sensitive

Python adalah bahasa pemrograman yang **case-sensitive** sehingga penamaan variable menjadi hal yang perlu diperhatikan. 

Misal kita ingin memanggil objek perusahaan tapi dengan huru p kapital.


```python
# Memanggil objek sebelumnya
perusahaan
# Case sensitive
```




    'Algoritma'



Berikut beberapa anjuran dalam memberikan nama variable pada Python:
- Special character `!, $ , &, dll` tidak dapat digunakan dalam penamaan variabel.
- Tidak boleh menggunakan angka di awal.
- Bersifat case-sensitive sehingga penamaan variable `algoritma`, `ALGORITMA`, dan `Algoritma` adalah 3 variable yang berbeda
- Tidak boleh menggunakan keyword pada Python

## 2.3 Keywords

**Keywords** adalah kata kunci yang sudah ditetapkan oleh Python sebagai nama yang tidak bisa dipakai baik untuk penamaan fungsi, variabel, dan lainnya. Keyword ditulis dalam lower-case (huruf kecil semua) kecuali keyword `True`, `False`, dan `None`. Sejauh ini (Python 3.10) keyword yang ada pada Python adalah sebagai berikut:


```python
#Cek daftar keyword
import keyword
keyword.kwlist
```




    ['False',
     'None',
     'True',
     'and',
     'as',
     'assert',
     'async',
     'await',
     'break',
     'class',
     'continue',
     'def',
     'del',
     'elif',
     'else',
     'except',
     'finally',
     'for',
     'from',
     'global',
     'if',
     'import',
     'in',
     'is',
     'lambda',
     'nonlocal',
     'not',
     'or',
     'pass',
     'raise',
     'return',
     'try',
     'while',
     'with',
     'yield']



Untuk membuktikan bahwa keyword tidak dapat digunakan sebagai nama variabel, mari kita coba untuk menyimpan nilai 1 pada variabel `True`.


```python
# Contoh ketika menyimpan kedalam sebuah keywords
True = 1
```


      Cell In[7], line 2
        True = 1
        ^
    SyntaxError: cannot assign to True
    


# 3. Introduction to Pandas Library

## 3.1 Import Library

`pandas` adalah library yang powerful sebagai tools analisis data dan struktur pada Python. Dengan `pandas`, mengolah data menjadi mudah karena disediakan salah satu objek bernama **DataFrame**. Dengan dataframe kita dapat membaca sebuah file, mengolah suatu data dengan menggunakan operasi seperti join, distinct, group by, agregasi, dan teknik lainnya.

> Lebih lengkapnya silahkan kunjungi [official documentation](https://pandas.pydata.org/)

Untuk menggunakan `pandas`, kita perlu import terlebih dahulu library dengan cara berikut ini:


```python
#code here
import pandas
```

Mari kita coba cek versi library dengan cara memanggil nama library lalu ditambahkan dengan syntax `.__version__`


```python
#code here
pandas.__version__
```




    '2.0.3'



Kita bisa menggunakan teknik **aliasing** agar pengetikan nama library tidak terlalu panjang, yaitu dengan `as`.


```python
#code here
import pandas as pd
```

Setelah menggunakan teknik aliasing kita hanya perlu memanggil nama asli library, melainkan kita bisa memanggil nama aliasing yang sudah kita tentukan sebelumnya.

Untuk lebih memahaminya mari kita coba cek versi pandas kita kembali.


```python
#code here
pd.__version__
```




    '2.0.3'



## 3.2 Membaca Data

Dalam membaca data kita akan menggunakan salah satu fungsi dari pandas dan dalam memanfaatkan fungsi dari pandas kita harus mengikuti syntax yang sudah disediakan.

Semua method pada `pandas` dapat dipanggil dengan syntax seperti: `pandas.function_name()`. Langkah pertama yang akan kita lakukan adalah membaca data. Kita dapat menggunakan method `.read_csv()` untuk membaca sebuah file dengan format `.csv`.


```python
# code here
rice = pd.read_csv("data_input/rice.csv", index_col=0)
rice.head()
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
      <th>receipt_id</th>
      <th>receipts_item_id</th>
      <th>purchase_time</th>
      <th>category</th>
      <th>sub_category</th>
      <th>format</th>
      <th>unit_price</th>
      <th>discount</th>
      <th>quantity</th>
      <th>yearmonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9622257</td>
      <td>32369294</td>
      <td>7/22/2018 21:19</td>
      <td>Rice</td>
      <td>Rice</td>
      <td>supermarket</td>
      <td>128000.0</td>
      <td>0</td>
      <td>1</td>
      <td>2018-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9446359</td>
      <td>31885876</td>
      <td>7/15/2018 16:17</td>
      <td>Rice</td>
      <td>Rice</td>
      <td>minimarket</td>
      <td>102750.0</td>
      <td>0</td>
      <td>1</td>
      <td>2018-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9470290</td>
      <td>31930241</td>
      <td>7/15/2018 12:12</td>
      <td>Rice</td>
      <td>Rice</td>
      <td>supermarket</td>
      <td>64000.0</td>
      <td>0</td>
      <td>3</td>
      <td>2018-07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9643416</td>
      <td>32418582</td>
      <td>7/24/2018 8:27</td>
      <td>Rice</td>
      <td>Rice</td>
      <td>minimarket</td>
      <td>65000.0</td>
      <td>0</td>
      <td>1</td>
      <td>2018-07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9692093</td>
      <td>32561236</td>
      <td>7/26/2018 11:28</td>
      <td>Rice</td>
      <td>Rice</td>
      <td>supermarket</td>
      <td>124500.0</td>
      <td>0</td>
      <td>1</td>
      <td>2018-07</td>
    </tr>
  </tbody>
</table>
</div>



Seperti yang kita lihat, pada data kita terdapat kolom yang bernama `RowNumber`. Hal tersebut sangatlah lumrah terjadi, karena biasanya pada data dengan format csv, memiliki sebuah kolom yang berisikan *index* atau urutan dari datanya.

Untuk menghilangkan kolom tersebut, pada fungsi `read_csv()` terdapat sebuah parameter yang bernama `index_col = `

Intuisi dari parameter `index_col` pada `read_csv()` adalah menjadikan kolom pada Dataframe sebagai index pada baris. Berikut beberapa nilai yang dapat ditampung oleh parameter `index_col`.
- Angka 0, 1, 2, dst : Menunjukkan *index* atau *urutan kolom* yang akan dijadikan sebagai index baris.
- `'nama_kolom'` : Selain menggunakan nilai index nya, kita juga dapat langsung mengetikkan nama kolomnya.

Mari kita coba implementasikan parameter tersebut.


```python
# code here
turnover = pd.read_csv("data_input/turnover.csv", index_col='RowNumber')
```

**Additional Information:**

Python menggunakan sistem **zero based indexing** yang berarti, urutan pada python dimulai dari angka 0.

## 3.3 Head & Tail

Daripada melihat keseluruhan data, lebih baik kita "mengintip" sebagian baris yang dapat merepresentasikan bentuk keseluruhan data.

Fungsi `head()` untuk melihat beberapa baris teratas pada data (default 5)


```python
# code here
turnover.head(5)
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Fungsi `tail()` untuk melihat beberapa data terakhir.


```python
# code here
turnover.tail(5)
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>9996</th>
      <td>15606229</td>
      <td>Obijiaku</td>
      <td>771</td>
      <td>France</td>
      <td>Male</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>15569892</td>
      <td>Johnstone</td>
      <td>516</td>
      <td>France</td>
      <td>Male</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101699.77</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>15628319</td>
      <td>Walker</td>
      <td>792</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
turnover.sample(5)
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>6190</th>
      <td>15572408</td>
      <td>Chambers</td>
      <td>714</td>
      <td>Germany</td>
      <td>Male</td>
      <td>39</td>
      <td>3</td>
      <td>149887.49</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>63846.36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4780</th>
      <td>15711843</td>
      <td>Pisani</td>
      <td>613</td>
      <td>Germany</td>
      <td>Male</td>
      <td>40</td>
      <td>1</td>
      <td>147856.82</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>107961.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2074</th>
      <td>15573309</td>
      <td>Ward</td>
      <td>626</td>
      <td>Spain</td>
      <td>Female</td>
      <td>48</td>
      <td>2</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>95794.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5404</th>
      <td>15733169</td>
      <td>Craig</td>
      <td>590</td>
      <td>Spain</td>
      <td>Male</td>
      <td>22</td>
      <td>7</td>
      <td>125265.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>161253.08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4363</th>
      <td>15618695</td>
      <td>Ts'ui</td>
      <td>571</td>
      <td>Spain</td>
      <td>Female</td>
      <td>22</td>
      <td>3</td>
      <td>108117.10</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>53328.70</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3.4 Tipe Data (Data Types)

Dataframe terdiri dari beberapa **Series** (mengacu pada satu kolom). Dalam satu series harus memiliki satu tipe data yang sama. `pandas`akan mencoba untuk menetapkan tipe data dari masing-masing Series, **tapi tidak selalu benar.**

### 3.4.1 Cara cek tipe data: 

Kita dapat memanfaakan kedua hal berikut ini untuk mengecek tipe data, `dtypes` atau `.info()` untuk lebih lengkapnya

🔎 **Method vs. Atribut**

- Secara fisik/terlihat mata, method selalu diikuti dengan tanda kurung ()
    * Contoh : head(). tail(), read_csv()
- Secara fisik/terlihat mata, atribut tidak diikuti oleh tanda kurung
    * Contoh : dtypes
- Didalam sebuah method, nilai parameter itu bisa diganti-ganti
    * Contoh : head(n=) -> parameter n bisa diganti/disesuaikan dengan jumlah baris yang mau ditampilkan
- Pada sebuah atribut, penggunaaan apa adanya/tidak ada nilai yang bisa diganti ganti


```python
# check tipe data dengan dtypes
print("Datatypes dari data Rice")
print(rice.dtypes)
print("\nDatatypes dari data Turnover")
print(turnover.dtypes)
```

    Datatypes dari data Rice
    receipt_id            int64
    receipts_item_id      int64
    purchase_time        object
    category             object
    sub_category         object
    format               object
    unit_price          float64
    discount              int64
    quantity              int64
    yearmonth            object
    dtype: object
    
    Datatypes dari data Turnover
    CustomerId           int64
    Surname             object
    CreditScore          int64
    Geography           object
    Gender              object
    Age                  int64
    Tenure               int64
    Balance            float64
    NumOfProducts        int64
    HasCrCard            int64
    IsActiveMember       int64
    EstimatedSalary    float64
    Exited               int64
    dtype: object
    


```python
# check tipe data dengan .info()
turnover.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10000 entries, 1 to 10000
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   CustomerId       10000 non-null  int64  
     1   Surname          10000 non-null  object 
     2   CreditScore      10000 non-null  int64  
     3   Geography        10000 non-null  object 
     4   Gender           10000 non-null  object 
     5   Age              10000 non-null  int64  
     6   Tenure           10000 non-null  int64  
     7   Balance          10000 non-null  float64
     8   NumOfProducts    10000 non-null  int64  
     9   HasCrCard        10000 non-null  int64  
     10  IsActiveMember   10000 non-null  int64  
     11  EstimatedSalary  10000 non-null  float64
     12  Exited           10000 non-null  int64  
    dtypes: float64(2), int64(8), object(3)
    memory usage: 1.1+ MB
    

### Gunain dtypes jika ingin mengambil value `dtype`nya, sedangkan `info` untuk melihat ringkasan null values, kolom, dan tipe datanya

### 3.4.2 Categorical and Numerical Variables

Karakteristik tipe data `category` :

- Dapat dikelompokkan menjadi beberapa kelompok (category)
- Berulang

Dua alasan mengapa kita perlu menggunakan tipe data categorical:

1. ***Dari sisi "business perspective"***, hal ini dapat menginformasikan dan memandu seorang Analyst pada pertanyaan seperti metode statistik atau tipe plot mana yang digunakan untuk mengolah data.

2. ***Dari sisi teknikal***, ketika kita bekerja dengan tipe data categorical pada pandas, hal ini akan jauh menghemat memori dan menambah kecepatan komputasional.



Mari kita cek kembali tipe data pada object `turnover`. Manakah yang seharusnya memiliki tipe data category?

Kita bisa menggunakan method berikut untuk mengidentifikasi kolom mana yang cocok untuk disimpan ke tipe data `category`

- `.unique()` to see unique values of a Series
- `.nunique()` to see number of unique values of a Series or DataFrame

Berikut contoh syntax untuk mengecek nilai unik pada sebuah kolom
> `df['nama_kolom'].unique()`


```python
# mengecek kolom format dengan fungsi unique()
for kolom in turnover.columns:
    if turnover[kolom].dtypes == 'object':
        print("Kolom : " + kolom)
        print(turnover[kolom].unique())
```

    Kolom : Surname
    ['Hargrave' 'Hill' 'Onio' ... 'Kashiwagi' 'Aldridge' 'Burbidge']
    Kolom : Geography
    ['France' 'Spain' 'Germany']
    Kolom : Gender
    ['Female' 'Male']
    


```python
# mengecek kolom format dengan fungsi nunique()
for kolom in turnover.columns:
    if turnover[kolom].dtypes == 'object':
        print("Kolom : " + kolom)
        print(turnover[kolom].nunique())
```

    Kolom : Surname
    2932
    Kolom : Geography
    3
    Kolom : Gender
    2
    


```python
for kolom in turnover.columns:
    if turnover[kolom].dtypes == 'object':
        print("Kolom : " + kolom)
        print(turnover[kolom].value_counts())
```

    Kolom : Surname
    Surname
    Smith       32
    Scott       29
    Martin      29
    Walker      28
    Brown       26
                ..
    Izmailov     1
    Bold         1
    Bonham       1
    Poninski     1
    Burbidge     1
    Name: count, Length: 2932, dtype: int64
    Kolom : Geography
    Geography
    France     5014
    Germany    2509
    Spain      2477
    Name: count, dtype: int64
    Kolom : Gender
    Gender
    Male      5457
    Female    4543
    Name: count, dtype: int64
    

Dari hasil pengecekan, kolom mana saja yang harus diubah menjadi tipe data kategori?

1. 

Untuk mengubah tipe data ke categorical pada pandas, Anda dapat melakukannya dengan method `astype()` berikut:

**Formula** 
```
df['column_name'] = df['column_name'].astype('new_data_types')
```

**Contoh**
```
employees['marital_status'] = employees['marital_status'].astype('category')
```


```python
# turnover['Tenure'].astype('object').value_counts().plot(kind ='bar')
```


```python
# code here
# Mengubah tipe data integer (angka) atau object ke category (text atau string)
turnover[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited']] = \
turnover[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited']].astype('category')
```


```python
# Buat manggil dataframe ke tipe data tertentu
turnover.select_dtypes(include='category')
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
      <th>Geography</th>
      <th>Gender</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spain</td>
      <td>Female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>France</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>France</td>
      <td>Male</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>France</td>
      <td>Female</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Germany</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
# cek tipe data
turnover.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10000 entries, 1 to 10000
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype   
    ---  ------           --------------  -----   
     0   CustomerId       10000 non-null  int64   
     1   Surname          10000 non-null  object  
     2   CreditScore      10000 non-null  int64   
     3   Geography        10000 non-null  category
     4   Gender           10000 non-null  category
     5   Age              10000 non-null  int64   
     6   Tenure           10000 non-null  int64   
     7   Balance          10000 non-null  float64 
     8   NumOfProducts    10000 non-null  int64   
     9   HasCrCard        10000 non-null  category
     10  IsActiveMember   10000 non-null  category
     11  EstimatedSalary  10000 non-null  float64 
     12  Exited           10000 non-null  category
    dtypes: category(5), float64(2), int64(5), object(1)
    memory usage: 752.6+ KB
    

**Additional Information:**

- Jika mengubah tipe data untuk ***angka tanpa koma*** kita bisa mengisi fungsi `.astype()` dengan ***int64***
- Jika mengubah tipe data untuk ***angka dengan koma*** kita bisa mengisi fungsi `.astype()` dengan ***float64***

## 3.5 Exploratory Data Analysis I

Exploratory Data Analysis (**EDA**) mengacu pada proses melakukan investigasi awal pada data, seringkali dengan tujuan untuk lebih mengenal dengan karakteristik data tertentu. EDA dilakukan dengan bantuan ringkasan statistik dan teknik grafis sederhana untuk melihat struktur data yang kita miliki.

Beberapa tools sederhana pada `pandas` yang dapat digunakan untuk melakukan EDA adalah sebagai berikut:
- `.describe()`

### 3.5.1  `describe()`

Method `describe()` menampilkan 8 ringkasan statistika deskriptif. Secara default menampilkan ringkasan untuk kolom numerik. 

Ringkasan statistika yang dimaksud adalah sebagai berikut:
- Count: banyaknya baris pada dataframe
- Mean: rata-rata nilai
- Standard Deviation: jarak rata-rata antara data ke mean (titik pusat data)
- Minimum Value: nilai terkecil dari keseluruhan data
- 25th Percentile (Q1)
- 50th Percentile (Q2/Median)
- 75th Percentile (Q3)
- Maximum Value: nilai terbesar dari keseluruhan data


```python
turnover['CustomerId'] = turnover['CustomerId'].astype('object')
```


```python
#code here
turnover.describe()
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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>EstimatedSalary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>650.528800</td>
      <td>38.921800</td>
      <td>5.012800</td>
      <td>76485.889288</td>
      <td>1.530200</td>
      <td>100090.239881</td>
    </tr>
    <tr>
      <th>std</th>
      <td>96.653299</td>
      <td>10.487806</td>
      <td>2.892174</td>
      <td>62397.405202</td>
      <td>0.581654</td>
      <td>57510.492818</td>
    </tr>
    <tr>
      <th>min</th>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>11.580000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>584.000000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>51002.110000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>652.000000</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>97198.540000</td>
      <td>1.000000</td>
      <td>100193.915000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>718.000000</td>
      <td>44.000000</td>
      <td>7.000000</td>
      <td>127644.240000</td>
      <td>2.000000</td>
      <td>149388.247500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>199992.480000</td>
    </tr>
  </tbody>
</table>
</div>



**Additional Information**

Kita bisa menambahkan parameter `include` ataupun `exclude` pada `describe()` untuk melihat statistika deskriptif dari variable non-numeric:

**Task 1:** Melihat statistika deskriptif untuk kolom bertipe data **selain angka**  menggunakan `exclude = 'number'`


```python
#code here
turnover.describe(exclude ='number')
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10000</td>
      <td>2932</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>15634602</td>
      <td>Smith</td>
      <td>France</td>
      <td>Male</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>32</td>
      <td>5014</td>
      <td>5457</td>
      <td>7055</td>
      <td>5151</td>
      <td>7963</td>
    </tr>
  </tbody>
</table>
</div>



**Task 2:** Melihat statistika deskriptif untuk kolom bertipe data **object** menggunakan `include = 'object'`


```python
#code here
turnover.describe(include =['object'])
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
      <th>CustomerId</th>
      <th>Surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10000</td>
      <td>2932</td>
    </tr>
    <tr>
      <th>top</th>
      <td>15634602</td>
      <td>Smith</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>



## 3.6 Indexing and Subsetting with Pandas

Indexing digunakan untuk memilih dan mengambil sebagian data yang hanya diperlukan dalam proses analisa data yang sedang dikerjakan. Contohnya:
- Compare sales pada tahun 2018 vs 2019
- Identifikasi peluang penjualan pada segment pasar (ex : Wholesale vs Retail)
- Melihat quarter terbaik untuk setiap tahun yang dapat digunakan untuk tujuan promosi
- dan sebagainya

### 3.6.1 `select_dtypes()`

Method `select_dtypes()` digunakan untuk memilih kolom sesuai dengan tipe datanya. Ada 2 parameter yang dapat digunakan di dalam method `select_dtypes()` yaitu parameter `include` dan `exclude` (seperti pada `describe()`.

Misal:
- parameter `include = 'category'` artinya kita memilih semua kolom dengan tipe data 'category'
- sebaliknya, ketika menggunakan parameter `exclude = 'category'` maka kolom-kolom dengan tipe data selain 'category' akan ditampilkan.


**Task 1:** Mengambil kolom dengan tipe data *category*


```python
# code here
turnover.select_dtypes(include=['category'])
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
      <th>Geography</th>
      <th>Gender</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spain</td>
      <td>Female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>France</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>France</td>
      <td>Male</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>France</td>
      <td>Female</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Germany</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>



**Task 2:** Mengambil kolom dengan tipe data *number*


```python
# code here
turnover.select_dtypes(include=['number'])
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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>EstimatedSalary</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>619</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>101348.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>608</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>112542.58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>502</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>113931.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>699</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>93826.63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>850</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>79084.10</td>
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
      <th>9996</th>
      <td>771</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>96270.64</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>516</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>101699.77</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>709</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>42085.58</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>772</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>92888.52</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>792</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>38190.78</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 6 columns</p>
</div>



### 3.6.2 `drop()`

Method `drop()` digunakan untuk membuang baris atau kolom yang tidak ingin digunakan untuk tujuan analisis. 

- Untuk menghapus baris pada fungsi tersebut dapat kita isi dengan parameter `index =`
- Untuk menghapus baris pada fungsi tersebut dapat kita isi dengan parameter `columns =`

**Task 1**: Hapus baris pertama pada DataFrame!


```python
kopi = turnover.copy()
kopi.head(5)
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# code here
turnover.drop(index=[1]).head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15574012</td>
      <td>Chu</td>
      <td>645</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>8</td>
      <td>113755.78</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>149756.71</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Task 2**: Hapus baris ke-2 dan ke-3 pada data!


```python
# code here
turnover.drop(index=[2,3]).head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15574012</td>
      <td>Chu</td>
      <td>645</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>8</td>
      <td>113755.78</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>149756.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15592531</td>
      <td>Bartlett</td>
      <td>822</td>
      <td>France</td>
      <td>Male</td>
      <td>50</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>10062.80</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Task 3**: Hapus kolom `Age` dan `Tenure`!


```python
# code here
turnover.drop(columns=['Age','Tenure']).head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
turnover.drop(['Age', 'Tenure'], axis=1).head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
turnover.columns[0]
```




    'CustomerId'



**Task 4**: Hapus baris ke 2 dan kolom `CustomerId`!


```python
# code here
turnover.drop(index=[2] ,columns=['CustomerId']).head()
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
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chu</td>
      <td>645</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>8</td>
      <td>113755.78</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>149756.71</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**NOTE:** Method `drop()` tidak mengubah objek dataframenya. Apabila ingin mengubah objek semulanya:
- Melakukan assignment kembali dengan nama objek yang sama: `turnover = turnover.drop(...)`, atau
- Menambahkan parameter inplace: `turnover.drop(..., inplace=True)` 

### 3.6.3 Slicing: **`[]` operator**

Digunakan untuk melakukan subsetting dengan cara mengiris (slicing) index pada dataframe. Formula penulisannya adalah `[start:end]` dengan mengikuti aturan indexing pada python (dimulai dari 0) dimana `start` inclusive dan `end` exclusive.

**Task 1:**  Dengan menggunakan metode slicing, silahkan tampilkan baris ke 1 dan ke 2

* baris ke 1 memiliki urutan index 0
* baris ke 2 memiliki urutan index 1


```python
# code here
turnover[0:2]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Task 2:** Dengan menggunakan metode slicing, silahkan tampilkan baris ke 10 sampai ke 15


```python
# code here
turnover[9:15]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>10</th>
      <td>15592389</td>
      <td>H?</td>
      <td>684</td>
      <td>France</td>
      <td>Male</td>
      <td>27</td>
      <td>2</td>
      <td>134603.88</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>71725.73</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15767821</td>
      <td>Bearce</td>
      <td>528</td>
      <td>France</td>
      <td>Male</td>
      <td>31</td>
      <td>6</td>
      <td>102016.72</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>80181.12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>15737173</td>
      <td>Andrews</td>
      <td>497</td>
      <td>Spain</td>
      <td>Male</td>
      <td>24</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>76390.01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15632264</td>
      <td>Kay</td>
      <td>476</td>
      <td>France</td>
      <td>Female</td>
      <td>34</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>26260.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15691483</td>
      <td>Chin</td>
      <td>549</td>
      <td>France</td>
      <td>Female</td>
      <td>25</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>190857.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15600882</td>
      <td>Scott</td>
      <td>635</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>65951.65</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Task 3:** Dengan menggunakan metode slicing, silahkan tampilkan baris ke 17 sampai ke 21


```python
# code here
turnover[16:21]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>17</th>
      <td>15737452</td>
      <td>Romeo</td>
      <td>653</td>
      <td>Germany</td>
      <td>Male</td>
      <td>58</td>
      <td>1</td>
      <td>132602.88</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5097.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>15788218</td>
      <td>Henderson</td>
      <td>549</td>
      <td>Spain</td>
      <td>Female</td>
      <td>24</td>
      <td>9</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>14406.41</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15661507</td>
      <td>Muldrow</td>
      <td>587</td>
      <td>Spain</td>
      <td>Male</td>
      <td>45</td>
      <td>6</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>158684.81</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>15568982</td>
      <td>Hao</td>
      <td>726</td>
      <td>France</td>
      <td>Female</td>
      <td>24</td>
      <td>6</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>54724.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>15577657</td>
      <td>McDonald</td>
      <td>732</td>
      <td>France</td>
      <td>Male</td>
      <td>41</td>
      <td>8</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>170886.17</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3.7 `.loc` dan `.iloc`

Dengan menggunakan `.loc` dan `iloc` kita dapat melakukan pengirisan pada index **baris dan kolom**. 

Perbedaan yang mendasar dari kedua operator ini adalah:
- `.iloc` merujuk pada lokasi **index** baris atau kolomnya sehingga harus **integer**, sedangkan
- `.loc` merujuk pada lokasi **nama** baris atau kolomnya

**Mari berfokus pada .iloc terlebih dahulu**

> Syntax: `df.iloc[baris, kolom]` 

**Task 1:** Tampilkan baris dengan index 3 dan kolom dengan index 1, artinya adalah data pada baris ke empat dan kolom `Surname`.


```python
# code here
turnover.iloc[3, 1] 
```




    'Boni'




```python
# Buat ngecek lokasi indeks kolom
turnover.columns.get_loc('Age')
```




    5



**Task 2:** Mengambil baris ke 2 sampai 5 dan kolom dengan `CustomerId` sampai `Age`


```python
# code here
turnover.iloc[1:5, turnover.columns.get_loc('CustomerId'):turnover.columns.get_loc('Age')+1] 
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



**Task 3:** Menampilkan semua data pada kolom `Surname` dan `Exited` saja


```python
# code here
turnover.iloc[::, [turnover.columns.get_loc('Surname'),turnover.columns.get_loc('Exited')]] 
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
      <th>Surname</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hargrave</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hill</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Onio</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boni</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mitchell</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Obijiaku</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>Johnstone</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>Liu</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Sabbatini</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>Walker</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>



**Mari kita pergi ke .loc**

> Syntax: `df.loc[baris, kolom]` 

Menggunakan `.loc`, kita bisa mengambil baris dan kolom berdasarkan namanya. 

**Task 1:** Mengambil baris ke 2 sampai 5 dan kolom dengan `CustomerId` sampai `Age`


```python
# code here
turnover.loc[2:5, 'CustomerId':'Age']
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



**Task 2:** Menampilkan semua data pada kolom `Surname` dan `Exited` saja


```python
%%time
# code here
turnover.loc[::, ['Surname','Exited']]
```

    CPU times: total: 0 ns
    Wall time: 998 µs
    




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
      <th>Surname</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hargrave</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hill</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Onio</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boni</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mitchell</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Obijiaku</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>Johnstone</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>Liu</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Sabbatini</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>Walker</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>




```python
%%time
turnover[['Surname','Exited']][::]
```

    CPU times: total: 0 ns
    Wall time: 1.01 ms
    




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
      <th>Surname</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hargrave</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hill</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Onio</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boni</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mitchell</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Obijiaku</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>Johnstone</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>Liu</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Sabbatini</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>Walker</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>



## 3.8 Conditional Subsetting

Selain menggunakan `.loc` dan `.iloc`, kita dapat melakukan subsetting berdasarkan kondisi tertentu. Misal pada dataframe `turnover`, kita ingin mengambil beberapa data dengan kondisi sebagai berikut:

- Customer yang memutuskan untuk *churn*: `.Exited == 1`
- Customer yang memiliki *balance* di atas 200000:`.Balance >= 200000`
- Customer dengan *credit score* sebesar 850: `.CreditScore != 0`

Syntax penulisan untuk conditional subsetting adalah:

**`df[df['column_name'] <comparison_operator> <value>]`**

Contoh comparison_operator adalah seperti `==`, `!=`, `>`, `>=`, `<`, `<=`.

**Task 1:** Tampilkan data customer yang memutuskan untuk *churn*


```python
# code here
# turnover.loc[turnover['Exited'] == 1].loc[::,'Surname']
# atau
turnover[turnover['Exited'] == 1]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>1</th>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15574012</td>
      <td>Chu</td>
      <td>645</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>8</td>
      <td>113755.78</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>149756.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15656148</td>
      <td>Obinna</td>
      <td>376</td>
      <td>Germany</td>
      <td>Female</td>
      <td>29</td>
      <td>4</td>
      <td>115046.74</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>119346.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>15737452</td>
      <td>Romeo</td>
      <td>653</td>
      <td>Germany</td>
      <td>Male</td>
      <td>58</td>
      <td>1</td>
      <td>132602.88</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5097.67</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>9982</th>
      <td>15672754</td>
      <td>Burbidge</td>
      <td>498</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>152039.70</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>53445.17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>15768163</td>
      <td>Griffin</td>
      <td>655</td>
      <td>Germany</td>
      <td>Female</td>
      <td>46</td>
      <td>7</td>
      <td>137145.12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>115146.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>15769959</td>
      <td>Ajuluchukwu</td>
      <td>597</td>
      <td>France</td>
      <td>Female</td>
      <td>53</td>
      <td>4</td>
      <td>88381.21</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>69384.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2037 rows × 13 columns</p>
</div>



**Task 2:** Tampilkan data customer yang memiliki *balance* di atas 200000


```python
# code here
turnover[turnover['Balance'] > 200000]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>139</th>
      <td>15594408</td>
      <td>Chia</td>
      <td>584</td>
      <td>Spain</td>
      <td>Female</td>
      <td>48</td>
      <td>2</td>
      <td>213146.20</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>75161.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>521</th>
      <td>15671256</td>
      <td>Macartney</td>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>211774.31</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>188574.12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>721</th>
      <td>15721658</td>
      <td>Fleming</td>
      <td>672</td>
      <td>Spain</td>
      <td>Female</td>
      <td>56</td>
      <td>2</td>
      <td>209767.31</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>150694.42</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>15599131</td>
      <td>Dilke</td>
      <td>650</td>
      <td>Germany</td>
      <td>Male</td>
      <td>26</td>
      <td>4</td>
      <td>214346.96</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>128815.33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>15588670</td>
      <td>Despeissis</td>
      <td>705</td>
      <td>Spain</td>
      <td>Female</td>
      <td>40</td>
      <td>5</td>
      <td>203715.15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>179978.68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1366</th>
      <td>15689514</td>
      <td>Kang</td>
      <td>625</td>
      <td>France</td>
      <td>Male</td>
      <td>43</td>
      <td>8</td>
      <td>201696.07</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>133020.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>15769818</td>
      <td>Moore</td>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>37</td>
      <td>3</td>
      <td>212778.20</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>69372.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2093</th>
      <td>15757408</td>
      <td>Lo</td>
      <td>655</td>
      <td>Spain</td>
      <td>Male</td>
      <td>38</td>
      <td>3</td>
      <td>250898.09</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>81054.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2598</th>
      <td>15668818</td>
      <td>Chidubem</td>
      <td>592</td>
      <td>Spain</td>
      <td>Female</td>
      <td>40</td>
      <td>2</td>
      <td>200322.45</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>113244.73</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2710</th>
      <td>15780212</td>
      <td>Mao</td>
      <td>592</td>
      <td>France</td>
      <td>Male</td>
      <td>37</td>
      <td>4</td>
      <td>212692.97</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>176395.02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>15795298</td>
      <td>Olisaemeka</td>
      <td>573</td>
      <td>Germany</td>
      <td>Female</td>
      <td>35</td>
      <td>9</td>
      <td>206868.78</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>102986.15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3281</th>
      <td>15715622</td>
      <td>To Rot</td>
      <td>583</td>
      <td>France</td>
      <td>Female</td>
      <td>57</td>
      <td>3</td>
      <td>238387.56</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>147964.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3589</th>
      <td>15571958</td>
      <td>McIntosh</td>
      <td>489</td>
      <td>Spain</td>
      <td>Male</td>
      <td>40</td>
      <td>3</td>
      <td>221532.80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>171867.08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3921</th>
      <td>15620268</td>
      <td>Thomson</td>
      <td>634</td>
      <td>Germany</td>
      <td>Male</td>
      <td>43</td>
      <td>3</td>
      <td>212696.32</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>115268.86</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4437</th>
      <td>15664498</td>
      <td>Golovanov</td>
      <td>508</td>
      <td>France</td>
      <td>Male</td>
      <td>26</td>
      <td>7</td>
      <td>205962.00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>156424.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4534</th>
      <td>15607275</td>
      <td>Ch'ang</td>
      <td>850</td>
      <td>Spain</td>
      <td>Male</td>
      <td>39</td>
      <td>6</td>
      <td>206014.94</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>42774.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5255</th>
      <td>15746664</td>
      <td>Ts'ui</td>
      <td>463</td>
      <td>Spain</td>
      <td>Male</td>
      <td>20</td>
      <td>8</td>
      <td>204223.03</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>128268.39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5872</th>
      <td>15709920</td>
      <td>Burke</td>
      <td>479</td>
      <td>France</td>
      <td>Female</td>
      <td>33</td>
      <td>2</td>
      <td>208165.53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>50774.81</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6272</th>
      <td>15620756</td>
      <td>Stokes</td>
      <td>747</td>
      <td>France</td>
      <td>Male</td>
      <td>49</td>
      <td>6</td>
      <td>202904.64</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>17298.72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6498</th>
      <td>15793688</td>
      <td>Bancks</td>
      <td>669</td>
      <td>France</td>
      <td>Male</td>
      <td>50</td>
      <td>9</td>
      <td>201009.64</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>158032.50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6718</th>
      <td>15586674</td>
      <td>Shaw</td>
      <td>663</td>
      <td>Spain</td>
      <td>Female</td>
      <td>58</td>
      <td>5</td>
      <td>216109.88</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>74176.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6914</th>
      <td>15784180</td>
      <td>Ku</td>
      <td>564</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>206329.65</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>46632.87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7354</th>
      <td>15736420</td>
      <td>Macdonald</td>
      <td>596</td>
      <td>France</td>
      <td>Male</td>
      <td>21</td>
      <td>4</td>
      <td>210433.08</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>197297.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7493</th>
      <td>15776545</td>
      <td>Napolitani</td>
      <td>682</td>
      <td>France</td>
      <td>Male</td>
      <td>28</td>
      <td>10</td>
      <td>200724.96</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>82872.64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7633</th>
      <td>15620570</td>
      <td>Sinnett</td>
      <td>736</td>
      <td>France</td>
      <td>Male</td>
      <td>43</td>
      <td>4</td>
      <td>202443.47</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>72375.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7888</th>
      <td>15745433</td>
      <td>Conti</td>
      <td>716</td>
      <td>Germany</td>
      <td>Female</td>
      <td>30</td>
      <td>2</td>
      <td>205770.78</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>65464.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8028</th>
      <td>15769412</td>
      <td>Atkinson</td>
      <td>684</td>
      <td>Spain</td>
      <td>Male</td>
      <td>39</td>
      <td>4</td>
      <td>207034.96</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>157694.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8064</th>
      <td>15663888</td>
      <td>Connor</td>
      <td>549</td>
      <td>Germany</td>
      <td>Male</td>
      <td>34</td>
      <td>6</td>
      <td>204017.40</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>109538.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8703</th>
      <td>15690589</td>
      <td>Udinesi</td>
      <td>541</td>
      <td>France</td>
      <td>Male</td>
      <td>37</td>
      <td>9</td>
      <td>212314.03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>148814.54</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8734</th>
      <td>15714241</td>
      <td>Haddon</td>
      <td>749</td>
      <td>Spain</td>
      <td>Male</td>
      <td>42</td>
      <td>9</td>
      <td>222267.63</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>101108.85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8795</th>
      <td>15578671</td>
      <td>Webb</td>
      <td>706</td>
      <td>Spain</td>
      <td>Female</td>
      <td>29</td>
      <td>1</td>
      <td>209490.21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>133267.69</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8983</th>
      <td>15627971</td>
      <td>Coates</td>
      <td>504</td>
      <td>France</td>
      <td>Female</td>
      <td>32</td>
      <td>8</td>
      <td>206663.75</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>16281.94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9834</th>
      <td>15807245</td>
      <td>McKay</td>
      <td>699</td>
      <td>Germany</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>200117.76</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>94142.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9921</th>
      <td>15673020</td>
      <td>Smith</td>
      <td>678</td>
      <td>France</td>
      <td>Female</td>
      <td>49</td>
      <td>3</td>
      <td>204510.94</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>738.88</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Task 3:** Tampilkan data customer dengan *credit score* sebesar 850


```python
# code here
turnover[turnover['CreditScore'] == 850]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>15717426</td>
      <td>Armstrong</td>
      <td>850</td>
      <td>France</td>
      <td>Male</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>40812.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>181</th>
      <td>15716334</td>
      <td>Rozier</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>45</td>
      <td>2</td>
      <td>122311.21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>19482.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>15604482</td>
      <td>Chiemezie</td>
      <td>850</td>
      <td>Spain</td>
      <td>Male</td>
      <td>30</td>
      <td>2</td>
      <td>141040.01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5978.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>224</th>
      <td>15733247</td>
      <td>Stevenson</td>
      <td>850</td>
      <td>France</td>
      <td>Male</td>
      <td>33</td>
      <td>10</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4861.72</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>9582</th>
      <td>15709256</td>
      <td>Glover</td>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>9</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>164864.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9647</th>
      <td>15603111</td>
      <td>Muir</td>
      <td>850</td>
      <td>Spain</td>
      <td>Male</td>
      <td>71</td>
      <td>10</td>
      <td>69608.14</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>97893.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9689</th>
      <td>15730579</td>
      <td>Ward</td>
      <td>850</td>
      <td>France</td>
      <td>Male</td>
      <td>68</td>
      <td>5</td>
      <td>169445.40</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>186335.07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9932</th>
      <td>15647800</td>
      <td>Greco</td>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>34</td>
      <td>6</td>
      <td>101266.51</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>33501.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9950</th>
      <td>15798615</td>
      <td>Wan</td>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>47</td>
      <td>9</td>
      <td>137301.87</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>44351.77</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>233 rows × 13 columns</p>
</div>



**Additional Information:**

Kita juga dapat menggunakan operator `&` (AND) dan `|` (OR) untuk melakukan subsetting lebih dari 1 kondisi. Misalnya kita ingin melihat data penjualan dari seorang pegawai bernama Moana yang jumlahnya lebih dari 5000, maka kita dapat menggunakan syntax:
```
sales[(sales.salesperson == 'Moana') & (sales.amount > 5000)]
```

Untuk subsetting dengan kondisi lebih dari 1, setiap kondisi diletakkan **di dalam tanda kurung `()`** atau bisa ditulis dengan syntax berikut:

```
df[(kondisi pertama) operator (kondisi kedua) operator (kondisi ketiga) dan seterusnya...]
```

**Poin:**
- Operator AND: harus semua kondisi terpenuhi dalam satu baris agar muncul
- Operator OR: salah satu kondisi saja sudah terpenuhi maka baris tsb muncul

**Task 1:** Lakukan subsetting untuk mengambil semua informasi customer yang memiliki *credit score* di atas 700 dan memutuskan untuk *churn*


```python
#code here
turnover[(turnover['CreditScore'] > 700) & (turnover['Exited'] == 1)]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>44</th>
      <td>15755196</td>
      <td>Lavine</td>
      <td>834</td>
      <td>France</td>
      <td>Female</td>
      <td>49</td>
      <td>2</td>
      <td>131394.56</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>194365.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>15602280</td>
      <td>Martin</td>
      <td>829</td>
      <td>Germany</td>
      <td>Female</td>
      <td>27</td>
      <td>9</td>
      <td>112045.67</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>119708.21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>15703793</td>
      <td>Konovalova</td>
      <td>738</td>
      <td>Germany</td>
      <td>Male</td>
      <td>58</td>
      <td>2</td>
      <td>133745.44</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>28373.86</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>15663706</td>
      <td>Leonard</td>
      <td>777</td>
      <td>France</td>
      <td>Female</td>
      <td>32</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>136458.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15762418</td>
      <td>Gant</td>
      <td>750</td>
      <td>Spain</td>
      <td>Male</td>
      <td>22</td>
      <td>3</td>
      <td>121681.82</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>128643.35</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>9885</th>
      <td>15686974</td>
      <td>Sergeyeva</td>
      <td>751</td>
      <td>France</td>
      <td>Female</td>
      <td>48</td>
      <td>4</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30165.06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9961</th>
      <td>15681026</td>
      <td>Lucciano</td>
      <td>795</td>
      <td>Germany</td>
      <td>Female</td>
      <td>33</td>
      <td>9</td>
      <td>104552.72</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>120853.83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9963</th>
      <td>15594612</td>
      <td>Flynn</td>
      <td>702</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>9</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>59207.41</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>619 rows × 13 columns</p>
</div>



**Task 2:**  Tampilkan informasi customer perempuan yang berdomisili di Jerman dan Spanyol 


```python
%%time
# code here
# Buat ngecek list dalam list
turnover[(turnover['Geography'].isin(['Germany', 'Spain'])) & (turnover['Gender'] == 'Female')]
```

    CPU times: total: 0 ns
    Wall time: 2.98 ms
    




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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15656148</td>
      <td>Obinna</td>
      <td>376</td>
      <td>Germany</td>
      <td>Female</td>
      <td>29</td>
      <td>4</td>
      <td>115046.74</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>119346.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15600882</td>
      <td>Scott</td>
      <td>635</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>65951.65</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>15788218</td>
      <td>Henderson</td>
      <td>549</td>
      <td>Spain</td>
      <td>Female</td>
      <td>24</td>
      <td>9</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>14406.41</td>
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
    </tr>
    <tr>
      <th>9951</th>
      <td>15638494</td>
      <td>Salinas</td>
      <td>625</td>
      <td>Germany</td>
      <td>Female</td>
      <td>39</td>
      <td>10</td>
      <td>129845.26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>96444.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9961</th>
      <td>15681026</td>
      <td>Lucciano</td>
      <td>795</td>
      <td>Germany</td>
      <td>Female</td>
      <td>33</td>
      <td>9</td>
      <td>104552.72</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>120853.83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9966</th>
      <td>15690164</td>
      <td>Shao</td>
      <td>627</td>
      <td>Germany</td>
      <td>Female</td>
      <td>33</td>
      <td>4</td>
      <td>83199.05</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>159334.93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9969</th>
      <td>15733491</td>
      <td>McGregor</td>
      <td>512</td>
      <td>Germany</td>
      <td>Female</td>
      <td>40</td>
      <td>8</td>
      <td>153537.57</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>23101.13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>15768163</td>
      <td>Griffin</td>
      <td>655</td>
      <td>Germany</td>
      <td>Female</td>
      <td>46</td>
      <td>7</td>
      <td>137145.12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>115146.40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2282 rows × 13 columns</p>
</div>




```python
# Kunci Jawaban
turnover[(turnover['CreditScore'] > turnover['CreditScore'].quantile(0.75)) \
& (turnover['Balance'] <= turnover['Balance'].quantile(0.25)) \
& (turnover['HasCrCard'] == 0) \
& (turnover['EstimatedSalary'] > turnover['EstimatedSalary'].mean()) \
& (turnover['Geography'] == turnover['Geography'].value_counts().tail(1).index[0])] 
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>218</th>
      <td>15786308</td>
      <td>Millar</td>
      <td>730</td>
      <td>Spain</td>
      <td>Female</td>
      <td>33</td>
      <td>9</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>176576.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>403</th>
      <td>15781589</td>
      <td>Carpenter</td>
      <td>751</td>
      <td>Spain</td>
      <td>Male</td>
      <td>52</td>
      <td>8</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>179291.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>15744517</td>
      <td>Esposito</td>
      <td>735</td>
      <td>Spain</td>
      <td>Male</td>
      <td>50</td>
      <td>9</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166677.35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1552</th>
      <td>15749177</td>
      <td>Maslow</td>
      <td>730</td>
      <td>Spain</td>
      <td>Female</td>
      <td>52</td>
      <td>7</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>122398.84</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1782</th>
      <td>15771636</td>
      <td>Marshall</td>
      <td>793</td>
      <td>Spain</td>
      <td>Female</td>
      <td>36</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>148993.47</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>15604588</td>
      <td>Li Fonti</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>38</td>
      <td>3</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>179360.76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>15775803</td>
      <td>Cawker</td>
      <td>841</td>
      <td>Spain</td>
      <td>Male</td>
      <td>41</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>193093.77</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>15735246</td>
      <td>Norman</td>
      <td>798</td>
      <td>Spain</td>
      <td>Female</td>
      <td>58</td>
      <td>9</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>119071.56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2388</th>
      <td>15595588</td>
      <td>Chukwunonso</td>
      <td>773</td>
      <td>Spain</td>
      <td>Female</td>
      <td>39</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>182081.45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2630</th>
      <td>15711789</td>
      <td>Davey</td>
      <td>768</td>
      <td>Spain</td>
      <td>Female</td>
      <td>42</td>
      <td>3</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>161242.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2657</th>
      <td>15713267</td>
      <td>Zimmer</td>
      <td>779</td>
      <td>Spain</td>
      <td>Female</td>
      <td>34</td>
      <td>5</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>111676.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2672</th>
      <td>15611105</td>
      <td>Castella</td>
      <td>799</td>
      <td>Spain</td>
      <td>Male</td>
      <td>35</td>
      <td>7</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>140780.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4082</th>
      <td>15762821</td>
      <td>Udinese</td>
      <td>721</td>
      <td>Spain</td>
      <td>Male</td>
      <td>33</td>
      <td>5</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>117626.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4106</th>
      <td>15701392</td>
      <td>Lucciano</td>
      <td>815</td>
      <td>Spain</td>
      <td>Male</td>
      <td>28</td>
      <td>6</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>185547.71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4701</th>
      <td>15773709</td>
      <td>Hung</td>
      <td>838</td>
      <td>Spain</td>
      <td>Male</td>
      <td>35</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>197305.91</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4779</th>
      <td>15582246</td>
      <td>Rowe</td>
      <td>737</td>
      <td>Spain</td>
      <td>Female</td>
      <td>45</td>
      <td>2</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>177695.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4907</th>
      <td>15571244</td>
      <td>Tung</td>
      <td>809</td>
      <td>Spain</td>
      <td>Female</td>
      <td>33</td>
      <td>3</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>141426.78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5015</th>
      <td>15773731</td>
      <td>John</td>
      <td>758</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>5</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>100365.51</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5041</th>
      <td>15749727</td>
      <td>Chukwufumnanya</td>
      <td>829</td>
      <td>Spain</td>
      <td>Male</td>
      <td>50</td>
      <td>7</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>178458.86</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5128</th>
      <td>15644796</td>
      <td>Dyer</td>
      <td>821</td>
      <td>Spain</td>
      <td>Female</td>
      <td>38</td>
      <td>8</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>126241.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5754</th>
      <td>15608328</td>
      <td>Sutherland</td>
      <td>760</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>6</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>101491.23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6020</th>
      <td>15697045</td>
      <td>Pisani</td>
      <td>726</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>9</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>100556.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6419</th>
      <td>15801924</td>
      <td>Browne</td>
      <td>754</td>
      <td>Spain</td>
      <td>Female</td>
      <td>27</td>
      <td>8</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>121821.16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6673</th>
      <td>15660403</td>
      <td>Fleming</td>
      <td>827</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>184514.01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7055</th>
      <td>15682860</td>
      <td>Lo</td>
      <td>769</td>
      <td>Spain</td>
      <td>Male</td>
      <td>38</td>
      <td>6</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>104393.78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7524</th>
      <td>15733602</td>
      <td>Rubin</td>
      <td>814</td>
      <td>Spain</td>
      <td>Female</td>
      <td>72</td>
      <td>2</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>130853.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7740</th>
      <td>15689952</td>
      <td>Zuyeva</td>
      <td>724</td>
      <td>Spain</td>
      <td>Male</td>
      <td>41</td>
      <td>5</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>115753.94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8374</th>
      <td>15785167</td>
      <td>Padovano</td>
      <td>795</td>
      <td>Spain</td>
      <td>Male</td>
      <td>29</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>155711.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8605</th>
      <td>15646942</td>
      <td>Meng</td>
      <td>786</td>
      <td>Spain</td>
      <td>Female</td>
      <td>39</td>
      <td>7</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>100929.59</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8707</th>
      <td>15717770</td>
      <td>Marcelo</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>55</td>
      <td>7</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>171762.87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8859</th>
      <td>15668009</td>
      <td>Hendley</td>
      <td>747</td>
      <td>Spain</td>
      <td>Male</td>
      <td>37</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>180551.76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9126</th>
      <td>15604138</td>
      <td>Iheanacho</td>
      <td>749</td>
      <td>Spain</td>
      <td>Male</td>
      <td>34</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>174189.04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9204</th>
      <td>15774401</td>
      <td>Chambers</td>
      <td>773</td>
      <td>Spain</td>
      <td>Male</td>
      <td>51</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>123587.83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9706</th>
      <td>15572374</td>
      <td>Hopetoun</td>
      <td>733</td>
      <td>Spain</td>
      <td>Male</td>
      <td>36</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>108377.82</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Jawaban Kelompok 1
turnover[(turnover['Surname'].str.lower().apply(lambda x:x[0]).isin(['a', 'i', 'u', 'e', 'o'])) \
& (turnover['CreditScore'] < turnover['CreditScore'].mean()) \
& (turnover['Balance'] > turnover['Balance'].quantile(0.75)) \
& (turnover['Exited'] == 1)].sort_values(by=['Age'], ascending=False)
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>400</th>
      <td>15646372</td>
      <td>Outhwaite</td>
      <td>616</td>
      <td>France</td>
      <td>Female</td>
      <td>66</td>
      <td>1</td>
      <td>135842.41</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>183840.51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>15684196</td>
      <td>Aitken</td>
      <td>627</td>
      <td>France</td>
      <td>Female</td>
      <td>55</td>
      <td>2</td>
      <td>159441.27</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>100686.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7809</th>
      <td>15649033</td>
      <td>Echezonachukwu</td>
      <td>603</td>
      <td>Germany</td>
      <td>Female</td>
      <td>55</td>
      <td>7</td>
      <td>127723.25</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>139469.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>15685372</td>
      <td>Azubuike</td>
      <td>350</td>
      <td>Spain</td>
      <td>Male</td>
      <td>54</td>
      <td>1</td>
      <td>152677.48</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>191973.49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>15589793</td>
      <td>Onwuamaeze</td>
      <td>604</td>
      <td>France</td>
      <td>Male</td>
      <td>53</td>
      <td>8</td>
      <td>144453.75</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>190998.96</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>15692416</td>
      <td>Aikenhead</td>
      <td>358</td>
      <td>Spain</td>
      <td>Female</td>
      <td>52</td>
      <td>8</td>
      <td>143542.36</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>141959.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6602</th>
      <td>15655213</td>
      <td>Udinese</td>
      <td>591</td>
      <td>Germany</td>
      <td>Female</td>
      <td>51</td>
      <td>8</td>
      <td>132508.30</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>161304.68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>857</th>
      <td>15693864</td>
      <td>Iheanacho</td>
      <td>567</td>
      <td>Germany</td>
      <td>Female</td>
      <td>49</td>
      <td>5</td>
      <td>134956.02</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>93953.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>15654673</td>
      <td>Onyinyechukwuka</td>
      <td>625</td>
      <td>France</td>
      <td>Male</td>
      <td>49</td>
      <td>6</td>
      <td>173434.90</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>165580.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>868</th>
      <td>15756804</td>
      <td>O'Loghlen</td>
      <td>636</td>
      <td>France</td>
      <td>Female</td>
      <td>48</td>
      <td>1</td>
      <td>170833.46</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>110510.28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3616</th>
      <td>15639357</td>
      <td>Allan</td>
      <td>415</td>
      <td>France</td>
      <td>Male</td>
      <td>46</td>
      <td>9</td>
      <td>134950.19</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>178587.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4625</th>
      <td>15710543</td>
      <td>Okwuoma</td>
      <td>629</td>
      <td>France</td>
      <td>Male</td>
      <td>46</td>
      <td>1</td>
      <td>130666.20</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>161125.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4672</th>
      <td>15808674</td>
      <td>Ejikemeifeuwa</td>
      <td>616</td>
      <td>Germany</td>
      <td>Female</td>
      <td>45</td>
      <td>6</td>
      <td>128352.59</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>144000.59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8505</th>
      <td>15743245</td>
      <td>Agafonova</td>
      <td>624</td>
      <td>France</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>145155.37</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>72169.95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>15703820</td>
      <td>Endrizzi</td>
      <td>552</td>
      <td>France</td>
      <td>Male</td>
      <td>42</td>
      <td>9</td>
      <td>133701.07</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>101069.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2382</th>
      <td>15774151</td>
      <td>Iadanza</td>
      <td>614</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>7</td>
      <td>179915.85</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>14666.35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2478</th>
      <td>15760294</td>
      <td>Endrizzi</td>
      <td>512</td>
      <td>France</td>
      <td>Female</td>
      <td>41</td>
      <td>8</td>
      <td>145150.28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>64869.32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9637</th>
      <td>15613048</td>
      <td>Anderson</td>
      <td>648</td>
      <td>Germany</td>
      <td>Female</td>
      <td>40</td>
      <td>5</td>
      <td>139973.65</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>667.66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>15771573</td>
      <td>Okagbue</td>
      <td>637</td>
      <td>Germany</td>
      <td>Female</td>
      <td>39</td>
      <td>9</td>
      <td>137843.80</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>117622.80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>15593969</td>
      <td>Abramovich</td>
      <td>630</td>
      <td>Spain</td>
      <td>Female</td>
      <td>39</td>
      <td>7</td>
      <td>135483.17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>140881.20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8831</th>
      <td>15810444</td>
      <td>Aksenov</td>
      <td>562</td>
      <td>Germany</td>
      <td>Female</td>
      <td>39</td>
      <td>6</td>
      <td>130565.02</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>9854.72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6440</th>
      <td>15583371</td>
      <td>Artemiev</td>
      <td>632</td>
      <td>Spain</td>
      <td>Male</td>
      <td>37</td>
      <td>1</td>
      <td>138207.08</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>60778.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3400</th>
      <td>15633352</td>
      <td>Okwukwe</td>
      <td>628</td>
      <td>France</td>
      <td>Female</td>
      <td>31</td>
      <td>6</td>
      <td>175443.75</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>113167.17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>15676242</td>
      <td>Artemova</td>
      <td>632</td>
      <td>Spain</td>
      <td>Male</td>
      <td>31</td>
      <td>3</td>
      <td>136556.44</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>82152.83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2580</th>
      <td>15597896</td>
      <td>Ozoemena</td>
      <td>365</td>
      <td>Germany</td>
      <td>Male</td>
      <td>30</td>
      <td>0</td>
      <td>127760.07</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>81537.85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5904</th>
      <td>15677317</td>
      <td>Ankudinova</td>
      <td>570</td>
      <td>France</td>
      <td>Female</td>
      <td>29</td>
      <td>4</td>
      <td>153040.03</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>131363.57</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
turnover[(turnover['CreditScore'] > turnover['CreditScore'].mean()) & (turnover['Balance'] < turnover['Balance'].mean())]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>4</th>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15592531</td>
      <td>Bartlett</td>
      <td>822</td>
      <td>France</td>
      <td>Male</td>
      <td>50</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>10062.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>15568982</td>
      <td>Hao</td>
      <td>726</td>
      <td>France</td>
      <td>Female</td>
      <td>24</td>
      <td>6</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>54724.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>15577657</td>
      <td>McDonald</td>
      <td>732</td>
      <td>France</td>
      <td>Male</td>
      <td>41</td>
      <td>8</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>170886.17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>15725737</td>
      <td>Mosman</td>
      <td>669</td>
      <td>France</td>
      <td>Male</td>
      <td>46</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8487.75</td>
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
    </tr>
    <tr>
      <th>9993</th>
      <td>15657105</td>
      <td>Chukwualuka</td>
      <td>726</td>
      <td>Spain</td>
      <td>Male</td>
      <td>36</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>195192.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>15719294</td>
      <td>Wood</td>
      <td>800</td>
      <td>France</td>
      <td>Female</td>
      <td>29</td>
      <td>2</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>167773.55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>15606229</td>
      <td>Obijiaku</td>
      <td>771</td>
      <td>France</td>
      <td>Male</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2052 rows × 13 columns</p>
</div>




```python
%%time
# code here
turnover[((turnover['Geography'] == 'Germany') | (turnover['Geography'] == 'Spain')) & (turnover['Gender'] == 'Female')]
```

    CPU times: total: 0 ns
    Wall time: 2.99 ms
    




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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
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
      <th>2</th>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15656148</td>
      <td>Obinna</td>
      <td>376</td>
      <td>Germany</td>
      <td>Female</td>
      <td>29</td>
      <td>4</td>
      <td>115046.74</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>119346.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15600882</td>
      <td>Scott</td>
      <td>635</td>
      <td>Spain</td>
      <td>Female</td>
      <td>35</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>65951.65</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>15788218</td>
      <td>Henderson</td>
      <td>549</td>
      <td>Spain</td>
      <td>Female</td>
      <td>24</td>
      <td>9</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>14406.41</td>
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
    </tr>
    <tr>
      <th>9951</th>
      <td>15638494</td>
      <td>Salinas</td>
      <td>625</td>
      <td>Germany</td>
      <td>Female</td>
      <td>39</td>
      <td>10</td>
      <td>129845.26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>96444.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9961</th>
      <td>15681026</td>
      <td>Lucciano</td>
      <td>795</td>
      <td>Germany</td>
      <td>Female</td>
      <td>33</td>
      <td>9</td>
      <td>104552.72</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>120853.83</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9966</th>
      <td>15690164</td>
      <td>Shao</td>
      <td>627</td>
      <td>Germany</td>
      <td>Female</td>
      <td>33</td>
      <td>4</td>
      <td>83199.05</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>159334.93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9969</th>
      <td>15733491</td>
      <td>McGregor</td>
      <td>512</td>
      <td>Germany</td>
      <td>Female</td>
      <td>40</td>
      <td>8</td>
      <td>153537.57</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>23101.13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>15768163</td>
      <td>Griffin</td>
      <td>655</td>
      <td>Germany</td>
      <td>Female</td>
      <td>46</td>
      <td>7</td>
      <td>137145.12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>115146.40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2282 rows × 13 columns</p>
</div>



<!-- ### Dive Deeper

1. Baca kembali data `rice` menggunakan perintah `pd.read_csv()`, kemudian simpan kedalam variable baru bernama `rice_new`. Dengan menggunakan parameter `index_col`, jadikan kolom `receipt_id` sebagai index baris nya!
2. `pandas.DataFrame.head(n)` dapat digunakan untuk menampilkan sebagian data teratas, dengan asumsi bahwa nilai `n` adalah jumlah baris yang ingin kita tampilkan. Silahkan set `head()` dengan nilai `n=8` pada data `rice_new`, kemudian lihat apa yang terjadi! 
3. Lawan dari `head()` adalah `tail()`. Method `tail()` akan menampilkan data dari urutan paling bawah. Silahkan set `tail()` dengan nilai `n=4`, kemudian lihat apa yang terjadi! -->
