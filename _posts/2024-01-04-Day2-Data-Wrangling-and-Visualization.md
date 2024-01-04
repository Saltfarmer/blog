---
title: "Day 2 Algorit.ma : Data Wrangling and Visualization"
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

Day 2, here I will share my notes of Inclass notebook. For further example you can check out on https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/tree/main

**Inclass: Data Wrangling and Visualization**

This notebook was made based on main materials `Data Wrangling and Visualization.ipynb`

Version: BRI Audit Analytics - January 2024

---

# Training Objectives

- Reproducible Environment
- Working with Multi-Index DataFrames
    - Subsetting Multi-Index DataFrames
- Visual Data Exploratory
- Using Group By Effectively

# Reproducible Environment

Ada beberapa paket baru yang akan digunakan dalam materi ini. Biasanya, kita dapat menggunakan `pip install`/`conda install` untuk menginstal library baru ke environment kita. Namun untuk saat ini, mari kita coba pendekatan lain dalam mempersiapkan library yang diperlukan untuk proyek tertentu.

Bayangkan Anda sedang bekerja dengan tim Anda dalam sebuah proyek kolaboratif. Anda menginisialisasi proyek dengan dependensi dan versi tertentu di komputer Anda dan semuanya berjalan dengan baik. Nantinya, Anda perlu 'mengirimkan' proyek itu ke tim Anda yang mengharuskan mereka menyiapkan environment yang sama dengan Anda. Lalu apa yang akan Anda lakukan untuk memastikan program itu juga berjalan lancar di mesin mereka?

Di sinilah Anda perlu membuat environment Anda dapat direproduksi dengan membuat file `requirements.txt`.

Lihat pada folder material utama, Anda akan menemukan file `requirements.txt` yang isinya seperti ini:
```
matplotlib==3.8.1
numpy==1.26.1
pandas==2.0.0
yfinance==0.2.31
```

Perhatikan kita memiliki baris untuk setiap library, lalu nomor versi. Hal ini penting karena saat Anda mulai mengembangkan aplikasi python, Anda akan mengembangkan aplikasi dengan mempertimbangkan versi library tertentu. Sederhananya, `requirements.txt` membantu melacak versi setiap library yang Anda gunakan untuk mencegah perubahan yang tidak terduga.

## Importing Requirements

Kita sudah membahas untuk apa file persyaratan itu, tetapi bagaimana cara menggunakannya? Karena kita tidak ingin menginstal dan melacak secara manual setiap library yang diperlukan untuk proyek tertentu, mari kita coba mengimpor persyaratan dengan langkah-langkah berikut:

1. Aktifkan environment yang ingin digunakan: 

    ```
    conda activate <ENV_NAME>
    ```
    
    <div class="alert alert-warning">
    
    <b> Apabila belum ada, maka perlu membuat environment baru:</b><br>

    <code>conda create -n [ENV_NAME] python=[PYTHON_VERSION]</code>

    </div>


    <div class="alert alert-warning">

    <b> Jangan lupa instalasi kernel di dalam environment tersebut apabila ingin dapat diakses menggunakan jupyter notebook:</b><br>

    <code>&gt; pip install ipykernel  </code> 
    
    <code>&gt; python -m ipykernel install --user --name=[ENV_NAME]</code>
    
    </div>


2. Navigasikan path ke folder di mana file `requirements.txt` berada

    ```
    cd <PATH_TO_REQUIREMENTS>
    ```

3. Instalasi packages dari file tersebut

    ```
    pip install -r requirements.txt
    ```

## Exporting Requirements

Perintah `pip install` selalu menginstal versi terbaru dari sebuah library, namun terkadang, Anda mungkin ingin menginstal versi tertentu yang Anda tahu berfungsi pada proyek Anda.

File persyaratan memungkinkan Anda menentukan dengan tepat library dan versi mana yang harus diinstal. Anda dapat mengikuti langkah-langkah berikut untuk membuat file kebutuhan Anda:

1. Aktifkan environment

  ```
  conda activate <ENV_NAME>
  ```

2. Navigasikan path ke folder tempat di mana file `requirements.txt` ingin disimpan

  ```
  cd <PATH_TO_REQUIREMENTS_FOLDER>
  ```

3. Export environment: membuat daftar packages beserta versinya.

  ```
  pip list --format=freeze > requirements.txt
  ```

<div class="alert alert-info">
  <b>Notes!</b>
    <p>Anda dapat menyimpan file dengan nama lain, namun sebagai <b>konvensi</b> biasa digunakan penamaan <code>requirements.txt</code></p>
</div> 

# Data Wrangling dan Reshaping

Pada materi sebelumnya, kita sudah mempelajari beberapa teknik yang biasa digunakan untuk eksplorasi data pada `pandas`. Secara spesifik, pada materi P4DA, berbagai tools yang digunakan untuk inspeksi, diagnostic, dan exploratory yaitu:

**Data Inspection**
- `.head()` and `.tail()`
- `.describe()`
- `.dtypes`
- Subsetting using `.loc`, `.iloc` and conditionals

---

## Load Data

### `yfinance`

Kita akan menggunakan library `yfinance` untuk mengakses data saham yang tersedia pada [Yahoo! Finance](https://finance.yahoo.com/). Penarikan data menggunakan `yfinance` membutuhkan koneksi internet.

Dokumentasi: https://pypi.org/project/yfinance/


```python
import pandas as pd
import yfinance as data
from datetime import date
# Formatting output ke bentuk 2 desimal
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```


```python
symbol = ['BRIS.JK' ,'BBRI.JK', 'BMRI.JK']
start_date = '2020-01-01' # 1 Januari 2020
end_date = date.today() # 4 Januari 2024 hari ini
stock = data.download(tickers = symbol, start = start_date, end = end_date)
stock.columns.names = ['Attributes', 'Symbols']
stock.head()
```

    [*********************100%%**********************]  3 of 3 completed
    




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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>3683.65</td>
      <td>3176.63</td>
      <td>318.32</td>
      <td>4370.00</td>
      <td>3800.00</td>
      <td>324.00</td>
      <td>4390.00</td>
      <td>3837.50</td>
      <td>334.00</td>
      <td>4320.00</td>
      <td>3762.50</td>
      <td>320.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>328.00</td>
      <td>44225100</td>
      <td>61892000</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>3708.94</td>
      <td>3176.63</td>
      <td>312.42</td>
      <td>4400.00</td>
      <td>3800.00</td>
      <td>318.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>4380.00</td>
      <td>3787.50</td>
      <td>316.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>103948100</td>
      <td>70895600</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>3692.08</td>
      <td>3134.83</td>
      <td>306.53</td>
      <td>4380.00</td>
      <td>3750.00</td>
      <td>312.00</td>
      <td>4400.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>4340.00</td>
      <td>3687.50</td>
      <td>312.00</td>
      <td>4380.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>171751200</td>
      <td>105080600</td>
      <td>4058800</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock.tail()
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-12-27</th>
      <td>5542.47</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5625.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5725.00</td>
      <td>6025.00</td>
      <td>1705.00</td>
      <td>5625.00</td>
      <td>5925.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>122236700</td>
      <td>43114900</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>5641.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6150.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6050.00</td>
      <td>1695.00</td>
      <td>121434600</td>
      <td>75118700</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>5641.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1710.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1735.00</td>
      <td>93126000</td>
      <td>63097100</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5625.00</td>
      <td>6025.00</td>
      <td>1710.00</td>
      <td>5650.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>91143100</td>
      <td>26235700</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5650.00</td>
      <td>6150.00</td>
      <td>1830.00</td>
      <td>5600.00</td>
      <td>6050.00</td>
      <td>1730.00</td>
      <td>5625.00</td>
      <td>6100.00</td>
      <td>1735.00</td>
      <td>83659700</td>
      <td>30053900</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
</div>



<button type="button" class="btn btn-primary">Symbols:</button>
- `BRIS.JK` : PT Bank Syariah Indonesia (Persero) Tbk
- `BBRI.JK`: PT Bank Rakyat Indonesia (Persero) Tbk
- `BMRI.JK`: PT Bank Mandiri (Persero) Tbk

<button type="button" class="btn btn-primary">Data description:</button>
- `Date` - tanggal dalam format `yyyy-mm-dd`
- `High` - nilai saham **tertinggi** pada hari tersebut 
- `Low` - nilai saham **terendah** pada hari tersebut
- `Open` - nilai saham saat **trading hours dibuka** pada hari tersebut
- `Close` - nilai saham saat **trading hours ditutup** pada hari tersebut
- `Adj Close` - nilai `Close` yang telah disesuaikan setelah stock split maupun pembagian dividen 
- `Volume` - jumlah lembar saham yang ditransaksikan pada hari tersebut

[Trading hours](https://www.maybank-ke.com.sg/markets/markets-listing/trading-hours/) dapat berbeda-beda pada tiap tempat. Di Indonesia (IDX/BEI), trading hours dibuka pada Senin - Jumat jam 09:00 WIB - 04:00 WIB.

Untuk membuat analisis kedepannya lebih mudah, kita akan melakukan rename kolom pada `Symbols`


```python
stock = stock.rename(columns = {'BRIS.JK' : 'BRIS', 
                                'BBRI.JK': 'BBRI',
                                'BMRI.JK' : 'BMRI'})
stock.head(2)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
  </tbody>
</table>
</div>



## Slicing Multi-Index DataFrame

Multi-Index Dataframe adalah bentuk dataframe yang memiliki level indexing lebih dari 1 baik pada baris, kolom, ataupun keduanya. Hal yang perlu diperhatikan dalam MultiIndex Dataframe adalah bentuk dataframe ini terkadang tidak bisa langsung kita gunakan untuk menganalisis data, sehingga akan ada beberapa perlakuan untuk kita mengiris atau mengubah bentuknya ke dataframe yang lebih sederhana. Berikut contoh bentuk multi-index dataframe:

<img src="assets/multiindex dataframe.png" width = 600>

Perhatikan bahwa data `stock` adalah Multi-Index DataFrame, di mana level dari columnnya terdiri dari:
- `Attributes` = level 0
- `Symbols` = level 1


```python
stock.head()
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
      <th>BBRI.JK</th>
      <th>BMRI.JK</th>
      <th>BRIS.JK</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>3683.65</td>
      <td>3176.63</td>
      <td>318.32</td>
      <td>4370.00</td>
      <td>3800.00</td>
      <td>324.00</td>
      <td>4390.00</td>
      <td>3837.50</td>
      <td>334.00</td>
      <td>4320.00</td>
      <td>3762.50</td>
      <td>320.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>328.00</td>
      <td>44225100</td>
      <td>61892000</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>3708.94</td>
      <td>3176.63</td>
      <td>312.42</td>
      <td>4400.00</td>
      <td>3800.00</td>
      <td>318.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>4380.00</td>
      <td>3787.50</td>
      <td>316.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>103948100</td>
      <td>70895600</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>3692.08</td>
      <td>3134.83</td>
      <td>306.53</td>
      <td>4380.00</td>
      <td>3750.00</td>
      <td>312.00</td>
      <td>4400.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>4340.00</td>
      <td>3687.50</td>
      <td>312.00</td>
      <td>4380.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>171751200</td>
      <td>105080600</td>
      <td>4058800</td>
    </tr>
  </tbody>
</table>
</div>



> Ketika kita subset menggunakan `[]`, maka kita hanya bisa mengakses kolom dengan **level teratas saja**, yaitu untuk `Attributes`.

❓ Subset pada kolom `High` akan menghasilkan DataFrame Single Index dengan `Symbols` sebagai levelnya


```python
# Ambil kolom High dengan semua attributnya
stock['High']
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
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>4390.00</td>
      <td>3837.50</td>
      <td>334.00</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>4400.00</td>
      <td>3775.00</td>
      <td>318.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>5725.00</td>
      <td>6025.00</td>
      <td>1705.00</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>5750.00</td>
      <td>6150.00</td>
      <td>1745.00</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>5650.00</td>
      <td>6150.00</td>
      <td>1830.00</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 3 columns</p>
</div>



❓ **Masalah**: Bagaimana jika kita ingin mengambil semua nilai `Attributes` untuk saham `BRIS`?


```python
# Pake cross section
# stock.xs(key='BRIS', level= 1, axis = 1) 
# Pake loc dan slice
# stock.loc[:, (slice(None), "BRIS")]
# Pake cara swaplevel terlebih dahulu baru panggil BRIS
stock.swaplevel(axis=1)['BRIS']
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2020-01-02</th>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>322.25</td>
      <td>328.00</td>
      <td>336.00</td>
      <td>326.00</td>
      <td>334.00</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>318.32</td>
      <td>324.00</td>
      <td>334.00</td>
      <td>320.00</td>
      <td>328.00</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>312.42</td>
      <td>318.00</td>
      <td>324.00</td>
      <td>316.00</td>
      <td>324.00</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>306.53</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>4058800</td>
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
      <th>2023-12-27</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 6 columns</p>
</div>



### Cross Section

Menggunakan method `.xs()` (cross section) untuk mengambil kolom (`axis = 1`) pada level dalam. Parameter:

- `key` : nama kolom/baris yang kita ingin ambil
- `level` : kolom/baris tersebut ada di level apa?
- `axis` : levelnya terdapat pada index kolom/baris 
    + `0` untuk baris
    + `1` untuk kolom


```python
stock.head(2)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
  </tbody>
</table>
</div>




```python
# mengambil seluruh nilai BRIS
bris = stock.xs(key='BRIS', level=1, axis=1)

bris
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2020-01-02</th>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>322.25</td>
      <td>328.00</td>
      <td>336.00</td>
      <td>326.00</td>
      <td>334.00</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>318.32</td>
      <td>324.00</td>
      <td>334.00</td>
      <td>320.00</td>
      <td>328.00</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>312.42</td>
      <td>318.00</td>
      <td>324.00</td>
      <td>316.00</td>
      <td>324.00</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>306.53</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>4058800</td>
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
      <th>2023-12-27</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 6 columns</p>
</div>



### ✅ **Knowledge Check**

Coba ambil seluruh nilai saham BBRI pada tanggal 12 Desember 2022! 

**Hint**: Gunakan metode `.loc` untuk mengambil tanggal yang bersesuaian


```python
# your code here
stock.loc['2022-12-12' ,(slice(None), "BBRI")]
```




    Attributes  Symbols
    Adj Close   BBRI           4496.79
    Close       BBRI           4850.00
    High        BBRI           4850.00
    Low         BBRI           4760.00
    Open        BBRI           4800.00
    Volume      BBRI      139582400.00
    Name: 2022-12-12 00:00:00, dtype: float64



**[Extra Challenge]** Ambil history data harga saham BRIS pada 2 minggu terakhir dari data!


```python
from datetime import datetime, timedelta
# your code here
# stock.iloc[-14: ,:].loc[:, (slice(None), "BRIS")]
# Ngambil dari BRIS variabel

checkpoint = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
bris.loc[checkpoint:]
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2023-12-21</th>
      <td>1690.00</td>
      <td>1690.00</td>
      <td>1765.00</td>
      <td>1680.00</td>
      <td>1765.00</td>
      <td>25310000</td>
    </tr>
    <tr>
      <th>2023-12-22</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1725.00</td>
      <td>1685.00</td>
      <td>1690.00</td>
      <td>9421600</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
</div>



## Reshaping DataFrames

Reshaping data adalah salah satu komponen penting dalam tahapan data wrangling, karena memungkinkan seorang **analis untuk mempersiapkan data menjadi bentuk yang sesuai untuk tahap analisa data berikutnya**.

## `stack()` and `unstack()`

`stack()` menumpuk level yang ditentukan dari kolom ke indeks dan sangat berguna pada DataFrames yang memiliki kolom multi-level. Ia melakukannya dengan "menggeser" kolom untuk membuat level baru pada indeksnya.

Hal ini lebih mudah dipahami bila kita hanya melihat contohnya. Perhatikan bahwa `stock` memiliki kolom 2 tingkat (Atribut dan Simbol) dan indeks 1 tingkat (Tanggal):


```python
stock.head(2)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
  </tbody>
</table>
</div>



Saat kita menumpuk `stock` DataFrame, kita mengecilkan jumlah level pada kolomnya sebanyak satu: `stock` sekarang memiliki 1 kolom level bernama `Attributes`:


```python
stock.stack()
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Symbols</th>
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
      <th rowspan="3" valign="top">2020-01-02</th>
      <th>BBRI</th>
      <td>3717.37</td>
      <td>4410.00</td>
      <td>4410.00</td>
      <td>4360.00</td>
      <td>4400.00</td>
      <td>41714100</td>
    </tr>
    <tr>
      <th>BMRI</th>
      <td>3239.33</td>
      <td>3875.00</td>
      <td>3887.50</td>
      <td>3825.00</td>
      <td>3837.50</td>
      <td>37379800</td>
    </tr>
    <tr>
      <th>BRIS</th>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2020-01-03</th>
      <th>BBRI</th>
      <td>3725.80</td>
      <td>4420.00</td>
      <td>4440.00</td>
      <td>4390.00</td>
      <td>4420.00</td>
      <td>82898300</td>
    </tr>
    <tr>
      <th>BMRI</th>
      <td>3228.88</td>
      <td>3862.50</td>
      <td>3912.50</td>
      <td>3812.50</td>
      <td>3875.00</td>
      <td>70294600</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2024-01-02</th>
      <th>BMRI</th>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6025.00</td>
      <td>6050.00</td>
      <td>26235700</td>
    </tr>
    <tr>
      <th>BRIS</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2024-01-03</th>
      <th>BBRI</th>
      <td>5600.00</td>
      <td>5600.00</td>
      <td>5650.00</td>
      <td>5600.00</td>
      <td>5625.00</td>
      <td>83659700</td>
    </tr>
    <tr>
      <th>BMRI</th>
      <td>6100.00</td>
      <td>6100.00</td>
      <td>6150.00</td>
      <td>6050.00</td>
      <td>6100.00</td>
      <td>30053900</td>
    </tr>
    <tr>
      <th>BRIS</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>2928 rows × 6 columns</p>
</div>



`unstack()` melakukan yang sebaliknya: ia "menggeser" level dari sumbu indeks ke sumbu kolom. **Coba dan buat tumpukan DataFrame, lalu terapkan `unstack` pada DataFrame baru untuk melihatnya kembali ke bentuk aslinya:**


```python
## Write your code to try out .unstack() method here
stock.unstack()
```




    Attributes  Symbols  Date      
    Adj Close   BBRI     2020-01-02       3717.37
                         2020-01-03       3725.80
                         2020-01-06       3683.65
                         2020-01-07       3708.94
                         2020-01-08       3692.08
                                          ...    
    Volume      BRIS     2023-12-27   10923600.00
                         2023-12-28   23222700.00
                         2023-12-29   21099100.00
                         2024-01-02   13118700.00
                         2024-01-03   76511200.00
    Length: 17568, dtype: float64



**Dive Deeper**

Jawablah pertanyaan-pertanyaan berikut ini untuk memastikan Anda dapat melanjutkan sesi berikutnya:

1. Bagaimana cara menukar posisi (level) Symbols dan Attributes?

2. Berdasarkan pengetahuan Anda, (simbol) perusahaan apa yang layak untuk diinvestasikan? 
(Anda dapat melihat fluktuasinya, artinya, dll)

<!--
# answer 1
stock.stack(level=0).unstack(level=1)

# answer 2
# Overal Growth Values
stock['Close'].iloc[-1,:] - stock['Close'].iloc[0,:]

# Oveal Growth Percentage
(stock['Close'].iloc[-1,:] - stock['Close'].iloc[0,:]) / stock['Close'].iloc[0,:]

# Standard Deviation 
stock['Close'].std() / stock['Close'].mean()
-->


```python
# Write your solution code here 
# stock.swaplevel(axis=1)
# stock stack + unstack
# stock.stack(level=0).unstack()
stock.stack([1,0]).unstack([1,2])
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Symbols</th>
      <th colspan="6" halign="left">BBRI</th>
      <th colspan="6" halign="left">BMRI</th>
      <th colspan="6" halign="left">BRIS</th>
    </tr>
    <tr>
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>4410.00</td>
      <td>4410.00</td>
      <td>4360.00</td>
      <td>4400.00</td>
      <td>41714100.00</td>
      <td>3239.33</td>
      <td>3875.00</td>
      <td>3887.50</td>
      <td>3825.00</td>
      <td>3837.50</td>
      <td>37379800.00</td>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400.00</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>4420.00</td>
      <td>4440.00</td>
      <td>4390.00</td>
      <td>4420.00</td>
      <td>82898300.00</td>
      <td>3228.88</td>
      <td>3862.50</td>
      <td>3912.50</td>
      <td>3812.50</td>
      <td>3875.00</td>
      <td>70294600.00</td>
      <td>322.25</td>
      <td>328.00</td>
      <td>336.00</td>
      <td>326.00</td>
      <td>334.00</td>
      <td>4989600.00</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>3683.65</td>
      <td>4370.00</td>
      <td>4390.00</td>
      <td>4320.00</td>
      <td>4360.00</td>
      <td>44225100.00</td>
      <td>3176.63</td>
      <td>3800.00</td>
      <td>3837.50</td>
      <td>3762.50</td>
      <td>3825.00</td>
      <td>61892000.00</td>
      <td>318.32</td>
      <td>324.00</td>
      <td>334.00</td>
      <td>320.00</td>
      <td>328.00</td>
      <td>6937900.00</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>3708.94</td>
      <td>4400.00</td>
      <td>4410.00</td>
      <td>4380.00</td>
      <td>4410.00</td>
      <td>103948100.00</td>
      <td>3176.63</td>
      <td>3800.00</td>
      <td>3862.50</td>
      <td>3787.50</td>
      <td>3862.50</td>
      <td>70895600.00</td>
      <td>312.42</td>
      <td>318.00</td>
      <td>324.00</td>
      <td>316.00</td>
      <td>324.00</td>
      <td>6319400.00</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>3692.08</td>
      <td>4380.00</td>
      <td>4400.00</td>
      <td>4340.00</td>
      <td>4380.00</td>
      <td>171751200.00</td>
      <td>3134.83</td>
      <td>3750.00</td>
      <td>3775.00</td>
      <td>3687.50</td>
      <td>3775.00</td>
      <td>105080600.00</td>
      <td>306.53</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>4058800.00</td>
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
      <th>2023-12-27</th>
      <td>5542.47</td>
      <td>5625.00</td>
      <td>5725.00</td>
      <td>5625.00</td>
      <td>5700.00</td>
      <td>122236700.00</td>
      <td>6000.00</td>
      <td>6000.00</td>
      <td>6025.00</td>
      <td>5925.00</td>
      <td>6000.00</td>
      <td>43114900.00</td>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600.00</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>5641.00</td>
      <td>5725.00</td>
      <td>5750.00</td>
      <td>5675.00</td>
      <td>5700.00</td>
      <td>121434600.00</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6150.00</td>
      <td>6000.00</td>
      <td>6050.00</td>
      <td>75118700.00</td>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700.00</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>5641.00</td>
      <td>5725.00</td>
      <td>5750.00</td>
      <td>5675.00</td>
      <td>5750.00</td>
      <td>93126000.00</td>
      <td>6050.00</td>
      <td>6050.00</td>
      <td>6125.00</td>
      <td>6000.00</td>
      <td>6125.00</td>
      <td>63097100.00</td>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100.00</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>5675.00</td>
      <td>5675.00</td>
      <td>5675.00</td>
      <td>5625.00</td>
      <td>5650.00</td>
      <td>91143100.00</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6025.00</td>
      <td>6050.00</td>
      <td>26235700.00</td>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700.00</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>5600.00</td>
      <td>5600.00</td>
      <td>5650.00</td>
      <td>5600.00</td>
      <td>5625.00</td>
      <td>83659700.00</td>
      <td>6100.00</td>
      <td>6100.00</td>
      <td>6150.00</td>
      <td>6050.00</td>
      <td>6100.00</td>
      <td>30053900.00</td>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200.00</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 18 columns</p>
</div>




```python
stock.groupby(level = 1, axis = 1).describe()
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
      <th></th>
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
    <tr>
      <th>Symbols</th>
      <th>Attributes</th>
      <th>Symbols</th>
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
      <th rowspan="6" valign="top">BBRI</th>
      <th>Adj Close</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>3983.76</td>
      <td>825.93</td>
      <td>1893.82</td>
      <td>3530.16</td>
      <td>3950.26</td>
      <td>4450.15</td>
      <td>5675.00</td>
    </tr>
    <tr>
      <th>Close</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>4341.34</td>
      <td>734.56</td>
      <td>2170.00</td>
      <td>3997.50</td>
      <td>4400.00</td>
      <td>4772.50</td>
      <td>5725.00</td>
    </tr>
    <tr>
      <th>High</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>4394.92</td>
      <td>727.07</td>
      <td>2270.00</td>
      <td>4070.00</td>
      <td>4440.00</td>
      <td>4820.00</td>
      <td>5750.00</td>
    </tr>
    <tr>
      <th>Low</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>4293.33</td>
      <td>737.23</td>
      <td>2160.00</td>
      <td>3930.00</td>
      <td>4350.00</td>
      <td>4722.50</td>
      <td>5675.00</td>
    </tr>
    <tr>
      <th>Open</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>4347.60</td>
      <td>731.63</td>
      <td>2250.00</td>
      <td>4000.00</td>
      <td>4400.00</td>
      <td>4780.00</td>
      <td>5750.00</td>
    </tr>
    <tr>
      <th>Volume</th>
      <th>BBRI</th>
      <td>976.00</td>
      <td>162823853.00</td>
      <td>99249723.95</td>
      <td>27676500.00</td>
      <td>99034175.00</td>
      <td>136537400.00</td>
      <td>191963800.00</td>
      <td>898453700.00</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">BMRI</th>
      <th>Adj Close</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>3723.45</td>
      <td>1214.71</td>
      <td>1633.40</td>
      <td>2775.37</td>
      <td>3280.89</td>
      <td>4780.97</td>
      <td>10225.00</td>
    </tr>
    <tr>
      <th>Close</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>3974.56</td>
      <td>1110.13</td>
      <td>1860.00</td>
      <td>3075.00</td>
      <td>3775.00</td>
      <td>5015.62</td>
      <td>10225.00</td>
    </tr>
    <tr>
      <th>High</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>4024.61</td>
      <td>1110.79</td>
      <td>1900.00</td>
      <td>3125.00</td>
      <td>3837.50</td>
      <td>5075.00</td>
      <td>10400.00</td>
    </tr>
    <tr>
      <th>Low</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>3925.31</td>
      <td>1106.74</td>
      <td>1830.00</td>
      <td>3046.88</td>
      <td>3750.00</td>
      <td>4975.00</td>
      <td>10225.00</td>
    </tr>
    <tr>
      <th>Open</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>3977.00</td>
      <td>1106.57</td>
      <td>1880.00</td>
      <td>3087.50</td>
      <td>3793.75</td>
      <td>5025.00</td>
      <td>10350.00</td>
    </tr>
    <tr>
      <th>Volume</th>
      <th>BMRI</th>
      <td>976.00</td>
      <td>105844761.99</td>
      <td>65129933.21</td>
      <td>0.00</td>
      <td>64435950.00</td>
      <td>91211050.00</td>
      <td>131406100.00</td>
      <td>770252400.00</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">BRIS</th>
      <th>Adj Close</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>1500.02</td>
      <td>671.26</td>
      <td>132.63</td>
      <td>1310.69</td>
      <td>1561.67</td>
      <td>1790.75</td>
      <td>3703.88</td>
    </tr>
    <tr>
      <th>Close</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>1517.88</td>
      <td>682.52</td>
      <td>135.00</td>
      <td>1318.75</td>
      <td>1580.00</td>
      <td>1810.00</td>
      <td>3770.00</td>
    </tr>
    <tr>
      <th>High</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>1555.41</td>
      <td>705.24</td>
      <td>155.00</td>
      <td>1350.00</td>
      <td>1607.50</td>
      <td>1848.75</td>
      <td>3980.00</td>
    </tr>
    <tr>
      <th>Low</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>1489.19</td>
      <td>665.76</td>
      <td>135.00</td>
      <td>1300.00</td>
      <td>1550.00</td>
      <td>1780.00</td>
      <td>3710.00</td>
    </tr>
    <tr>
      <th>Open</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>1519.86</td>
      <td>684.72</td>
      <td>136.00</td>
      <td>1315.00</td>
      <td>1580.00</td>
      <td>1805.00</td>
      <td>3800.00</td>
    </tr>
    <tr>
      <th>Volume</th>
      <th>BRIS</th>
      <td>976.00</td>
      <td>66363241.09</td>
      <td>124768967.30</td>
      <td>664700.00</td>
      <td>10885025.00</td>
      <td>23884104.00</td>
      <td>60038475.00</td>
      <td>1318651800.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock['Close'].plot()
```




    <Axes: xlabel='Date'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_36_1.png)
    


## Sepertinya saham BBRI paling GACOR !!! Karena selisih high and low secara median dan mean yang cukup bagus, std tidak begitu variatif jadi lebih stabil dan volume yang tinggi sehingga lebih dipercaya masyarakat

### `melt()`

**Goal**: Melebur beberapa kolom menjadi 1 kolom (variabel) dan nilai di dalamnnya menjadi value
`
> Syntax: `Dataframe.melt()`

<img src="assets/reshaping_melt.png" width="600"/>

❓ Dari dataframe `bris` hasil metode cross section (`xs`) sebelumnya, aplikasikan method `melt()`!


```python
# dataframe bris
bris
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2020-01-02</th>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>322.25</td>
      <td>328.00</td>
      <td>336.00</td>
      <td>326.00</td>
      <td>334.00</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>318.32</td>
      <td>324.00</td>
      <td>334.00</td>
      <td>320.00</td>
      <td>328.00</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>312.42</td>
      <td>318.00</td>
      <td>324.00</td>
      <td>316.00</td>
      <td>324.00</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>306.53</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>312.00</td>
      <td>318.00</td>
      <td>4058800</td>
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
      <th>2023-12-27</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 6 columns</p>
</div>



DataFrame di atas memiliki format *wide*: terdiri dari __ baris dan 6 kolom. Fungsi `melt()` menggabungkan semua kolom menjadi satu dan menyimpan nilai yang sesuai dengan masing-masing kolom, sehingga DataFrame yang dihasilkan memiliki ___ * 6 = 5.382 baris. Selain itu, DataFrame ini juga memiliki kolom `Attributes` dan kolom `value`.


```python
# your code here
bris.melt(ignore_index=False)
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
      <th>Attributes</th>
      <th>value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>Adj Close</td>
      <td>326.18</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>Adj Close</td>
      <td>322.25</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>Adj Close</td>
      <td>318.32</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>Adj Close</td>
      <td>312.42</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>Adj Close</td>
      <td>306.53</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>Volume</td>
      <td>10923600.00</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>Volume</td>
      <td>23222700.00</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>Volume</td>
      <td>21099100.00</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>Volume</td>
      <td>13118700.00</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>Volume</td>
      <td>76511200.00</td>
    </tr>
  </tbody>
</table>
<p>5856 rows × 2 columns</p>
</div>



Data di atas kurang informatif sebab kita tidak tahu kapan nilai-nilai tersebut muncul. Hal ini dikarenakan `Date` merupakan index baris.

❓ Bagaimana jika kita ingin Date tetap dimunculkan? Kita perlu mengubah `Date` dari index menjadi kolom

#### Identifier and Value

Dalam method `melt()`, terdapat dua parameter yang sering digunakan:
- `id_vars`: kolom yang dipertahankan
- `value_vars`: kolom yang di-melt



```python
# menggunakan reset_index() sehingga Date menjadi kolom
stock_kopi = stock.reset_index().copy()
stock_kopi.tail(5)
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
      <th>Attributes</th>
      <th>Date</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th></th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>971</th>
      <td>2023-12-27</td>
      <td>5542.47</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5625.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5725.00</td>
      <td>6025.00</td>
      <td>1705.00</td>
      <td>5625.00</td>
      <td>5925.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>122236700</td>
      <td>43114900</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>972</th>
      <td>2023-12-28</td>
      <td>5641.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6150.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6050.00</td>
      <td>1695.00</td>
      <td>121434600</td>
      <td>75118700</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>973</th>
      <td>2023-12-29</td>
      <td>5641.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1710.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1735.00</td>
      <td>93126000</td>
      <td>63097100</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>974</th>
      <td>2024-01-02</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5625.00</td>
      <td>6025.00</td>
      <td>1710.00</td>
      <td>5650.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>91143100</td>
      <td>26235700</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>975</th>
      <td>2024-01-03</td>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5650.00</td>
      <td>6150.00</td>
      <td>1830.00</td>
      <td>5600.00</td>
      <td>6050.00</td>
      <td>1730.00</td>
      <td>5625.00</td>
      <td>6100.00</td>
      <td>1735.00</td>
      <td>83659700</td>
      <td>30053900</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
</div>



📌 Note: 

Fungsi `reset_index` di Pandas digunakan untuk mengembalikan indeks dari DataFrame atau Series ke indeks default yang berurutan (0, 1, 2, ...) dan menghapus indeks yang sudah ada


```python
# reset index & melt
stock_kopi.melt()
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
      <th>Attributes</th>
      <th>Symbols</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Date</td>
      <td></td>
      <td>2020-01-02 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Date</td>
      <td></td>
      <td>2020-01-03 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Date</td>
      <td></td>
      <td>2020-01-06 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Date</td>
      <td></td>
      <td>2020-01-07 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Date</td>
      <td></td>
      <td>2020-01-08 00:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18539</th>
      <td>Volume</td>
      <td>BRIS</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>18540</th>
      <td>Volume</td>
      <td>BRIS</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>18541</th>
      <td>Volume</td>
      <td>BRIS</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>18542</th>
      <td>Volume</td>
      <td>BRIS</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>18543</th>
      <td>Volume</td>
      <td>BRIS</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>18544 rows × 3 columns</p>
</div>




```python
# id_vars
stock_kopi.melt(id_vars=['Date'])
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
      <th>Date</th>
      <th>Attributes</th>
      <th>Symbols</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>Adj Close</td>
      <td>BBRI</td>
      <td>3717.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-03</td>
      <td>Adj Close</td>
      <td>BBRI</td>
      <td>3725.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-06</td>
      <td>Adj Close</td>
      <td>BBRI</td>
      <td>3683.65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-07</td>
      <td>Adj Close</td>
      <td>BBRI</td>
      <td>3708.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-08</td>
      <td>Adj Close</td>
      <td>BBRI</td>
      <td>3692.08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17563</th>
      <td>2023-12-27</td>
      <td>Volume</td>
      <td>BRIS</td>
      <td>10923600.00</td>
    </tr>
    <tr>
      <th>17564</th>
      <td>2023-12-28</td>
      <td>Volume</td>
      <td>BRIS</td>
      <td>23222700.00</td>
    </tr>
    <tr>
      <th>17565</th>
      <td>2023-12-29</td>
      <td>Volume</td>
      <td>BRIS</td>
      <td>21099100.00</td>
    </tr>
    <tr>
      <th>17566</th>
      <td>2024-01-02</td>
      <td>Volume</td>
      <td>BRIS</td>
      <td>13118700.00</td>
    </tr>
    <tr>
      <th>17567</th>
      <td>2024-01-03</td>
      <td>Volume</td>
      <td>BRIS</td>
      <td>76511200.00</td>
    </tr>
  </tbody>
</table>
<p>17568 rows × 4 columns</p>
</div>



❓ Lakukan melt terhadap data `bris` hanya pada kolom `Close` dan `Open`, serta setiap observasinya dibedakan berdasarkan `Date`.


```python
# bris.reset_index().melt(id_vars=['Date'], value_vars=['Close', 'Open'])
bris.reset_index().melt(id_vars=['Date'], value_vars=['Close', 'Open']).set_index('Date')
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
      <th>Attributes</th>
      <th>value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>Close</td>
      <td>332.00</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>Close</td>
      <td>328.00</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>Close</td>
      <td>324.00</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>Close</td>
      <td>318.00</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>Close</td>
      <td>312.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>Open</td>
      <td>1695.00</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>Open</td>
      <td>1695.00</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>Open</td>
      <td>1735.00</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>Open</td>
      <td>1740.00</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>Open</td>
      <td>1735.00</td>
    </tr>
  </tbody>
</table>
<p>1952 rows × 2 columns</p>
</div>



# Visualisasi Data

Tujuan visualisasi: 
- **Exploratory**: proses **mengeksplorasi data** untuk **mendapatkan sebuah insight**. 
    + Visualisasi yang ditampilkan sederhana
    + Analogi: mencari dan mendapatkan batu permata di antara ratusan batu biasa

- **Explanatory**: proses untuk menjelaskan atau **menyajikan insight** (*explain*) yang didapat dari hasil exploratory kepada audience.
    + Visualisasi biasanya lebih menarik
    + Visualisasi meng-*highlight* insight secara spesifik
    + Analogi: mempoles batu permata dan menawarkannya kepada pembeli
    
Pada course ini, dititikberatkan pada visualisasi untuk eksplorasi, yaitu menampilkan visualisasi data yang **informatif dan tepat** sehingga mendapatkan insight.

## `Pandas` dan `Matplotlib`

Sampai tahap ini mungkin Anda tidak sabar untuk melakukan visualisasi data di Python. Dengan cukup mudah, kita bisa membuat objek plot `matplotlib` dengan hanya menggunakan method `.plot()`

❗️ Sekarang mari kita coba melakukan visualisasi untuk **50 observasi (baris) terakhir `Volume` pada `stock`**

- Index akan menjadi sumbu horizontal pada plot (`Date`)
- Nilai akan menjadi sumbu vertikal pada plot
- Masing-masing kolom akan menjadi 1 komponen pada plot, dalam hal ini 1 `Symbols` menjadi 1 garis


```python
stock['Volume']
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
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>44225100</td>
      <td>61892000</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>103948100</td>
      <td>70895600</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>171751200</td>
      <td>105080600</td>
      <td>4058800</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>122236700</td>
      <td>43114900</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>121434600</td>
      <td>75118700</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>93126000</td>
      <td>63097100</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>91143100</td>
      <td>26235700</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>83659700</td>
      <td>30053900</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 3 columns</p>
</div>




```python
# ambil 50 data terakhir dari kolom Volume kemudian lakukan .plot()
stock['Volume'][-50:].plot()
```




    <Axes: xlabel='Date'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_53_1.png)
    


📈 **Insight**:

- 
- 

🔻 Method `plot()` mempermudah kita dalam melakukan visualisasi langsung pada DataFrame, tanpa perlu mengerti cara penggunaan `matplotlib`. Kunjungi [dokumentasi matplotlib](https://matplotlib.org/stable/tutorials/introductory/quick_start.html) untuk detail mengenai `matplotlib`.

🔻 Namun, keterbatasan dari penggunaan `plot()` adalah minim kustomisasi dari visualisasi yang ada. Hanya terbatas pada parameter yang ada di dalam method tersebut. Kunjungi [dokumentasi method plot](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html).

🔻 Salah satu kustomisasi yang dapat kita lakukan untuk memperindah visualisasi adalah melalui [matplotlib style sheet](https://matplotlib.org/stable/tutorials/introductory/customizing.html). Kita dapat mengganti nilai 'default' pada method `plt.style.use()` dengan salah satu style yang tersedia, kemudian jalankan kembali code visualisasi untuk menerapkan style yang dipilih.


```python
import matplotlib.pyplot as plt
print(plt.style.available)
```

    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    


```python
# misalnya gunakan style seaborn-v0_8
plt.style.use('seaborn-v0_8-poster')
```


```python
# ambil 50 data terakhir dari kolom Volume kemudian lakukan .plot()
stock['Volume'].tail(50).plot()
```




    <Axes: xlabel='Date'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_58_1.png)
    


🧐 **Task**: Ambil data BRIS untuk tanggal 8 September 2022 s.d. 8 September 2023 dan assign ke variabel `bris_sept`. Setelah itu lakukan visualisasi dari data `bris_sept` tersebut.

*Hint: gunakan `.loc[start_date : end_date]`*


```python
bris.tail()
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
      <th>Attributes</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2023-12-27</th>
      <td>1695.00</td>
      <td>1695.00</td>
      <td>1705.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1685.00</td>
      <td>1695.00</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1735.00</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# gunakan .loc untuk mensubset tanggal
bris_sept = bris.loc['2022-09-08':'2023-09-08']

# melakukan visualisasi
bris_sept.plot()
```




    <Axes: xlabel='Date'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_61_1.png)
    



```python

```


```python
bris_sept.drop(columns='Volume').plot(style='.-')
```




    <Axes: xlabel='Date'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_63_1.png)
    


💭 **Diskusi:** Apakah visualisasi tersebut sudah cukup informatif dan tepat? 

## Kurang detail untuk nominalnya karena nilai dari volume terlalu tinggi dibanding yang lain

## 📌 Types of Visualization

Visualisasi berikut hanya perlu menggunakan **satu** kolom:

- Data kategorik:
    - **`.plot(kind='bar')` untuk barplot (diagram batang)**
    - **`.plot(kind='barh')` untuk horizontal barplot**
    - **`.plot(kind='box')` untuk boxplot (five number summary)** 
    - `.plot(kind='pie')` untuk pie chart
    

- Data numerik:
    - **`.plot(kind='hist')` untuk histogram**
    - `.plot(kind='density')` untuk density plot
    - `.plot(kind='area')` untuk area plot

Visualisasi berikut perlu menggunakan **dua** kolom:

- `.plot(kind='scatter')` untuk scatter plot
- `.plot(kind='hexbin')` untuk hexagonal bin plot

💡 Panduan untuk menentukan tipe visualisasi yang tepat: https://www.data-to-viz.com/

Silakan mengacu referensi lengkapnya di [official documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html) untuk method `plot` apabila ingin eksplor visualisasi yang ada di luar lingkup course ini

### 📊 Barplot

> Visualisasi untuk melihat perbandingan nilai dari beberapa kategori

❓ Menggunakan data `stock`, tampilkan visualisasi untuk **membandingkan** fluktuasi (menggunakan coefficient of variance) nilai `Close` pada masing-masing `Symbols`. **Mana saham yang paling berfluktuasi?**

Info lebih lanjut terkait [coefficient of variance](https://www.investopedia.com/terms/c/coefficientofvariation.asp)


```python
# mengambil nilai coefficient of variation
coef_of_var = stock['High'].std() / stock['High'].mean()
coef_of_var
```




    Symbols
    BBRI   0.17
    BMRI   0.28
    BRIS   0.45
    dtype: float64




```python
(stock['Close']-stock['Open']).plot(kind='box')
```




    <Axes: >




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_69_1.png)
    



```python
# Visualisasi barplot
coef_of_var.plot(kind='bar')
```




    <Axes: xlabel='Symbols'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_70_1.png)
    


📈 **Insight**:

> ...


```python
import matplotlib.pyplot as plt


fig, ax = plt.subplots()


ax.plot(bris_sept.loc[:, bris_sept.columns != 'Volume'])

ax_vol = ax.twinx()
ax_vol.bar(bris_sept.index, bris_sept['Volume'], color='blue', alpha=0.3, label='Volume')
ax_vol.set_ylabel('Volume')
ax_vol.legend(loc='upper right')


plt.show()

```


    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_72_0.png)
    



```python
# (opsional) ingin diurutkan
coef_of_var.sort_values(ascending = False).plot(kind='bar')
```




    <Axes: xlabel='Symbols'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_73_1.png)
    


### Histogram

> Visualisasi untuk melihat persebaran data

Menggunakan data `bbri`, tampilkan visualisasi histogram untuk mengetahui **persebaran** `Volume` pada saham `BBRI`:


```python
# slicing volume BBRI, masukan ke variabel vol_bbri
bbri = stock['Volume']['BBRI']
```


```python
# visualisasi
bbri.hist().xaxis.set_major_formatter('{x:1.0f}')
```


    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_76_0.png)
    


📈 **Insight**:

- Transaksi terbanyak terjadi di persebaran di 0-200 juta
- 

# Group By: Aggregation Table

Teknik yang tak kalah penting adalah operasi **group by**. Mungkin untuk Anda yang sudah pernah menggunakan SQL akan familiar dengan operasi group by ini.

❗️ Misalkan kita punya dataframe `close_melted` yang ingin kita bandingkan nilai `Close` hariannya pada saham BRIS, BBRI, dan BMRI:


```python
close = stock['Close']
close_melted = close.reset_index().melt(id_vars = 'Date', value_name = 'Close')
close_melted
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
      <th>Date</th>
      <th>Symbols</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>BBRI</td>
      <td>4410.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-03</td>
      <td>BBRI</td>
      <td>4420.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-06</td>
      <td>BBRI</td>
      <td>4370.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-07</td>
      <td>BBRI</td>
      <td>4400.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-08</td>
      <td>BBRI</td>
      <td>4380.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>2023-12-27</td>
      <td>BRIS</td>
      <td>1695.00</td>
    </tr>
    <tr>
      <th>2924</th>
      <td>2023-12-28</td>
      <td>BRIS</td>
      <td>1740.00</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>2023-12-29</td>
      <td>BRIS</td>
      <td>1740.00</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2024-01-02</td>
      <td>BRIS</td>
      <td>1740.00</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2024-01-03</td>
      <td>BRIS</td>
      <td>1800.00</td>
    </tr>
  </tbody>
</table>
<p>2928 rows × 3 columns</p>
</div>



❓ Di antara saham BRIS, BBRI, maupun BMRI, manakah saham yang memiliki rata-rata `Close` harian tertinggi? 

Cobalah preprocessing datanya menggunakan `.groupby()`.

Syntax: `[df_name].groupby(by=[column_name]).aggfunc_name()`


```python
# coba pakai method yang sudah dipelajari
stock.groupby(level=1, axis=1).mean() 
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
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>6955899.56</td>
      <td>6233077.39</td>
      <td>243009.03</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>13819949.30</td>
      <td>11718881.90</td>
      <td>831874.37</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>7374370.61</td>
      <td>10318400.27</td>
      <td>1156587.39</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>17328234.82</td>
      <td>11819014.86</td>
      <td>1053499.07</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>28628732.01</td>
      <td>17516453.72</td>
      <td>676727.75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-12-27</th>
      <td>20377486.24</td>
      <td>7190808.33</td>
      <td>1822012.50</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>20243848.50</td>
      <td>12524858.33</td>
      <td>3871884.17</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>15525756.83</td>
      <td>10521241.67</td>
      <td>3517961.67</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>15195233.33</td>
      <td>4377691.67</td>
      <td>2187895.83</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>13947962.50</td>
      <td>5014066.67</td>
      <td>12753349.17</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 3 columns</p>
</div>




```python
# coba pakai groupby
avg_close = close_melted.groupby(by = 'Symbols').mean()['Close']
avg_close
```




    Symbols
    BBRI   4341.34
    BMRI   3974.56
    BRIS   1517.88
    Name: Close, dtype: float64




```python
# visualisasi bar chart
avg_close.plot(kind = 'bar')
```




    <Axes: xlabel='Symbols'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_83_1.png)
    


## Grouped Bar Chart

Kita akan coba melakukan visualisasi **grouped bar chart** untuk membandingkan rata-rata nilai `Close` untuk ketiga saham **setiap bulannya**

1️⃣ Step 1: Panggil kembali dataframe `close` yang sudah dibuat sebelumnya


```python
# dataframe close
close.head()
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
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>4370.00</td>
      <td>3800.00</td>
      <td>324.00</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>4400.00</td>
      <td>3800.00</td>
      <td>318.00</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>4380.00</td>
      <td>3750.00</td>
      <td>312.00</td>
    </tr>
  </tbody>
</table>
</div>



2️⃣ Step 2: Buatlah kolom `Month` yang berisikan nama bulan pada dataframe `close`

📌 Note: untuk mengambil nama bulan tidak perlu `.dt` lagi, karena sudah berupa objek DatetimeIndex. Jika date ada di dalam kolom maka perlu menggunakan `.dt`


```python
close.loc[:,'Month'] = close.index.month_name()

close.head()
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
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>Month</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>4370.00</td>
      <td>3800.00</td>
      <td>324.00</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>4400.00</td>
      <td>3800.00</td>
      <td>318.00</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>4380.00</td>
      <td>3750.00</td>
      <td>312.00</td>
      <td>January</td>
    </tr>
  </tbody>
</table>
</div>



3️⃣ Step 3: Membuat `.groupby()` untuk mendapatkan mean Close tiap bulan


```python
close_mean = close.groupby(by=['Month']).mean()
```

4️⃣ Step 4: Visualisasikan group barchart


```python
# visualisasi barchart
close_mean.plot(kind = 'bar')
```




    <Axes: xlabel='Month'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_92_1.png)
    


5️⃣ Step 5: Improvement Visualisasi


```python
# perbaiki urutan bulan
months= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

close_mean.loc[months,].plot(kind='bar')
```




    <Axes: xlabel='Month'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_94_1.png)
    



```python
stock.loc[:, stock.columns!='Volume']
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Attributes</th>
      <th colspan="3" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th colspan="3" halign="left">High</th>
      <th colspan="3" halign="left">Low</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="3" halign="left">Volume</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
      <th>BBRI</th>
      <th>BMRI</th>
      <th>BRIS</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>3717.37</td>
      <td>3239.33</td>
      <td>326.18</td>
      <td>4410.00</td>
      <td>3875.00</td>
      <td>332.00</td>
      <td>4410.00</td>
      <td>3887.50</td>
      <td>336.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>330.00</td>
      <td>4400.00</td>
      <td>3837.50</td>
      <td>330.00</td>
      <td>41714100</td>
      <td>37379800</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>3725.80</td>
      <td>3228.88</td>
      <td>322.25</td>
      <td>4420.00</td>
      <td>3862.50</td>
      <td>328.00</td>
      <td>4440.00</td>
      <td>3912.50</td>
      <td>336.00</td>
      <td>4390.00</td>
      <td>3812.50</td>
      <td>326.00</td>
      <td>4420.00</td>
      <td>3875.00</td>
      <td>334.00</td>
      <td>82898300</td>
      <td>70294600</td>
      <td>4989600</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>3683.65</td>
      <td>3176.63</td>
      <td>318.32</td>
      <td>4370.00</td>
      <td>3800.00</td>
      <td>324.00</td>
      <td>4390.00</td>
      <td>3837.50</td>
      <td>334.00</td>
      <td>4320.00</td>
      <td>3762.50</td>
      <td>320.00</td>
      <td>4360.00</td>
      <td>3825.00</td>
      <td>328.00</td>
      <td>44225100</td>
      <td>61892000</td>
      <td>6937900</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>3708.94</td>
      <td>3176.63</td>
      <td>312.42</td>
      <td>4400.00</td>
      <td>3800.00</td>
      <td>318.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>4380.00</td>
      <td>3787.50</td>
      <td>316.00</td>
      <td>4410.00</td>
      <td>3862.50</td>
      <td>324.00</td>
      <td>103948100</td>
      <td>70895600</td>
      <td>6319400</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>3692.08</td>
      <td>3134.83</td>
      <td>306.53</td>
      <td>4380.00</td>
      <td>3750.00</td>
      <td>312.00</td>
      <td>4400.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>4340.00</td>
      <td>3687.50</td>
      <td>312.00</td>
      <td>4380.00</td>
      <td>3775.00</td>
      <td>318.00</td>
      <td>171751200</td>
      <td>105080600</td>
      <td>4058800</td>
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
      <th>2023-12-27</th>
      <td>5542.47</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5625.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>5725.00</td>
      <td>6025.00</td>
      <td>1705.00</td>
      <td>5625.00</td>
      <td>5925.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6000.00</td>
      <td>1695.00</td>
      <td>122236700</td>
      <td>43114900</td>
      <td>10923600</td>
    </tr>
    <tr>
      <th>2023-12-28</th>
      <td>5641.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6150.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1685.00</td>
      <td>5700.00</td>
      <td>6050.00</td>
      <td>1695.00</td>
      <td>121434600</td>
      <td>75118700</td>
      <td>23222700</td>
    </tr>
    <tr>
      <th>2023-12-29</th>
      <td>5641.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5725.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5675.00</td>
      <td>6000.00</td>
      <td>1710.00</td>
      <td>5750.00</td>
      <td>6125.00</td>
      <td>1735.00</td>
      <td>93126000</td>
      <td>63097100</td>
      <td>21099100</td>
    </tr>
    <tr>
      <th>2024-01-02</th>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1740.00</td>
      <td>5675.00</td>
      <td>6125.00</td>
      <td>1745.00</td>
      <td>5625.00</td>
      <td>6025.00</td>
      <td>1710.00</td>
      <td>5650.00</td>
      <td>6050.00</td>
      <td>1740.00</td>
      <td>91143100</td>
      <td>26235700</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2024-01-03</th>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5600.00</td>
      <td>6100.00</td>
      <td>1800.00</td>
      <td>5650.00</td>
      <td>6150.00</td>
      <td>1830.00</td>
      <td>5600.00</td>
      <td>6050.00</td>
      <td>1730.00</td>
      <td>5625.00</td>
      <td>6100.00</td>
      <td>1735.00</td>
      <td>83659700</td>
      <td>30053900</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 18 columns</p>
</div>



## Menggabungkan `agg` dan `groupby`


```python
stock_long = stock.stack().reset_index()
stock_long
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
      <th>Attributes</th>
      <th>Date</th>
      <th>Symbols</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-02</td>
      <td>BBRI</td>
      <td>3717.37</td>
      <td>4410.00</td>
      <td>4410.00</td>
      <td>4360.00</td>
      <td>4400.00</td>
      <td>41714100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>BMRI</td>
      <td>3239.33</td>
      <td>3875.00</td>
      <td>3887.50</td>
      <td>3825.00</td>
      <td>3837.50</td>
      <td>37379800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-02</td>
      <td>BRIS</td>
      <td>326.18</td>
      <td>332.00</td>
      <td>336.00</td>
      <td>330.00</td>
      <td>330.00</td>
      <td>1456400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-03</td>
      <td>BBRI</td>
      <td>3725.80</td>
      <td>4420.00</td>
      <td>4440.00</td>
      <td>4390.00</td>
      <td>4420.00</td>
      <td>82898300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-03</td>
      <td>BMRI</td>
      <td>3228.88</td>
      <td>3862.50</td>
      <td>3912.50</td>
      <td>3812.50</td>
      <td>3875.00</td>
      <td>70294600</td>
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
    </tr>
    <tr>
      <th>2923</th>
      <td>2024-01-02</td>
      <td>BMRI</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6125.00</td>
      <td>6025.00</td>
      <td>6050.00</td>
      <td>26235700</td>
    </tr>
    <tr>
      <th>2924</th>
      <td>2024-01-02</td>
      <td>BRIS</td>
      <td>1740.00</td>
      <td>1740.00</td>
      <td>1745.00</td>
      <td>1710.00</td>
      <td>1740.00</td>
      <td>13118700</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>2024-01-03</td>
      <td>BBRI</td>
      <td>5600.00</td>
      <td>5600.00</td>
      <td>5650.00</td>
      <td>5600.00</td>
      <td>5625.00</td>
      <td>83659700</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2024-01-03</td>
      <td>BMRI</td>
      <td>6100.00</td>
      <td>6100.00</td>
      <td>6150.00</td>
      <td>6050.00</td>
      <td>6100.00</td>
      <td>30053900</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2024-01-03</td>
      <td>BRIS</td>
      <td>1800.00</td>
      <td>1800.00</td>
      <td>1830.00</td>
      <td>1730.00</td>
      <td>1735.00</td>
      <td>76511200</td>
    </tr>
  </tbody>
</table>
<p>2928 rows × 8 columns</p>
</div>



Misalkan kita ingin membuat tabel agregasi dengan `aggfunc` yang berbeda-beda untuk masing-masing `Symbols` berupa:
- Maximum `stock` price (`max` dari `High`)
- Minimum `stock` price (`min` dari `Low`)
- Rata-rata closing price (`mean` dari `Close`)

Untuk mendapat hasil tersebut, kita harus melakukan chaining `groupby` dengan method `agg`. Kita harus menyertakan mapping (**dictionary**) untuk setiap kolom dengan fungsi agregasinya seperti berikut ini:

Syntax:

```
.agg({
    'NAMA_KOLOM': 'FUNGSI_AGREGASI'
})
```


```python
summary_stock = stock_long.groupby('Symbols').agg({
    'High' : 'max',
    'Low' : 'min',
    'Close' : 'mean'
})
summary_stock
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
      <th>Attributes</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Symbols</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BBRI</th>
      <td>5750.00</td>
      <td>2160.00</td>
      <td>4341.34</td>
    </tr>
    <tr>
      <th>BMRI</th>
      <td>10400.00</td>
      <td>1830.00</td>
      <td>3974.56</td>
    </tr>
    <tr>
      <th>BRIS</th>
      <td>3980.00</td>
      <td>135.00</td>
      <td>1517.88</td>
    </tr>
  </tbody>
</table>
</div>



❓ Visualisasikan tabel agregasi di atas untuk membandingkan nilai tersebut.


```python
# visualisasi
summary_stock.plot(kind = 'bar')
```




    <Axes: xlabel='Symbols'>




    
![png](https://github.com/Saltfarmer/Algoritma-BFLP-DS-Audit/blob/main/3_dwv-dev/output_102_1.png)