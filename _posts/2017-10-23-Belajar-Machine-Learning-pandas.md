---
title: "0001 Belajar Machine Learning : Pandas"
header:
  image: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Python
  - Pandas
---

Midnight post nih gan mumpung lagi gabut. Pikir-pikir enaknya lanjut bahas ML kayak kemaren ( ͡° ͜ʖ ͡°). **Pandas** adalah semacam library dari Python yang biasanya digunakan untuk manipulasi data. Pandas secara umumnya digunakan seperti membuat tabel, mengubah dimensi data, mengecek data, dan semacamnya. Python sudah dikenal sebelumnya dalam modifkasi data, tapi kurang baik untuk analisis dan modelling. Pandas membantu memperbaiki ini memungkinkan anda mengerjakan alur kerja dalam analisis data. Dalam post ini akan menyebutkan beberapa macam function yang sering digunakan. Pertama-tama kita harus mengimport terlebih dahulu pandas ke dalam python

```python
import pandas as pd
```

<figure>
  <img src="https://blogunik.com/wp-content/uploads/2017/04/keunikan-bulu-panda-1.jpg">
  <figcaption>Lucu ya pandanya :v</figcaption>
</figure>

## Input Output
Yang pertama dan paling penting ialah input dan output (iyalah nggak ada isinya lu mau ngapain tong :v). Digunakan untuk mengimport data atau mengexport data. Data yang disupport pandas seperti format tabel kebanyakan seperti csv dan excel.

```python
pd.read_csv("namafile")
pd.read_table("namafile")
pd.read_excel("namafile")
pd.read_html("url")
```

Untuk import jangan lupa juga menyertakan lokasi direktori dari data kalian. Untuk data yang dari web tentu saja memasukkan link dari data tersebut

```python
df.to_csv("namafile")
df.to_table("namafile")
df.to_excel("namafile")
df.to_html("namafile")
```

*df* yang dimaksud disini adalah nama variabel dari dataframe kalian.

## Melihat/mengecek data

```python
df.head(n)
df.tail(n)
df.shape()
df.info()
df.describe()
```

* head(n) : Berfungsi untuk melihat data sebanyak **n** pada kolom awal (jika tidak diisi, secara random n=6).
* tail(n) : Berfungsi untuk melihat data sebanyak **n** pada kolom akhit (jika tidak diisi, secara random n=6).
* shape() : Melihat jumlah baris dan kolom.
* info() : Nomor index beserta tipe datanya.
* describe() : Menunjukkan rangkuman statistik seperti rata-rata, median, dll pada kolom.

## Memilih baris dan kolom

```python
df.iloc([nomor_baris], [nomor kolom])
df.loc([nomor_baris], ["Nama kolom"])
df.loc([:], ["Nama kolom"])
``` 

Pada index yang berisi ":" itu menandakan memilih semua semua urutan pada index. Lalu kenapa terkadang menggunakan **iloc()** atau **loc()**. Yang pertama untuk iloc() karena dia akan memilih data berdasarkan index, sedangkan untuk loc() menghiraukan kolom index nya. 

## Data cleaning

```python
df.columns =['a', 'b', 'c']
pd.isnull()
df.dropna()
df.dropna(axis=1)
df.dropna(axis=1, thresh=n)
df.fillna(x)
df.fillna(df.mean())
s = df.ix[:,nomor kolom yang dipilih]
s.astype(tipedata)
s.replace([1,3], ['one', 'three'])
df.rename(columns={'nama lama': 'nama baru'})
```

keterangan dari awal sampai akhir :
1. Mengubah nama kolom jadi 'a', 'b', 'c'.
2. Mengecek apakah ada nilai NULL dengan keluaran boolean.
3. Menghapus baris yang berisi NULL.
4. Menghapus semua kolom yang berisi NULL.
5. Menghapus semua kolom dengan batas n.
6. Mengisi nilai NULL dengan x.
7. Mengisi nilai NULL dengan rata-rata.
8. Membuat series dengan dataframe.
9. Mengubah tipe data pada series.
10. Mengubah nilai 1 dan 3 pada series dengan 'one' dan 'three'
11. Merename nama kolom.

## Filter dan sort

```python
df[(df[kolom] > 0.5) & {df[kolom] < 0.7)}]
df.sort_values(nomorkolom, ascending=True)
```

Yang pertama adalah melihat kolom yang lebih dari 0.5 dan kurang dari 0.7. Lalu yang kedua adalah mengurutkan kolom secara ascending (kecil terlebih dahulu).

## Combine

```python
df1.append(df2)
df1.append([df1, df2], axis=1)
df1.join(df2, on="kolom_yang_sama", how='inner')
```

* Menggabungkan baris df1 ke akhir df2 (jumlah kolom harus sama)
* Menggabungkan kolom df1 ke akhir df2 (jumlah baris harus sama)
* Metode SQL : melakukan **inner join** pada df1 dan df2 dengan mensyaratkan kolom yang sama. Seperti di SQL, terdapat juga Right join, Left Joim, outer join, dan tentunya inner join.

Untuk lebih lengkapnya dalam penggunaan pandas bisa dilihat di [Dokumentasi pandas]("https://pandas.pydata.org/pandas-docs/stable/index.html"). Cukup sekian post dari saya. Sebenarnya masih banyak lagi fungsi - fungsi di pandas. Asalkan anda sering latihan maka saya yakin anda akan bisa menguasai library ini. Semoga bermanfaat postingan saya pada malam ini. Saya tunggu di postingan berikutnya

>Sing penting yakin
