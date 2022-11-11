---
title: "0000 Belajar Algoritma dan Struktur data : Pengenalan Python 3"
header:
  teaser: /assets/images/cp-header.jpg
comments : true
share : true
categories:
  - Algoritma dan Struktur Data
tags:
  - Algoritma dan Struktur Data
  - Python
---

Pagi gan, semoga udah pada bangun. Jadi untuk mengisi kegabutan saya di kampus, pada hari ini saya akan mulai posting tentang algoritma dan struktur data. Karena sudah kebanyakan orang *ngoding* di bahasa c++, pada postingan ini dan seterusnya saya akan menjelaskannya dalam bahasa Python3. Kenapa Python ??? karena Python sangat mudah bagi para nubie seperti saya (ya, saya masih noob :v) untuk membaca alur programnya. Lalu kelebihan selanjutnya adalah komunitasnya. Alasannya adalah karena Python adalah basa paling populer di dunia. 

<figure>
	<img src='https://static1.squarespace.com/static/51361f2fe4b0f24e710af7ae/51363c0de4b0b95e028b2e8c/51363c0ee4b0b95e028b2ea9/1362508821226/'>
</figure>

Lalu kelebihan selanjutnya dari Python adalah penerapan yang bisa dilakukan di banyak bidang seperti di bagian back-end pada *web developing* dan membuat software desktop. Terus apa bedanya Python 2 dan Python 3 ??? sebenarnya tidak ada perbedaan yang cukup mencolok. Contoh perbedaanya adalah saat pemanggilan function.

```python
print 'Hello world' #python2.x
print ('Hello world') #python3.x
```

Langsung saja kita membahas hal-hal dasar untuk ngoding di Python. Untuk melakukan comment bisa menulis '#' di awal baris dan indentasi python adalah perbaris programnnya.

## 1.Input dan Output

<figure>
	<img src='https://upload.wikimedia.org/wikipedia/commons/9/92/CPT_Hardware-InputOutput.svg'>
</figure>

```python
text = input('')
print (text)
```

## 2.Operator

<figure>
	<img src="https://d1e4pidl3fu268.cloudfront.net/f20083ef-a2fb-4673-ac88-13d58ba68133/Arithmeticoperators.png">
</figure>

```python
#Operator aritmatika
a = 1+1 #tambah
a = 2-1 #kurang
a = 2*3 #perkalian
a = 2**3 #pangkat
a = 4/2 #pembagian
a = 8//2 #akar
a = 7 % 3 #modulo atau sisa bagi
```

<figure>
	<img src="https://d1e4pidl3fu268.cloudfront.net/48295477-61ac-4b5e-a6dc-456acd3ab0a6/comparisonoperators.png">
</figure>

```python
#selanjutnya operator perbandingan
a <= b #kurang dari sama dengan
a >= b #lebih dari sama dengan
a > b #lebih dari
a < b #kurang dari
a == b #sama dengan
a != b #tidak sama dengan
a and b = c #dan pada operasi boolean
a or b = c #atau pada operasi boolean
not a != b #negasi pada operasi boolean
```
CATATAN : Untuk operasi penghitungan terhadap dirinya sendiri bisa menggunnakan operator sebelum sama dengan. Lalu untuk perbandingan kurang sama dengan atau lebih sama dengan operator pembanding kurang atau lebih ditulis telebih dahulu. Contoh :

```python
a += b #Maksudnya adalah a = a+b
a -= b
a *= b
a /= b

a <= b #benar
a =< b #error
```

## 3.Percabangan

<figure>
	<img src="https://cdn.programiz.com/sites/tutorial2program/files/c-if-else.jpg">
</figure>

```python
if (persyaratannya):
	perintahnya
elif (persyaratannya):
	perintahnya
else:
	perintahnya
```

Contoh konkrit penggunaan percabangan pada menentukan negatif atau positif :

```python
n = input('Masukkan angka : ')
if (n>0):
	print ('Positif')
elif (n == 0):
	print ('Nol')
else:
	print ('Negatif')
```

## 4. Pengulangan

<figure>
	<img src="https://ramsgatearts.org/wp-content/uploads/2015/01/LOOPING-THE-LOOP.jpg">
</figure>

```python
for i in range(nilai_awal, batas_atas, perubahan):
	print('sesuatu sebanyak batas_atas - nilai_awal')

while (persyaratan):
	print('sampai persyaratannya tidak terpenuhi lagi')
```

contoh konkrit for dan while :

```python
for i in range(0,5,1):
	print("halo")
#Mencetak halo 5x
i = 0
while (i<5):
	print("halo")
	i += 1
#Sama seperti yang diatas
```

Karena pada pengulangan diawali dengan nilai angka awal namun tidak menyentuh batas atas. Sekian postingan tentang dasar - dasar penggunaan python. Untuk jenis array akan dibahas di postingan berikutnya beserta dari contoh soal yang menarik dan juga solusyen nya :v. Semoga bermanfaat.

>Sing penting yakin
