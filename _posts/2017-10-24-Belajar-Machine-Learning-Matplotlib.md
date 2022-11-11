---
title: "0010 Belajar Machine Learning : Matplotlib"
header:
  teaser: /assets/images/machinelearning_header.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - Python
  - Matplotlib
---

Midnight post lagi -_-. Enaknya bahas apaan ya ??? Karena bakal nggak asik kalo ML tanpa illustrasi ( ͡° ͜ʖ ͡°), mendingan bahas tentang illustrasi yang bisa dilakukan di python. Library yang akan digunakan pada hari ini yaitu **matplotlib**. Jadi berikut akan saya berikan contoh contoh penggunaan matplotlib. Untuk postingan akan membahas sedikit tentang penggunaan numpy juga. Untuk seaborn hanya digunakan sebagaia background dari grafik yang digambarkan. Silahkan dinikmati :v

```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

## Garis sederhana

```python
plt.plot(x, y, color='warna_yang_diinginkan', linestyle='bentuk_garis', label='nama_garisnya')
plt.title("Judul")
plt.xlabel("Label yang horizontal")
plt.ylabel("Label yang vertical")
plt.legend()
```

Contoh konkrit : 

```python
plt.plot(x, np.sin(x), color='blue', linestyle='dashed', label="sin(x)")       
plt.plot(x, np.cos(x), color='#FFDD44', linestyle='-', label="sin(x)") 
plt.title("Sinus dan Cos")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()  
```

<figure>
<img src="https://raw.githubusercontent.com/Saltfarmer/blog/master/assets/images/plot1.png">
</figure>

<figure>
<img src='https://static.esea.net/global/images/teams/135421.1467088198.jpg'>
<figcaption>Awas tertypu gan :v</figcaption>
</figure>

## Scatter plot

```python
plt.plot(x, y, 'bentuk', color='warna');
```

Untuk penggunaan bentuk bisa lihat grafik dibawah :

```python
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);
```

<figure>
<img src='https://raw.githubusercontent.com/Saltfarmer/blog/master/assets/images/plot2.png'>
</figure>

Contoh konkrit :
```python
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')
plt.plot(x, np.cos(x), '-p', color='green')
```

<figure>
<img src='https://raw.githubusercontent.com/Saltfarmer/blog/master/assets/images/plot3.png'>
</figure>

## Histogram

```python
plt.hist(data)
```

Contoh konkrit :

```python
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

plt.hist(x1, histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(x2, histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(x3, histtype='stepfilled', alpha=0.3, normed=True, bins=40)
```

<figure>
<img src='https://raw.githubusercontent.com/Saltfarmer/blog/master/assets/images/plot4.png'>
</figure>

Jadi akhir dari post ini hanya menunjukan ke3 grafik tersebut. Untuk detail lebih lengkapnya bisa melihat contoh grafik di <a href='https://pandas.pydata.org/pandas-docs/stable/index.html'>Dokumentasi matplotlib</a>. Semoga postingan ini bermanfaat dan jangan lupa untuk mencobanya di rumah :v

>Sing penting yakin
