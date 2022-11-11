---
title: "Python Numpy"
header :
  teaser: /assets/images/numpy-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Machine Learning
  - Python
  - Numpy
 

---

Let's continue with Numpy. NumPy is a python library used for working with arrays. It also has functions for working in domain of linear algebra and matrices. In Python we have lists that serve the purpose of arrays, but they are slow to process. NumPy aims to provide an array object that is up to 50x faster that  traditional Python lists. The array object in NumPy is called `ndarray`, it provides a lot of supporting functions that make working with  `ndarray` very easy. 

Arrays are very frequently used in data science, where speed and resources  are very important. NumPy arrays are stored at one continuous place in memory unlike lists,  so processes can access and manipulate them very efficiently.

## Creating Array

Let start creating numpy array

```python
import numpy as np

x_list = [1, 2, 3, 4, 5]

arr = np.array(x_list)

print(arr)
print(type(arr))
```

```
[1 2 3 4 5]
<class 'numpy.ndarray'>
```

It is pretty much the same as list. So what is the difference ? 

- **Arrays need to be declared. Lists don't**, since they are built into Python. In the examples above, you saw that lists  are created by simply enclosing a sequence of elements into square  brackets. Creating an array, on the other hand, requires a specific  function from either the array module (i.e., `array.array()`) or NumPy package (i.e., `numpy.array()`). Because of this, lists are used more often than arrays.
- **Arrays can store data very compactly** and are more efficient for storing large amounts of data.
- **Arrays are great for numerical operations**; lists  cannot directly handle math operations. For example, you can divide each element of an array by the same number with just one line of code. If  you try the same with a list, you'll get an error.

 Here we try to make multidimensional array in Numpy

```python
import numpy as np

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr)
print(arr.ndim)
```

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]
2
```

You can also use `np.arange` to generate interval value. `np.arange` works with this formula

> `np.arange(start, stop, step)` 

For example

```python
import numpy as np

print(np.arange(1,10))
```

```
[1 2 3 4 5 6 7 8 9]
```

with steps

```python
import numpy as np

print(np.arange(1,10,2))
```

```
[2 4 6 8]
```

You can also generate with `np.linspace` to generate evenly spaced numbers over a specified interval. For Example

```python
import numpy as np

print(np.linspace(1,10,11))
```

```
[ 1.   1.9  2.8  3.7  4.6  5.5  6.4  7.3  8.2  9.1 10. ]
```



you can also generate zero matrices, one matrices, and eye matrices (one diagonal and zeros everywhere). The formula is

> np.zeros((number of dimension))
>
> np.ones((number of dimension))
>
> np.eye((number of dimension))

Example

```python
import numpy as np

print(np.zeros((3,3)))
print(np.ones((2,2)))
print(np.eye(2))
```

```
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
[[1. 1.]
 [1. 1.]]
[[1. 0.]
 [0. 1.]]
```

 ## Numpy Random

Random number does NOT mean a different number every time. Random means something that can not be predicted logically. Computers work on programs, and programs are definitive set of instructions. So it means there must be some algorithm to generate a random number as well.

If there is a program to generate random number it can be predicted, thus it is not truly random. Random numbers generated through a generation algorithm are called *pseudo random*. Example of random numbers in numpy

```python
import numpy as np

print(np.random.rand(10))
```

```
[0.23310967 0.40175505 0.33707093 0.99442824 0.63863446 0.96654872
 0.2247678  0.00794046 0.29332574 0.23675905]
```

then for Integer number, here is the example

```python
import numpy as np

print(np.random.randint(1, 20, 10))
```

```
[18  1 16  9  1  3  5 19 10 16]
```

There are many different random distribution like Normal, Binomial, Poisson, Uniform, Logistic, Multinomial, Exponential, Chi Square, Rayleigh, Pareto, and Zipf Distribution. For the details you can check the differences [here](https://www.w3schools.com/python/numpy_random.asp).

## Shape and Reshaping

The shape of an array is the number of elements in each dimension. NumPy arrays have an attribute called `shape` that returns a tuple with each index having the number of corresponding elements. For example

```python
import numpy as np

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr.shape)
```

```
(3, 3)
```

That means we have 3x3 dimension. Now lets try reshape it to flat 1D dimension.

```python
print(arr.reshape(9))
```

```
[1 2 3 4 5 6 7 8 9]
```

And then back to multidimension

```python
import numpy as np

arr = np.arange(1,13)
arr = arr.reshape(4,3)
print(arr)
```

```
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

## Indexing and Slicing

Array indexing is the same as accessing an array element. You can access an array element by referring to its index number. Because we already discuss it on last post, lets try something different by indexing 2D element.  For example, lets try pick coordinate number [1,1]

```python
import numpy as np

arr = np.arange(1,13)
arr = arr.reshape(4,3)

print(arr[1,1])
```

```
[5]
```

Slicing in python means taking elements from one given index to another given index. We pass slice instead of index like this: `[*start*:*end*]`. We can also define the step, like this: `[*start*:*end*:*step*]`. If we don't pass start its considered 0 If we don't pass end its considered length of array in that dimension. If we don't pass step its considered 1

For Example lets try picking 5, 8, 11 from last example

```python
import numpy as np

arr = np.arange(1,13)
arr = arr.reshape(4,3)

print(arr[1:,1])
```

```
[5 8 11]
```

