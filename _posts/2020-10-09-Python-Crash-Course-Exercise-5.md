---
title: "Python Crash Course Exercise 5"
header :
  teaser: /assets/images/matplotlib-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Exercise
  - Matplotlib
 

---

This day i will completing Matplotlib Exercise. If you want to solve it all by yourself, you can download notebooks file [here](https://drive.google.com/file/d/1LJAoRsI-F1UbY1gjjnQo5jATBMjua_qO/view?usp=sharing)

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

/

Now Lets get started

# Matplotlib Exercises 

Welcome to the exercises for reviewing matplotlib! Take your time with these, Matplotlib can be tricky to understand at first. These are relatively simple plots, but they can be hard if this is your first time with matplotlib, feel free to reference the solutions as you go along.

Also don't worry if you find the matplotlib syntax frustrating, we actually won't be using it that often throughout the course, we will switch to using seaborn and pandas built-in visualization capabilities. But, those are built-off of matplotlib, which is why it is still important to get exposure to it!

** * NOTE: ALL THE COMMANDS FOR PLOTTING A FIGURE SHOULD ALL GO IN THE SAME CELL. SEPARATING THEM OUT INTO MULTIPLE CELLS MAY CAUSE NOTHING TO SHOW UP. * **

# Exercises

Follow the instructions to recreate the plots using this data:

## Data


```python
import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2
```

** Import matplotlib.pyplot as plt and set %matplotlib inline if you are using the jupyter notebook. What command do you use if you aren't using the jupyter notebook?**


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

## Exercise 1

** Follow along with these steps: **

* ** Create a figure object called fig using plt.figure() **
* ** Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. **
* ** Plot (x,y) on that axes and set the labels and titles to match the plot below:**


```python
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')
```


    Text(0.5, 1.0, 'title')




![png](https://i.ibb.co/Cny1XT7/output-5-1.png)


## Exercise 2

** Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.**


```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])
```


![png](https://i.ibb.co/Gv5jy97/output-7-0.png)


** Now plot (x,y) on both axes. And call your figure object to show it.**


```python
ax1.plot(x,y)
ax2.plot(x,y)

fig
```




![png](https://i.ibb.co/XF9vymt/output-9-0.png)



## Exercise 3

** Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]**


```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.4,.4])
```


![png](https://i.ibb.co/XDFkhpk/output-11-0.png)


** Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:**


```python
ax1.plot(x,z)
ax2.plot(x,y)
ax2.set_xlim(20,25)
ax2.set_ylim(30,50)

fig
```




![png](https://i.ibb.co/yhW1fzH/output-13-0.png)



## Exercise 4

** Use plt.subplots(nrows=1, ncols=2) to create the plot below.**


```python
fig, axes = plt.subplots(nrows=1, ncols=2)
```


![png](https://i.ibb.co/tx6zvv1/output-15-0.png)


** Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style**


```python
axes[0].plot(x,y,color="b", lw=4, ls="--")
axes[1].plot(x,z,color="r", lw=4, ls="-")
fig
```




![png](https://i.ibb.co/mGBPRVq/output-17-0.png)



** See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code.**


```python
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,4))

axes[0].plot(x,y,color="blue", lw=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

axes[1].plot(x,z,color="red", lw=3, ls='--')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
```


    Text(0, 0.5, 'z')




![png](https://i.ibb.co/YTJf557/output-19-1.png)

# Great Job!