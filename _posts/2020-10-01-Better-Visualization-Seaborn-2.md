---
title: "Better Visualization : Seaborn 2"
header :
  teaser: /assets/images/matplotlib-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Machine Learning
  - Python
  - Matplotlib
  - Seaborn
 

---

Multiplot grid are general types of plots that allow you to map plot types to rows and columns of a grid, this helps you create similar plots separated by features.

## Facet Grids

Faceting is the act of breaking data variables up across multiple subplots and combining those subplots into a single figure.

Faceting is really helpful if you want to quickly explore your dataset.

To use one kind of faceting in Seaborn we can use the `FacetGrid`. First of all, we need to define the `FacetGrid` and pass it our data as well as a row or column, which will be used to split the data. Then we need to call the `map `function on our `FacetGrid` object and define the plot type we want to use, as well as the column we want to graph.

Lets continue from Tips dataset from last post as example

```python
tips = sns.load_dataset("tips")

g = sns.FacetGrid(tips, hue='smoker', col="time",  row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()
```

![](https://i.ibb.co/KrRTbFJ/facet1.png)

You can try `FacetGrid()` with different types of plot in Seaborn

```python
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.histplot, "total_bill")
```

![](https://i.ibb.co/Dz2sxdw/facet2.png)

## Pair Grids

This Pair Grids let you visualize plots of every combination of various attributes together, so you can look for interesting patterns between features. The PairGrid does require further setup, but it also provides greater control of the output. This comes in the form of specifying the plots to be used in relation to the grid. It feels good when the best plot form can be used for the data relation in question. 

Let see the example

```python
g = sns.PairGrid(tips)
g.map(sns.scatterplot)
```

![](https://i.ibb.co/9vRV6V8/pair1.png)

Passing separate functions to [`PairGrid.map_diag()`](https://seaborn.pydata.org/generated/seaborn.PairGrid.map_diag.html#seaborn.PairGrid.map_diag) and [`PairGrid.map_offdiag()`](https://seaborn.pydata.org/generated/seaborn.PairGrid.map_offdiag.html#seaborn.PairGrid.map_offdiag) will show each variable’s marginal distribution on the diagonal:

```python
g = sns.PairGrid(tips, hue='sex')
g.map_diag(sns.boxplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
```

![](https://i.ibb.co/hMgVMtx/pair2.png)

As same as Facet Grids, you can also customize the pair grid with different kinds of plot in Seaborn as long as you can differ univariate and bivariate plot.

## Joint Grids

Joint Grids are useful when we want to visualize the relationship between two variable as well as their univariate relationship. Many plots can be drawn by using the figure-level interface [`jointplot()`](https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot). Use this class directly when you need more flexibility.

Example

```python
g = sns.JointGrid(data=tips, x="total_bill", y="tip", hue='sex')
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.histplot, kde=True)
```

![](https://i.ibb.co/qW4PyP9/joint1.png)





