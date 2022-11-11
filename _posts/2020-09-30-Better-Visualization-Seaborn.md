---
title: "Better Visualization : Seaborn"
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

After discussing basic visualization with Matplotlib, now let's try another but more attractive visualization library called Seaborn. Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

There are 5 main plots that really useful for your visualization.

1. Relational Plot
2. Distribution Plot
3. Categorical Plot
4. Regression Plot
5. Matrix Plot

and 3 multi-plot grids which are

1. Facet grids
2. Pair grids
3. Joint grids

But first, let's use Seaborn own dataset sample called tips

```python
tips = sns.load_dataset('tips')
```

## Relational Plot

This function provides access to several different axes-level functions that show the relationship between two variables  or features with semantic mappings of subsets. The `kind` parameter selects the underlying axes-level function like `scatter` and `line`

Example

```python
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter')
```

![](https://i.ibb.co/DpKjD8J/sns-scatter.png)

```python
sns.relplot(data=tips, x='total_bill', y='tip', kind='line')
```

![](https://i.ibb.co/Chndn9X/sns-line.png)

When thinking about how to assign variables to different facets, a general rule is that it makes sense to use `hue` for the most important comparison, followed by `col` and `row`. However, always think about your particular dataset and the goals of the visualization you are creating.

You can visualize comparison between 2 features with another categorical features with `hue` (by color), `column`, and `row` (by another plot).

Example

```python
sns.relplot(data=tips, x="total_bill", y="tip", hue="day", col="time", row="sex")
```

![](https://i.ibb.co/dKbBRgN/rel-group.png)

## Distribution Plot

This function provides access to several approaches for visualizing the univariate (single feature) or bivariate (double features) distribution of data, including subsets of data defined by semantic mapping and faceting across multiple subplots. The `kind` parameter selects the approach to use are `hist` (Histogram), `kde` (A kernel density estimate (KDE) plot is a method for visualizing the density of dataset), and `ecdf` (stair-like, univariate only).

Example

```python
sns.displot(data=tips, x='total_bill', kind='hist')
```

![](https://i.ibb.co/jvW9R7y/sns-hist.png)

```python
sns.displot(data=tips, x='total_bill', kind='kde')
```

![](https://i.ibb.co/ZdkbD1v/sns-kde.png)

```python
sns.displot(data=tips, x='total_bill', kind='ecdf')
```

![](https://i.ibb.co/2vCkhDn/sns-ecdf.png)

You can also showing histogram or ecdf with kde.

```python
sns.displot(data=tips, x='total_bill', kind='hist', kde=True)
```

![](https://i.ibb.co/zJTTsQN/sns-histkde.png)

Then, if you want to visualize between 2 features

```python
sns.displot(data=tips, x='total_bill', y='tip', kind='kde')
```

![](https://i.ibb.co/ydDgKXQ/sns-bivkde.png)

## Categorical Plot

This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. The `kind` parameter selects the underlying axes-level function to use:

### Categorical scatterplots:

- [`stripplot()`](https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot) (with `kind="strip"`; the default)

  ```python
  sns.catplot(data=tips, x='sex', y='total_bill', kind='strip')
  ```

  ![](https://i.ibb.co/DCV3Sf3/sns-strip.png)

- [`swarmplot()`](https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot) (with `kind="swarm"`)

```python
sns.catplot(data=tips, x='sex', y='total_bill', kind='swarm')
```

![](https://i.ibb.co/JHRq5D7/sns-swarm.png)

### Categorical distribution plots:

- [`boxplot()`](https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot) (with `kind="box"`)

  ```python
  sns.catplot(data=tips, x='sex', y='total_bill', kind='box')
  ```

  ![](https://i.ibb.co/9GHk2YB/sns-box.png)

- [`violinplot()`](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot) (with `kind="violin"`)

  ```python
  sns.catplot(data=tips, x='sex', y='total_bill', kind='violin')
  ```

    ![](https://i.ibb.co/RHq7wJs/sns-violin.png)

- [`boxenplot()`](https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot) (with `kind="boxen"`)

  ```python
  sns.catplot(data=tips, x='sex', y='total_bill', kind='boxen')
  ```

    ![](https://i.ibb.co/9yCSq42/sns-boxen.png)

### Categorical estimate plots:

- [`pointplot()`](https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot) (with `kind="point"`)

  ```python
  sns.catplot(data=tips, x='day', y='total_bill', kind='point')
  ```

    ![](https://i.ibb.co/S63GM3y/sns-point.png)

- [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot) (with `kind="bar"`)

  ```python
  sns.catplot(data=tips, x='day', y='total_bill', kind='bar')  
  ```

    ![](https://i.ibb.co/8sQnjRV/sns-bar.png)

- [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) (with `kind="count"`)

  ```python
  sns.catplot(data=tips, x='sex', hue='day', kind='count')
  ```

    ![](https://i.ibb.co/djyVjhY/sns-count.png)

## Regression Plot

To plot data with regression model fits across a FacetGrid.

```python
sns.lmplot(data=tips, x='tip', y='total_bill', hue='sex', col='smoker')
```

![](https://i.ibb.co/wCqvkn5/sns-lm.png)

## Matrix Plot

Plot rectangular data as a color-encoded matrix. This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the `ax` argument. Part of this Axes space will be taken and used to plot a colormap, unless `cbar` is False or a separate Axes is provided to `cbar_ax`.

Example

```python
sns.heatmap(tips.corr())
```

![](https://i.ibb.co/Y8V4B0Q/sns-heatmap.png)













