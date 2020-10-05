---
title: "Interactive Visualization Plotly"
header :
  image: /assets/images/matplotlib-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Machine Learning
  - Python
  - Matplotlib
  - Plotly
 

---

The plotly Python library is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases. Built on top of the Plotly JavaScript library (plotly.js), plotly enables Python users to create beautiful interactive web-based visualizations that can be displayed in Jupyter notebooks, saved to standalone HTML files, or served as part of pure Python-built web applications using Dash. However, Cufflinks connects Plotly with pandas to produce the interactive data visualizations.

Let's get start with

```python
!pip install plotly
!pip install cufflinks

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

# For Notebooks
init_notebook_mode(connected=True)

# For offline use
cf.go_offline()
```

We will be using download_plotlyjs, init_notebook_mode, plot and iplot from plotly.offline and the .go_offline() method to allow us interactive visualizations offline.

- download_plotlyjs -> Allows for us to work with the visualizations offline
- init_notebook_mode -> Allows for us to plot graphs offline inside a Jupyter Notebook Environment

Let's see the example of Plotly visualization using available sample datasets from Seaborn

## Line Plots

Use the .iplot() method to generate a line plot with the dataset. This plot allows us to click on the elements in the legend to hide and display context which is pretty neat. My the cursor to the top right of the plot to observe the various features of the plot. We can also use the zoom feature of specific areas of the plot.

```python
import seaborn as sns

flights = sns.load_dataset('flights')
flights.iplot(kind='line', y='passengers')
```

<figure>
    <img src="https://i.ibb.co/XV381jr/ezgif-2-1791205089be.gif">
</figure>

This is really useful for understanding TimeSeries data 

## Scatter Plots

Use the .iplot() method with arguments kind (plot type), x (x-axis variable), y (y-axis variable), and mode argument removes the line connections setup by default with plotly. The plot can be zoomed in or out depending on need.

```python
iris = sns.load_dataset('iris')
iris.iplot(kind ='scatter', x ='sepal_length', y ='sepal_width', mode ='markers') 
```

<figure>
    <img src="https://i.ibb.co/zrX1SKX/ezgif-2-a281e19dfc8e.gif">
</figure>


