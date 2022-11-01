---
title: "Python Data Visualization Guide"
header : 
  overlay_image : /assets/images/python-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - EDA
  - Visualization
---

Creating a visualization may not as easier as it looks. Some of the visualizations may look cool but not interpret what they mean. Imagine after a hard and long time working on some projects. Next days, you need to present your project to your executives or your boss. You thought "Ah it's fine, my visualization is cool and eye-catching".  On judgement day, all your optimism while naive thought was wrong. Your boss doesn't get any point and don't understand what you are presenting.  Then you are thinking, what should i do to make them understand what i am presenting. 

A visualization should speak for itself. You should not need to spend time trying to tell its story or do its job. Stephen Few writes that “An effective [visualization] is the product not of cute gauges, meters, and traffic lights, but rather of informed design: more science than art, more simplicity than dazzle. It is above all else, communication.” 

Visualization type and what visualization to use is the main point of this post. There are mainly 4 narratives in "storytelling" with visualization.

## Time Series

Imagine you work for a Financial company or Banking. Or at least working as a Data Analyst in the Marketing Division. You would need to use time series analysis to set the initial context for the change. 

**Line charts** would be the key to visualization. They could show the unprecedented change in a single financial metric on a daily, monthly or annual granularity. 

**Multiple line charts** are for multiple data sets that all share common units of measure, such as showing the decline in revenue, expense, and income over the same time period. 

**Stacked area charts** would come into play show change over time for multiple data sets, groups, or cluster that together make up a whole. For Example, the banker, he could show how each of the regions changed over time while showing how the combination of the regional totals added up to the corporate total. 

**Combination bar and line charts** would be used when multiple data sets need to be shown together over time, but they differ in units of measure. This would be how the banker would show staffing levels on the left Y-axis in units of people while showing net profit on the right Y-axis in units of dollars.  The combination of all these line-based visualizations would set the context for why radical measures are needed to deal with unprecedented change.

## Categorical Comparison

Many data stories involve comparing categories, such as comparing multiple lines, different types of data, or same type but not quite the same in distribution of data. 

Your visualization could involve **drawn objects** like a process flow or entity relationship diagram to show the flow of money from liquidity providers to loan applicants. 

**Tables** are the best way to show a large collection of numbers across categories in a dense format. 

**Bar charts** would be used to compare different groups in data. 

**Tree maps** are used for a similar purpose but show proportionally how one data compared for each type of groups of data. 

**Spatial maps** would be helpful to regionally display the data. For a nationwide bank, a spatial map could display income by region with individual measures overlaid on each region.

**Small multiples** would be helpful to compare the branches on multiple levels, in a format seen frequently in Tableau. Three side by sidebar graphs could represent income, expenditures, and revenue. Within these bars, graphs would be barred representing each banking branch, set up for easy comparison on multiple levels. 

**Stacked bar graphs** could also be helpful to show multiple types of expenditures with emphasis on their contribution to the overall total. Within the bar displaying each expenditure type, divisions would display the percentage of that expenditure made by each region. 

## Trends in Qualitative Data

Some data stories involve finding trends in a collection of measurements in data. 

**Scatter plots** would be a way to show clusters of those measures and reveal correlations between 2 (or more with grouping and multidimensional) data. 

**Box plots** could also be useful to show the distribution of those data. These types of visualizations are the backdrop to visualize the results of a regression analysis. 

## A Singular Highlight

Some measures demand special attention and can be presented as a single number in extra large or coloured font, such as a total revenue number or per cent change in deposits. These are your story headlines. A **scorecard** could be used to emphasize a sharp decline in profit. By colouring this number in red and letting it stand alone, those present at the presentation will understand its importance. **Sparklines** is a mini line chart with little or no labelling of details like axis scale that could be used to give a very high-level trend in a number. These often accompany a large scorecard value. **Bullet charts** could be used to show several dimensions of a single metric, like the target profit, year-to-date profit, and projected profit. By using these single-metric visualizations, a presenter can draw focus to one story detail at a time. 

## Summary

Thanks to Mr Stan Pugsley, a data warehouse and analytics consultant with Eide Bailly Technology Consultingbased in Salt Lake City, UT. He is also an adjunct faculty member at the University of Utah Eccles School of Business. He also made this Decision Tree in making Data Visualization.

![](https://i.ibb.co/0V4qddq/pugsley-visualization-decision-tree-kd-v12.jpg)