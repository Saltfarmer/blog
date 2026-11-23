---
title: "Intermediate SQL 1 : Getting Started & The Dataset"
lang: en
series: "Intermediate SQL"
series_order: 1
excerpt: "Kicking off the Intermediate SQL series! Moving past basic SELECT queries, multiple ways to solve problems, and exploring the Apple stock dataset."
takeaways:
  - "In intermediate SQL, there is rarely just one right query—focus on accuracy, readability, and performance."
  - "We will use historical Apple (AAPL) stock price data to practice aggregations, filtering, and joins."
  - "Real-world datasets always have missing values (NULLs), so inspecting your table first is mandatory."
header:
  teaser: /assets/images/intermediate-sql-banner.jpg
comments: true
share: true
categories:
  - Data Engineering
tags:
  - sql
  - intermediate sql
  - data analysis
  - database
---

Hello everyone! Tonight in my free time, I am going to start a brand-new series on **Intermediate SQL**.

If you have been following my blog, you know I use SQL almost every single day at work. While Python and pandas are awesome for heavy data wrangling and machine learning, SQL is still the undisputed king when you need to pull, filter, and aggregate millions of rows directly from a data warehouse before your laptop fans start sounding like a jet engine.

I am writing this series partly to sharpen my own fundamentals and document patterns so I can remember them properly[cite: 4], and partly as a reference for anyone trying to level up from basic `SELECT * FROM table` queries.

## Why Intermediate SQL Feels Different

When you first learn Basic SQL (`SELECT`, `WHERE`, `ORDER BY`, `LIMIT`), most practice problems feel pretty rigid. Usually, there are only one or two obvious ways to write the query and get the right answer.

Once we step into **Intermediate SQL**, things get way more interesting—and honestly, a bit trickier. As the problems get harder, there are suddenly *many* different ways to produce the exact same result.

> **Note:** Keep in mind that any query solutions I share in this series should be treated as a reference, not the *only* way to answer a question. Two queries can return identical tables while being written completely differently.

At this stage, writing SQL isn't just about getting the right numbers; it's about writing queries that are readable for your teammates (and your future self six months from now) and don't choke the database.

## What We Will Cover in This Series

To move past the basics, we are going to break down the core building blocks that make up 80% of day-to-day analytical SQL work:

1. **Aggregate Functions:** Summarizing data using `COUNT`, `SUM`, `MIN`, `MAX`, and `AVG` alongside `GROUP BY`.
2. **Filtering Aggregations & Logic:** Using `HAVING` to filter grouped results, `DISTINCT` for unique values, and `CASE` statements for custom `IF/THEN` logic inside queries.
3. **SQL Joins & Unions:** Combining multiple tables together using `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, self-joins, and stacking datasets with `UNION` / `UNION ALL`.

## The Dataset: Apple Historical Stock Prices

![](https://media.appleinsider.com/gallery/69028-145625-Stock-high-xl.jpg)

For the first several lessons—especially while we master aggregations—we will be working with a historical **Apple (`AAPL`) stock price** dataset (originally pulled from Google Finance).

I actually really enjoy working with stock price data considering my last bonus that i get is from Stock. Financial time-series tables are super intuitive because everyone understands open, high, low, close, and volume.

In our table (`tutorial.aapl_historical_stock_price`), each row represents a single trading day:

| Column | Description |
| --- | --- |
| `date` | The specific trading date |
| `year` / `month` | Extracted date parts for easier grouping |
| `open` | Stock price when the market opened that day |
| `high` | Highest price reached during that day |
| `low` | Lowest price reached during that day |
| `close` | Stock price when the market closed that day |
| `volume` | Total number of shares traded that day |
| `id` | Unique row identifier |

Let's run a quick sanity check query to inspect the table structure and see what the first few rows look like:

```sql
SELECT *
FROM tutorial.aapl_historical_stock_price
LIMIT 5;
```

This pulls the first 5 rows from the table so we can verify the column names and formatting without dumping thousands of rows onto our screen.

One important thing to watch out for: **some data has been intentionally removed** from this dataset. Just like real-world banking or production tables, you are going to run into `NULL` values—which will become super important once we start talking about how functions like `COUNT` and `AVG` behave when rows are missing data.

## In Summary

That's our starting line! Basic SQL teaches you how to fetch raw rows, while Intermediate SQL teaches you how to summarize, reshape, and combine them to actually answer business questions.

In the next post, we will dive straight into **SQL Aggregate Functions (`COUNT`, `SUM`, `MIN`, `MAX`, and `AVG`)** using this exact Apple stock table. See you in the next post!

> "Success is the sum of small efforts, repeated day in and day out."
>
> – Robert Collier
