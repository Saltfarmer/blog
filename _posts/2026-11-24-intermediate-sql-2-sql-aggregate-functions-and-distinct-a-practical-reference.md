---
title: "Intermediate SQL 2 : SQL Aggregate Functions and DISTINCT: A Practical Reference"
lang: en
series: "Intermediate SQL"
series_order: 2
excerpt: "Moving from raw rows to real summaries! In this post, we break down COUNT, SUM, MIN, MAX, AVG, and DISTINCT using our Apple (AAPL) stock dataset—including how each handles NULL values."
takeaways:
  - "Aggregate functions summarize data vertically down columns, unlike standard arithmetic which calculates horizontally across a single row."
  - "Most aggregate functions silently ignore NULL values—except COUNT(*), which counts every row regardless of missing cells."
  - "Use SELECT DISTINCT to inspect unique combinations and COUNT(DISTINCT col) to measure cardinality, but avoid putting DISTINCT inside MIN or MAX."
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

Hello everyone! Welcome back to the **Intermediate SQL** series.

In the previous post, we set up our game plan and inspected our historical Apple (`AAPL`) stock price dataset (`tutorial.aapl_historical_stock_price`). Remember how we noticed that some values in that table were intentionally removed to simulate real-world `NULL`s? Tonight, that detail is going to matter a lot.

When you move from Basic SQL into Intermediate SQL, your job shifts from simply fetching rows to summarizing them. Instead of dumping 3,500+ daily trading rows onto your screen, you want SQL to do the heavy lifting—just like a pivot table in Excel, except way faster and repeatable.

In this post, we are going to cover the five core **SQL Aggregate Functions** (`COUNT`, `SUM`, `MIN`, `MAX`, and `AVG`) alongside the **`DISTINCT`** keyword, using our exact Apple stock table.

## Horizontal Math vs. Vertical Aggregation

Before jumping into the functions, it helps to visualize the direction SQL calculates things:

- **Horizontal (row-level) math:** Standard arithmetic operators (`+`, `-`, `*`, `/`) work across columns *within the same row*. For example, `close - open` calculates the price change for one single trading day at a time.
- **Vertical (column-level) aggregation:** Aggregate functions collapse values *down an entire column* to return a single summary value.

As a quick refresher, here are the columns we have in `tutorial.aapl_historical_stock_price`:

```
date | year | month | open | high | low | close | volume | id
```

And here is a cheat sheet of how our five aggregate functions behave before we test them one by one:

| Function | What It Does | Supported Data Types | How It Treats `NULL` Values |
| --- | --- | --- | --- |
| `COUNT(*)` | Counts total rows | All rows (table-level) | Includes rows containing `NULL` |
| `COUNT(col)` | Counts non-null entries | Numeric, text, date, boolean | Ignores `NULL` |
| `SUM(col)` | Totals all values | Numeric only | Ignores `NULL` (treats as $0$ in addition) |
| `MIN(col)` | Finds lowest / earliest value | Numeric, text, date | Ignores `NULL` |
| `MAX(col)` | Finds highest / latest value | Numeric, text, date | Ignores `NULL` |
| `AVG(col)` | Calculates arithmetic mean | Numeric only | Ignores `NULL` in both sum and count |

## `COUNT`: Measuring Row Volume and Catching `NULL`s

`COUNT` is usually the first function I run whenever I touch a new table in a data warehouse. It works on any data type and immediately tells you how big the table is and where data might be missing.

### Counting All Rows: `COUNT(*)` vs. `COUNT(1)`

To count every single row in `tutorial.aapl_historical_stock_price`, pass `*` into `COUNT`:

```sql
SELECT COUNT(*) AS total_rows
FROM tutorial.aapl_historical_stock_price;
```

You will sometimes see people write `COUNT(1)` instead of `COUNT(*)`. In modern SQL engines, both produce the exact same execution plan and return the exact same number—they count every row in the table, even if every column in that row is `NULL`. I always stick with `COUNT(*)` because it is cleaner and immediately obvious to anyone reading the query.

### Counting Specific Columns (`COUNT(col)`)

Things change when you put an actual column name inside the parentheses. Instead of counting every row in the table, `COUNT(column_name)` only counts rows where that specific column is **not `NULL`**:

```sql
SELECT COUNT(high) AS non_null_highs
FROM tutorial.aapl_historical_stock_price;
```

If `COUNT(*)` gives you `3,555` total rows, while `COUNT(high)` gives you `3,510`, you just proved that `45` trading days are missing a `high` price!

Think about how useful this is at work. Suppose you have an `email_campaigns` table with `recipient_email` and `opened_at`:

- `COUNT(*)` tells you how many emails were sent.
- `COUNT(opened_at)` tells you how many emails were actually opened, because unopened rows leave `opened_at` as `NULL`.

### Counting Non-Numerical Columns & Aliasing Best Practices

Because `COUNT` only checks whether a value exists (it doesn't do any math), you can run it on dates and text strings without any issues:

```sql
SELECT COUNT(date) AS count_of_dates
FROM tutorial.aapl_historical_stock_price;
```

Two quick habits worth building early:

1. **`COUNT(col)` does not deduplicate.** If the same date or category appears ten times, `COUNT(col)` counts it all ten times.
2. **Always alias your aggregations with `AS`.** Otherwise, your output column shows up as a generic `count` or `?column?`. Stick to lowercase `snake_case` (like `AS count_of_dates`) so you don't have to wrap column names in quotes later. If you *do* want spaces for a final report header, use double quotes (`AS "Count Of Dates"`), and save single quotes strictly for string values.

> **Pro Tip:** You can audit data completeness across an entire table in one shot by lining up `COUNT(col)` next to `COUNT(*)`. Any column with a smaller number has missing data:
>
> ```sql
> SELECT
>   COUNT(*) AS total_rows,
>   COUNT(open) AS open_cnt,
>   COUNT(high) AS high_cnt,
>   COUNT(low) AS low_cnt,
>   COUNT(close) AS close_cnt,
>   COUNT(volume) AS volume_cnt
> FROM tutorial.aapl_historical_stock_price;
> ```

## `SUM`: Totaling Numerical Metrics Vertically

`SUM` adds up every non-null value in a column. Unlike `COUNT`, `SUM` strictly requires **numerical** columns—if you try to `SUM(date)`, the database will throw a type error right away.

```sql
SELECT SUM(volume) AS total_share_volume
FROM tutorial.aapl_historical_stock_price;
```

### How `SUM` Handles `NULL`s and Arithmetic

`SUM` skips `NULL` rows without breaking, effectively treating missing entries as $0$ during addition (though if *every* row in a column is `NULL`, `SUM` returns `NULL`).

You can also combine `SUM` and `COUNT` with standard arithmetic operators to calculate metrics manually. For example, here is how you could compute the average opening price by hand:

```sql
SELECT SUM(open) / COUNT(open) AS manual_avg_open
FROM tutorial.aapl_historical_stock_price;
```

Notice that the denominator uses `COUNT(open)` instead of `COUNT(*)`. Since `open` has `NULL` rows in our Apple table, dividing by `COUNT(*)` would divide the total opening price by too many rows and artificially drag the average down.

## `MIN` and `MAX`: Finding Boundaries Across Data Types

`MIN` and `MAX` return the lowest and highest non-null values in a column. Just like `COUNT`, they work across several data types:

- **Numbers:** `MIN` gives the smallest number; `MAX` gives the largest.
- **Dates & Timestamps:** `MIN` gives the earliest date in the table; `MAX` gives the most recent date.
- **Text / Strings:** `MIN` gives the value closest alphabetically to `"A"`; `MAX` gives the value closest to `"Z"`.

```sql
SELECT
  MIN(volume) AS min_volume,
  MAX(volume) AS max_volume,
  MIN(date) AS first_trade_date,
  MAX(date) AS latest_trade_date
FROM tutorial.aapl_historical_stock_price;
```

### Nesting Row-Level Math Inside `MIN` and `MAX`

Because SQL evaluates row-level arithmetic before aggregating vertically, you can put expressions directly inside `MIN` or `MAX`. Want to know Apple's biggest single-day intraday gain in the dataset?

```sql
SELECT MAX(close - open) AS max_single_day_gain
FROM tutorial.aapl_historical_stock_price;
```

## `AVG`: Computing Means and the `NULL` Denominator Trap

`AVG` calculates the arithmetic mean of a numerical column. Like `SUM`, it only works on numbers.

```sql
SELECT AVG(high) AS avg_high_price
FROM tutorial.aapl_historical_stock_price;
```

### Watch Out for How `AVG` Treats `NULL`s

Here is a sneaky trap that trips up a lot of analysts: **`AVG` completely ignores `NULL` rows**—both in the numerator (the sum) and the denominator (the count).

That means these two queries return the exact same result:

```sql
-- Query A: Implicitly ignores NULLs
SELECT AVG(high)
FROM tutorial.aapl_historical_stock_price;

-- Query B: Explicitly filters out NULLs
SELECT AVG(high)
FROM tutorial.aapl_historical_stock_price
WHERE high IS NOT NULL;
```

Let's test this with a tiny example. Imagine a column has three values: `[10, 20, NULL]`.

- Standard `AVG` ignores the `NULL` and calculates $\frac{10 + 20}{2} = 15$.
- However, if a `NULL` in your business table actually means *zero activity* (for example, days where a store had zero sales and someone left the cell blank), ignoring it inflates your average! To treat `NULL` as $0$ so all three rows count in the denominator ($\frac{10 + 20 + 0}{3} = 10$), wrap the column in `COALESCE`:

```sql
SELECT AVG(COALESCE(high, 0)) AS avg_high_with_zeros
FROM tutorial.aapl_historical_stock_price;
```

> **Note:** Always ask yourself what a `NULL` means in your table before running `AVG`. If `NULL` means *"data wasn't recorded / unknown,"* leave `AVG(col)` alone. If `NULL` means *"zero,"* use `AVG(COALESCE(col, 0))`.

## `DISTINCT`: Inspecting Unique Values and Cardinality

When you are exploring a dataset for the first time, you often want to see what unique categories or time periods exist rather than looking at duplicate rows. `DISTINCT` handles two main jobs: deduplicating query output and counting unique items inside aggregations.

### Finding Unique Values and Unique Pairs

Putting `DISTINCT` right after `SELECT` removes duplicate values from your results:

```sql
SELECT DISTINCT month
FROM tutorial.aapl_historical_stock_price
ORDER BY month;
```

If you include multiple columns after `SELECT DISTINCT`, SQL returns every **unique combination** across all those columns. You only write `DISTINCT` once right after `SELECT`—never in front of each column:

```sql
SELECT DISTINCT year, month
FROM tutorial.aapl_historical_stock_price
ORDER BY year, month;
```

### Counting Unique Entries with `COUNT(DISTINCT ...)`

To count how many unique values exist in a column, place `DISTINCT` **inside** the `COUNT` parentheses:

```sql
SELECT COUNT(DISTINCT month) AS unique_months
FROM tutorial.aapl_historical_stock_price;
```

Checking `COUNT(DISTINCT ...)` (known as checking column **cardinality**) is a great habit before grouping data. Once you confirm `unique_months` returns `12`, you know a monthly breakdown will give you clean, predictable buckets.

### Two Quick Rules for `DISTINCT`

1. **Never put `DISTINCT` inside `MIN` or `MAX`.** Writing `MAX(DISTINCT volume)` returns the exact same answer as `MAX(volume)` because removing duplicates doesn't change the highest or lowest number in a list. It just forces the database to do extra work for nothing.
2. **Be careful with `DISTINCT` on massive tables.** Under the hood, `SELECT DISTINCT` and `COUNT(DISTINCT col)` require the database to sort or hash values in memory to weed out duplicates. On high-cardinality columns with tens of millions of rows (like `user_id` or `transaction_id`), `COUNT(DISTINCT ...)` can noticeably slow down your query.

## In Summary

We just covered the core toolkit for summarizing data in SQL! Before wrapping up, here is how you can combine all of these functions into a single profiling query whenever you inspect a new table:

```sql
SELECT
  COUNT(*) AS total_records,
  COUNT(close) AS non_null_closes,
  COUNT(DISTINCT year) AS distinct_years,
  MIN(date) AS earliest_date,
  MAX(date) AS latest_date,
  SUM(volume) AS total_volume,
  AVG(close) AS avg_close_price
FROM tutorial.aapl_historical_stock_price;
```

Right now, these functions collapse the entire Apple stock table into a single summary row. In the next post, we will pair these exact aggregate functions with **`GROUP BY`** and **`HAVING`** so we can slice Apple's performance by year and month! See you in the next post!

> "Without data, you're just another person with an opinion."
>
> – W. Edwards Deming
