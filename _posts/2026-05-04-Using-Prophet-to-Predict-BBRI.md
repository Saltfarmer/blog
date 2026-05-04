---
title: "Predicting BBRI Stocks with Meta’s Prophet"  
header :
  teaser : https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRM_Fs2pjAqnLAHBnRSiVUXXAvgURKPytD3Yw&s
comments : true  
share : true  
categories:
    - Forecasting
---
The IHSG so shit recently not gonna lie/ Have you ever wondered how businesses predict future sales, how weather apps forecast the temperature, or how data scientists try to anticipate stock market trends? The magic behind these predictions often lies in Time Series Forecasting.

Tonight in my free time, I am going to try to dive into forecasting using Prophet, a powerful and open-source library developed by Meta (formerly Facebook). Designed to be intuitive and robust to missing data, Prophet is arguably the best entry point for anyone looking to learn forecasting.

To make this hands-on, we will build a model to analyze the historical stock prices of Bank Rakyat Indonesia (BBRI), one of the largest and most widely traded banks on the Indonesian Stock Exchange (IDX).

> **Disclaimer before we start**: This tutorial is strictly for educational purposes. Predicting the stock market is notoriously difficult because prices are influenced by unpredictable real-world events, news, and human psychology. Do not use this model to make actual financial or investment decisions!

# What is Meta’s Prophet?

Prophet is an additive regression model. In simple terms, it breaks down a time series into three main components:

- Trend: The overall non-periodic growth or decline.

- Seasonality: Periodic changes (e.g., weekly, yearly).

- Holidays: Irregular events that happen on specific dates.

Prophet is fantastic because it works right out of the box with default parameters, but also allows advanced users to tweak the model to their specific domain knowledge.
Prerequisites

Before we write the code, ensure you have the necessary Python libraries installed. You can install them via your terminal or Jupyter Notebook:

```bash
pip install prophet yfinance pandas matplotlib
```

- `prophet`: Our forecasting engine.

- `yfinance`: A handy library to fetch historical stock data from Yahoo Finance.

- `pandas`: For data manipulation.

- `matplotlib`: For visualizing our results.

# Step 1: Fetching the BBRI Data

First, we need historical data. We will use `yfinance` to download the daily closing prices of BBRI. The ticker symbol for BBRI on the Jakarta Stock Exchange is `BBRI.JK`.

```python
import yfinance as yf
import pandas as pd

print("Downloading BBRI.JK data...")
bbri_data = yf.download('BBRI.JK', start='2024-08-01', end='2026-05-01')

# Let's look at the first few rows
print(bbri_data.head())
```

```
import yfinance as yf
import pandas as pd

print("Downloading BBRI.JK data...")
bbri_data = yf.download('BBRI.JK', start='2024-08-01', end='2026-05-01')

# Let's look at the first few rows
print(bbri_data.head())
```

# Step 2: Preparing Data for Prophet

Prophet is famously strict about its input data format. It requires a Pandas DataFrame with exactly two columns:

- `ds`: The datestamp (must be YYYY-MM-DD or a datetime format).

- `y`: The numeric value we want to predict (in our case, the 'Close' price).

Let's clean up our yfinance data to meet Prophet's requirements.

```python
# Reset the index to bring the 'Date' out of the index and into a column
df = bbri_data.reset_index()

# Keep only the Date and Close price columns
df = df[['Date', 'Close']]

# Rename the columns to 'ds' and 'y' as required by Prophet
df.columns = ['ds', 'y']

# Remove timezone information from dates (Prophet doesn't like timezones)
df['ds'] = df['ds'].dt.tz_localize(None)

# Drop any rows with missing values (NaNs)
df = df.dropna()

print("Data prepared for Prophet:")
print(df.head())
```

```
Data prepared for Prophet:
          ds            y
0 2024-08-01  3933.282227
1 2024-08-02  3900.159668
2 2024-08-05  3751.109131
3 2024-08-06  3809.073242
4 2024-08-07  3825.634766
```

# Step 3: Training the Prophet Model

Now comes the exciting part—training the machine learning model. With Prophet, this is incredibly straightforward.

Because the stock market is only open on weekdays, we don't have true "daily" seasonality that includes weekends. We will let Prophet figure out the weekly and yearly trends.

```python
from prophet import Prophet

# Initialize the Prophet model
# We turn off daily seasonality because stock markets close at night
model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)

# Fit the model to our historical BBRI data
model.fit(df)
print("Model training complete!")
```

```
from .autonotebook import tqdm as notebook_tqdm
20:23:48 - cmdstanpy - INFO - Chain [1] start processing
20:23:48 - cmdstanpy - INFO - Chain [1] done processing
Model training complete!
```

# Step 4: Peering into the Future

To make a forecast, we first need to create a dataframe that extends into the future. Prophet has a built-in helper function for this. Let's ask our model to predict the next 365 days.

```python
# Create a dataframe holding dates for the next 365 days
future_dates = model.make_future_dataframe(periods=365)

# Generate the forecast
forecast = model.predict(future_dates)

# The forecast dataframe contains a lot of info. Let's look at the most important ones:
# 'ds' (date), 'yhat' (prediction), 'yhat_lower' & 'yhat_upper' (uncertainty intervals)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

```
ds         yhat  yhat_lower   yhat_upper
776 2027-04-26  2086.632914  843.793064  3287.292599
777 2027-04-27  2096.495340  827.606054  3331.130274
778 2027-04-28  2109.974118  871.862353  3356.915706
779 2027-04-29  2115.109520  906.432581  3354.548906
780 2027-04-30  2117.840023  877.550252  3340.736764
```

Seems like the BBRI Stocks is sad :( 

# Step 5: Visualizing the Forecast

Numbers on a screen are great, but visuals are where the story comes alive. Prophet provides built-in plotting tools to see exactly what the model is thinking.

## Plotting the Overall Prediction

```python
import matplotlib.pyplot as plt

# Plot the forecast
fig1 = model.plot(forecast)
plt.title("BBRI.JK Stock Price Forecast (1 Year)")
plt.xlabel("Date")
plt.ylabel("Stock Price (IDR)")
plt.show()
```

![bbri next year](https://i.ibb.co.com/qLy2R5Fd/bbri-next-year.png)

In the output chart, the black dots are the actual historical BBRI prices, the dark blue line is Prophet's prediction (`yhat`), and the light blue shaded area represents the confidence interval (the margin of error).
Uncovering the Hidden Patterns (Components)

One of Prophet's best features is its ability to break down the forecast into components. This allows us to see the overarching trend and seasonal behaviors of BBRI stocks.

```python
# Plot the forecast components
fig2 = model.plot_components(forecast)
plt.show()
```

![](https://i.ibb.co.com/v6r8SZPf/bbri-analysis.png)

The bottom plot in the provided Prophet forecast—the yearly seasonality component—reveals a highly distinct, cyclical pattern that aligns perfectly with the traditional rhythm of the Indonesian stock market.

If this model is analyzing an Indonesian blue-chip stock (like a major banking entity), the peaks and troughs tell a clear story of corporate actions and investor behavior throughout the year.

Here is a breakdown of the patterns shown in the yearly component:

## 1. The May–June Surge (The Dividend Rally)

The most striking feature of the entire chart is the massive, steep spike that begins climbing in late April and peaks violently in early June. Academic research on the Indonesian stock market has actually noted that May and June historically exhibit statistically significant positive returns.  

This is almost entirely driven by the Indonesian Dividend Season. Major companies typically hold their Annual General Meetings (RUPST) in March or April. Following these meetings, the "Cum-Dividend" dates (the final day to buy a stock and still be entitled to the payout) are usually scheduled between late April and early June. Investors aggressively buy into these stocks during this window to capture what are often very high dividend yields.

## 2. The Late-June Cliff (The Ex-Dividend Drop)

Immediately following the June peak, the trend plummets rapidly back below zero. This represents the Ex-Dividend effect.

On the ex-dividend date, the stock price technically drops by an amount roughly equal to the dividend paid out. Furthermore, many short-term traders utilize a "dividend-timing strategy" where they buy shares a few days before the cum-date and sell them immediately at the opening bell on the ex-dividend date. This mass profit-taking creates the sharp downward cliff you see extending into July.  

## 3. The February & September Mini-Peaks (Earnings Seasons)

You can spot two smaller, secondary peaks forming around mid-February and again in September/October. These correlate directly with earnings reporting seasons:

> The February Peak: Investors are anticipating the release of audited Full Year/Q4 financial statements from the previous year. Strong banking sector results usually create a mini-rally here.

> The September/October Peak: This upward slope aligns with the anticipation and release of Q3 financial results. For some companies, this is also when smaller, interim dividends are announced.

## 4. The November/December Trough (Year-End Rebalancing)

The chart drops to its lowest annual point in late November through early December. While the Indonesian market is famous for late-December "Window Dressing" (where fund managers buy up stocks in the final weeks of the year to make their portfolios look better), the period immediately preceding it often faces heavy selling pressure. This deep trough reflects foreign portfolio rebalancing, tax-loss harvesting, and general profit-taking before the final holiday weeks.

In Summary: This Prophet model has successfully learned that the most critical driver of this asset's yearly price action isn't necessarily the end-of-year holidays, but rather the massive influx of capital surrounding the mid-year dividend distribution cycle.
