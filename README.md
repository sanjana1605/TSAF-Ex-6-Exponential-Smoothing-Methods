# TSAF-Ex-6-Exponential-Smoothing-Methods

## AIM:
To implement and compare three exponential smoothing techniques for time-series
forecasting:

1. Simple Exponential Smoothing

2. Holt's Linear Trend Method

3. Holt-Winters Seasonal Method


## PROGRAM:

```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# 1. Simple Exponential Smoothing (Stable traffic)
views_stable = [1200, 1250, 1220, 1240, 1210, 1260, 1230, 1255, 1215, 1245]
data1 = pd.Series(views_stable)
model1 = SimpleExpSmoothing(data1)
fit1 = model1.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)
print("1. Website Traffic Forecast (SES):")
print(forecast1)

# 2. Holt's Linear Method (Steadily growing traffic)
views_trend = [1200, 1310, 1420, 1550, 1680, 1800, 1920, 2050, 2180, 2300]
data2 = pd.Series(views_trend)
model2 = Holt(data2)
fit2 = model2.fit()
forecast2 = fit2.forecast(3)
print("\n2. Growing Website Traffic Forecast (Holt):")
print(forecast2)

# 3. Holt-Winters Method (Weekly cycle = 7 days)
views_seasonal = [
 1500, 1600, 1580, 1620, 1400, 900, 850, # Week 1
 1550, 1650, 1620, 1670, 1450, 930, 880, # Week 2
 1600, 1710, 1680, 1720, 1500, 960, 910 # Week 3
]
data3 = pd.Series(views_seasonal)
model3 = ExponentialSmoothing(data3, trend="add", seasonal="add", seasonal_periods=7)
fit3 = model3.fit()
forecast3 = fit3.forecast(7)
print("\n3. Weekly Website Traffic Forecast (Holt-Winters):")
print(forecast3)

```

## OUTPUT:

<img width="542" height="360" alt="image" src="https://github.com/user-attachments/assets/3386c174-5b2b-4ab3-bf18-654051dbdf61" />
