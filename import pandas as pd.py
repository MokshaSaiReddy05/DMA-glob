import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv(r"C:\Users\knsha\Downloads\dataset.csv")

# Data Preprocessing
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Calculate monthly totals for revenue and profit
monthly_data = df.groupby(df.index.to_period('M')).agg({
    'Revenue': 'sum',
    'Profit': 'sum'
})

# 1. Monthly Revenue Trends Visualization
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_data, x=monthly_data.index.astype(str), y='Revenue', marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()

# 2. Monthly Profit Trends Visualization
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_data, x=monthly_data.index.astype(str), y='Profit', marker='o', color='green')
plt.title('Monthly Profit Trend')
plt.xlabel('Month')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.show()

# 3. Profitability by Product
product_data = df.groupby(['Product_Name', df.index.to_period('M')]).agg({
    'Revenue': 'sum',
    'Profit': 'sum'
}).unstack(level=0)

# Plot product-wise profitability (Revenue & Profit)
product_data['Revenue'].plot(kind='bar', figsize=(12, 6), title='Revenue by Product')
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

product_data['Profit'].plot(kind='bar', figsize=(12, 6), title='Profit by Product', color='green')
plt.ylabel('Profit')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

# 4. Growth Rate (Month-on-Month)
monthly_data['Revenue_Growth'] = monthly_data['Revenue'].pct_change() * 100
monthly_data['Profit_Growth'] = monthly_data['Profit'].pct_change() * 100

# Plot Growth Rate of Revenue and Profit
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_data, x=monthly_data.index.astype(str), y='Revenue_Growth', marker='o', label='Revenue Growth')
sns.lineplot(data=monthly_data, x=monthly_data.index.astype(str), y='Profit_Growth', marker='o', color='green', label='Profit Growth')
plt.title('Monthly Growth Rate (Revenue vs Profit)')
plt.xlabel('Month')
plt.ylabel('Growth Rate (%)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 5. Time Series Forecasting with ARIMA (12-month Forecast)
model = ARIMA(monthly_data['Revenue'], order=(5, 1, 0))  # Adjust parameters if needed
model_fit = model.fit()

# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=monthly_data.index[-1].end_time + pd.Timedelta(days=1), periods=13, freq='M')[1:]

# Visualize forecast
plt.figure(figsize=(10,6))
plt.plot(monthly_data.index, monthly_data['Revenue'], label='Historical Data')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('Revenue Forecast for Next 12 Months')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()
plt.xticks(rotation=45)
plt.show()
