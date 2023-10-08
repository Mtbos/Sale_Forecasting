
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Load your data from the CSV file
data = 'give ur csv file path here'

# Create a DataFrame from the data
df = pd.read_csv(data, parse_dates=['Date'], index_col='Date')

# Sort the data by date if it's not already sorted
df = df.sort_index()

# Visualize the original data using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')


# Using the Dickey-Fuller Test to check for stationarity

def test():
    df_diff = df['Close']
    differencing_order = 0

    while True:
        df_diff = df_diff.diff().dropna()
        differencing_order += 1

        r = adfuller(df_diff)

        print(f"ADF Statistics after {differencing_order} differencing:", r[0])
        print(f'p-value after {differencing_order} differencing:', r[1])

        if r[1] < 0.05:
            print(f"{differencing_order} differencing step(s) made the data stationary.")
            break  # Exit the loop if stationary

    # Define ARIMA model parameters
    p, d, q = 1, 2, 5

    # Create and fit the ARIMA model
    model = ARIMA(df_diff, order=(p, d, q))
    result = model.fit()

    # Make forecasts (for example, forecasting the next 5 days)
    forecast_periods = 5
    forecasts = result.forecast(steps=forecast_periods)

    # Generate a date range for forecasts
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_periods + 1, freq='D')

    # Visualize forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(df_diff, label='Differenced Data')
    plt.plot(forecast_dates, [df_diff.iloc[-1]] + list(forecasts), label='Forecasts', linestyle='--', marker='o')
    plt.title('ARIMA Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()


# Call the test function to perform differencing and ARIMA modeling
test()
