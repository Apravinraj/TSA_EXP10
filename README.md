# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/BTC-USD(1).csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the Closing Price to inspect for trends
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Bitcoin Closing Price Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of Closing Price
check_stationarity(data['Close'])

# Plot ACF and PACF for Close prices
plot_acf(data['Close'])
plt.show()
plot_pacf(data['Close'])
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Define and fit the SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE for evaluation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('SARIMA Model Predictions for Bitcoin Closing Price')
plt.legend()
plt.show()

```


### OUTPUT:
![download](https://github.com/user-attachments/assets/0e12db29-1233-4bc4-8160-a05336d6d0f6)
![download](https://github.com/user-attachments/assets/44fcff30-0aa3-4c95-ad1c-935f6bedab12)
![download](https://github.com/user-attachments/assets/20dc52cd-70f3-4885-a3d7-21340b8e3ab2)
![download](https://github.com/user-attachments/assets/fffe1f1a-3397-46f8-817e-8ad2c0fae3b5)


### RESULT:
Thus the program run successfully based on the SARIMA model.
