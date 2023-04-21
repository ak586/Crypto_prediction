import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import plotly.graph_objs as go
from datetime import datetime


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from the CSV file
data = pd.read_csv('btc_data.csv')

# Extract the "Close" column as the target variable
y = data['Close']

# Extract the "Open", "High", "Low", and "Volume" columns as the features
X = data[['Open', 'High', 'Low', 'Volume']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data by scaling the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest regressor on the training data
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)


# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)
accuracy = rf.score(X_test, y_test) * 100
print('Accuracy:', accuracy, '%')

# Create a DataFrame with the predicted and actual values
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test}).reset_index(drop=True)

# Print the DataFrame
print(results)

symbol = 'BTC-USD'
start_date = '2023-04-21'
end_date = '2023-04-22'
data = yf.download(tickers=symbol, start=start_date, end=end_date, interval='15m')
print(data)

# Preprocess the data using the same scaler object as before
preprocessed_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume']])
print(preprocessed_data)

# Make predictions using the preprocessed data
predictions = rf.predict(preprocessed_data)

# Print the predictions
print('Predicted Close Price for', end_date, ':', predictions[0])

