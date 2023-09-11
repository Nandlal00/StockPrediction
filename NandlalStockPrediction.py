#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[10]:


# Load historical stock price data (replace 'stock_data.csv' with your dataset)
data = pd.read_csv('C:/Users/hp/Downloads/stock prediction/Tata-steel.csv')


# In[11]:


# Explore and preprocess the data
# For simplicity, let's assume 'Close' prices are our target variable, and we'll use 'Open' as a feature.
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)


# In[16]:


# Feature engineering (you can add more features here)
data['Next_Close'] = data['Close Price'].shift(-1)  # Predicting the next day's closing price
data.dropna(inplace=True)


# In[20]:


# Split the data into training and testing sets
X = data[['Open Price']]  # Feature(s)
y = data['Next_Close']  # Target variable


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[27]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[30]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')


# In[48]:


# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Open Price')
plt.ylabel('Next Close Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()

