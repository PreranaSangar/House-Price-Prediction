# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
dataset = pd.read_csv(r"D:/Full Stack Data Science & AI/Notes/September/6 September/Slr - House price prediction/House data.csv")

# Use only 'sqft_living' and 'price'
x = dataset[['sqft_living']].values  # Feature (2D array)
y = dataset['price'].values          # Target (1D array)

# Split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Create and train the model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predict on test data
pred = regressor.predict(xtest)

# Visualize training results
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Linear Regression - Training Set")
plt.xlabel("Square Foot Living")
plt.ylabel("House Price")
plt.show()

# Visualize test results
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')  # Same regression line
plt.title("Linear Regression - Test Set")
plt.xlabel("Square Foot Living")
plt.ylabel("House Price")
plt.show()

# Model parameters
m_slope = regressor.coef_[0]
c_intercept = regressor.intercept_
print(f"Slope (Coefficient): {m_slope}")
print(f"Intercept: {c_intercept}")

# Predict price for 12 and 20 sqft (for demonstration, normally values are much higher in sqft)
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted price for 12 sqft: ${y_12[0]:,.2f}")
print(f"Predicted price for 20 sqft: ${y_20[0]:,.2f}")

# Evaluate model
train_score = regressor.score(xtrain, ytrain)
test_score = regressor.score(xtest, ytest)
train_mse = mean_squared_error(ytrain, regressor.predict(xtrain))
test_mse = mean_squared_error(ytest, pred)

print(f"Training Score (R^2): {train_score:.2f}")
print(f"Testing Score (R^2): {test_score:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the model using pickle
with open('house_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
print("Model saved as 'house_model.pkl'")
