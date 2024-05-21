import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming you have data in the form of numpy arrays
# X should contain features: square footage, number of bedrooms, number of bathrooms
# y should contain the target variable: house prices

# Example data (replace with your actual data)
square_footage = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)  # Reshaping to make it a 2D array
bedrooms = np.array([2, 3, 3, 4, 4]).reshape(-1, 1)
bathrooms = np.array([1, 2, 2, 2.5, 3]).reshape(-1, 1)
prices = np.array([200000, 250000, 300000, 350000, 400000])

# Concatenating features into one array
X = np.concatenate((square_footage, bedrooms, bathrooms), axis=1)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, prices, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# You can now use this trained model to predict house prices for new data
# For example:
new_data = np.array([[1800, 3, 2]])  # New house with 1800 sqft, 3 bedrooms, and 2 bathrooms
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])