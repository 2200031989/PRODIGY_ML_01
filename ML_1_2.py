# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Collect Data
# Updated dataset with different values
data = {
    'square_footage': [1800, 2400, 3000, 2000, 2700, 1500],
    'bedrooms': [4, 3, 5, 4, 3, 2],
    'bathrooms': [3, 2, 4, 3, 2, 2],
    'price': [450000, 350000, 500000, 400000, 380000, 320000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess Data
# Check for missing values (for simplicity, we assume there are none)
print(df.isnull().sum())

# Step 3: Split Data
X = df[['square_footage', 'bedrooms', 'bathrooms']]  # Features
y = df['price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate Model
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 6: Predict Prices
# Example prediction
example_house = np.array([[2500, 3, 2]])
predicted_price = model.predict(example_house)
print(f'Predicted Price for example house: {predicted_price[0]}')

# Print model coefficients
print(f'Model Coefficients: {model.coef_}')
print(f'Model Intercept: {model.intercept_}')
