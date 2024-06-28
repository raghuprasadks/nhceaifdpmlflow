import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Load the data
data = pd.read_csv('houseprices.csv')

# Step 3: Preprocess the data
# Assuming 'Price' is the target and the rest are features
# This step might need adjustments based on the actual data structure
X = data.drop('Price', axis=1)
y = data['Price']

# Handling missing values, encoding, etc., should be done here

# Step 4 & 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a linear regression model
model = LinearRegression()

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Step 9: Use the model for predictions (example)
# Replace X_new with new data points for prediction
# X_new = ...
# predictions_new = model.predict(X_new)