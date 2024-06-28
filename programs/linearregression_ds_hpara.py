from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load the data
data = pd.read_csv('houseprices.csv')

# Step 3: Preprocess the data
# Assuming 'Price' is the target and the rest are features
# This step might need adjustments based on the actual data structure
X = data.drop('price', axis=1)
y = data['price']

# Handling missing values, encoding, etc., should be done here

# Step 4 & 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a linear regression model
model = DecisionTreeRegressor()

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Step 9: Use the model for predictions (example)
# Replace X_new with new data points for prediction
X_new = [[7000]]
predictions_new = model.predict(X_new)
print("prediced value ",predictions_new)

#mlflow.set_experiment("House Price Prediction with decision tree")


# Assuming the DecisionTreeRegressor is being used
# Define hyperparameters
max_depth = 5
min_samples_split = 2
min_samples_leaf = 1

# Initialize MLflow experiment
mlflow.set_experiment("House Price Prediction with Decision Tree")

with mlflow.start_run():
    # Create a Decision Tree model with hyperparameters
    model = DecisionTreeRegressor(max_depth=max_depth, 
                                  min_samples_split=min_samples_split, 
                                  min_samples_leaf=min_samples_leaf)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Log metrics
    mlflow.log_metric("mse", mse)

    # Log hyperparameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)

    # Example prediction
    X_new = [[7000]]  # Replace with new data points for prediction
    predictions_new = model.predict(X_new)
    print("Predicted value ", predictions_new)

    # Log the model
    mlflow.sklearn.log_model(model, "decision_tree_model")