# 1. Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# 2. Load datasets
train_df = pd.read_csv("/content/sample_data/california_housing_train.csv")
test_df = pd.read_csv("/content/sample_data/california_housing_test.csv")

# 3. Separate features and target
X_train = train_df.drop("median_house_value", axis=1)
y_train = train_df["median_house_value"]

X_test = test_df.drop("median_house_value", axis=1)
y_test = test_df["median_house_value"]

# 4. Create a pipeline: scaling + linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Standardize features
    ('regressor', LinearRegression())   # Linear Regression model
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Make predictions
y_pred = pipeline.predict(X_test)

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 8. Save the pipeline (scaling + model)
joblib.dump(pipeline, "/content/sample_data/house_price_pipeline.pkl")
print("Pipeline saved as house_price_pipeline.pkl")

# 9. Load the pipeline later (optional)
loaded_pipeline = joblib.load("/content/sample_data/house_price_pipeline.pkl")

# 10. Predict using the loaded pipeline
predictions = loaded_pipeline.predict(X_test)
print(f"Predictions for first 5 test samples: {predictions[:5]}")
