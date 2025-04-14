import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv("train.csv")

# Drop columns with too many missing values
missing_threshold = 0.4  # Drop columns with more than 40% missing
data = data.loc[:, data.isnull().mean() < missing_threshold]

# Fill numerical NaNs with median
num_cols = data.select_dtypes(include=[np.number]).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Fill categorical NaNs with mode
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data)

# Split into features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save column names for later use
joblib.dump(X.columns, "model_columns.pkl")
joblib.dump(model, "property_model.pkl")


def predict_price(user_input_dict):
    # Load model and columns
    model = joblib.load("property_model.pkl")
    model_columns = joblib.load("model_columns.pkl")

    # Create dataframe from user input
    input_df = pd.DataFrame([user_input_dict])

    # Match columns using one-hot encoding
    input_df = pd.get_dummies(input_df)

    # Add any missing columns that were in training data
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]  # Reorder to match model input
    return model.predict(input_df)[0]


# === Main Interface ===
def main():
    print("\nWelcome to the Property Price Estimator!")
    print("Please enter the house specifications:")

    sqft = int(input("Enter the square footage of the house (GrLivArea): "))
    bedrooms = int(input("Enter the number of bedrooms (BedroomAbvGr): "))
    bathrooms = int(input("Enter the number of full bathrooms (FullBath): "))
    year_built = int(input("Enter the year the house was built (YearBuilt): "))
    garage_cars = int(input("Enter the garage capacity in cars (GarageCars): "))
    overall_qual = int(input("Enter the overall quality (1-10) (OverallQual): "))
    neighborhood = input("Enter the neighborhood (e.g., 'NAmes', 'CollgCr'): ")
    house_style = input("Enter the house style (e.g., '1Story', '2Story'): ")

    # Create user input dictionary
    user_input = {
        "GrLivArea": sqft,
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "OverallQual": overall_qual,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style
    }

    predicted_price = predict_price(user_input)
    print(f"\n💰 Estimated Property Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()
