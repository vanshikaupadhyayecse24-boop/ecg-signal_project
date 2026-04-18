import joblib
import pandas as pd

# Load model
model = joblib.load("heart_model.pkl")

# Create input as dictionary (BEST METHOD)
input_data = {
    'age': 52,
    'trestbps': 125,
    'chol': 212,
    'thalach': 168,
    'oldpeak': 1.0,
    # add ALL columns you saw from X.columns
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Fill missing columns with 0
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
prediction = model.predict(input_df)

if prediction[0] == 1:
    print("⚠️ Patient may have Heart Disease")
else:
    print("✅ Patient is Normal")