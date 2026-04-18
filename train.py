import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# synthetic training data (same structure as your CSV)
n = 300

X = pd.DataFrame({
    "mean": np.random.normal(0, 1, n),
    "std": np.random.normal(1, 0.2, n),
    "max": np.random.normal(2, 0.5, n),
    "min": np.random.normal(-2, 0.5, n)
})

# fake labels (0 = normal, 1 = risk)
y = np.random.randint(0, 2, n)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "heart_model.pkl")

print("Model retrained successfully")