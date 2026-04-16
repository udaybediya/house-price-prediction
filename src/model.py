import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib


df = pd.read_csv("Data/processed/clened_data.csv")

# Drop unwanted column
df.drop(columns=["Unnamed: 0"], inplace=True)


X = df.drop(columns=["price"])
y = df["price"]


ohe_cols = ['area_type', 'location']
ordinal_cols = ['availability']


preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_cols),
        ('ord', OrdinalEncoder(categories=[['Available Soon', 'Ready To Move']]), ordinal_cols)
    ],
    remainder='passthrough'
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


pipelines = {
    "Linear": Pipeline([
        ("preprocessing", preprocessor),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    
    "Lasso": Pipeline([
        ("preprocessing", preprocessor),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
}

print("=== Test Set Performance ===")

best_model = None
best_score = -1

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    print(f"{name}: {score}")
    
    if score > best_score:
        best_score = score
        best_model = pipe

joblib.dump(best_model, "house_price_model.pkl")
