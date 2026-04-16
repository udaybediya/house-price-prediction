import pandas as pd

df = pd.read_csv("Data/processed/clened_data.csv")

df = df.drop(columns=["Unnamed: 0"])

for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())
