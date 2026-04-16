import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/raw/Bengaluru_House_Data.csv")
df = df.drop(columns=["society", "balcony"])

# ['area_type', 'availability', 'location', 'size', 'total_sqft', 'bath','price'],


# null values
df["location"] = df["location"].fillna("Sarjapur Road")
df["size"] = df["size"].fillna("2 BHK")
df["bath"] = df["bath"].fillna(df["bath"].median())
print(df.shape)


df["availability"] = df["availability"].apply(
    lambda x: "Ready To Move" if x == "Ready To Move" else "Available Soon"
)


def convertrange(x):
    temp = x.split("-")
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except:
        return None


df["total_sqft"] = df["total_sqft"].apply(convertrange)
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
df["bhk"] = pd.to_numeric(df["size"].str.split().str[0], errors="coerce")

df["location"] = df["location"].apply(lambda x: x.strip())
top_locations = df["location"].value_counts().head(5).index
df["location"] = df["location"].apply(lambda x: x if x in top_locations else "Other")

df = df[((df["total_sqft"] / df["bhk"]) >= 300)]


def remove_out_sqft(df):
    df_output = pd.DataFrame()

    for key, subdf in df.groupby("location"):
        m = subdf.price_per_sqft.mean()
        st = subdf.price_per_sqft.std()

        gen_df = subdf[
            (subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft < (m + st))
        ]

        df_output = pd.concat([df_output, gen_df], ignore_index=True)

    return df_output


df = remove_out_sqft(df)


import numpy as np


def remove_out_bhk(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby("location"):

        bhk_stats = {}

        # Step 1: Calculate mean, std, count for each bhk
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0],
            }

        # Step 2: Compare with previous bhk
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)

            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values,
                )

    return df.drop(exclude_indices, axis="index")


df = remove_out_bhk(df)
df.drop(columns=["size", "price_per_sqft"], inplace=True)

print(df.columns)
print(df.shape)

# df.to_csv('clened_data.csv')