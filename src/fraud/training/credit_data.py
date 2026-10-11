import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

print(df.shape)
print(df.head())

print("missing:", df.isna().sum().sum())
print("duplicates:", df.duplicated().sum())

# frauds
print(df["Class"].value_counts())
print("fraud rate:", df["Class"].mean())

print("time range:", df["Time"].min(), "-", df["Time"].max())
print("hours:", df["Time"].max() / 3600)
print("sorted by time?", df["Time"].is_monotonic_increasing)

df["hour"] = df["Time"] // 3600
print(df.groupby("hour")["Class"].agg(["count", "sum"]))

# amounts for normal vs fraud
print(df.groupby("Class")["Amount"].describe())
