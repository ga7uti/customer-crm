import pandas as pd

base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

df = pd.read_csv(base_url + "train.csv")

print("Dataset shape:", df.shape)

print("Missing Values: ")
print(df.isnull().sum())


print("\n Number of unique values: ")
print(df["category"].nunique())

print("\n Examples from unique categories: ")
print(df.groupby("category")["text"]
.first()
.head(10))


print("\n Most common categories: ")
print(df["category"].value_counts().head(10))