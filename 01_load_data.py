import pandas as pd

base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

train_url = base_url + "train.csv"
test_url = base_url + "test.csv"

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

print("Training examples:", len(train_data))
print("Test examples:", len(test_data))

print("\nTraining columns:", train_data.columns.tolist())
print("\nFirst training example:")
print(train_data.iloc[0])

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

