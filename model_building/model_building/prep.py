# Cleans the raw tourism dataset and splits it into train/test, then uploads the
# four splits back to the same Hugging Face Dataset repo.
# Invoked by the `data-prep` job in the GitHub Actions pipeline.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

DATASET_REPO = "creativitysupreme/tourism-data"  

# Step 1: Load the raw dataset directly from the HF Hub.
raw_path = f"hf://datasets/{DATASET_REPO}/tourism.csv"
df = pd.read_csv(raw_path)
print(f"Loaded raw dataset: {df.shape[0]:,} rows, {df.shape[1]} cols.")

# Step 2: Drop identifier-like columns with no predictive value.
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Step 3: Normalize dirty categorical values found during EDA.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Step 4: Drop content duplicates (same row repeated after ID removal).
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate rows; {len(df):,} rows remain.")

# Step 5: Define the feature lists used here and in train.py.
TARGET = "ProdTaken"

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

X = df[numeric_features + categorical_features]
y = df[TARGET]

# Step 6: Stratified train/test split (preserves the ~19% positive ratio).
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Train: {Xtrain.shape}  |  Test: {Xtest.shape}")
print(f"Train positive rate: {ytrain.mean():.4f}")
print(f"Test  positive rate: {ytest.mean():.4f}")

# Step 7: Save splits locally as CSVs (no index column).
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Step 8: Upload the four split files to the same HF dataset repo.
api = HfApi(token=os.getenv("HF_TOKEN"))
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {fname} -> {DATASET_REPO}")
