# Uploads the raw tourism dataset to a Hugging Face Dataset repository.
# Invoked by the `register-dataset` job in the GitHub Actions pipeline.

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "creativitysupreme/tourism-data"
repo_type = "dataset"

# HF_TOKEN is injected via GitHub Actions secrets (or set locally for testing).
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo on first run; reuse if it already exists.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the local data folder (contains tourism.csv) to the HF dataset repo.
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded 'tourism_project/data/' to '{repo_id}'.")
