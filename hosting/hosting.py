# Uploads the deployment artifacts (Dockerfile, app.py, requirements.txt) to the
# Hugging Face Space that serves the Streamlit front-end.
# Invoked by the `deploy-hosting` job in the GitHub Actions pipeline.

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

SPACE_REPO = "tourism-package-predictor"   # e.g. "<HF_USERNAME>/tourism-app"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Space on first run as a Docker space; reuse it if it already exists.
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating new Docker Space...")
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        private=False,
    )
    print(f"Space '{SPACE_REPO}' created.")

# Upload the entire deployment folder to the Space root so HF Spaces' Docker
# builder finds the Dockerfile and rebuilds the container automatically.
api.upload_folder(
    folder_path="deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Uploaded 'tourism_project/deployment/' -> Space '{SPACE_REPO}'.")
