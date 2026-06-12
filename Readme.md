# ANN Churn Classification

A Streamlit app that predicts customer churn using a pre-trained artificial neural network.

## Summary
This repository contains a small demo app that loads a trained Keras model and preprocessing artifacts to estimate the probability a customer will churn.

## Quickstart

1. Create & activate a Python environment (recommended):
	```bash
	python -m venv .venv
	# Windows
	.venv\Scripts\activate
	# macOS / Linux
	source .venv/bin/activate
	```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

3. Run the Streamlit app:
	```bash
	streamlit run app.py
	```

## Files
- `app.py` — Streamlit application (UI + prediction logic)
- `model.h5` — Trained Keras model (binary; recommended to store with Git LFS)
- `ohe_geo.pkl`, `label_encoder_gender.pkl`, `scaler.pkl` — Preprocessing artifacts (pickle files)
- `requirements.txt` — Python dependencies
- `README.md` / `Readme.md` — This file

## Git LFS (recommended for large model files)
If `model.h5` is large, use Git LFS to avoid committing big binaries directly to Git.

Install and enable Git LFS, then track the model file:
```bash
# Install git-lfs (one-time)
# Linux/macOS (example)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Or on macOS
brew install git-lfs

# Or Windows: download from https://git-lfs.github.com/ and run installer
git lfs install

# Track the model file
git lfs track "model.h5"

# Add the .gitattributes entry created by git-lfs
git add .gitattributes
```

## Suggested .gitignore
You may want to ignore virtual envs and local caches. Example `.gitignore`:
```text
__pycache__/
.venv/
env/
.env
*.pyc
.ipynb_checkpoints/
.streamlit/
logs/
.vscode/
.DS_Store
```

## Example Git workflow (use your remote URL)
Replace `<YOUR_REMOTE_URL>` below with:
`https://github.com/AadeeshRS/ann-churn-classification.git`

```bash
# Initialize repo (if needed)
git init
git branch -M main

# Install and configure LFS, track the model (if you haven't yet)
git lfs install
git lfs track "model.h5"
git add .gitattributes

# Add files and commit
git add .
git commit -m "Initial commit: Streamlit churn classifier"

# Add remote and push
git remote add origin https://github.com/AadeeshRS/ann-churn-classification.git
git push -u origin main
```

If `model.h5` is already large and present, ensure it is added after `git lfs track` so it will be stored in LFS.

## Notes
- Ensure `model.h5` and the pickle preprocessors are present in the repository root (or adjust `app.py` paths).
- `runtime.txt` pins Streamlit Cloud to Python 3.11 because TensorFlow does not provide wheels for Python 3.14.
- To silence TensorFlow oneDNN messages in some environments, set:
  - Linux/macOS: `export TF_ENABLE_ONEDNN_OPTS=0`
  - Windows (PowerShell): `$env:TF_ENABLE_ONEDNN_OPTS = "0"`

## License
Choose and add your preferred license (e.g., MIT).