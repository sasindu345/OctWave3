# First-time setup

## 1. Push this repo to GitHub

```bash
git add .
git commit -m "initial project structure"
gh repo create OctWave3 --private --source=. --push
# or: git remote add origin https://github.com/sasindu345/OctWave3.git && git push -u origin main
```

## 2. Open the notebook in Colab

Any of these work:

- **URL shortcut (fastest):**
  `https://colab.research.google.com/github/sasindu345/OctWave3/blob/main/notebooks/01_train_colab.ipynb`
- **From Colab:** `File → Open notebook → GitHub tab → paste repo URL`.
  For a private repo, click *Include private repos* and authorize Colab once.
- **Badge:** add this to the README for one-click access:
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sasindu345/OctWave3/blob/main/notebooks/01_train_colab.ipynb)
  ```

## 3. Turn on the GPU

`Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save`.
Verify with cell 1 (`nvidia-smi`). Do this **every new session** — it is not remembered.

## 4. Kaggle API token

Kaggle → your avatar → *Settings* → *API* → **Create New API Token** → downloads `kaggle.json`.
Upload it when cell 5 prompts you.

Better, so you only do it once: open Colab's **🔑 Secrets** panel, add a secret named
`KAGGLE_JSON` with the file contents, then replace the upload block with:

```python
from google.colab import userdata
import os, json
os.makedirs('/root/.kaggle', exist_ok=True)
open('/root/.kaggle/kaggle.json', 'w').write(userdata.get('KAGGLE_JSON'))
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

## 5. Private repo access from Colab

GitHub → Settings → Developer settings → **Fine-grained tokens** → repo-scoped, Contents: Read.
Store it as a Colab Secret named `GH_TOKEN` and clone with:

```python
from google.colab import userdata
TOKEN = userdata.get('GH_TOKEN')
REPO_URL = f'https://{TOKEN}@github.com/sasindu345/OctWave3.git'
```

Never paste a token into a notebook cell — it gets committed.

## Colab limits worth knowing

| | Free tier |
|---|---|
| Session length | ~12h max, often less; idle tabs get killed sooner |
| GPU | T4 16GB (when available) |
| Disk | ~78GB at `/content`, wiped every session |
| Drive | persistent, but slow for many small file reads |

Consequences baked into this repo: checkpoints go to Drive and training auto-resumes;
the dataset stays on fast local disk and is re-downloaded each session.
