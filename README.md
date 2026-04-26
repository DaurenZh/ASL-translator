# ASL Translator

Standalone hand-sign translator prototype.

## Structure

```text
backend/    FastAPI inference API
frontend/   React webcam UI
models/     Trained model artifacts
notebooks/  Original notebook and experiments
scripts/    Training and utility scripts
```

## Train The Model

Your dataset should use the `ImageFolder` layout:

```text
asl_alphabet_train/
  A/
  B/
  C/
  ...
  space/
  del/
  nothing/
```

Run training from the project root:

```bash
cd /Users/j-19group/asl-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.train.txt
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 12
```

The script automatically uses `asl_alphabet_train/` if your dataset path points at its parent folder.

It saves:

```text
models/asl_model.pt
models/asl_model.metrics.json
```

The checkpoint format matches the backend:

```python
{
    "model_state_dict": ...,
    "classes": [...],
    "img_size": 96,
}
```

## Run Backend

```bash
cd /Users/j-19group/asl-translator/backend
source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check the API:

```bash
curl http://localhost:8000/health
```

## Run Frontend

```bash
cd /Users/j-19group/asl-translator/frontend
npm install
npm run dev
```

The frontend expects the backend at:

```text
http://localhost:8000
```

## Deploy

For a simple demo deploy:

1. Train locally and commit or upload `models/asl_model.pt` to the server.
2. Build the backend Docker image with `Dockerfile.backend`.
3. Host the frontend with Vite build output or a static host.
4. Set the frontend API URL to your backend domain before production.

