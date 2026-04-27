# ASL Translator

Standalone hand-sign translator prototype for ASL alphabet classification.

## Start Here After Clone

Use the full setup guide:

```text
docs/getting_started_after_clone.md
```

Short version:

```bash
cd asl-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.train.txt
pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

## Structure

```text
backend/    FastAPI inference API
frontend/   React webcam and upload UI
models/     trained model artifacts and metrics
notebooks/  original and improved notebooks
scripts/    training and comparison scripts
docs/       checkpoint report, presentation, and setup materials
```

## Notebooks

Checkpoint 1 baseline notebook:

```text
notebooks/checkpoint1.ipynb
```

Improved Checkpoint 1 notebook for Checkpoint 2:

```text
notebooks/checkpoint1_improved_for_checkpoint2.ipynb
```

Checkpoint 2 experiment notebook:

```text
notebooks/checkpoint2_experiments.ipynb
```

Use `checkpoint1.ipynb` as evidence of the first baseline. Use the improved notebook and Checkpoint 2 notebook for the final explanation, model comparison, and presentation.

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
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --architecture simple_cnn --epochs 8
```

The script automatically uses `asl_alphabet_train/` if your dataset path points at its parent folder.

It saves:

```text
models/asl_model.pt
models/asl_model.metrics.json
```

## Compare Models

Checkpoint #2 requires comparison with at least two other models:

```bash
python scripts/compare_models.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 5 --pretrained
```

This trains and compares:

- `simple_cnn`
- `resnet18`
- `mobilenet_v3_small`

If pretrained weights cannot be downloaded, run without `--pretrained`.

It saves:

```text
models/model_comparison.json
```

After choosing the best model, copy it to the backend default path:

```bash
cp models/resnet18.pt models/asl_model.pt
```

Replace `resnet18.pt` with the best model from `model_comparison.json`.

## Run Backend

```bash
cd backend
source ../.venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check the API:

```bash
curl http://localhost:8000/health
```

You want:

```json
"model_ready": true
```

## Run Frontend

```bash
cd frontend
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

The UI supports:

- webcam prediction
- image upload prediction
- confidence display
- top predictions
- sentence builder
- clear, copy, and speech output

## Checkpoint #2 Materials

Use these files:

```text
docs/getting_started_after_clone.md
docs/task_division.md
docs/report_template.md
docs/presentation_outline.md
docs/experiment_plan.md
docs/result_explanation_guide.md
```

## Deploy

For a simple demo deploy:

1. Train locally and upload `models/asl_model.pt` to the server.
2. Build the backend Docker image with `Dockerfile.backend`.
3. Host the frontend with Vite build output or a static host.
4. Set the frontend API URL to your backend domain before production.
