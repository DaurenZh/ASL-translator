# Getting Started After Clone

This guide starts from a fresh clone of the ASL translator repository.

## 1. Open The Project

```bash
cd asl-translator
```

## 2. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.train.txt
pip install -r backend/requirements.txt
```

If `torch` or `torchvision` installation fails, install the correct PyTorch build for your system from the official PyTorch install command page, then rerun the requirements command.

## 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## 4. Prepare Dataset

The dataset should look like this:

```text
ASL_Alphabet_Dataset/
  asl_alphabet_train/
    A/
    B/
    C/
    ...
    Z/
    del/
    nothing/
    space/
```

You can pass either:

```text
/path/to/ASL_Alphabet_Dataset
```

or:

```text
/path/to/ASL_Alphabet_Dataset/asl_alphabet_train
```

to the training command.

## 5. Quick Smoke Test

Run one epoch to check that the dataset path and dependencies work:

```bash
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --architecture simple_cnn --epochs 1
```

Expected result:

```text
models/asl_model.pt
models/asl_model.metrics.json
```

## 6. Run Model Comparison For Checkpoint #2

Required comparison:

```bash
python scripts/compare_models.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 5 --pretrained
```

This trains and compares:

- `simple_cnn`
- `resnet18`
- `mobilenet_v3_small`

If there is no internet for downloading pretrained weights:

```bash
python scripts/compare_models.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 5
```

Outputs:

```text
models/simple_cnn.pt
models/resnet18.pt
models/mobilenet_v3_small.pt
models/model_comparison.json
```

## 7. Choose Best Model For The App

Open:

```text
models/model_comparison.json
```

Pick the model with the best test accuracy and copy it as:

```bash
cp models/resnet18.pt models/asl_model.pt
```

Replace `resnet18.pt` with the actual best model file.

## 8. Run Backend

```bash
cd backend
source ../.venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl http://localhost:8000/health
```

You should see:

```json
"model_ready": true
```

## 9. Run Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open the URL printed by Vite, usually:

```text
http://localhost:5173
```

The UI supports:

- webcam prediction
- image upload prediction
- confidence display
- top predictions
- sentence builder
- copy, clear, and speech output

## 10. Use The Notebooks

Checkpoint 1 baseline notebook:

```text
notebooks/checkpoint1.ipynb
```

Improved Checkpoint 1 notebook:

```text
notebooks/checkpoint1_improved_for_checkpoint2.ipynb
```

Checkpoint 2 experiment notebook:

```text
notebooks/checkpoint2_experiments.ipynb
```

Use the improved notebook for explanation and presentation. Use `checkpoint1.ipynb` only as evidence of the initial baseline work.

## 11. Report And Presentation

Use:

```text
docs/task_division.md
docs/report_template.md
docs/presentation_outline.md
docs/experiment_plan.md
docs/result_explanation_guide.md
```

Fill in:

- real team member names
- dataset statistics
- final model comparison table
- screenshots of the UI
- explanation of errors and limitations

## 12. Recommended Checkpoint #2 Workflow

1. Clone repo.
2. Install Python and frontend dependencies.
3. Run one-epoch smoke test.
4. Run three-model comparison.
5. Choose best model and copy it to `models/asl_model.pt`.
6. Start backend.
7. Start frontend.
8. Take UI screenshots.
9. Fill report template.
10. Build PowerPoint from presentation outline.
