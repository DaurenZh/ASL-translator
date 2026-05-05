# Hugging Face Spaces Deployment

This project can run as one free Docker Space: React frontend + FastAPI backend + PyTorch model.

## 1. Choose Final Model

The Docker Space expects:

```text
models/asl_model.pt
```

For the final Checkpoint 2 model:

```bash
cp models/resnet18.pt models/asl_model.pt
```

## 2. Create Hugging Face Space

1. Go to Hugging Face.
2. Create a new Space.
3. Select SDK: `Docker`.
4. Hardware: free CPU is enough for demo, but prediction may be slower.

## 3. Push Project

Use Git LFS for the `.pt` model file:

```bash
git lfs install
git add .
git commit -m "Deploy ASL translator Space"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

If the branch is named `master`:

```bash
git push space master:main
```

## 4. What The Dockerfile Does

The root `Dockerfile`:

- builds the React frontend
- installs the FastAPI/PyTorch backend
- copies `models/asl_model.pt`
- serves the UI and API from the same app on port `7860`

Endpoints:

```text
/          React UI
/health    API health check
/predict   prediction API
```

## 5. Important Notes

- The frontend uses same-origin API calls in deployment.
- Webcam needs HTTPS; Hugging Face Spaces provides HTTPS.
- MediaPipe hand detection downloads its model in the browser from Google's model storage/CDN.
- If the Space sleeps, first load can be slow.

