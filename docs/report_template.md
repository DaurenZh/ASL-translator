# ASL Translator Checkpoint #2 Report

## 1. Project Overview

This project classifies American Sign Language alphabet images and displays predictions through a user interface. The system contains a PyTorch training pipeline, FastAPI inference backend, and React frontend for webcam/upload prediction.

## 2. Team Task Division

Use `docs/task_division.md` and replace placeholder participant names with real names.

## 3. Dataset

- Dataset name:
- Number of classes:
- Number of images:
- Train/validation/test split:
- Image size:
- Preprocessing:
- Augmentation:

## 4. Models Compared

| Model | Complexity | Pretrained | Validation Accuracy | Test Accuracy | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| Simple CNN | fill from metrics | No | | | baseline model |
| ResNet18 | fill from metrics | Optional | | | deeper residual model |
| MobileNetV3 Small | fill from metrics | Optional | | | efficient mobile-friendly model |

## 5. Model Complexity

Discuss trainable parameter count from each `*.metrics.json` file. Explain the tradeoff between accuracy, speed, and deployment size.

## 6. Evaluation

Include:

- best validation accuracy
- test accuracy
- per-class accuracy
- top misclassification pairs
- confusion matrix from notebook or metrics

## 7. Explanation Of Results

Explain which classes are easy and which are confused. Common causes include similar hand shapes, lighting, hand position, background, and static-frame limitations.

## 8. User Interface

The UI supports:

- webcam prediction
- image upload prediction
- confidence display
- top prediction list
- sentence builder
- clear/copy/speak controls

Add screenshots of the UI here.

## 9. Limitations

- Current model classifies static alphabet images, not full ASL grammar.
- Real webcam images may differ from dataset images.
- Dynamic signs such as J and Z are difficult in a static-image classifier.
- Hand detection/cropping is not yet implemented.

## 10. Bonus Work

Choose at least one:

- Demonstration on unseen images
- Deployment with Docker or hosted frontend/backend
- SOTA/adapted pretrained model comparison
- Advanced UI features such as webcam sentence builder and speech output

## 11. Conclusion

Summarize the best model, final accuracy, practical usefulness, and future improvements.

