# Result Explanation Guide

Use this structure when explaining results in the report and presentation.

## Accuracy

Report both validation and test accuracy. Validation accuracy is used during training; test accuracy is the final estimate on held-out data.

## Model Comparison

Compare:

- accuracy
- number of trainable parameters
- training time
- suitability for deployment

Simple CNN is the baseline. ResNet18 is deeper and usually more accurate. MobileNetV3 Small is designed to be efficient and is a good candidate for deployment.

## Error Analysis

Use the `top_mistakes` section in each metrics file. Explain whether mistakes are caused by:

- visually similar signs
- poor lighting
- background noise
- hand position
- class imbalance
- dynamic signs represented as static images

## Conclusion Pattern

Example:

The best performing model was `{model}` with `{test_accuracy}` test accuracy. It improved over the baseline because `{reason}`. The main remaining errors were `{mistakes}`, likely caused by `{explanation}`. For deployment, `{model}` is preferred because `{accuracy/speed/size tradeoff}`.

