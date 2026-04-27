# Experiment Plan

Run all three models on the same dataset split and settings.

## Fast Comparison

```bash
cd /Users/j-19group/asl-translator
source .venv/bin/activate
python scripts/compare_models.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 3
```

## Better Comparison

```bash
python scripts/compare_models.py --dataset "/path/to/ASL_Alphabet_Dataset" --epochs 8 --pretrained
```

## Individual Training

```bash
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --architecture simple_cnn --epochs 8
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --architecture resnet18 --epochs 8 --pretrained --output models/resnet18.pt
python scripts/train.py --dataset "/path/to/ASL_Alphabet_Dataset" --architecture mobilenet_v3_small --epochs 8 --pretrained --output models/mobilenet_v3_small.pt
```

## Choosing The UI Model

After comparison, copy the best checkpoint to:

```bash
cp models/best_model_name.pt models/asl_model.pt
```

The backend loads `models/asl_model.pt`.

