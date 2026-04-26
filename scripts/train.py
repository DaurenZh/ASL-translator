import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import ASLModel  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset_path(path: Path):
    if (path / "asl_alphabet_train").is_dir():
        return path / "asl_alphabet_train"
    return path


def make_loaders(dataset_path: Path, img_size: int, batch_size: int, seed: int, num_workers: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_train = datasets.ImageFolder(dataset_path, transform=train_transform)
    full_eval = datasets.ImageFolder(dataset_path, transform=eval_transform)

    if len(full_train.classes) < 2:
        raise ValueError(
            f"Dataset path {dataset_path} does not look like an ImageFolder dataset. "
            "It should contain one subfolder per sign class."
        )

    train_size = int(0.70 * len(full_train))
    val_size = int(0.15 * len(full_train))
    test_size = len(full_train) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_train,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # Evaluation subsets must use deterministic transforms, so point them at the eval dataset.
    val_subset.dataset = full_eval
    test_subset.dataset = full_eval

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return full_train.classes, train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate_predictions(model, loader, device, classes):
    model.eval()
    correct = 0
    total = 0
    per_class_total = torch.zeros(len(classes), dtype=torch.long)
    per_class_correct = torch.zeros(len(classes), dtype=torch.long)
    mistakes = {}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)

            correct_mask = preds == labels
            correct += correct_mask.sum().item()
            total += labels.size(0)

            for label, pred, is_correct in zip(labels.cpu(), preds.cpu(), correct_mask.cpu()):
                per_class_total[label] += 1
                if is_correct:
                    per_class_correct[label] += 1
                else:
                    pair = (classes[label], classes[pred])
                    mistakes[pair] = mistakes.get(pair, 0) + 1

    per_class = []
    for index, name in enumerate(classes):
        count = per_class_total[index].item()
        accuracy = per_class_correct[index].item() / count if count else 0.0
        per_class.append({"class": name, "accuracy": round(accuracy, 4), "count": count})

    top_mistakes = [
        {"true": true, "predicted": predicted, "count": count}
        for (true, predicted), count in sorted(mistakes.items(), key=lambda item: item[1], reverse=True)[:15]
    ]

    return {
        "accuracy": round(correct / total, 4),
        "total": total,
        "per_class": per_class,
        "top_mistakes": top_mistakes,
    }


def train(args):
    seed_everything(args.seed)

    dataset_path = resolve_dataset_path(Path(args.dataset).expanduser().resolve())
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = output_path.resolve()
    else:
        output_path = PROJECT_ROOT / "models" / "asl_model.pt"
    metrics_path = output_path.with_suffix(".metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes, train_loader, val_loader, test_loader = make_loaders(
        dataset_path=dataset_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    device = get_device()
    model = ASLModel(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Device: {device}")
    print(f"Saving best checkpoint to: {output_path}")

    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_predictions(model, test_loader, device, classes)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "img_size": args.img_size,
        "architecture": "ASLModel",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    torch.save(checkpoint, output_path)

    metrics = {
        "dataset": str(dataset_path),
        "output": str(output_path),
        "classes": classes,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "test": test_metrics,
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Metrics saved to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ASL translator image classifier.")
    parser.add_argument("--dataset", required=True, help="Path to ASL ImageFolder dataset or its parent folder.")
    parser.add_argument("--output", help="Checkpoint output path. Defaults to models/asl_model.pt.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
