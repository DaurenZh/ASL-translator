import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_training(args, architecture: str):
    output = PROJECT_ROOT / "models" / f"{architecture}.pt"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--dataset",
        args.dataset,
        "--architecture",
        architecture,
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--img-size",
        str(args.img_size),
        "--patience",
        str(args.patience),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.pretrained and architecture != "simple_cnn":
        command.append("--pretrained")

    print(f"\n=== Training {architecture} ===")
    subprocess.run(command, check=True)
    return output.with_suffix(".metrics.json")


def summarize(metrics_paths):
    rows = []
    for path in metrics_paths:
        data = json.loads(path.read_text())
        rows.append(
            {
                "architecture": data["architecture"],
                "pretrained": data["pretrained"],
                "parameters": data["trainable_parameters"],
                "best_val_accuracy": data["best_val_accuracy"],
                "test_accuracy": data["test"]["accuracy"],
                "metrics": str(path),
            }
        )

    rows.sort(key=lambda row: row["test_accuracy"], reverse=True)
    summary_path = PROJECT_ROOT / "models" / "model_comparison.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    print("\n=== Model Comparison ===")
    for row in rows:
        print(
            f"{row['architecture']:20s} "
            f"test={row['test_accuracy']:.4f} "
            f"val={row['best_val_accuracy']:.4f} "
            f"params={row['parameters']:,}"
        )
    print(f"\nSaved summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare three ASL classifier architectures.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    architectures = ["simple_cnn", "resnet18", "mobilenet_v3_small"]
    summarize([run_training(parsed, architecture) for architecture in architectures])

