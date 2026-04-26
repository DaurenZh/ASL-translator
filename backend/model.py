from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


DEFAULT_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]


class ASLModel(nn.Module):
    def __init__(self, num_classes: int = 29):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((10, 10)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ASLClassifier:
    def __init__(self, checkpoint_path: str = "../models/asl_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.classes = DEFAULT_CLASSES
        self.img_size = 96
        self.model = ASLModel(num_classes=len(self.classes)).to(self.device)
        self.ready = False

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.classes = checkpoint.get("classes", self.classes)
            self.img_size = checkpoint.get("img_size", self.img_size)
            self.model = ASLModel(num_classes=len(self.classes)).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.ready = True

        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image: Image.Image, top_k: int = 3):
        if not self.ready:
            return {
                "ready": False,
                "label": "model_missing",
                "confidence": 0.0,
                "predictions": [],
            }

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probabilities, k=min(top_k, len(self.classes)))

        predictions = [
            {
                "label": self.classes[index.item()],
                "confidence": round(value.item(), 4),
            }
            for value, index in zip(values, indices)
        ]

        return {
            "ready": True,
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "predictions": predictions,
        }

