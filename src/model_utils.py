import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from PIL import ImageOps
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = int((max_wh - w) / 2)
        p_top = int((max_wh - h) / 2)
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        padding = (p_left, p_top, p_right, p_bottom)
        return ImageOps.expand(image, padding, fill=0)

class CarTypeViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CarTypeViTClassifier, self).__init__()

        self.base_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        for param in self.base_model.parameters():
            param.requires_grad = False

        num_features = self.base_model.heads.head.in_features

        self.base_model.heads = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def load_classifier(path, num_classes):
    print(f"Loading Classifier from {path}...")
    if not os.path.exists(path):
         raise FileNotFoundError(f"Classifier model not found at {path}")
         
    model = CarTypeViTClassifier(num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
