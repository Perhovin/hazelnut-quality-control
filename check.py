import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

import paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(category="hazelnut"):
    ckpt = torch.load(paths.BASE / f"model_{category}.pth",
                      map_location=DEVICE)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    tf = T.Compose([T.ToTensor(), T.Normalize(ckpt["mean"], ckpt["std"])])
    return model, tf, ckpt["img_size"], ckpt["threshold"]


def predict(model, tf, img_size, threshold, photo_path):
    img = Image.open(photo_path).convert("RGB").resize((img_size, img_size))
    with torch.no_grad():
        p_ng = torch.sigmoid(model(tf(img).unsqueeze(0).to(DEVICE))).item()
    is_ng = p_ng >= threshold
    return is_ng, p_ng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--category", default="hazelnut")
    args = parser.parse_args()

    model, tf, img_size, threshold = load_model(args.category)
    is_ng, p_ng = predict(model, tf, img_size, threshold, args.photo)
    confidence = p_ng if is_ng else 1 - p_ng
    print(f"{'NG' if is_ng else 'OK'}  (confidence {confidence:.1%}, defect probability {p_ng:.1%})")


if __name__ == "__main__":
    main()
