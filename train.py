import argparse
import json
import os
from pathlib import Path
import paths
# keep the downloaded pretrained weights inside the project folder
os.environ["TORCH_HOME"] = str(paths.BASE / "torch_cache")

import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

CATEGORY = "hazelnut"
CSV_PATH = str(paths.BASE / f"dataset_{CATEGORY}.csv")
MODEL_PATH = str(paths.BASE / f"model_{CATEGORY}.pth")

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
SEED = 42
THRESHOLD = 0.5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhotoDataset(Dataset):
    def __init__(self, df, transform):
        self.images = [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)) for p in df["path"]]
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def make_transforms():
    normalize = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        *normalize,
    ])
    eval_tf = T.Compose(normalize)
    return train_tf, eval_tf


def metrics(labels, probs, loss_sum):
    labels, preds = np.array(labels), (np.array(probs) >= THRESHOLD).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    recall_ng = tp / max(tp + fn, 1)
    recall_ok = tn / max(tn + fp, 1)
    return {
        "loss": loss_sum / len(labels),
        "acc": (tp + tn) / len(labels),
        "recall_ng": recall_ng,
        "precision_ng": tp / max(tp + fp, 1),
        "bal_acc": (recall_ng + recall_ok) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, all_labels, all_probs = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).float().unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            all_labels += y.squeeze(1).int().tolist()
            all_probs += torch.sigmoid(logits).squeeze(1).tolist()
    return metrics(all_labels, all_probs, loss_sum)


def metric_dict(m):
    return {k: m[k] for k in ("loss", "acc", "recall_ng", "precision_ng",
                              "bal_acc", "tp", "tn", "fp", "fn")}


def main():
    global CATEGORY, CSV_PATH, MODEL_PATH, IMG_SIZE, BATCH_SIZE, EPOCHS, LR
    parser = argparse.ArgumentParser(
        description="Train the defect model for one MVTec category."
    )
    parser.add_argument(
        "category", nargs="?", default="hazelnut",
        help="MVTec category name (default: hazelnut)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--out", default=None,
        help="output model path (default: model_<category>.pth next to "
             "this script)",
    )
    args = parser.parse_args()

    CATEGORY = args.category
    CSV_PATH = str(paths.BASE / f"dataset_{CATEGORY}.csv")
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    MODEL_PATH = args.out or str(
        paths.BASE / f"model_{CATEGORY}.pth")
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"Category: {CATEGORY} | device: {DEVICE} ({gpu_name})", flush=True)

    df = pd.read_csv(CSV_PATH)
    train_tf, eval_tf = make_transforms()
    print("Loading photos into memory...", flush=True)
    train_ds = PhotoDataset(df[df["split"] == "train"], train_tf)
    val_ds = PhotoDataset(df[df["split"] == "val"], eval_tf)
    test_ds = PhotoDataset(df[df["split"] == "test"], eval_tf)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(DEVICE)

    n_ng = sum(train_ds.labels)
    n_ok = len(train_ds) - n_ng
    pos_weight = torch.tensor([n_ok / n_ng], device=DEVICE)
    print(f"ok={n_ok} ng={n_ng} -> pos_weight={pos_weight.item():.2f}", flush=True)
    info = {
        "category": CATEGORY,
        "device": str(DEVICE),
        "gpu_name": gpu_name,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_ok": n_ok,
        "n_ng": n_ng,
        "pos_weight": pos_weight.item(),
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }
    print("@@INFO " + json.dumps(info), flush=True)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_key, best_epoch = (-1.0, 0.0), 0
    epoch_rows = []
    print(f"\n{'epoch':>5} | {'train loss':>10} {'acc':>6} | {'val loss':>8} {'acc':>6} {'recall ng':>9} {'prec ng':>7} {'bal acc':>7}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, loss_fn, optimizer)
        va = run_epoch(model, val_loader, loss_fn)
        seconds = time.perf_counter() - t0
        key = (va["bal_acc"], -va["loss"])
        is_best = key > best_key
        saved = ""
        if is_best:
            best_key, best_epoch, saved = key, epoch, "  <- best, saved"
            torch.save({
                "model_state": model.state_dict(), "category": CATEGORY, "img_size": IMG_SIZE,
                "mean": MEAN, "std": STD, "threshold": THRESHOLD, "epoch": epoch, "val": va,
            }, MODEL_PATH)
        print(f"{epoch:>5} | {tr['loss']:>10.4f} {tr['acc']:>6.1%} | {va['loss']:>8.4f} {va['acc']:>6.1%} "
              f"{va['recall_ng']:>9.1%} {va['precision_ng']:>7.1%} {va['bal_acc']:>7.1%}{saved}", flush=True)
        row = {
            "epoch": epoch,
            "seconds": seconds,
            "best": is_best,
            "train": {"loss": tr["loss"], "acc": tr["acc"]},
            "val": metric_dict(va),
        }
        epoch_rows.append(row)
        print("@@EPOCH " + json.dumps(row), flush=True)

    print(f"\nBest epoch: {best_epoch}. Final exam on test (the model has never seen these photos):", flush=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model_state"])
    te = run_epoch(model, test_loader, loss_fn)
    test_metrics = metric_dict(te)
    print(f"  accuracy={te['acc']:.1%}  recall ng={te['recall_ng']:.1%}  precision ng={te['precision_ng']:.1%}  balanced acc={te['bal_acc']:.1%}", flush=True)
    print(f"  confusion matrix: defects found={te['tp']}, defects missed={te['fn']}, good flagged={te['fp']}, good passed={te['tn']}", flush=True)
    print(f"\nModel: {MODEL_PATH}", flush=True)
    print("@@FINAL " + json.dumps({
        "best_epoch": best_epoch, "test": test_metrics, "model": MODEL_PATH,
    }), flush=True)

    sidecar = {
        "category": CATEGORY,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "epochs": EPOCHS, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "lr": LR, "seed": SEED, "threshold": THRESHOLD,
        },
        "info": info,
        "epochs": epoch_rows,
        "best_epoch": best_epoch,
        "test": test_metrics,
    }
    Path(MODEL_PATH).with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
