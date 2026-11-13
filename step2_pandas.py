import sys
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import paths

CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "hazelnut"
ROOT = paths.get_mvtec_root() / CATEGORY
OUT = paths.BASE / f"dataset_{CATEGORY}.csv"
SEED = 42

rows = []
for folder in sorted(p for s in ["train", "test"] for p in (ROOT / s).iterdir()):
    for img in sorted(folder.glob("*.png")):
        rows.append({
            "path": str(img),
            "label": 0 if folder.name == "good" else 1,
            "defect": folder.name,
        })
df = pd.DataFrame(rows)

train_val, test = train_test_split(df, test_size=0.15, stratify=df["defect"], random_state=SEED)
train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=train_val["defect"], random_state=SEED)
df.loc[train.index, "split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

assert df["split"].notna().all()
assert df["path"].is_unique
assert df["path"].map(lambda p: Path(p).exists()).all()
sizes = df["path"].map(lambda p: (lambda im: (im.size, im.mode))(Image.open(p)))
print("Image sizes and modes:", sizes.value_counts().to_dict())
df["md5"] = df["path"].map(lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest())
assert df["md5"].is_unique, "duplicate files by content found"
df = df.drop(columns="md5")

df.to_csv(OUT, index=False)

print("\nTotal photos:", len(df))
print("\nok/ng per split:")
print(pd.crosstab(df["split"], df["label"].map({0: "ok", 1: "ng"}), margins=True))
print("\nDefect types per split:")
print(pd.crosstab(df["split"], df["defect"]))
print("\nSaved:", OUT)
