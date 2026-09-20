from datasets import load_dataset
from pathlib import Path

dataset = load_dataset("flwrlabs/pacs", split="train")

output_root = Path("data/PACS")

label_names = dataset.features["label"].names

counters = {}

for sample in dataset:
    image = sample["image"].convert("RGB")
    domain = sample["domain"]
    label = label_names[sample["label"]]

    key = (domain, label)

    if key not in counters:
        counters[key] = 0

    counters[key] += 1

    folder = output_root / domain / label
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"pic_{counters[key]:04d}.jpg"

    image.save(folder / filename, quality=95)

print(f"Done. Exported {len(dataset)} images to {output_root}")