import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm

from config import (
    DATASET_PATH,
    EXPORT_PATH,
    GAIN_WEIGHTS,
    NEW_URL,
    OLD_URL,
    SPLIT_RATIO,
    YOLO_CONF,
)
from splitter import greedy_loc_split


def encode(p):
    return hashlib.sha1(str(p).encode()).hexdigest()


tasks = json.load(open(EXPORT_PATH, encoding="utf-8"))

loc_data = defaultdict(lambda: [0, 0])
for task in tqdm(tasks):
    url = task["image"].replace(OLD_URL, NEW_URL)
    path = Path(urlparse(url).path)

    img_hash = encode(path)
    loc_hash = encode(path.parent)
    task["loc_hash"] = loc_hash

    loc_data[loc_hash][0] += 1

    entry = {
        "type": "image",
        "file": f"{img_hash}{path.suffix}",
        "url": url,
        "width": 0,
        "height": 0,
        "split": "",
        "annotations": {"boxes": []},
    }
    task["entry"] = entry

    label = task.get("label")
    if label is not None:
        loc_data[loc_hash][1] += len(label)

        entry["width"] = label[0]["original_width"]
        entry["height"] = label[0]["original_height"]

        for annotation in label:
            w = annotation["width"] / 100.0
            h = annotation["height"] / 100.0
            x = annotation["x"] / 100.0 + w / 2
            y = annotation["y"] / 100.0 + h / 2

            entry["annotations"]["boxes"].append([0, x, y, w, h])
    else:
        resp = requests.get(url)
        image = Image.open(BytesIO(resp.content))
        entry["width"], entry["height"] = image.size

mapping = greedy_loc_split(loc_data, SPLIT_RATIO, GAIN_WEIGHTS)

with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(YOLO_CONF, f)

    for task in tasks:
        entry = task["entry"]
        entry["split"] = mapping[task["loc_hash"]]

        f.write("\n")
        json.dump(entry, f, ensure_ascii=False)
