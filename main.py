import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm

from splitter import greedy_loc_split

EXPORT_PATH = Path('./data/label-studio-export.json')
DATASET_PATH = Path('./data/dataset.ndjson')
OLD_URL = 'http://192.168.0.104:1337'
NEW_URL = 'https://barstatic.oops.wtf'

tasks = json.load(open(EXPORT_PATH, encoding="utf-8"))

loc_data = defaultdict(lambda: {"total": 0, "pos": 0})
total_pos = 0
for task in tqdm(tasks):
    task_id = task["id"]
    url = task["image"].replace(OLD_URL, NEW_URL)
    path = Path(urlparse(url).path)

    loc_hash = hashlib.sha1(
        str(path.parent).encode(), usedforsecurity=False
    ).hexdigest()
    task["loc_hash"] = loc_hash

    loc_data[loc_hash]["total"] += 1

    entry = {
        "type": "image",
        "file": f"{task_id}{path.suffix}",
        "url": url,
        "width": 0,
        "height": 0,
        "split": "",
        "annotations": {"boxes": []},
    }

    label = task.get("label")
    if label is not None:
        total_pos += len(label)
        loc_data[loc_hash]["pos"] += len(label)

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

    task["entry"] = entry

mapping = greedy_loc_split(len(tasks), total_pos, loc_data, (0.7, 0.15, 0.15), (1, 3))

with open(DATASET_PATH, "w", encoding="utf-8") as f:
    config = {
        "type": "dataset",
        "task": "detect",
        "name": "WildlifeCV",
        "class_names": {"0": "snow_leopard"},
        "version": 1,
    }

    json.dump(config, f)
    f.write("\n")

    for task in tasks:
        entry = task["entry"]
        entry["split"] = mapping[task["loc_hash"]]

        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")
