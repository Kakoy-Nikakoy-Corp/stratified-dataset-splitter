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

tasks = json.load(open('data/label-studio-export.json', 'r', encoding='utf-8'))

loc_data = defaultdict(lambda: {'total': 0, 'pos': 0})
for task in tqdm(tasks):
    task_id = task['id']
    url = task['image']
    path = Path(urlparse(url).path)

    entry = {
        'type': 'image',
        'file': f'{task_id}{path.suffix}',
        'url': task['image'],
        'width': 0,
        'height': 0,
        'split': '',
        'annotations': {'boxes': []}
    }

    label = task.get('label')
    if label is not None:
        entry['width'] = label[0]['original_width']
        entry['height'] = label[0]['original_height']

        for annotation in label:
            w = annotation['width'] / 100.0
            h = annotation['height'] / 100.0
            x = annotation['x'] / 100.0 + w / 2
            y = annotation['y'] / 100.0 + h / 2

            entry['annotations']['boxes'].append([0, x, y, w, h])
    else:
        resp = requests.get(url)
        image = Image.open(BytesIO(resp.content))
        entry["width"], entry["height"] = image.size

    loc_hash = hashlib.sha1(str(path.parent).encode(), usedforsecurity=False).hexdigest()
    task['loc_hash'] = loc_hash

    loc_data[loc_hash]['total'] += 1
    loc_data[loc_hash]['pos'] += len(label) if label else 0

    task['entry'] = entry

mapping = greedy_loc_split(len(tasks), loc_data, (0.7, 0.15, 0.15), (1, 3))

with open('data/dataset.ndjson', 'w', encoding='utf-8') as f:
    config = {
        'type': 'dataset',
        'task': 'detect',
        'name': 'WildlifeCV',
        'class_names': {'0': 'snow_leopard'},
        'version': 1
    }

    json.dump(config, f)
    f.write('\n')

    for task in tasks:
        entry = task['entry']
        entry['split'] = mapping[task['loc_hash']]

        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
