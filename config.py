from pathlib import Path

YOLO_CONF = {
    "type": "dataset",
    "task": "detect",
    "name": "WildlifeCV",
    "class_names": {"0": "snow_leopard"},
    "version": 1,
}

EXPORT_PATH = Path("./data/label-studio-export.json")
DATASET_PATH = Path("./data/dataset.ndjson")
OLD_URL = "http://192.168.0.104:1337"
NEW_URL = "https://dataset.irbis.wild1.net"

SPLIT_RATIO = [0.7, 0.15, 0.15]
GAIN_WEIGHTS = [1, 3]
