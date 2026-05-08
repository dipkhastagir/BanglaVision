import json

def load_coco_annotations(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data