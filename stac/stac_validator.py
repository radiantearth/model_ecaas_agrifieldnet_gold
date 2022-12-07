import json
from pystac.validation import validate_dict

with open('item.json') as f:
    js = json.load(f)
validate_dict(js)

with open('collection.json') as f:
    js = json.load(f)
validate_dict(js)
