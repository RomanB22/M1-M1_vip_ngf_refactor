# m1_model/utils/csv_helpers.py
import csv

def csv_to_dict(path: str):
    """
    Returns {variant_name: {param: value_str, ...}, ...}
    Assumes the first column is a variant key (header 'variant' or similar).
    Other columns are parameter names. All values left as strings; cast later.
    """
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        key_field = None
        # pick first column as the key if we can't find 'variant'
        if "variant" in reader.fieldnames:
            key_field = "variant"
        else:
            key_field = reader.fieldnames[0]
        for row in reader:
            key = row[key_field]
            vals = {k: v for k, v in row.items() if k != key_field}
            out[key] = vals
    return out
