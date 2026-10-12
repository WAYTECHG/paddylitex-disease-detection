"""Add the nine thesis class labels while preserving existing per-class scores."""
import argparse
import copy
import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    'Rice__BacterialLeafBlight',
    'Rice__BrownSpot',
    'Rice__Healthy',
    'Rice__Hispa',
    'Rice__LeafBlast',
    'Rice__LeafScald',
    'Rice__LeafSmut',
    'Rice__NarrowBrownLeafSpot',
    'Rice__NeckBlast',
]


def apply_labels(payload):
    result = copy.deepcopy(payload)
    if result.get('schema_version') != 1 or result.get('unit') != 'percent':
        raise ValueError('Expected schema_version: 1 and unit: percent. No file was changed.')
    rows = result.get('classes')
    if not isinstance(rows, list):
        raise ValueError('Expected a classes list. No file was changed.')
    existing = {}
    for row in rows:
        class_id = row.get('id')
        if type(class_id) is not int or not 0 <= class_id < 9 or class_id in existing:
            raise ValueError('Class IDs must be distinct integers from 0 to 8. No file was changed.')
        if not isinstance(row.get('models', {}), dict):
            raise ValueError('Class models must be objects. No file was changed.')
        existing[class_id] = row
    result['classes'] = []
    for class_id, name in enumerate(CLASS_NAMES):
        row = existing.get(class_id, {'id': class_id, 'models': {}})
        row['name'] = name
        row.setdefault('models', {})
        result['classes'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=APP_ROOT / 'app/static/per_class.json')
    args = parser.parse_args()
    try:
        raw = args.file.read_bytes() if args.file.exists() else None
        payload = json.loads(raw.decode('utf-8-sig')) if raw is not None else {
            'schema_version': 1, 'unit': 'percent', 'context': {'split': 'test'}, 'classes': []}
        updated = apply_labels(payload)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            # Retain the first backup; repeating the update must not overwrite it.
            backup = args.file.with_name(args.file.name + '.before-labels.bak')
            if not backup.exists():
                backup.write_bytes(raw)
        temporary = args.file.with_name(args.file.name + '.tmp')
        temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(args.file)
    except (ValueError, OSError, TypeError, AttributeError) as error:
        parser.exit(1, f'Cannot update labels: {error}\n')
    print(f'Updated nine labels in {args.file.resolve()}')
    print('Existing class metrics were preserved by class ID. Refresh the browser with Ctrl + F5.')


if __name__ == '__main__':
    main()
