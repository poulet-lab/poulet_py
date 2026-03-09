# JSON schemas for experiment metadata

This folder holds **JSON Schema** definitions for metadata produced by different experiment types in `poulet_py`. Use them to validate metadata files, document the expected structure, or generate types in other languages.

## Widefield

- **Schema:** [widefield.json](widefield.json)  
- **Description:** Metadata extracted from widefield imaging (e.g. from `data.h5` or saved `metadata.json`). Includes `source_h5`, `trial_name`, `file_attributes`, `datasets`, and optional `parsed` (e.g. temperature range).

### Validating in Python

Use the Pydantic models and parser from the package (no extra dependency):

```python
from pathlib import Path
import json
from poulet_py.schemas import parse_widefield_trial_metadata, WidefieldTrialMetadata

# From a JSON file
with open(Path("trial/metadata.json")) as f:
    data = json.load(f)
meta = parse_widefield_trial_metadata(data)  # Validated WidefieldTrialMetadata

# From extract_trial_metadata()
from poulet_py.widefield.io import extract_trial_metadata, load_trial_metadata
raw = extract_trial_metadata(Path("trial/data.h5"))
meta = parse_widefield_trial_metadata(raw)
```

To validate a dict without raising (e.g. for optional checks), use `WidefieldTrialMetadata.model_validate(data)` inside a try/except for `ValidationError`.

### Validating with JSON Schema (optional)

For strict validation against the JSON Schema file (e.g. in CI or from another language), use the [jsonschema](https://pypi.org/project/jsonschema/) library and this file:

```python
import json
from pathlib import Path
import jsonschema

schema_path = Path(__file__).resolve().parents[1] / "schemas" / "widefield_trial_metadata.json"
with open(schema_path) as f:
    schema = json.load(f)
jsonschema.validate(instance=metadata_dict, schema=schema)
```

## Adding a new experiment schema

1. Add a new `.json` file here (e.g. `my_data.json`) following [JSON Schema](https://json-schema.org/) (draft 2020-12 or draft-07).
2. Optionally add a Pydantic model and `parse_*` function under `poulet_py/schemas/` (e.g. `poulet_py/schemas/my_experiment.py`) and export them from `poulet_py/schemas/__init__.py`.
