# Poulet Py – Neuroscience Python Library from Poulet Lab
A modular Python library for neuroscientific hardware control, data analysis, and experimental setups.

[![Unit Tests](https://github.com/poulet-lab/poulet_py/actions/workflows/test-suite.yml/badge.svg?branch=main)](https://github.com/poulet-lab/poulet_py/actions/workflows/test-suite.yml)

## Overview
Poulet provides a collection of tools for neuroscience researchers, including:
* Hardware communication (e.g., serial devices, DAQ interfaces)
* Data analysis utilities
* Experimental workflow automation

The library is organized into these core modules:

|Module        |Description|
| --------     | ------- |
|**config**    |It contains the SETTINGS and LOGGER constants for loading environmental variables and logging.|
|**converters**|For converting different filetypes to common data science (h5, plk, etc.).|
|**hardware**  |Low-level drivers and interfaces for external devices (e.g., amplifiers, stimulator).|
|**tools**     |Helper functions (serializers, generators, file I/O) – typically class-free.|
|**utils**     |High-level interactive tools for experiments and data processing.|

## Installation
### Minimal Installation
```sh
pip install -U .
```

### Optional Dependencies
Install specific modules:
```sh
pip install -U .[hardware]          # Only hardware dependencies
pip install -U .[tools,utils]       # Tools + utilities

# Special case for seq or all:
pip install -U git+https://github.com/LJMUAstroecology/flirpy.git
pip install -U .[seq]

pip install -U .[all]               # Everything
```

### Development Mode (Editable Install)
```sh
pip install -U git+https://github.com/LJMUAstroecology/flirpy.git
pip install -U -e .[all]            # For contributors
```

## Configuration
The config/ folder includes:
* settings.py – Environment variables for dynamic configuration (avoid hardcoded values).
* Custom logger – Use instead of print() for better debugging.

Example:
```python
from poulet_py import LOGGER
LOGGER.info("Starting trial...")
```

## Examples
Check the examples/ folder for usage scripts, such as:
* examples/oscilloscope.py – Oscilloscope like visualization.

## Contributing
We welcome contributions! Follow these guidelines:

1. Branching
  * Work on a dedicated branch (git checkout -b your-feature).
  * Submit PRs to the dev branch (PRs to main will be rejected).
  
2. Code Standards
  * Tests: Add unit tests in tests/ (mirroring the module structure).
  * Documentation: Use docstrings and update __init__.py for lazy loading.
  * Type hints: Recommended for new functions.

3. Commit Messages
  * Use semantic prefixes (e.g., feat:, fix:, docs:).

