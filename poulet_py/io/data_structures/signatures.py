try:
    from poulet_py.io.data_structures.widefield import WidefieldDataV1
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


DATA_SIGNATURES = {
    "widefield_v1": WidefieldDataV1.DATA_SIGNATURE,
}
