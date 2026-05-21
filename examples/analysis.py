from pathlib import Path

from poulet_py import PathPattern, PatternBasedDiscovery
from poulet_py.utils.analysis.common import Session

session = Session(
    path=Path("."),
    discovery_strategy=PatternBasedDiscovery(
        patterns=[
            PathPattern(
                pattern="{date}/{subject_id}/{experiment}/{method}/{trial}",
                custom_fields={
                    "date": (r"\d{4}-\d{2}-\d{2}", "date"),
                    "method": (r"(widefield|ephys|behavior)", "str"),
                },
            )
        ]
    ),
)


session.iloc[:5]
