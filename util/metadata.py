from pathlib import Path
from typing import Any, Dict
from collections.abc import Mapping
import textwrap

import pandas as pd

from .geofeatures import CO_BOUNDS

class MetadataIndex(Mapping):
    # A quick helper for navigating the metadata index
    def __init__(self, src: Path, **kwargs):
        self.tif_index = pd.read_csv(src, **kwargs)
        
    @property
    def valid_keys(self):
        return list(self.tif_index['variable_key'].unique())

    def __getitem__(self, key: str) -> Dict[str, Any]:
        rows = self.tif_index[self.tif_index['variable_key'] == key]
        if len(rows) != 1:
            raise IndexError(
                f"Key `{key}` could not be uniquely found. Use the `tif_index` directly."
            )
        else:
            return rows.iloc[0].to_dict()
        
    def __iter__(self):
        return iter(self.valid_keys)
        
    def __len__(self):
        return len(self.valid_keys)

    
def write_legend_txt(
    out_path: Path,
    meta: Dict[str, str],
    export_date: pd.Timestamp,
    interactive: bool = False
) -> None:
    lines = [
        "Colorado Climate Hazard Map Legend – Technical Metadata",
        "=" * 55,
        "",
        f"Export date         : {export_date:%Y-%m-%d %H:%M %Z}",
        f"Hazard type         : {meta['base_variable']}",
        f"Subtype variety     : {meta['variable_subtype']}",
        f"Unique ID           : {meta['variable_key']}",
        f"Units               : {meta['units']}"
    ] + textwrap.wrap(
        f"Hazard description  : {meta['technical_description']}", subsequent_indent="    "
    ) + [
        "",
        "Color ramp",
        f"    Minimum value   : {meta['symbology_min_value']} {meta['units']}",
        f"    Maximum value   : {meta['symbology_max_value']} {meta['units']}",
        f"    Colormap name   : {meta['symbology_colormap']}",
        "",
        "Colorado bounding box (WGS84)",
        f"    lonMin          : {CO_BOUNDS['min_lon']}",
        f"    lonMax          : {CO_BOUNDS['max_lon']}",
        f"    latMin          : {CO_BOUNDS['min_lat']}",
        f"    latMax          : {CO_BOUNDS['max_lat']}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if interactive:
        print(f"[legend] TXT → {out_path}")
