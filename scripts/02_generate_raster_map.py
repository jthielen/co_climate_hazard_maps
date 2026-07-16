#!/usr/bin/env python
"""
Process CO Climate Hazard Data - 02 Generate Maps

...TODO notes
"""

import argparse
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from util.geofeatures import (
    CO_BOUNDS, MAP_H_PX_DEFAULT, MAP_W_PX_DEFAULT, load_counties, load_hillshade, load_rivers
)
from util.transforms import CO_BOUNDS_3857, CRS_3857
from util.metadata import MetadataIndex, write_legend_txt
from util.plotting import (
    register_colormaps, render_cities, render_county_borders, render_hillshade,
    render_legend_png, render_rioxarray, render_rivers
)

# Instantiate
register_colormaps()
OUTPUT_CRS = ccrs.epsg(3857)
now = pd.Timestamp.now()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render Colorado climate hazard maps → nested directories of figures"
    )
    parser.add_argument(
        "--tif-index", type=Path, required=True,
        help="Location of GeoTIFF index at root of collection"
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Output path for directories of figures"
    )
    parser.add_argument(
        "--counties-file", type=Path, default=None,
        help="Local counties GeoJSON (auto-fetched if omitted)"
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-hillshade", action="store_true")
    parser.add_argument("--no-rivers", action="store_true")
    parser.add_argument("--map-h-px", type=int, default=MAP_H_PX_DEFAULT)
    parser.add_argument("--map-w-px", type=int, default=MAP_W_PX_DEFAULT)
    parser.add_argument("--create-json-ref", type=bool, default=True)
    parser.add_argument("--skip", nargs="*", type=str, default="")
    parser.add_argument("--only", nargs="*", type=str, default="")

    args = parser.parse_args()
    map_h_px = args.map_h_px
    map_w_px = args.map_w_px
    include_hillshade = not args.no_hillshade
    include_rivers = not args.no_rivers
    dpi = args.dpi

    if include_hillshade:
        hillshade_img = load_hillshade(
            bbox=CO_BOUNDS_3857, width_px=map_w_px, height_px=map_h_px, interactive=True
        )

    if include_rivers:
        rivers_gdf = load_rivers(CO_BOUNDS, interactive=True)

    counties_gdf = load_counties(args.counties_file, interactive=True)

    index = MetadataIndex(args.tif_index)
    tif_base_path = args.tif_index.parent
    output_path = args.output_path
    output_meta = {}

    # Create the figures!
    for key, meta in index.items():
        if any(bool(skip) and key.startswith(skip) for skip in args.skip):
            print(f"[warn] Skipping {key} as specified in --skip")
            continue
        if meta['symbology_min_value'] == '--':
            print(f"[warn] Skipping {key} due to presently-unsupported symbology")
            continue
        if args.only and not any(bool(only) and key.startswith(only) for only in args.only):
            # silently continue, since this is not an "only" spec
            continue
        # Load epsg3857 geotiff and prep this output collection
        print(f"[tif] Opening epsg3857 geotiff for: {key}")
        rda = rioxarray.open_rasterio(
            tif_base_path / meta['geotiff_filepath_wgs84_pseudomercator']
        )
        this_output_path = (
            output_path / (meta['geotiff_filepath_wgs84_pseudomercator'].split("/")[1]) / key
        )
        this_output_path.mkdir(parents=True, exist_ok=True)

        # Create map
        fig = plt.figure(figsize=(map_w_px / dpi, map_h_px / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection=OUTPUT_CRS)
        ax.set_extent(
            [
                CO_BOUNDS_3857["min_x"],
                CO_BOUNDS_3857["max_x"],
                CO_BOUNDS_3857["min_y"],
                CO_BOUNDS_3857["max_y"]
            ],
            crs=OUTPUT_CRS
        )
        ax.set_facecolor("#f0f3f8")

        render_rioxarray(ax, rda, meta)

        if include_hillshade:
            render_hillshade(ax, hillshade_img, CO_BOUNDS_3857)

        if include_rivers:
            render_rivers(ax, rivers_gdf)

        render_county_borders(ax, counties_gdf)
        render_cities(ax)

        ax.axis('off')

        map_path = this_output_path / f"{key}_{now:%Y%m%d}.png"
        fig.savefig(map_path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="#f0f3f8")
        plt.close(fig)
        print(f"[render] Map → {map_path}  ({map_path.stat().st_size / 1024:.0f} KB)")

        # Create legend
        cbar_path = this_output_path / f"LEGEND_{key}.png"
        render_legend_png(meta['symbology_colormap'], cbar_path, interactive=True)

        # Create legend text
        legend_txt_path = this_output_path / f"LEGEND_{key}.txt"
        legend_txt = write_legend_txt(legend_txt_path, meta, now, interactive=True)

        # Add to output json collection
        if meta['base_variable'] not in output_meta:
            output_meta[meta['base_variable']] = {}
        output_meta[meta['base_variable']][meta['variable_subtype']] = {
            "variable_key": key,
            "collection_path": str(this_output_path.relative_to(output_path)),
            "figure_path": str(map_path.relative_to(output_path)),
            "colorbar_path": str(cbar_path.relative_to(output_path)),
            "legend_path": str(legend_txt_path.relative_to(output_path)),
            "vmin": meta['symbology_min_value'],
            "vmax": meta['symbology_max_value'],
            "colormap_name": meta['symbology_colormap'],
            "legend_text": legend_txt,
            "public_facing_name": meta['public_facing_name'],
            "export_date": f"{now:%Y-%m-%d %H:%M %Z}",
            "hazard_type": meta['base_variable'],
            "hazard_subtype": meta['variable_subtype'],
            "units": meta['units'],
            "hazard_technical_description": meta['technical_description'],
        }

    # create collection index
    if args.create_json_ref:
        print("[json] Exporting collection reference")
        json_out = output_path / (
            f"subset_{'_'.join(args.only)}_reference.json"
            if args.only else "collection_reference.json"
        )
        json_out.write_text(json.dumps(output_meta, indent=4))
