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
import geopandas as gpd
import rioxarray

from geocube.api.core import make_geocube

from util.transforms import CO_BOUNDS_3857
from util.metadata import MetadataIndex
from util.geofeatures import COLORADO_CLIMATE_REGIONS_URL

# Instantiate
now = pd.Timestamp.now()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Colorado climate hazard variable zonal statistics"
    )
    parser.add_argument(
        "--tif-index", type=Path, required=True,
        help="Location of GeoTIFF index at root of collection"
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Output directory of geojson vector files"
    )
    parser.add_argument(
        "--source-vector", type=Path, default=None,
        help="Local GeoJSON vector regions (auto-fetches climate regions if omitted)"
    )
    parser.add_argument(
        "--id-column", type=str, default=None,
        help="Unique integer ID column in source vector, for association in analysis"
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix to append to base variable in output geojson names"
    )
    parser.add_argument("--skip", nargs="*", type=str, default="")

    args = parser.parse_args()

    # open index
    index = MetadataIndex(args.tif_index)
    tif_base_path = args.tif_index.parent

    # open the vector source
    if args.source_vector:
        regions = gpd.read_file(args.source_vector)
        id_col = args.id_column
        if not id_col or id_col not in regions.columns or not pd.api.types.is_numeric_dtype(regions[id_col]):
            raise ValueError("Must provide valid numeric unique ID column when using custom vector source")
    else:
        regions = gpd.read_file(COLORADO_CLIMATE_REGIONS_URL)
        id_col = "REGION_ID"

    # parse other inputs
    output_path = args.output_path
    suffix = args.suffix if args.suffix else ''

    # open the data
    out_grids = {}
    out_paths = {}
    for key, meta in index.items():
        if any(bool(skip) and key.startswith(skip) for skip in args.skip):
            print(f"[warn] Skipping {key} as specified in --skip")
            continue
        if "categorical" in meta['symbology_colormap']:
            print(f"[warn] Skipping {key} due to categorical symbology")
            continue
        # Load epsg3857 geotiff and prep this output collection
        print(f"[tif] Opening original projection geotiff for: {key}")
        rda = rioxarray.open_rasterio(
            tif_base_path / meta['geotiff_filepath_lcc_original']
        ).squeeze().drop("band")

        if meta['base_variable'] not in out_grids:
            # set up the out_grid for this base variable
            out_grids[meta['base_variable']] = make_geocube(
                vector_data=regions,
                measurements=[id_col],
                like=rda, # ensure the data are on the same grid
            )
            out_paths[meta['base_variable']] = output_path / (
                meta['geotiff_filepath_lcc_original'].split("/")[1] + suffix + ".geojson"
            )

        # Append to out_grids
        out_grids[meta['base_variable']][key] = rda.variable

    # loop through groups to compute zonal stats and output
    for base_var, out_grid in out_grids.items():
        # stats
        print(f"[stats] Calculating zonal stats for {base_var}")
        grouped = out_grid.drop("spatial_ref").groupby(out_grid[id_col])
        data_mean = grouped.mean()
        data_mean = data_mean.rename({k: f"MEAN_{k}" for k in data_mean.data_vars})
        data_max = grouped.max()
        data_max = data_max.rename({k: f"MAX_{k}" for k in data_max.data_vars})
        data_min = grouped.min()
        data_min = data_min.rename({k: f"MIN_{k}" for k in data_min.data_vars})
        data_std = grouped.std()
        data_std = data_std.rename({k: f"STD_{k}" for k in data_std.data_vars})
        zonal_stats = xr.merge([data_mean, data_min, data_max, data_std]).to_dataframe()
        regions_with_these_data = regions.merge(zonal_stats, on=id_col)

        # output
        print(f"[geojson] Exporting {base_var} stats to {out_paths[base_var]}")
        regions_with_these_data.to_file(out_paths[base_var], driver='GeoJSON')

    # for each base_var in collection (which should be a geodataframe), export to geojson for group
