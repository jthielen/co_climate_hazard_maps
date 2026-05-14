#!/usr/bin/env python
"""
Process CO Climate Hazard Data - 01 GeoTiff Creation

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
import xesmf
import rioxarray
import geopandas
import pyproj

from util.transforms import (
    mask_to_co,
    generate_co_raster_target_3857,
    generate_co_raster_target_4326,
    create_crs_and_trf_from_cf_to_latlon
)

####
# todo functions and such
####

# ...

#######
# CLI #
#######


if __name__ == "__main__":
    # Take input
    parser = argparse.ArgumentParser(
        description="Process CO Climate Hazard Data - Step 01 GeoTiff Creation"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--data", required=True, help="Path to zarr data store of source data"
    )
    parser.add_argument(
        "--symbology", required=True, help="Path to symbology specifications tsv file"
    )
    parser.add_argument(
        "--index-filename",
        default="geotiff_index.csv",
        help="Filename of index file within output directory"
    )

    args = parser.parse_args()

    # Prepare directories
    output_path = Path(args.output)
    (output_path / "lcc_original").mkdir(exist_ok=True)
    (output_path / "wgs84_equirectangular").mkdir(exist_ok=True)
    (output_path / "wgs84_pseudomercator").mkdir(exist_ok=True)

    # Open source data collection
    dt = xr.open_datatree(args.data, decode_timedelta=False)

    # Open symbology reference
    symbology_ref = pd.read_csv(args.symbology, sep="\t")

    # Create WRF grid crs and transformers
    crs_wusd3, trf_wusd3 = create_crs_and_trf_from_cf_to_latlon(
        dt['wusd3_grid']['lambert_conformal'].attrs
    )
    crs_conus404, trf_conus404 = create_crs_and_trf_from_cf_to_latlon(
        dt['conus404_grid']['lambert_conformal'].attrs
    )

    # Begin to loop through grids to create geotiffs
    for grid, spacing_meters, points_per_degree in (
        ("wusd3_grid", 9.0e3, 11),
        ("conus404_coarsened_grid", 1.2e4, 11),
        ("conus404_grid", 4.0e3, 33),
    ):
        # Get target dataset templates
        ds_out_4326 = generate_co_raster_target_4326(points_per_degree)
        ds_out_3857 = generate_co_raster_target_3857(spacing_meters)
        
        # Loop through variables within this grid collection
        for base_variable in dt[grid].children:
            print(f"Processing {base_variable}...")
            # Create subdirectories for output
            for subdir in ("lcc_original", "wgs84_equirectangular", "wgs84_pseudomercator"):
                (output_path / subdir / base_variable).mkdir(exist_ok=True)
            # Get this dataset
            ds = mask_to_co(dt[grid][base_variable].to_dataset().drop_encoding())
            # Create the regridders (pre-loop for efficiency)
            regridders = {}
            for varname in ds.data_vars:
                if varname in ds.coords:
                    # skip coords
                    continue
                elif (
                    ds[varname].attrs.get("regrid_method") == "bilinear"
                    and "bilinear" not in regridders
                ):
                    regridders["bilinear_4326"] = xesmf.Regridder(ds, ds_out_4326, "bilinear")
                    regridders["bilinear_3857"] = xesmf.Regridder(ds, ds_out_3857, "bilinear")
                elif (
                    ds[varname].attrs.get("regrid_method") == "nearest_s2d"
                    and "nearest_s2d" not in regridders
                ):
                    regridders["nearest_s2d_4326"] = xesmf.Regridder(ds, ds_out_4326, "nearest_s2d")
                    regridders["nearest_s2d_3857"] = xesmf.Regridder(ds, ds_out_3857, "nearest_s2d")
            # Save tif output and update dt with symbology
            this_crs = crs_wusd3 if grid == "wusd3_grid" else crs_conus404
            for varname in ds.data_vars:
                if varname in ds.coords:
                    # skip coords
                    continue

                # Save original
                f = (output_path / "lcc_original" / base_variable / f"{varname}_lcc.tif")
                print(f"[tif] Generating {f}")
                ds[varname].rio.set_spatial_dims(
                    "west_east", "south_north"
                ).rio.write_crs(
                    this_crs
                ).rio.to_raster(
                    f, driver="GTiff"
                )
                dt[grid][base_variable][varname].attrs["tif_out_lcc"] = str(f)

                # Save reprojected
                method = ds[varname].attrs.get("regrid_method")
                # epsg4326
                f = (
                    output_path / "wgs84_equirectangular" / base_variable /
                    f"{varname}_epsg4326.tif"
                )
                print(f"[tif] Generating {f}")
                da_transformed = regridders[f"{method}_4326"](ds[varname], keep_attrs=False)
                da_transformed.rio.set_spatial_dims(
                    "lon", "lat"
                ).rio.write_crs(
                    "EPSG:4326"
                ).rio.to_raster(
                    f, driver="GTiff"
                )
                dt[grid][base_variable][varname].attrs["tif_out_4326"] = str(f)
                # epsg3857
                f = (
                    output_path / "wgs84_pseudomercator" / base_variable /
                    f"{varname}_epsg3857.tif"
                )
                print(f"[tif] Generating {f}")
                da_transformed = regridders[f"{method}_3857"](ds[varname], keep_attrs=False)
                da_transformed.rio.set_spatial_dims(
                    "west_east", "south_north"
                ).rio.write_crs(
                    "EPSG:3857"
                ).rio.to_raster(
                    f, driver="GTiff"
                )
                dt[grid][base_variable][varname].attrs["tif_out_3857"] = str(f)

                # Search for symbology ref and insert into dt
                print(f"[index] Fetching symbology metadata for {varname}")
                for _, row in (
                    symbology_ref.loc[
                        symbology_ref['variable_group'] == ds[varname].variable_base
                    ].iterrows()
                ):
                    if ds[varname].variable_subtype in json.loads(row['variable_subtype_list'].replace("\'", "\"")):
                        this_symbology = {k:v for k,v in row.to_dict().items() if k.startswith("symb")}
                        # load this symbology into dt
                        for k, v in this_symbology.items():
                            dt[grid][base_variable][varname].attrs[k] = v

    # Now that all geotiffs are created, make the geotiff index
    csv_rows = []
    print("[index] Creating geotiff index")
    for grid in ('wusd3_grid', 'conus404_coarsened_grid', 'conus404_grid'):
        for base_variable in dt[grid].children:
            ds = dt[grid][base_variable].to_dataset()
            for varname in ds.data_vars:
                if varname in ds.coords:
                    continue
                csv_rows.append(",".join([f'"{v}"' for v in (
                     varname,
                     getattr(ds[varname], "tif_out_lcc", "--"),
                     getattr(ds[varname], "tif_out_4326", "--"),
                     getattr(ds[varname], "tif_out_3857", "--"),
                     getattr(ds[varname], "variable_base", "--"),
                     getattr(ds[varname], "variable_subtype", "--"),
                     getattr(ds[varname], "units", "--"),
                     getattr(ds[varname], "technical_description", "--"),
                     getattr(ds[varname], "symbology_colormap_type", "--"),
                     getattr(ds[varname], "symbology_colormap", "--"),
                     getattr(ds[varname], "symbology_min_value", "--"),
                     getattr(ds[varname], "symbology_max_value", "--"),
                     getattr(ds[varname], "symbology_fixed_value_list", "--")
                )]))
    with open(output_path / args.index_filename, "w") as f:
        f.write((
            '"variable_key","geotiff_filepath_lcc_original",'
            '"geotiff_filepath_wgs84_equirectangular",'
            '"geotiff_filepath_wgs84_pseudomercator","base_variable","variable_subtype","units",'
            '"technical_description","symbology_colormap_type","symbology_colormap",'
            '"symbology_min_value","symbology_max_value","symbology_fixed_value_list"\n'
        ))
        for row in csv_rows:
            f.write(row + "\n")
