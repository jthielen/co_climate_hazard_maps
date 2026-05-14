import warnings

import numpy as np
import xarray as xr
import pyproj

#######################
# Specified Constants #
#######################

CO_BOUNDS = {
    "min_lat": 37.0,
    "min_lon": -109.046667,
    "max_lat": 41.0,
    "max_lon": -102.046667
}
CLIP_MARGIN_DEGREES = 0.5

############################
# General xarray Utilities #
############################

def mask_to_co(ds):
    return ds.where(
        ~(
            (ds.lat < CO_BOUNDS['min_lat'] - CLIP_MARGIN_DEGREES)
            | (ds.lat > CO_BOUNDS['max_lat'] + CLIP_MARGIN_DEGREES)
            | (ds.lon < CO_BOUNDS['min_lon'] - CLIP_MARGIN_DEGREES)
            | (ds.lon > CO_BOUNDS['max_lon'] + CLIP_MARGIN_DEGREES)
        ),
        drop=True
    )

########################
# Reference Transforms #
########################

CRS_4326 = pyproj.CRS(4326)
CRS_3857 = pyproj.CRS(3857)
TRF_4326_TO_3857 = pyproj.Transformer.from_crs(CRS_4326, CRS_3857, always_xy=True)
TRF_3857_TO_4326 = pyproj.Transformer.from_crs(CRS_3857, CRS_4326, always_xy=True)
CO_BOUNDS_3857 = {}
CO_BOUNDS_3857["min_x"], CO_BOUNDS_3857["min_y"] = TRF_4326_TO_3857.transform(
    CO_BOUNDS["min_lon"], CO_BOUNDS["min_lat"]
)
CO_BOUNDS_3857["max_x"], CO_BOUNDS_3857["max_y"] = TRF_4326_TO_3857.transform(
    CO_BOUNDS["max_lon"], CO_BOUNDS["max_lat"]
)

def create_crs_and_trf_from_cf_to_latlon(cf_attrs):
    crs = pyproj.CRS.from_cf(cf_attrs)
    trf = pyproj.Transformer.from_crs(crs, CRS_4326, always_xy=True)
    return crs, trf


###################
# xesmf Utilities #
###################

# Define the output raster target, for a given points_per_degree / grid_spacing, for use with xesmf
def generate_co_raster_target_4326(points_per_degree):
    latlon_spacing = 1.0 / points_per_degree
    return xr.Dataset(
    {
        "lat": (
            ["lat"],
            np.arange(CO_BOUNDS["min_lat"] + latlon_spacing / 2, CO_BOUNDS["max_lat"], latlon_spacing),
            {"units": "degrees_north"}
        ),
        "lon": (
            ["lon"],
            np.arange(CO_BOUNDS["min_lon"] + latlon_spacing / 2, CO_BOUNDS["max_lon"], latlon_spacing),
            {"units": "degrees_east"}
        ),
    }
)
def generate_co_raster_target_3857(grid_spacing_meters):
    e = np.arange(CO_BOUNDS_3857["min_x"] + grid_spacing_meters / 2, CO_BOUNDS_3857["max_x"], grid_spacing_meters)
    n = np.arange(CO_BOUNDS_3857["min_y"] + grid_spacing_meters / 2, CO_BOUNDS_3857["max_y"], grid_spacing_meters)
    nn, ee = np.meshgrid(n, e)
    lon2d, lat2d = (a.T for a in TRF_3857_TO_4326.transform(ee, nn))
    return xr.Dataset({
        "south_north": xr.Variable(["south_north"], n, {"units": "meters"}),
        "west_east": xr.Variable(["west_east"], e, {"units": "meters"}),
        "lat": xr.Variable(["south_north", "west_east"], lat2d, {"units": "degrees_north"}),
        "lon": xr.Variable(["south_north", "west_east"], lon2d, {"units": "degrees_east"})
    })

# Add coordinate bounds, given usual input data south_north / west_east dimension coordinates
def add_coordinate_bounds(ds, trf):
    n_diff = np.diff(ds['south_north'].data).mean()
    n_bounds = np.concatenate([[ds['south_north'].data[0] - n_diff / 2], ds['south_north'].data + n_diff / 2])
    e_diff = np.diff(ds['west_east'].data).mean()
    e_bounds = np.concatenate([[ds['west_east'].data[0] - e_diff / 2], ds['west_east'].data + e_diff / 2])
    nn_bounds, ee_bounds = np.meshgrid(n_bounds, e_bounds)
    lon_bounds, lat_bounds = (a.T for a in trf.transform(ee_bounds, nn_bounds))
    return ds.assign_coords({
        "south_north_b": n_bounds,
        "west_east_b": e_bounds,
        "lat_b": xr.Variable(("south_north_b","west_east_b"),lat_bounds,{"units":"degrees_north"}),
        "lon_b": xr.Variable(("south_north_b","west_east_b"),lon_bounds,{"units":"degrees_east"})
    })
