from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import shapely
from shapely.ops import unary_union
import requests
from io import BytesIO
from PIL import Image

#######################
# Specified Constants #
#######################

CO_BOUNDS = {
    "min_lat": 37.0,
    "min_lon": -109.046667,
    "max_lat": 41.0,
    "max_lon": -102.046667
}

COLORADO_COUNTIES_URL = (
    "https://raw.githubusercontent.com/kbazlen/co-risk-assessment/"
    "main/jobs/data/colorado_counties.geojson"
)

MAP_W_PX_DEFAULT = 1200
MAP_H_PX_DEFAULT = 800

HILLSHADE_REST = (
    "https://services.arcgisonline.com/arcgis/rest/services/"
    "Elevation/World_Hillshade/MapServer/export"
)

RIVERS_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_rivers_lake_centerlines.geojson"
)

CITIES = [
    {"name": "Aurora",           "lon": -104.7275, "lat": 39.7084},
    {"name": "Boulder",          "lon": -105.2515, "lat": 40.0273},
    {"name": "Colorado Springs", "lon": -104.7606, "lat": 38.8674},
    {"name": "Craig",            "lon": -107.5557, "lat": 40.5170},
    {"name": "Denver",           "lon": -104.9893, "lat": 39.7627},
    {"name": "Durango",          "lon": -107.8703, "lat": 37.2750},
    {"name": "Fort Collins",     "lon": -105.0657, "lat": 40.5478},
    {"name": "Glenwood Springs", "lon": -107.3344, "lat": 39.5454},
    {"name": "Grand Junction",   "lon": -108.5675, "lat": 39.0878},
    {"name": "Greeley",          "lon": -104.7707, "lat": 40.4149},
    {"name": "Gunnison",         "lon": -106.9246, "lat": 38.5490},
    {"name": "Lamar",            "lon": -102.6152, "lat": 38.0737},
    {"name": "Montrose",         "lon": -107.8594, "lat": 38.4688},
    {"name": "Pueblo",           "lon": -104.6131, "lat": 38.2706},
    {"name": "Trinidad",         "lon": -104.4908, "lat": 37.1749},
]

MAJOR_RIVERS = {"colorado", "arkansas", "platte", "rio grande", "gunnison"}


############
# Features #
############


def load_counties(
    counties_file: Optional[Path] = None,
    interactive: bool = False,
    state_column: str = "STATEFP"
) -> Optional[gpd.GeoDataFrame]:
    # 1. Explicit local path
    if counties_file is not None:
        if counties_file.exists():
            gdf = gpd.read_file(counties_file)
            if state_column in gdf.columns:
                gdf = gdf[gdf[state_column] == "08"]
            if interactive:
                print(f"[counties] Loaded {len(gdf)} counties from {counties_file}")
            return gdf
        if interactive:
            print(f"[warn] Counties file not found: {counties_file}")

    # 2. Auto-fetch from GitHub
    if interactive:
        print("[counties] Fetching from GitHub…")
    try:
        r = requests.get(COLORADO_COUNTIES_URL, timeout=15)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(
            r.json()["features"], crs="EPSG:4326"
        )
        if interactive:
            print(f"[counties] Loaded {len(gdf)} counties from GitHub")
        return gdf
    except Exception as exc:
        if interactive:
            print(f"[warn] GitHub fetch failed: {exc}")
        else:
            raise exc

    # 3. Interactive fallback
    print("\n[counties] Could not auto-fetch counties.")
    path_str = input(
        "  Enter path to a Colorado counties GeoJSON/Shapefile "
        "(or press Enter to skip): "
    ).strip()
    if path_str:
        p = Path(path_str)
        if p.exists():
            gdf = gpd.read_file(p)
            if state_column in gdf.columns:
                gdf = gdf[gdf[state_column] == "08"]
            print(f"[counties] Loaded {len(gdf)} counties from {p}")
            return gdf
        print(f"[warn] Not found: {p} — skipping counties layer")


def get_colorado_polygon(crs: str = "EPSG:3857") -> shapely.Geometry:
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )
    states = gpd.read_file(shp_path)
    co = states[states["postal"] == "CO"].to_crs(crs)
    return unary_union(co.geometry.values)


def load_hillshade(
    bbox: Dict[str, float],
    epsg: int = 3857,
    width_px: int = MAP_W_PX_DEFAULT,
    height_px: int = MAP_H_PX_DEFAULT,
    interactive: bool = False
) -> Optional[Image.Image]:
    bbox_str = ",".join(f"{bbox[key]:.2f}" for key in ("min_x", "min_y", "max_x", "max_y"))
    params = {
        "bbox": bbox_str, "bboxSR": epsg, "imageSR": epsg,
        "size": f"{width_px},{height_px}",
        "format": "png32", "transparent": "true", "f": "image",
    }
    url = HILLSHADE_REST + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        if interactive:
            print("[hillshade] Fetching from ESRI…")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        if interactive:
            print(f"[hillshade] Got {img.size[0]}×{img.size[1]} px")
        return img
    except Exception as exc:
        if interactive:
            print(f"[warn] Hillshade fetch failed: {exc}")
        else:
            raise exc


def load_rivers(
    bbox_wgs84: Dict[str, float], bleed: float = 1.0, interactive: bool = False
) -> Optional[gpd.GeoDataFrame]:
    try:
        if interactive:
            print("[rivers] Fetching…")
        r = requests.get(RIVERS_URL, timeout=20)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        gdf = gdf.cx[
            bbox_wgs84["min_lon"] - bleed : bbox_wgs84["max_lon"] + bleed,
            bbox_wgs84["min_lat"] - bleed : bbox_wgs84["max_lat"] + bleed,
        ]
        return gdf
    except Exception as exc:
        if interactive:
            print(f"[warn] Rivers fetch failed: {exc}")
        else:
            raise exc
