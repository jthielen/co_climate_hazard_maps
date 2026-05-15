from pathlib import Path
from typing import Any, Union, Dict, SupportsFloat

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
import requests
import cmocean
from PIL import Image
import cartopy.crs as ccrs

from .geofeatures import CITIES, MAJOR_RIVERS, get_colorado_polygon

#######################
# Specified Constants #
#######################


COLORMAP_URL = (
    "https://raw.githubusercontent.com/jthielen/co_climate_hazard_maps/refs/heads/main/"
    "custom_colormaps.json"
)

# Default layer draw order
Z_TIF            = 3
Z_HILLSHADE      = 4
Z_COUNTY_FILL    = 2
Z_COUNTY_BORDERS = 6
Z_RIVERS         = 5
Z_CITIES         = 7

GEODETIC = ccrs.Geodetic()


###################
# Colormap Import #
###################

def register_colormaps(interactive: bool = False) -> None:
    # Fetch the color lists
    try:
        r = requests.get(COLORMAP_URL)
        r.raise_for_status()
        color_data = r.json()
    except Exception as exc:
        if interactive:
            print(f"[warn] Could not access colormap reference - {exc}")
        else:
            raise exc

    # Register them with matplotlib
    for var_name, this_var in color_data['colormaps'].items():
        lutsize = (
            mpl.rcParams['image.lut']
            if this_var["quantization_levels"] == "LUT"
            else this_var["quantization_levels"]
        )
        cmap = colors.LinearSegmentedColormap.from_list(
            var_name, this_var["color_list"], lutsize
        )
        cmap_reversed = colors.LinearSegmentedColormap.from_list(
            var_name + "_r", this_var["color_list"][::-1], lutsize
        )
        mpl.colormaps.register(name=var_name, cmap=cmap, force=True)
        mpl.colormaps.register(name=var_name + "_r", cmap=cmap_reversed, force=True)


####################
# Render Functions #
####################


def render_rioxarray(
    ax: Axes,
    rda: xr.DataArray,
    symbology: Union[pd.Series, Dict[str, Any]],
    zorder: SupportsFloat = Z_TIF
) -> None:
    ax.pcolormesh(
        rda.x.data,
        rda.y.data,
        rda.squeeze().data,
        vmin=symbology['symbology_min_value'],
        vmax=symbology['symbology_max_value'],
        cmap=symbology['symbology_colormap'],
        zorder=zorder
    )


def render_hillshade(
    ax: Axes,
    img: Image.Image,
    bbox: Dict[str, float],
    zorder: SupportsFloat = Z_HILLSHADE,
    transform: Optional[ccrs.CRS] = None,
    interpolation: Optional[str] = None
) -> None:
    """Shadow-only multiply — transparent where flat, dark where shaded."""
    gray         = np.array(img.convert("L")).astype(np.float32) / 255.0
    shadow_alpha = np.power(1.0 - gray, 0.8) * 0.85
    shadow_alpha = np.clip(shadow_alpha, 0.0, 1.0)
    rgba         = np.zeros((*gray.shape, 4), dtype=np.float32)
    rgba[:, :, 3] = shadow_alpha
    ax.imshow(
        rgba,
        extent=[bbox[key] for key in ("min_x", "max_x", "min_y", "max_y")],
        zorder=zorder,
        transform=transform,
        interpolation=interpolation
    )


"""
# TODO revisit
def render_counties(ax, gdf, county_values, cmap, vmin, vmax):
    if gdf is None or len(gdf) == 0:
        return
    gdf_3857  = gdf.to_crs("EPSG:3857")
    norm      = Normalize(vmin=vmin, vmax=vmax)
    geoid_col = next(
        (c for c in ("GEOID", "FIPS", "geoid", "fips") if c in gdf_3857.columns),
        None,
    )
    for _, row in gdf_3857.iterrows():
        geoid = str(row[geoid_col]) if geoid_col else str(row.name)
        val   = (county_values or {}).get(geoid)
        color = "#d8dde8" if (val is None or not np.isfinite(val)) else cmap(norm(val))
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax, facecolor=color, edgecolor="white",
            linewidth=0.6, zorder=Z_COUNTY_FILL)
"""


def render_county_borders(
    ax: Axes,
    gdf: gpd.GeoDataFrame,
    zorder: SupportsFloat = Z_COUNTY_BORDERS,
    crs: str = "EPSG:3857"
) -> None:
    gdf.to_crs(crs).boundary.plot(
        ax=ax, edgecolor="#1a2756", linewidth=0.5, alpha=0.35, zorder=zorder
    )


def render_rivers(
    ax: Axes, gdf: gpd.GeoDataFrame, zorder: SupportsFloat = Z_RIVERS, crs: str = "EPSG:3857"
) -> None:
    """Rivers clipped to Colorado border via intersection."""
    co_poly = get_colorado_polygon(crs)
    rivers_plotgeom = gdf.to_crs(crs)
    clipped = rivers_plotgeom[rivers_plotgeom.intersects(co_poly)].copy()
    clipped["geometry"] = clipped.geometry.intersection(co_poly)
    clipped = clipped[~clipped.is_empty]
    for _, row in clipped.iterrows():
        name = str(row.get("name", "")).lower()
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax,
            edgecolor=(0,0,0.745,0.5),
            linewidth=1.2 if any(r in name for r in MAJOR_RIVERS) else 0.8,
            zorder=zorder
        )


def render_cities(ax: Axes, zorder: SupportsFloat = Z_CITIES) -> None:
    for city in CITIES:
        lon, lat = city["lon"], city["lat"]
        ax.plot(
            lon, lat, marker="o", markersize=6, color="#dc143c", markeredgecolor=(1,1,1,1.0),
            markeredgewidth=1.0, transform=GEODETIC, zorder=zorder
        )
        ax.text(
            lon + 0.06, lat + 0.04, city["name"], fontsize=7.5, fontweight="bold",
            color="#111111", path_effects=[pe.withStroke(linewidth=2.2, foreground=(1,1,1,0.75))],
            transform=GEODETIC, zorder=zorder
        )
        

def render_legend_png(
    cmap: str,
    out_path: Path,
    scale: int = 4,
    base_w_in: float = 3.0,
    base_h_in: float = 0.28,
    base_dpi: SupportsFloat = 100,
    interactive: bool = False
) -> None:
    """
    Render a standalone horizontal color gradient bar at ``scale``× resolution.
    No labels, no axes, transparent background.
    Base size: 300 × 28 px → at 4×: 1200 × 112 px.
    """
    fig, ax = plt.subplots(
        figsize=(base_w_in, base_h_in),
        dpi=base_dpi * scale,
    )
    fig.patch.set_alpha(0.0)    # transparent figure background
    ax.set_facecolor("none")    # transparent axes background

    # Draw the gradient bar
    gradient = np.linspace(0, 1, 512).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        norm=colors.Normalize(vmin=0, vmax=1),
    )

    # Rounded end caps via a white rectangle mask on each side (gives pill shape)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(
        out_path,
        dpi=base_dpi * scale,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        format="png",
    )
    plt.close(fig)
    if interactive:
        print(f"[legend] PNG → {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
