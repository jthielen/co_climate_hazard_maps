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
import xesmf
import rioxarray
import requests
from io import BytesIO
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
import cmocean

# todo local imports

# todo copy from temp notebook
