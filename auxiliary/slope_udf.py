#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# slope_udf.py
from openeo.udf import XarrayDataCube
import xarray as xr
import numpy as np

def compute_slope(da, resolution=30):
    # Calculate gradient in x and y
    dz_dx, dz_dy = np.gradient(da, resolution, resolution)

    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)

    return slope

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()
    band = data.sel(bands="DEM")

    # Compute slope and aspect
    slope = compute_slope(band)

    # Wrap as DataArray with new bands
    new_data = xr.concat(
        [
            xr.DataArray(slope, dims=band.dims, coords=band.coords, name="slope")
        ],
        dim="bands"
    )
    new_data["bands"] = ["slope"]
    return XarrayDataCube(new_data)