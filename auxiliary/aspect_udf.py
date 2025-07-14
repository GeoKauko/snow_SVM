#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from openeo.udf import XarrayDataCube
import xarray as xr
import numpy as np

def compute_aspect(da, resolution=30):
    # Calculate gradient in x and y
    dz_dx, dz_dy = np.gradient(da, resolution, resolution)

    # Calculate aspect in degrees
    aspect = np.arctan2(-dz_dy, dz_dx) * (180 / np.pi)
    aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)
    aspect = np.where(aspect >= 360, aspect - 360, aspect)

    return aspect

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()
    band = data.sel(bands="DEM")

    # Compute slope and aspect
    aspect = compute_aspect(band)

    # Wrap as DataArray with new bands
    new_data = xr.concat(
        [
            xr.DataArray(aspect, dims=band.dims, coords=band.coords, name="aspect")
        ],
        dim="bands"
    )
    new_data["bands"] = ["aspect"]
    return XarrayDataCube(new_data)



