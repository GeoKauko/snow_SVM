import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()  

    snow = data.sel(bands="value")         
    dem = data.sel(bands="DEM")            

    time_steps = data.coords["t"].values
    results = []

    for t in time_steps:
        snow_mask = snow.sel(t=t).data.astype(bool) 
        dem_at_time = dem.sel(t=t).data             

        snow_elev = np.where(snow_mask, dem_at_time, np.nan)
        min_elev = np.nanmin(snow_elev)

        threshold = min_elev - 200
        masked = np.where(dem_at_time >= threshold, dem_at_time, np.nan)

        da = xr.DataArray(
            masked[np.newaxis, np.newaxis, :, :],  
            coords={
                "t": [t],
                "bands": ["thresholded_dem"],
                "y": data.coords["y"],
                "x": data.coords["x"]
            },
            dims=["t", "bands", "y", "x"]
        )
        results.append(da)

    final = xr.concat(results, dim="t")
    return XarrayDataCube(final)


