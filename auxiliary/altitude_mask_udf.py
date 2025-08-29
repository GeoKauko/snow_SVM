import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()  

    snow = data.sel(bands="snow_sure")         
    dem = data.sel(bands="DEM")            
    # valid = data.sel(bands="valid_mask")

    time_steps = data.coords["t"].values
    results = []

    for t in time_steps:
        snow_mask = snow.sel(t=t).data == 1
        # valid_mask = valid.sel(t=t).data == 1
        combined_mask = snow_mask #& valid_mask

        dem_at_time = dem.sel(t=t).data             

        # Use only DEM values where snow is present AND valid
        snow_elev = np.where(combined_mask, dem_at_time, np.nan)
        min_elev = np.nanmin(snow_elev)

        threshold = min_elev - 200

        final_mask = (dem_at_time >= threshold) #& valid_mask
        masked = np.where(final_mask, dem_at_time, np.nan)

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
