from openeo.udf import XarrayDataCube
import xarray as xr
import numpy as np

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()

    # Extract distance index and valid mask
    distance = data.sel(bands="snow_sure") 
    # valid = data.sel(bands="valid_mask")

    time_steps = data.coords["t"].values
    results = []

    for t in time_steps:
        dist_t = distance.sel(t=t)
        # valid_t = valid.sel(t=t).data == 1

        dist_data = dist_t.data
        # dist_data[~valid_t] = np.nan 

        min_val = np.nanmin(dist_data)
        max_val = np.nanmax(dist_data)
        range_val = max_val - min_val
        safe_range = range_val if range_val > 0 else 1.0

        scaled = ((dist_data - min_val) / safe_range) * 254.0
        scaled = np.where(np.isnan(dist_data), 255.0, scaled)

        da = xr.DataArray(
            scaled[np.newaxis, np.newaxis, :, :].round().astype(np.float32),
            coords={
                "t": [t],
                "bands": ["scaled_distance"],
                "y": data.coords["y"],
                "x": data.coords["x"]
            },
            dims=["t", "bands", "y", "x"]
        )
        results.append(da)

    output = xr.concat(results, dim="t")
    return XarrayDataCube(output)
