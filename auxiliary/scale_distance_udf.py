from openeo.udf import XarrayDataCube
import xarray as xr
import numpy as np

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()

    if "t" in data.dims:
        for t in data["t"]:
            time_slice = data.sel(t=t)

            min_val = time_slice.min(skipna=True)
            max_val = time_slice.max(skipna=True)
            range_val = max_val - min_val
            safe_range = range_val if range_val > 0 else 1.0

            scaled = ((time_slice - min_val) / safe_range) * 254.0
            scaled = scaled.where(~xr.ufuncs.isnan(time_slice), 255.0)

            data.loc[dict(t=t)] = scaled.round().astype(np.uint8)
    else:
        min_val = data.min(skipna=True)
        max_val = data.max(skipna=True)
        range_val = max_val - min_val
        safe_range = range_val if range_val > 0 else 1.0

        scaled = ((data - min_val) / safe_range) * 254.0
        data = scaled.where(~xr.ufuncs.isnan(data), 255.0).round().astype(np.uint8)

    return XarrayDataCube(data)
