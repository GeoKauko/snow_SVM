import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from openeo.udf import XarrayDataCube

def compute_distance(mask: np.ndarray) -> np.ndarray:
    inverse_mask = ~mask
    return distance_transform_edt(inverse_mask).astype(np.float32)

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()  
    result_list = []
    for t in data.coords['t'].values:
        snow_mask = data.sel(t=t, bands="snow_sure").data == 1
        # valid_mask = data.sel(t=t, bands="valid_mask").data == 1
        combined_mask = snow_mask #& valid_mask

        distance = compute_distance(combined_mask)

        min_val = np.nanmin(distance)
        max_val = np.nanmax(distance)
        norm_distance = (distance - min_val) / (max_val - min_val) if max_val > min_val else distance

        da = xr.DataArray(
            norm_distance[np.newaxis, np.newaxis, :, :],  # shape (1, 1, y, x)
            coords={
                "t": [t],
                "bands": ["distance"],
                "y": data.coords["y"],
                "x": data.coords["x"],
            },
            dims=["t", "bands", "y", "x"]
        )
        result_list.append(da)

    output = xr.concat(result_list, dim="t")
    return XarrayDataCube(output)