import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from openeo.udf import XarrayDataCube

def compute_distance(mask: np.ndarray) -> np.ndarray:
    inverse_mask = ~mask
    return distance_transform_edt(inverse_mask).astype(np.float32)

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    data = cube.get_array()
    t_coords = data.coords['t'].values

    result = []
    for t in t_coords:
        mask = data.sel(t=t).data.astype(bool)
        distance = compute_distance(mask)

        # normalize
        min_val = np.nanmin(distance)
        max_val = np.nanmax(distance)
        norm_distance = (distance - min_val) / (max_val - min_val) if max_val > min_val else distance

        # Insert a dummy 'bands' dimension
        da = xr.DataArray(
            norm_distance[np.newaxis, :, :],  # Add bands axis
            coords={
                't': [t],
                'bands': ['distance'],
                'y': data.coords['y'],
                'x': data.coords['x']
            },
            dims=['t', 'bands', 'y', 'x']
        )

        result.append(da)

    normalized = xr.concat(result, dim='t')
    return XarrayDataCube(normalized)
