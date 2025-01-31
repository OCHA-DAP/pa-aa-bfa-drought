import numpy as np
import xarray as xr


def upsample_dataarray(
    da: xr.DataArray, resolution: float = 0.1
) -> xr.DataArray:
    new_lat = np.arange(da.latitude.min(), da.latitude.max(), resolution)
    new_lon = np.arange(da.longitude.min(), da.longitude.max(), resolution)
    return da.interp(latitude=new_lat, longitude=new_lon, method="nearest")
