import numpy as np
import xarray as xr


def upsample_dataarray(
    da: xr.DataArray,
    resolution: float = 0.1,
    x_var: str = "x",
    y_var: str = "y",
) -> xr.DataArray:
    new_x = np.arange(da[x_var].min(), da[x_var].max(), resolution)
    new_y = np.arange(da[y_var].min(), da[y_var].max(), resolution)
    return da.interp(
        {x_var: new_x, y_var: new_y},
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
