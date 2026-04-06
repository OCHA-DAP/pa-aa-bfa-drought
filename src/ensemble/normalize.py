"""
Per-pixel z-score normalization for the MME regression.

All normalization is computed over the year dimension so that each pixel's
mean and standard deviation reflect local climatology.

Usage
-----
    from src.ensemble.normalize import compute_zscores, apply_zscores

    ds_z, clim_mean, clim_std = compute_zscores(ds)
    ds_new_z = apply_zscores(ds_new, clim_mean, clim_std)
"""

import xarray as xr


def compute_zscores(
    da: xr.DataArray,
    dim: str = "year",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Standardize a DataArray per-pixel over ``dim``.

    Parameters
    ----------
    da : xr.DataArray
        Input data.  Must contain ``dim``.
    dim : str
        Dimension along which to compute mean and std (default 'year').

    Returns
    -------
    da_z : xr.DataArray
        Standardized values.
    clim_mean : xr.DataArray
        Per-pixel mean used for standardization (drop the ``dim`` dimension).
    clim_std : xr.DataArray
        Per-pixel std used for standardization.
    """
    clim_mean = da.mean(dim=dim)
    clim_std = da.std(dim=dim)
    # Avoid division by zero at pixels with zero variance
    clim_std = clim_std.where(clim_std > 0, other=1.0)
    da_z = (da - clim_mean) / clim_std
    return da_z, clim_mean, clim_std


def apply_zscores(
    da: xr.DataArray,
    clim_mean: xr.DataArray,
    clim_std: xr.DataArray,
) -> xr.DataArray:
    """
    Apply pre-computed climatological statistics to a new DataArray.

    Useful for standardizing real-time forecasts using the hindcast
    climatology.

    Parameters
    ----------
    da : xr.DataArray
        New data to standardize.
    clim_mean, clim_std : xr.DataArray
        Statistics computed by :func:`compute_zscores`.

    Returns
    -------
    xr.DataArray
        Standardized DataArray.
    """
    return (da - clim_mean) / clim_std


def invert_zscores(
    da_z: xr.DataArray,
    clim_mean: xr.DataArray,
    clim_std: xr.DataArray,
) -> xr.DataArray:
    """
    Back-transform z-scores to original units.

    Parameters
    ----------
    da_z : xr.DataArray
        Standardized data.
    clim_mean, clim_std : xr.DataArray
        Statistics from :func:`compute_zscores`.

    Returns
    -------
    xr.DataArray
        Data in original units.
    """
    return da_z * clim_std + clim_mean
