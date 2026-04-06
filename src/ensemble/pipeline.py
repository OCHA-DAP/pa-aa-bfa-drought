"""
End-to-end MME pipeline: fit and predict.

The pipeline ties together loading, normalization, regression, and
persistence of fitted parameters in Azure Blob Storage.

Usage
-----
    from src.ensemble.pipeline import fit, predict

    # Train on hindcast data
    fit_result = fit(issue_month=4, valid_months=[7, 8, 9])

    # Apply to a new real-time forecast (xr.DataArray with model dim)
    forecast_z = predict(X_new, issue_month=4)
"""

import io
import logging

import numpy as np
import xarray as xr

from src.datasources.era5 import open_era5_rasters
from src.ensemble.load import load_forecasts
from src.ensemble.normalize import apply_zscores, compute_zscores
from src.ensemble.regression import fit_ridge_per_pixel, predict_per_pixel
from src.ensemble.skill import skill_vs_baseline
from src.utils import blob_utils

logger = logging.getLogger(__name__)

_FIT_BLOB_TEMPLATE = (
    "{prefix}/processed/ensemble/mme_fit_i{issue_month:02d}.nc"
)


def _fit_blob_name(issue_month: int) -> str:
    return _FIT_BLOB_TEMPLATE.format(
        prefix=blob_utils.PROJECT_PREFIX, issue_month=issue_month
    )


def _load_era5_jas(valid_months: list[int], years: np.ndarray) -> xr.DataArray:
    """
    Load ERA5 precipitation, aggregate over valid_months, and return a
    DataArray with dims (year, lat, lon).
    """
    da_era5 = open_era5_rasters(months=valid_months)
    # open_era5_rasters returns dims (year, issued_month, x, y)
    # Sum over the season and select the hindcast years
    da_jas = da_era5.sum(dim="issued_month")
    da_jas = da_jas.rename({"x": "lon", "y": "lat"})
    da_jas = da_jas.sel(year=years)
    return da_jas


def fit(
    issue_month: int = 4,
    valid_months: list[int] = None,
    centres: list[str] = None,
    stage: str = "dev",
    save_to_blob: bool = True,
) -> dict:
    """
    Train the per-pixel Ridge MME on hindcast data.

    Steps:
    1. Load each centre's hindcast DataArray from blob.
    2. Load ERA5 JAS observations from blob.
    3. Align to a common year range and pixel grid.
    4. Z-score both predictors and predictand per pixel.
    5. Fit per-pixel RidgeCV with LOO cross-validation.
    6. Compute skill maps.
    7. Optionally save fitted parameters to blob.

    Parameters
    ----------
    issue_month : int
        Forecast issue month (default 4 = April).
    valid_months : list of int, optional
        Valid season months (default [7, 8, 9] = JAS).
    centres : list of str, optional
        Models to include.  Defaults to all in CENTRE_SYSTEMS.
    stage : str
        Blob storage stage.
    save_to_blob : bool
        Whether to persist the fitted parameters as a netCDF blob.

    Returns
    -------
    dict with keys:
        - 'coefs'       : xr.DataArray (model, lat, lon)
        - 'alpha'       : xr.DataArray (lat, lon)
        - 'loo_preds'   : xr.DataArray (year, lat, lon)
        - 'X_clim_mean' : xr.DataArray (model, lat, lon)
        - 'X_clim_std'  : xr.DataArray (model, lat, lon)
        - 'y_clim_mean' : xr.DataArray (lat, lon)
        - 'y_clim_std'  : xr.DataArray (lat, lon)
        - 'skill'       : dict of skill DataArrays
    """
    if valid_months is None:
        valid_months = [7, 8, 9]

    logger.info("Loading ensemble hindcasts …")
    ds_fc = load_forecasts(
        centres=centres, valid_months=valid_months, stage=stage
    )
    # ds_fc.precip has dims (year, model, lat, lon)
    # Drop years where any model has NaN (centres have different year ranges)
    da_fc = ds_fc["precip"]
    valid_year_mask = da_fc.notnull().all(dim=["model", "lat", "lon"])
    da_fc = da_fc.sel(year=valid_year_mask)

    logger.info("Loading ERA5 observations …")
    common_years = da_fc.year.values
    da_era5 = _load_era5_jas(valid_months, common_years)

    # Regrid ERA5 to match forecast grid
    da_era5 = da_era5.interp(
        lat=da_fc.lat.values, lon=da_fc.lon.values, method="linear"
    )

    # Restrict to common years
    shared_years = np.intersect1d(da_fc.year.values, da_era5.year.values)
    da_fc = da_fc.sel(year=shared_years)
    da_era5 = da_era5.sel(year=shared_years)

    logger.info("Normalizing …")
    # Z-score each model independently over year dim
    da_fc_z, X_clim_mean, X_clim_std = compute_zscores(da_fc, dim="year")
    da_era5_z, y_clim_mean, y_clim_std = compute_zscores(da_era5, dim="year")

    logger.info("Fitting per-pixel Ridge regression …")
    coefs, alpha, loo_preds = fit_ridge_per_pixel(da_fc_z, da_era5_z)

    logger.info("Computing skill maps …")
    skill = skill_vs_baseline(loo_preds, da_fc_z, da_era5_z)

    result = {
        "coefs": coefs,
        "alpha": alpha,
        "loo_preds": loo_preds,
        "X_clim_mean": X_clim_mean,
        "X_clim_std": X_clim_std,
        "y_clim_mean": y_clim_mean,
        "y_clim_std": y_clim_std,
        "skill": skill,
    }

    if save_to_blob:
        _save_fit(result, issue_month=issue_month, stage=stage)

    return result


def _save_fit(result: dict, issue_month: int, stage: str = "dev") -> None:
    """Serialize fitted parameters to a single netCDF blob."""
    ds = xr.Dataset(
        {
            "coefs": result["coefs"],
            "alpha": result["alpha"],
            "loo_preds": result["loo_preds"],
            "X_clim_mean": result["X_clim_mean"],
            "X_clim_std": result["X_clim_std"],
            "y_clim_mean": result["y_clim_mean"],
            "y_clim_std": result["y_clim_std"],
            "mme_r": result["skill"]["mme_r"],
            "equal_weight_r": result["skill"]["equal_weight_r"],
        }
    )
    buf = io.BytesIO()
    ds.to_netcdf(buf)
    buf.seek(0)
    blob_name = _fit_blob_name(issue_month)
    blob_utils._upload_blob_data(buf, blob_name, stage=stage)
    logger.info("Saved fit to blob: %s", blob_name)


def load_fit(issue_month: int = 4, stage: str = "dev") -> xr.Dataset:
    """
    Load a previously saved fitted model from blob storage.

    Parameters
    ----------
    issue_month : int
        Issue month used when the model was fitted.
    stage : str
        Blob storage stage.

    Returns
    -------
    xr.Dataset
        Dataset containing coefs, alpha, loo_preds, clim stats, and skill.
    """
    blob_name = _fit_blob_name(issue_month)
    raw = blob_utils._load_blob_data(blob_name, stage=stage)
    ds = xr.open_dataset(io.BytesIO(raw), engine="scipy")
    return ds


def predict(
    X_new: xr.DataArray,
    issue_month: int = 4,
    stage: str = "dev",
    fit_ds: xr.Dataset = None,
    back_transform: bool = False,
) -> xr.DataArray:
    """
    Produce an MME forecast from a new real-time predictor array.

    Parameters
    ----------
    X_new : xr.DataArray, dims (model, lat, lon) or (year, model, lat, lon)
        Absolute precipitation ensemble means for each centre, on the same
        grid as the fitted model.
    issue_month : int
        Issue month, used to load the correct fitted model if ``fit_ds``
        is not provided.
    stage : str
        Blob storage stage.
    fit_ds : xr.Dataset, optional
        Pre-loaded fit dataset (avoids re-downloading from blob).
    back_transform : bool
        If True, back-transform the z-score prediction to mm using the
        ERA5 climatological mean and std.

    Returns
    -------
    xr.DataArray
        Predicted z-score anomaly (or mm if ``back_transform=True``).
    """
    if fit_ds is None:
        fit_ds = load_fit(issue_month=issue_month, stage=stage)

    X_clim_mean = fit_ds["X_clim_mean"]
    X_clim_std = fit_ds["X_clim_std"]
    coefs = fit_ds["coefs"]

    X_new_z = apply_zscores(X_new, X_clim_mean, X_clim_std)
    y_hat_z = predict_per_pixel(X_new_z, coefs)

    if back_transform:
        from src.ensemble.normalize import invert_zscores

        y_hat = invert_zscores(
            y_hat_z, fit_ds["y_clim_mean"], fit_ds["y_clim_std"]
        )
        return y_hat

    return y_hat_z
