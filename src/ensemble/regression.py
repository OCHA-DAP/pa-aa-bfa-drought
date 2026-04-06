"""
Per-pixel Ridge regression for the multi-model ensemble.

Both predictors and predictand are assumed to be z-scored before calling
these functions (no intercept is fitted).

Usage
-----
    from src.ensemble.regression import fit_ridge_per_pixel, predict_per_pixel

    coefs, alphas, loo_preds = fit_ridge_per_pixel(X_z, y_z)
    y_hat = predict_per_pixel(X_new_z, coefs)
"""

import numpy as np
import xarray as xr
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.model_selection import KFold

# Log-spaced regularization grid for alpha selection
DEFAULT_ALPHAS = np.logspace(-2, 4, 50)
DEFAULT_LASSO_ALPHAS = np.logspace(
    -3, 1, 50
)  # 0.001–10; Lasso shrinks to 0 fast


def _fit_pixel(
    X: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a RidgeCV model for a single pixel.

    Parameters
    ----------
    X : ndarray, shape (n_years, n_models)
        Predictor matrix (z-scored ensemble means).
    y : ndarray, shape (n_years,)
        Observed z-scores.
    alphas : ndarray
        Regularization strengths to try.

    Returns
    -------
    coefs : ndarray, shape (n_models,)
    best_alpha : float
    loo_preds : ndarray, shape (n_years,)
    """
    # Skip pixels with no valid data
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < 3:
        n_models = X.shape[1]
        return (
            np.full(n_models, np.nan),
            np.nan,
            np.full(len(y), np.nan),
        )

    X_v, y_v = X[valid], y[valid]

    model = RidgeCV(
        alphas=alphas,
        fit_intercept=False,
    )  # cv=None → GCV (avoids NaN scores from single-point LOO folds)
    model.fit(X_v, y_v)

    coefs = model.coef_
    best_alpha = float(model.alpha_)

    # Explicitly refit with LOO to obtain out-of-sample predictions
    loo_pred_v = _loo_predictions(X_v, y_v, best_alpha)

    loo_preds = np.full(len(y), np.nan)
    loo_preds[valid] = loo_pred_v
    return coefs, best_alpha, loo_preds


def _loo_predictions(
    X: np.ndarray, y: np.ndarray, alpha: float, estimator_cls=None
) -> np.ndarray:
    """Leave-one-out predictions for a fixed alpha (no intercept)."""
    from sklearn.linear_model import Ridge

    if estimator_cls is None:
        estimator_cls = Ridge
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = estimator_cls(alpha=alpha, fit_intercept=False)
        m.fit(X[mask], y[mask])
        preds[i] = m.predict(X[[i]])[0]
    return preds


def _fit_pixel_lasso(
    X: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a LassoCV model for a single pixel.

    Parameters
    ----------
    X : ndarray, shape (n_years, n_models)
        Predictor matrix (z-scored ensemble means).
    y : ndarray, shape (n_years,)
        Observed z-scores.
    alphas : ndarray
        Regularization strengths to try.

    Returns
    -------
    coefs : ndarray, shape (n_models,)
    best_alpha : float
    loo_preds : ndarray, shape (n_years,)
    """
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < 3:
        n_models = X.shape[1]
        return (
            np.full(n_models, np.nan),
            np.nan,
            np.full(len(y), np.nan),
        )

    X_v, y_v = X[valid], y[valid]

    model = LassoCV(alphas=alphas, cv=KFold(n_splits=5), fit_intercept=False)
    model.fit(X_v, y_v)

    coefs = model.coef_
    best_alpha = float(model.alpha_)

    loo_pred_v = _loo_predictions(X_v, y_v, best_alpha, estimator_cls=Lasso)

    loo_preds = np.full(len(y), np.nan)
    loo_preds[valid] = loo_pred_v
    return coefs, best_alpha, loo_preds


def fit_ridge_per_pixel(
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    alphas: np.ndarray = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Fit per-pixel Ridge regression across the full BFA raster.

    Parameters
    ----------
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored ensemble predictor stack.
    y_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observation stack.
    alphas : array-like, optional
        Regularization strengths.  Defaults to a log-spaced grid.

    Returns
    -------
    coefs : xr.DataArray, dims (model, lat, lon)
        Per-pixel regression coefficients.
    best_alphas : xr.DataArray, dims (lat, lon)
        Per-pixel selected regularization strength.
    loo_preds : xr.DataArray, dims (year, lat, lon)
        Leave-one-out cross-validated predictions.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # Align years between X and y
    common_years = np.intersect1d(X_z.year.values, y_z.year.values)
    X_z = X_z.sel(year=common_years)
    y_z = y_z.sel(year=common_years)

    # Transpose to ensure correct core dimension order for apply_ufunc:
    # X_z: (lat, lon, year, model)  y_z: (lat, lon, year)
    X_z = X_z.transpose("lat", "lon", "year", "model")
    y_z = y_z.transpose("lat", "lon", "year")

    def _wrapper(x_pixel, y_pixel):
        # x_pixel: (year, model), y_pixel: (year,)
        coefs, alpha, loo = _fit_pixel(x_pixel, y_pixel, alphas)
        return coefs, np.array([alpha]), loo

    n_models = len(X_z.model)
    n_years = len(common_years)

    coefs_np = np.full((len(X_z.lat), len(X_z.lon), n_models), np.nan)
    alphas_np = np.full((len(X_z.lat), len(X_z.lon)), np.nan)
    loo_np = np.full((len(X_z.lat), len(X_z.lon), n_years), np.nan)

    X_vals = X_z.values  # (lat, lon, year, model)
    y_vals = y_z.values  # (lat, lon, year)

    for i in range(len(X_z.lat)):
        for j in range(len(X_z.lon)):
            c, a, p = _fit_pixel(X_vals[i, j], y_vals[i, j], alphas)
            coefs_np[i, j] = c
            alphas_np[i, j] = a
            loo_np[i, j] = p

    coefs = xr.DataArray(
        coefs_np.transpose(2, 0, 1),  # (model, lat, lon)
        dims=["model", "lat", "lon"],
        coords={
            "model": X_z.model,
            "lat": X_z.lat,
            "lon": X_z.lon,
        },
        name="coefs",
    )
    best_alphas = xr.DataArray(
        alphas_np,
        dims=["lat", "lon"],
        coords={"lat": X_z.lat, "lon": X_z.lon},
        name="alpha",
    )
    loo_preds = xr.DataArray(
        loo_np.transpose(2, 0, 1),  # (year, lat, lon)
        dims=["year", "lat", "lon"],
        coords={
            "year": common_years,
            "lat": X_z.lat,
            "lon": X_z.lon,
        },
        name="loo_preds",
    )
    return coefs, best_alphas, loo_preds


def fit_lasso_per_pixel(
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    alphas: np.ndarray = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Fit per-pixel Lasso regression across the full BFA raster.

    Uses 5-fold CV (LassoCV) for alpha selection, then explicit LOO for
    out-of-sample predictions.  Lasso produces sparse coefficients
    (typically 1–2 non-zero per pixel), which may outperform Ridge in
    pixels where one model clearly dominates.

    Parameters
    ----------
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored ensemble predictor stack.
    y_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observation stack.
    alphas : array-like, optional
        Regularization strengths.  Defaults to a log-spaced grid (0.001–10).

    Returns
    -------
    coefs : xr.DataArray, dims (model, lat, lon)
        Per-pixel regression coefficients (sparse).
    best_alphas : xr.DataArray, dims (lat, lon)
        Per-pixel selected regularization strength.
    loo_preds : xr.DataArray, dims (year, lat, lon)
        Leave-one-out cross-validated predictions.
    """
    if alphas is None:
        alphas = DEFAULT_LASSO_ALPHAS

    common_years = np.intersect1d(X_z.year.values, y_z.year.values)
    X_z = X_z.sel(year=common_years)
    y_z = y_z.sel(year=common_years)

    X_z = X_z.transpose("lat", "lon", "year", "model")
    y_z = y_z.transpose("lat", "lon", "year")

    n_models = len(X_z.model)
    n_years = len(common_years)

    coefs_np = np.full((len(X_z.lat), len(X_z.lon), n_models), np.nan)
    alphas_np = np.full((len(X_z.lat), len(X_z.lon)), np.nan)
    loo_np = np.full((len(X_z.lat), len(X_z.lon), n_years), np.nan)

    X_vals = X_z.values  # (lat, lon, year, model)
    y_vals = y_z.values  # (lat, lon, year)

    for i in range(len(X_z.lat)):
        for j in range(len(X_z.lon)):
            c, a, p = _fit_pixel_lasso(X_vals[i, j], y_vals[i, j], alphas)
            coefs_np[i, j] = c
            alphas_np[i, j] = a
            loo_np[i, j] = p

    coefs = xr.DataArray(
        coefs_np.transpose(2, 0, 1),  # (model, lat, lon)
        dims=["model", "lat", "lon"],
        coords={
            "model": X_z.model,
            "lat": X_z.lat,
            "lon": X_z.lon,
        },
        name="lasso_coefs",
    )
    best_alphas = xr.DataArray(
        alphas_np,
        dims=["lat", "lon"],
        coords={"lat": X_z.lat, "lon": X_z.lon},
        name="lasso_alpha",
    )
    loo_preds = xr.DataArray(
        loo_np.transpose(2, 0, 1),  # (year, lat, lon)
        dims=["year", "lat", "lon"],
        coords={
            "year": common_years,
            "lat": X_z.lat,
            "lon": X_z.lon,
        },
        name="lasso_loo_preds",
    )
    return coefs, best_alphas, loo_preds


def fit_ridge_global(
    da_fc_z: xr.DataArray,
    da_era5_z: xr.DataArray,
    alphas: np.ndarray = None,
) -> tuple[xr.DataArray, xr.DataArray, float]:
    """
    Fit a single global Ridge regression pooling all valid (year, pixel) pairs.

    A global model avoids the low obs/predictor ratio of per-pixel fits (~2.7)
    by treating every (year, pixel) pair as an independent observation with a
    shared coefficient vector.  The trade-off is losing spatial heterogeneity
    in model weights.

    Parameters
    ----------
    da_fc_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored ensemble predictor stack.
    da_era5_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observation stack.
    alphas : array-like, optional
        Regularization strengths.  Defaults to a log-spaced grid.

    Returns
    -------
    loo_preds : xr.DataArray, dims (year, lat, lon)
        Leave-one-out cross-validated predictions (LOO over years).
    global_coefs : xr.DataArray, dims (model,)
        Single global coefficient vector fitted on all pooled data.
    global_alpha : float
        Regularization strength selected by RidgeCV on pooled data.
    """
    from sklearn.linear_model import Ridge

    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # Align years
    common_years = np.intersect1d(da_fc_z.year.values, da_era5_z.year.values)
    da_fc_z = da_fc_z.sel(year=common_years)
    da_era5_z = da_era5_z.sel(year=common_years)

    n_years = len(common_years)
    n_models = len(da_fc_z.model)

    da_fc_z = da_fc_z.transpose("year", "model", "lat", "lon")
    da_era5_z = da_era5_z.transpose("year", "lat", "lon")

    X_vals = da_fc_z.values  # (year, model, lat, lon)
    y_vals = da_era5_z.values  # (year, lat, lon)

    n_lat = len(da_fc_z.lat)
    n_lon = len(da_fc_z.lon)
    n_pixels = n_lat * n_lon

    # Reshape: X -> (year, pixel, model), y -> (year, pixel)
    X_all = X_vals.transpose(0, 2, 3, 1).reshape(n_years, n_pixels, n_models)
    y_all = y_vals.reshape(n_years, n_pixels)

    # Valid-pixel mask: y non-NaN and X finite in ALL years
    valid_pixels = np.all(np.isfinite(y_all), axis=0) & np.all(
        np.all(np.isfinite(X_all), axis=2), axis=0
    )

    X_valid = X_all[:, valid_pixels, :]  # (year, n_valid, model)
    y_valid = y_all[:, valid_pixels]  # (year, n_valid)
    n_valid = int(valid_pixels.sum())

    # Select alpha via RidgeCV on full pooled (year*pixel, model) data
    X_pool = X_valid.reshape(-1, n_models)
    y_pool = y_valid.reshape(-1)
    cv_model = RidgeCV(alphas=alphas, fit_intercept=False)
    cv_model.fit(X_pool, y_pool)
    best_alpha = float(cv_model.alpha_)

    # LOO over years: train on all other years × all valid pixels, predict year t
    loo_preds_valid = np.full((n_years, n_valid), np.nan)
    for t in range(n_years):
        mask = np.ones(n_years, dtype=bool)
        mask[t] = False
        X_train = X_valid[mask].reshape(-1, n_models)
        y_train = y_valid[mask].reshape(-1)
        m = Ridge(alpha=best_alpha, fit_intercept=False)
        m.fit(X_train, y_train)
        loo_preds_valid[t] = m.predict(X_valid[t])

    # Global coefficients from full-data fit
    final_model = Ridge(alpha=best_alpha, fit_intercept=False)
    final_model.fit(X_pool, y_pool)
    global_coefs_np = final_model.coef_  # (n_models,)

    # Reconstruct (year, lat, lon), NaN for non-valid pixels
    loo_preds_np = np.full((n_years, n_pixels), np.nan)
    loo_preds_np[:, valid_pixels] = loo_preds_valid
    loo_preds_np = loo_preds_np.reshape(n_years, n_lat, n_lon)

    loo_preds = xr.DataArray(
        loo_preds_np,
        dims=["year", "lat", "lon"],
        coords={
            "year": common_years,
            "lat": da_fc_z.lat,
            "lon": da_fc_z.lon,
        },
        name="global_loo_preds",
    )
    global_coefs = xr.DataArray(
        global_coefs_np,
        dims=["model"],
        coords={"model": da_fc_z.model},
        name="global_coefs",
    )
    return loo_preds, global_coefs, best_alpha


def predict_per_pixel(
    X_new_z: xr.DataArray,
    coefs: xr.DataArray,
) -> xr.DataArray:
    """
    Apply stored per-pixel coefficients to a new z-scored predictor.

    Parameters
    ----------
    X_new_z : xr.DataArray, dims (model, lat, lon) or (year, model, lat, lon)
        New z-scored ensemble forecast(s).
    coefs : xr.DataArray, dims (model, lat, lon)
        Coefficients from :func:`fit_ridge_per_pixel`.

    Returns
    -------
    xr.DataArray
        Predicted z-score anomaly with dims (lat, lon) or (year, lat, lon).
    """
    # Dot product over model dimension
    y_hat = (X_new_z * coefs).sum(dim="model")
    y_hat.name = "precip_z_hat"
    return y_hat
