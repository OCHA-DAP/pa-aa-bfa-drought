"""
Skill metric maps for the MME regression.

Usage
-----
    from src.ensemble.skill import pearson_r_map, rmse_map, skill_vs_baseline

    r = pearson_r_map(loo_preds, y_z)
    rmse = rmse_map(loo_preds, y_z)
    summary = skill_vs_baseline(loo_preds, X_z, y_z)
"""

import numpy as np
import xarray as xr
from scipy import stats


def pearson_r_map(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    dim: str = "year",
) -> xr.DataArray:
    """
    Per-pixel Pearson correlation between predictions and observations.

    Parameters
    ----------
    predicted : xr.DataArray, dims include ``dim``
        Predicted values (e.g. LOO predictions from Ridge regression).
    observed : xr.DataArray, dims include ``dim``
        Observed values on the same grid.
    dim : str
        Dimension along which to correlate (default 'year').

    Returns
    -------
    xr.DataArray, dims (lat, lon)
        Pearson r values in [-1, 1].
    """
    common_years = np.intersect1d(predicted[dim].values, observed[dim].values)
    pred = predicted.sel({dim: common_years})
    obs = observed.sel({dim: common_years})

    def _pearsonr(p, o):
        valid = np.isfinite(p) & np.isfinite(o)
        if valid.sum() < 3:
            return np.nan
        r, _ = stats.pearsonr(p[valid], o[valid])
        return r

    r_map = xr.apply_ufunc(
        _pearsonr,
        pred,
        obs,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    r_map.name = "pearson_r"
    return r_map


def rmse_map(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    dim: str = "year",
) -> xr.DataArray:
    """
    Per-pixel Root Mean Squared Error.

    Parameters
    ----------
    predicted, observed : xr.DataArray
        Predicted and observed values, must share ``dim``.
    dim : str
        Dimension along which to compute RMSE (default 'year').

    Returns
    -------
    xr.DataArray, dims (lat, lon)
        RMSE values.
    """
    common_years = np.intersect1d(predicted[dim].values, observed[dim].values)
    pred = predicted.sel({dim: common_years})
    obs = observed.sel({dim: common_years})

    rmse = np.sqrt(((pred - obs) ** 2).mean(dim=dim))
    rmse.name = "rmse"
    return rmse


def equal_weight_ensemble(
    X_z: xr.DataArray,
    dim: str = "model",
) -> xr.DataArray:
    """
    Simple equal-weight ensemble mean baseline.

    Parameters
    ----------
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored predictor stack.
    dim : str
        Model dimension to average over (default 'model').

    Returns
    -------
    xr.DataArray, dims (year, lat, lon)
    """
    ew = X_z.mean(dim=dim)
    ew.name = "equal_weight_mean"
    return ew


def skill_vs_baseline(
    loo_preds: xr.DataArray,
    X_z: xr.DataArray,
    y_z: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Compare MME LOO skill against baselines.

    Returns a dict with keys:
    - 'mme_r'         : Pearson r of MME LOO predictions
    - 'equal_weight_r': Pearson r of equal-weight ensemble mean
    - 'per_model_r'   : Pearson r for each individual model (model dim retained)

    Parameters
    ----------
    loo_preds : xr.DataArray, dims (year, lat, lon)
        LOO cross-validated MME predictions.
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored predictor stack (individual model forecasts).
    y_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observations.
    """
    mme_r = pearson_r_map(loo_preds, y_z)

    ew = equal_weight_ensemble(X_z)
    ew_r = pearson_r_map(ew, y_z)

    # Per-model skill
    model_r_list = []
    for model in X_z.model.values:
        r = pearson_r_map(X_z.sel(model=model), y_z)
        r = r.assign_coords(model=model).expand_dims("model")
        model_r_list.append(r)
    per_model_r = xr.concat(model_r_list, dim="model")
    per_model_r.name = "pearson_r"

    return {
        "mme_r": mme_r,
        "equal_weight_r": ew_r,
        "per_model_r": per_model_r,
    }


def nested_loo_best3(
    loo_preds: xr.DataArray,
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    min_r: float = 0.0,
) -> dict[str, xr.DataArray]:
    """
    Nested LOO evaluation of the three-way best-of (Ridge, sel-EW, best-single).

    For each held-out year, method selection is done on the N-1 training years;
    the winning method's prediction is used for the held-out year.

    Ridge's test prediction comes from the pre-computed ``loo_preds`` (no extra
    fitting).  Ridge's training-fold r is approximated by correlating ``loo_preds``
    over the N-1 training years — valid because each LOO pred was computed leaving
    that year out.

    Returns dict with keys:
    - 'nested_best3_preds'  : (year, lat, lon)
    - 'nested_best3_r'      : (lat, lon)
    - 'nested_best3_method' : (year, lat, lon) int  [0=ridge, 1=sel_ew, 2=best_single]
    """
    common_years = np.intersect1d(
        np.intersect1d(loo_preds.year.values, X_z.year.values),
        y_z.year.values,
    )
    X_z = X_z.sel(year=common_years)
    y_z = y_z.sel(year=common_years)
    loo_preds = loo_preds.sel(year=common_years)
    n = len(common_years)

    pred_list = []
    method_list = []

    for i in range(n):
        yr = common_years[i]
        train_years = common_years[np.arange(n) != i]

        X_train = X_z.sel(year=train_years)  # (year, model, lat, lon)
        y_train = y_z.sel(year=train_years)  # (year, lat, lon)
        X_test = X_z.sel(year=yr)  # (model, lat, lon)

        # Per-model r on training years
        per_model_r_train = xr.concat(
            [
                pearson_r_map(X_train.sel(model=m), y_train)
                .assign_coords(model=m)
                .expand_dims("model")
                for m in X_z.model.values
            ],
            dim="model",
        )  # (model, lat, lon)

        # sel-EW training r and test prediction
        _n_pass = (per_model_r_train > min_r).sum(dim="model")
        sel_ew_train = X_train.where(per_model_r_train > min_r).mean(
            dim="model", skipna=True
        )
        sel_ew_train = sel_ew_train.where(_n_pass > 0)
        sel_ew_r_train = pearson_r_map(sel_ew_train, y_train)
        sel_ew_test = X_test.where(per_model_r_train > min_r).mean(
            dim="model", skipna=True
        )
        sel_ew_test = sel_ew_test.where(_n_pass > 0)  # (lat, lon)

        # best-single training r and test prediction
        _best_idx = per_model_r_train.argmax("model")  # (lat, lon) int
        best_single_r_train = per_model_r_train.max("model")  # (lat, lon)
        best_single_test = X_test.isel(model=_best_idx)  # (lat, lon)

        # Ridge training r (approximate from existing LOO preds) and test prediction
        ridge_r_train = pearson_r_map(loo_preds.sel(year=train_years), y_train)
        ridge_test = loo_preds.sel(year=yr)  # (lat, lon)

        # Method selection on training data
        _all_r = xr.concat(
            [ridge_r_train, sel_ew_r_train, best_single_r_train],
            dim=xr.DataArray(
                ["ridge", "sel_ew", "best_single"], dims="method"
            ),
        )
        best_method = _all_r.argmax("method")  # (lat, lon) int

        pred_test = xr.where(
            best_method == 0,
            ridge_test,
            xr.where(best_method == 1, sel_ew_test, best_single_test),
        )

        pred_list.append(pred_test.assign_coords(year=yr).expand_dims("year"))
        method_list.append(
            best_method.assign_coords(year=yr).expand_dims("year")
        )

    nested_preds = xr.concat(pred_list, dim="year")
    nested_preds.name = "nested_best3_preds"
    nested_method = xr.concat(method_list, dim="year")
    nested_method.name = "nested_best3_method"
    nested_r = pearson_r_map(nested_preds, y_z)

    return {
        "nested_best3_preds": nested_preds,
        "nested_best3_r": nested_r,
        "nested_best3_method": nested_method,
    }


def nested_loo_selected_ew(
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    min_r: float = 0.0,
) -> xr.DataArray:
    """
    Nested LOO evaluation of the selected equal-weight ensemble.

    For each held-out year, model selection (r > min_r threshold) is applied
    on the N-1 training years only; the selected models' test-year mean is the
    prediction.  This gives an honest out-of-sample r comparable to ``mme_r``
    and ``nested_best3_r``.

    Parameters
    ----------
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored ensemble predictor stack.
    y_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observations.
    min_r : float
        Minimum per-model r on training years to include a model (default 0.0).

    Returns
    -------
    xr.DataArray, dims (lat, lon)
        Nested LOO Pearson r of the selected equal-weight ensemble.
    """
    common_years = np.intersect1d(X_z.year.values, y_z.year.values)
    X_z = X_z.sel(year=common_years)
    y_z = y_z.sel(year=common_years)
    n = len(common_years)

    pred_list = []
    for i in range(n):
        yr = common_years[i]
        train_years = common_years[np.arange(n) != i]

        X_train = X_z.sel(year=train_years)
        X_test = X_z.sel(year=yr)
        y_train = y_z.sel(year=train_years)

        per_model_r_train = xr.concat(
            [
                pearson_r_map(X_train.sel(model=m), y_train)
                .assign_coords(model=m)
                .expand_dims("model")
                for m in X_z.model.values
            ],
            dim="model",
        )

        n_pass = (per_model_r_train > min_r).sum(dim="model")
        pred = X_test.where(per_model_r_train > min_r).mean(
            dim="model", skipna=True
        )
        pred = pred.where(n_pass > 0)
        pred_list.append(pred.assign_coords(year=yr).expand_dims("year"))

    nested_preds = xr.concat(pred_list, dim="year")
    nested_preds.name = "nested_ew_preds"
    return pearson_r_map(nested_preds, y_z)


def model_of_models_tercile(
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    tercile: float = 1 / 3,
) -> dict[str, xr.DataArray]:
    """
    Fraction-of-models-below-tercile probabilistic forecast.

    No fitted parameters → in-sample evaluation is honest (no LOO needed).

    Returns
    -------
    dict with keys:
    - 'frac_below'  : (year, lat, lon)  forecast probability in [0, 1]
    - 'obs_below'   : (year, lat, lon)  1 if ERA5 below tercile, else 0
    - 'thresh'      : (lat, lon)        per-pixel lower tercile threshold
    - 'r'           : (lat, lon)        Pearson r (frac_below vs obs_below)
    - 'brier'       : (lat, lon)        Brier score = MSE(frac_below, obs_below)
    - 'brier_clim'  : (lat, lon)        Brier score of climatological p=1/3 baseline
    - 'bss'         : (lat, lon)        Brier skill score = 1 - brier/brier_clim
    """
    common_years = np.intersect1d(X_z.year.values, y_z.year.values)
    X_z = X_z.sel(year=common_years)
    y_z = y_z.sel(year=common_years)

    thresh = y_z.quantile(tercile, dim="year").drop_vars("quantile")
    frac_below = (X_z < thresh).mean(dim="model").astype(float)
    obs_below = (y_z < thresh).astype(float)

    r = pearson_r_map(frac_below, obs_below)
    brier = ((frac_below - obs_below) ** 2).mean(dim="year")
    brier_clim = float(tercile) * (
        1 - float(tercile)
    )  # scalar: p*(1-p) for p=1/3
    bss = 1 - brier / brier_clim

    return {
        "frac_below": frac_below,
        "obs_below": obs_below,
        "thresh": thresh,
        "r": r,
        "brier": brier,
        "brier_clim": brier_clim,
        "bss": bss,
    }


def selected_equal_weight(
    X_z: xr.DataArray,
    y_z: xr.DataArray,
    min_r: float = 0.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Per-pixel equal-weight mean of models whose in-sample Pearson r exceeds
    ``min_r``.

    Individual model forecasts have no fitted parameters, so in-sample r == LOO r
    (no cross-validation needed).  Pixels where no model passes return NaN
    (climatology).

    Parameters
    ----------
    X_z : xr.DataArray, dims (year, model, lat, lon)
        Z-scored ensemble predictor stack.
    y_z : xr.DataArray, dims (year, lat, lon)
        Z-scored ERA5 observations.
    min_r : float
        Minimum per-model or to include a model at a given pixel (default 0.0).

    Returns
    -------
    selected : xr.DataArray, dims (year, lat, lon)
        Equal-weight mean of selected models.
    per_model_r : xr.DataArray, dims (model, lat, lon)
        Per-model per-pixel Pearson r used for selection.
    """
    r_list = [
        pearson_r_map(X_z.sel(model=m), y_z)
        .assign_coords(model=m)
        .expand_dims("model")
        for m in X_z.model.values
    ]
    per_model_r = xr.concat(r_list, dim="model")

    # Average only models that pass the threshold; NaN where none pass
    selected = X_z.where(per_model_r > min_r).mean(dim="model", skipna=True)
    n_passing = (per_model_r > min_r).sum(dim="model")
    selected = selected.where(n_passing > 0)
    selected.name = "selected_ew"
    return selected, per_model_r
