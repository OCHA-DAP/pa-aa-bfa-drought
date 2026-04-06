import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import marimo as mo

    warnings.filterwarnings("ignore")

    from pathlib import Path

    import cartopy.crs as ccrs
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import ocha_stratus as stratus
    import xarray as xr

    import src as _src
    from src.constants import AOI_ADM2_PCODES_2026
    from src.datasources import codab
    from src.datasources.era5 import open_era5_rasters
    from src.ensemble import (
        download,
        load,
        normalize,
        pipeline,
        regression,
        skill,
    )
    from src.utils.blob_utils import PROJECT_PREFIX

    NOTEBOOKS_DIR = Path(_src.__file__).resolve().parent.parent / "notebooks"
    return (
        AOI_ADM2_PCODES_2026,
        NOTEBOOKS_DIR,
        PROJECT_PREFIX,
        ccrs,
        codab,
        mo,
        normalize,
        np,
        open_era5_rasters,
        plt,
        regression,
        skill,
        stratus,
        xr,
    )


@app.cell
def _(AOI_ADM2_PCODES_2026, codab):
    adm2 = codab.load_codab_from_blob(admin_level=2)
    adm2_aoi = adm2[adm2["ADM2_PCODE"].isin(AOI_ADM2_PCODES_2026)]
    return (adm2_aoi,)


@app.cell
def _(PROJECT_PREFIX, stratus):
    blob_name = f"{PROJECT_PREFIX}/raw/codab/bfa_admin_boundaries.shp.zip"

    adm1_new = stratus.load_shp_from_blob(
        blob_name, shapefile="bfa_admin1.shp"
    )
    return (adm1_new,)


@app.cell
def _(adm1_new):
    adm1_new.plot()
    return


@app.cell
def _(adm2_aoi):
    adm2_aoi.plot()
    return


@app.cell
def _(adm1_new, adm2_aoi, ccrs):
    def add_boundaries(ax):
        adm1_new.boundary.plot(
            ax=ax, linewidth=0.3, color="black", transform=ccrs.PlateCarree()
        )
        adm2_aoi.boundary.plot(
            ax=ax, linewidth=1.0, color="black", transform=ccrs.PlateCarree()
        )

    return (add_boundaries,)


@app.cell
def _():
    reference_years = range(1993, 2016 + 1)
    return (reference_years,)


@app.cell
def _(NOTEBOOKS_DIR, np, open_era5_rasters, reference_years, xr):
    _cache = NOTEBOOKS_DIR / "data/cache/inputs.nc"
    _cache.parent.mkdir(exist_ok=True)

    if _cache.exists():
        _ds_cache = xr.open_dataset(_cache)
        da_fc = _ds_cache["da_fc"]
        da_era5 = _ds_cache["da_era5"]
        print("Loaded da_fc and da_era5 from cache:", _cache)
    else:
        _dss = []
        _target_ds = None

        for _provider in [
            "ecmwf",
            "bom",
            "jma",
            "meteofrance",
            "ukmetoffice",
            "cmcc",
            "dwd",
            "ncep",
            "eccc",
        ]:
            _filepath = (
                NOTEBOOKS_DIR / f"data/apr_issued_jas_valid/{_provider}.grib"
            )
            _ds_in = xr.load_dataset(
                str(_filepath),
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
            _ds_in_mean = (
                _ds_in["tprate"]
                .mean(dim=["number", "step"])
                .groupby("time.year")
                .mean("time")
                .assign_coords(provider=_provider)
            )
            if _target_ds is None:
                _target_ds = _ds_in_mean
            else:
                _ds_in_mean = _ds_in_mean.interp(
                    latitude=_target_ds.latitude,
                    longitude=_target_ds.longitude,
                )
            _dss.append(_ds_in_mean)

        ds_combined = xr.concat(_dss, dim="provider")
        ds_combined_ref = ds_combined.sel(year=reference_years)
        _ds_era5 = open_era5_rasters(months=[7, 8, 9])

        da_fc = ds_combined_ref.rename(
            {"latitude": "lat", "longitude": "lon", "provider": "model"}
        )
        da_era5 = (
            _ds_era5.sum("issued_month")
            .rename({"x": "lon", "y": "lat"})
            .interp(
                lat=da_fc.lat.values, lon=da_fc.lon.values, method="linear"
            )
            .compute()
        )
        _shared_years = np.intersect1d(da_fc.year.values, da_era5.year.values)
        da_fc = da_fc.sel(year=_shared_years)
        da_era5 = da_era5.sel(year=_shared_years)

        xr.Dataset({"da_fc": da_fc, "da_era5": da_era5}).to_netcdf(_cache)
        print("Computed and cached to:", _cache)

    print("Forecast:", da_fc)
    print("ERA5:    ", da_era5)
    return da_era5, da_fc


@app.cell
def _(da_era5, da_fc, normalize):
    da_fc_z, X_clim_mean, X_clim_std = normalize.compute_zscores(
        da_fc, dim="year"
    )
    da_era5_z, y_clim_mean, y_clim_std = normalize.compute_zscores(
        da_era5, dim="year"
    )

    print(da_fc_z)
    print(
        "Mean of z-scores over year (should ≈ 0):",
        float(da_fc_z.mean("year").mean()),
    )
    return da_era5_z, da_fc_z


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, da_fc_z, regression):
    coefs, alpha, loo_preds = regression.fit_ridge_per_pixel(
        da_fc_z, da_era5_z
    )
    print("Coefs: ", coefs)
    print("Alphas:", alpha)
    return alpha, coefs, loo_preds


@app.cell
def _(da_era5_z, da_fc_z, regression, skill):
    global_loo_preds, global_coefs, global_alpha_val = (
        regression.fit_ridge_global(da_fc_z, da_era5_z)
    )
    global_r = skill.pearson_r_map(global_loo_preds, da_era5_z)
    print("Global alpha:", global_alpha_val)
    print("Global coefs:", global_coefs.values)
    return (global_r,)


@app.cell
def _(add_boundaries, ccrs, global_r, mme_r, plt):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    _delta = global_r - mme_r
    for _ax, _da, _title, _vmin, _vmax, _cmap in zip(
        _axes,
        [mme_r, global_r, _delta],
        [
            "Per-pixel Ridge LOO r",
            "Global Ridge LOO r",
            "Δr (global − per-pixel)",
        ],
        [-0.8, -0.8, -0.3],
        [0.8, 0.8, 0.3],
        ["RdBu", "RdBu", "RdBu"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")
    plt.suptitle("Global vs per-pixel Ridge — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(add_boundaries, ccrs, da_era5_z, da_fc_z, loo_preds, plt, skill):
    skill_maps = skill.skill_vs_baseline(loo_preds, da_fc_z, da_era5_z)
    mme_r = skill_maps["mme_r"]
    ew_r = skill_maps["equal_weight_r"]
    per_model_r = skill_maps["per_model_r"]

    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    _cmap, _vmin, _vmax = "RdBu", -0.8, 0.8

    for _ax, _da, _title in zip(
        _axes,
        [mme_r, ew_r, per_model_r.mean("model")],
        ["MME Ridge (LOO r)", "Equal-weight mean (r)", "Per-model mean r"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")

    plt.suptitle("JAS Forecast Skill — Apr-issued", y=1.02)
    plt.tight_layout()

    _fig
    return ew_r, mme_r, per_model_r


@app.cell
def _():
    return


@app.cell
def _(add_boundaries, alpha, ccrs, np, plt):
    _fig, _ax = plt.subplots(
        figsize=(6, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    add_boundaries(_ax)
    _im = _ax.pcolormesh(
        alpha.lon,
        alpha.lat,
        np.log10(alpha.values),
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(_im, ax=_ax, label="log10(α)")
    _ax.set_title("Regularisation strength (log10 α)")
    plt.tight_layout()

    _fig
    return


@app.cell
def _(add_boundaries, ccrs, coefs, np, plt):
    _models = coefs.model.values
    _n = len(_models)
    _ncols = int(np.ceil(np.sqrt(_n)))
    _nrows = int(np.ceil(_n / _ncols))
    _fig, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(4 * _ncols, 4 * _nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _axes_flat = np.array(_axes).flatten()

    for _ax, _m in zip(_axes_flat, _models):
        add_boundaries(_ax)
        _c = coefs.sel(model=_m)
        _im = _ax.pcolormesh(
            _c.lon,
            _c.lat,
            _c.values,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_m)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="β")

    for _ax in _axes_flat[_n:]:
        _ax.set_visible(False)

    plt.suptitle("Per-model regression coefficients β", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(add_boundaries, ccrs, da_fc_z, np, per_model_r, plt):
    _models = da_fc_z.model.values
    _n = len(_models)
    _ncols = int(np.ceil(np.sqrt(_n)))
    _nrows = int(np.ceil(_n / _ncols))
    _fig, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(4 * _ncols, 4 * _nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _axes_flat = np.array(_axes).flatten()

    _cmap, _vmin, _vmax = "RdBu", -0.8, 0.8
    for _ax, _m in zip(_axes_flat, _models):
        add_boundaries(_ax)
        _r = per_model_r.sel(model=_m)
        _im = _ax.pcolormesh(
            _r.lon,
            _r.lat,
            _r.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_m)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")

    for _ax in _axes_flat[_n:]:
        _ax.set_visible(False)

    plt.suptitle("Per-model in-sample Pearson r vs ERA5", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, da_fc_z, skill):
    # Compute selected equal-weight ensemble (only models with r > 0 per pixel)
    sel_ew, per_model_r_ew = skill.selected_equal_weight(
        da_fc_z, da_era5_z, min_r=0.0
    )
    sel_ew_r = skill.pearson_r_map(sel_ew, da_era5_z)
    n_selected = (per_model_r_ew > 0).sum(dim="model")
    return n_selected, per_model_r_ew, sel_ew, sel_ew_r


@app.cell
def _(add_boundaries, ccrs, ew_r, mme_r, plt, sel_ew_r):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    _cmap, _vmin, _vmax = "RdBu", -0.8, 0.8

    for _ax, _da, _title in zip(
        _axes,
        [ew_r, sel_ew_r, mme_r],
        [
            "Equal-weight (all models)",
            "Selected equal-weight (r > 0)",
            "MME Ridge (LOO)",
        ],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")

    plt.suptitle("Skill comparison — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(da_fc_z, mo):
    model_picker = mo.ui.multiselect(
        options=list(da_fc_z.model.values),
        value=list(da_fc_z.model.values),
        label="Models to include in hand-picked EW",
    )
    model_picker
    return (model_picker,)


@app.cell
def _(da_era5_z, da_fc_z, model_picker, skill):
    if model_picker.value:
        picked_ew = da_fc_z.sel(model=model_picker.value).mean(dim="model")
        picked_ew_r = skill.pearson_r_map(picked_ew, da_era5_z)
    else:
        picked_ew_r = da_era5_z.isel(year=0) * float("nan")
    return (picked_ew_r,)


@app.cell
def _(add_boundaries, ccrs, mme_r, picked_ew_r, plt, sel_ew_r):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for _ax, _da, _title in zip(
        _axes,
        [picked_ew_r, sel_ew_r, mme_r],
        ["Hand-picked EW r (honest)", "Selected EW (r > 0)", "Ridge LOO r"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap="RdBu",
            vmin=-0.8,
            vmax=0.8,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")
    plt.suptitle("Hand-picked equal-weight ensemble — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(add_boundaries, ccrs, da_fc_z, n_selected, np, per_model_r_ew, plt):
    _models = da_fc_z.model.values
    _n = len(_models)
    _total = _n + 1  # models + count panel
    _ncols = int(np.ceil(np.sqrt(_total)))
    _nrows = int(np.ceil(_total / _ncols))
    _fig, _axes = plt.subplots(
        _nrows,
        _ncols,
        figsize=(4 * _ncols, 4 * _nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _axes_flat = np.array(_axes).flatten()

    for _ax, _m in zip(_axes_flat[:_n], _models):
        add_boundaries(_ax)
        _selected_mask = (per_model_r_ew.sel(model=_m) > 0).astype(float)
        _selected_mask = _selected_mask.where(
            per_model_r_ew.sel(model=_m).notnull()
        )
        _im = _ax.pcolormesh(
            _selected_mask.lon,
            _selected_mask.lat,
            _selected_mask.values,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(f"{_m}\n(r > 0?)")
        plt.colorbar(_im, ax=_ax, shrink=0.7, ticks=[0, 1], label="selected")

    _ax_last = _axes_flat[_n]
    add_boundaries(_ax_last)
    _cmap_count = plt.get_cmap("YlGn", _n + 1)
    _im = _ax_last.pcolormesh(
        n_selected.lon,
        n_selected.lat,
        n_selected.values.astype(float),
        cmap=_cmap_count,
        vmin=-0.5,
        vmax=_n + 0.5,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(
        _im,
        ax=_ax_last,
        shrink=0.7,
        ticks=range(_n + 1),
        label="# models selected",
    )
    _ax_last.set_title("# models with r > 0")

    for _ax in _axes_flat[_total:]:
        _ax.set_visible(False)

    plt.suptitle("Per-pixel model selection (r > 0 threshold)", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(da_fc_z, per_model_r):
    # per_model_r is already the honest r for each individual model (no fitted params)
    best_model_idx = per_model_r.argmax("model")  # (lat, lon) int
    best_model_r = per_model_r.max("model")  # (lat, lon) float
    best_model_name = da_fc_z.model[best_model_idx]  # (lat, lon) string
    # Predictions: select per-pixel best model from z-scored forecasts
    best_single_pred = da_fc_z.isel(model=best_model_idx)  # (year, lat, lon)
    return best_model_r, best_single_pred


@app.cell
def _(best_model_r, best_single_pred, loo_preds, mme_r, sel_ew, sel_ew_r, xr):
    # Stack r scores; pick method with highest r per pixel
    _all_r = xr.concat(
        [mme_r, sel_ew_r, best_model_r],
        dim=xr.DataArray(["ridge", "sel_ew", "best_single"], dims="method"),
    )
    best3_method = _all_r.argmax("method")  # 0=ridge, 1=sel_ew, 2=best_single
    best3_r = _all_r.max("method")

    # Build best prediction array per pixel
    best3_pred = xr.where(
        best3_method == 0,
        loo_preds,
        xr.where(best3_method == 1, sel_ew, best_single_pred),
    )
    return best3_method, best3_r


@app.cell
def _(add_boundaries, best3_r, best_model_r, ccrs, mme_r, plt, sel_ew_r):
    # Four-panel skill comparison: Ridge LOO, selected-EW, best-single, best3
    _fig, _axes = plt.subplots(
        1, 4, figsize=(20, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for _ax, _da, _title in zip(
        _axes,
        [mme_r, sel_ew_r, best_model_r, best3_r],
        ["Ridge LOO r", "Selected-EW r", "Best-single r", "Best-of-3 r"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap="RdBu",
            vmin=-0.8,
            vmax=0.8,
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")
        _ax.set_title(_title)

    plt.suptitle("Three-way skill comparison — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(add_boundaries, best3_method, ccrs, mme_r, plt):
    import matplotlib.colors as _mcolors

    # 3-colour discrete colormap: 0=Ridge, 1=sel-EW, 2=best-single
    _cmap3 = _mcolors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
    _bounds = [-0.5, 0.5, 1.5, 2.5]
    _norm3 = _mcolors.BoundaryNorm(_bounds, _cmap3.N)

    _fig, _ax = plt.subplots(
        1, 1, figsize=(7, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    add_boundaries(_ax)
    _method_plot = best3_method.astype(float).where(mme_r.notnull())
    _im = _ax.pcolormesh(
        _method_plot.lon,
        _method_plot.lat,
        _method_plot.values,
        cmap=_cmap3,
        norm=_norm3,
        transform=ccrs.PlateCarree(),
    )
    _cb = plt.colorbar(_im, ax=_ax, shrink=0.7, ticks=[0, 1, 2])
    _cb.set_ticklabels(["Ridge", "sel-EW", "best-single"])
    _ax.set_title("Three-way method selection")

    plt.suptitle("Best-of-3 method per pixel — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, da_fc_z, loo_preds, skill):
    _nested = skill.nested_loo_best3(loo_preds, da_fc_z, da_era5_z)
    nested_best3_r = _nested["nested_best3_r"]
    nested_best3_preds = _nested["nested_best3_preds"]
    nested_best3_method = _nested["nested_best3_method"]
    nested_ew_r = skill.nested_loo_selected_ew(da_fc_z, da_era5_z)
    return nested_best3_r, nested_ew_r


@app.cell
def _(
    add_boundaries,
    best3_r,
    ccrs,
    mme_r,
    nested_best3_r,
    nested_ew_r,
    plt,
    sel_ew_r,
):
    _fig, _axes = plt.subplots(
        1, 5, figsize=(25, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for _ax, _da, _title, _vmin, _vmax, _cmap in zip(
        _axes,
        [
            mme_r,
            sel_ew_r,
            nested_ew_r,
            nested_best3_r,
            best3_r - nested_best3_r,
        ],
        [
            "Ridge LOO r",
            "Naive sel-EW r",
            "Nested LOO sel-EW r",
            "Nested LOO best-of-3 r",
            "Optimism bias (naive − nested)",
        ],
        [-0.8, -0.8, -0.8, -0.8, -0.1],
        [0.8, 0.8, 0.8, 0.8, 0.5],
        ["RdBu", "RdBu", "RdBu", "RdBu", "OrRd"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar(_im, ax=_ax, shrink=0.7)
        _ax.set_title(_title)
    plt.suptitle("Honest vs naive skill — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(loo_preds, mme_r, sel_ew, sel_ew_r):
    # Best-of: use Ridge LOO where it beats selected-EW, otherwise fall back to selected-EW
    # Both are on equal footing: Ridge is evaluated LOO; selected-EW has no fitted
    # parameters so its in-sample r == LOO r.
    ridge_wins = mme_r > sel_ew_r
    best_pred = loo_preds.where(ridge_wins, other=sel_ew)
    best_r = mme_r.where(ridge_wins, other=sel_ew_r)
    return best_r, ridge_wins


@app.cell
def _(add_boundaries, best_r, ccrs, ew_r, mme_r, plt, ridge_wins):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Left: which method was chosen
    _ax = _axes[0]
    add_boundaries(_ax)
    _method = ridge_wins.astype(float).where(mme_r.notnull())
    _im = _ax.pcolormesh(
        _method.lon,
        _method.lat,
        _method.values,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(
        _im, ax=_ax, shrink=0.7, ticks=[0, 1], label="0=sel-EW  1=Ridge"
    )
    _ax.set_title("Method selected")

    # Middle: best-of skill
    _ax = _axes[1]
    add_boundaries(_ax)
    _im = _ax.pcolormesh(
        best_r.lon,
        best_r.lat,
        best_r.values,
        cmap="RdBu",
        vmin=-0.8,
        vmax=0.8,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")
    _ax.set_title("Best-of r")

    # Right: improvement over plain equal-weight
    _ax = _axes[2]
    add_boundaries(_ax)
    _delta = best_r - ew_r
    _im = _ax.pcolormesh(
        _delta.lon,
        _delta.lat,
        _delta.values,
        cmap="RdBu",
        vmin=-0.3,
        vmax=0.3,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(_im, ax=_ax, shrink=0.7, label="Δr vs equal-weight")
    _ax.set_title("Δr  (best-of minus equal-weight)")

    plt.suptitle("Final best-of product — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, loo_preds, np, plt):
    from scipy import stats as _stats

    _pred_flat = loo_preds.values.ravel()
    _obs_flat = da_era5_z.values.ravel()
    _mask = np.isfinite(_pred_flat) & np.isfinite(_obs_flat)

    _r, _p = _stats.pearsonr(_pred_flat[_mask], _obs_flat[_mask])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(
        _pred_flat[_mask], _obs_flat[_mask], alpha=0.1, s=5, c="steelblue"
    )
    _lim = max(np.abs([_pred_flat[_mask], _obs_flat[_mask]]).max(), 2)
    _ax.axline((0, 0), slope=1, color="k", lw=0.8, ls="--")
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel("LOO predicted z-score")
    _ax.set_ylabel("ERA5 observed z-score")
    _ax.set_title(
        f"Reliability scatter (all pixels × years) r = {_r:.3f}, p = {_p:.3e}"
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(alpha, da_era5_z, da_fc_z, np):
    from scipy.stats import pearsonr as _pearsonr

    from src.ensemble.regression import DEFAULT_ALPHAS
    from src.ensemble.regression import _fit_pixel as _fit_pixel_fn
    from src.ensemble.regression import _loo_predictions as _loo_pred_fn

    _X_z_t = da_fc_z.transpose("lat", "lon", "year", "model")
    _y_z_t = da_era5_z.transpose("lat", "lon", "year")

    print("alpha stats:")
    print(f"  median: {float(alpha.median()):.2f}")
    print(f"  min:    {float(alpha.min()):.2f}")
    print(f"  max:    {float(alpha.max()):.2f}")

    for _i in range(len(da_fc_z.lat)):
        for _j in range(len(da_fc_z.lon)):
            _xp = _X_z_t.values[_i, _j]
            _yp = _y_z_t.values[_i, _j]
            _valid = np.isfinite(_yp) & np.all(np.isfinite(_xp), axis=1)
            if _valid.sum() < 10:
                continue
            _xv, _yv = _xp[_valid], _yp[_valid]
            _c, _a, _loo = _fit_pixel_fn(_xp, _yp, DEFAULT_ALPHAS)
            print(f"\nPixel ({_i},{_j}): selected alpha={_a:.4f}")
            print(f"  coefs: {_c}")
            print(
                f"  loo pred std: {np.nanstd(_loo):.4f}  (obs std: {np.nanstd(_yp):.4f})"
            )
            for _test_alpha in [0.01, 0.1, 1, 10, 100, 1000]:
                _lp = _loo_pred_fn(_xv, _yv, _test_alpha)
                _r_val, _ = _pearsonr(_lp, _yv)
                print(f"  alpha={_test_alpha:6.2f}: LOO r = {_r_val:.3f}")
            break
        else:
            continue
        break
    return


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, da_fc_z, np, regression, skill):
    lasso_coefs, lasso_alpha, lasso_loo = regression.fit_lasso_per_pixel(
        da_fc_z, da_era5_z
    )
    lasso_r = skill.pearson_r_map(lasso_loo, da_era5_z)
    print(
        "Lasso alpha — median:",
        float(lasso_alpha.median()),
        "  min:",
        float(lasso_alpha.min()),
        "  max:",
        float(lasso_alpha.max()),
    )
    n_nonzero = (np.abs(lasso_coefs) > 1e-6).sum(dim="model")
    print(
        "Non-zero coefs per pixel — median:",
        float(n_nonzero.median()),
        "  max:",
        float(n_nonzero.max()),
    )
    return (lasso_r,)


@app.cell
def _(add_boundaries, ccrs, lasso_r, mme_r, plt, sel_ew_r):
    # 4-panel skill comparison: Ridge LOO, Lasso LOO, Selected-EW, Δr Lasso − selected-EW
    _fig, _axes = plt.subplots(
        1, 4, figsize=(20, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    _cmap, _vmin, _vmax = "RdBu", -0.8, 0.8

    for _ax, _da, _title in zip(
        _axes[:3],
        [mme_r, lasso_r, sel_ew_r],
        ["Ridge LOO r", "Lasso LOO r", "Selected equal-weight r"],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7, label="Pearson r")

    # Delta: Lasso minus selected equal-weight
    _ax = _axes[3]
    add_boundaries(_ax)
    _delta = lasso_r - sel_ew_r
    _im = _ax.pcolormesh(
        _delta.lon,
        _delta.lat,
        _delta.values,
        cmap="RdBu",
        vmin=-0.3,
        vmax=0.3,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(_im, ax=_ax, shrink=0.7, label="Δr")
    _ax.set_title("Δr  Lasso − selected-EW")

    plt.suptitle("Skill comparison incl. Lasso — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(adm2_aoi, da_era5_z, da_fc_z, np, regression, xr):
    from scipy.stats import pearsonr as _pearsonr
    from shapely.geometry import Point
    from shapely.ops import unary_union

    # Build a boolean (lat, lon) mask for AOI pixels
    _union = unary_union(adm2_aoi.geometry)
    _lats = da_era5_z.lat.values
    _lons = da_era5_z.lon.values
    _mask = np.array(
        [[_union.contains(Point(lo, la)) for lo in _lons] for la in _lats]
    )  # (lat, lon)

    # Spatial mean over AOI pixels → (year, model) and (year,)
    _mask_da = xr.DataArray(
        _mask,
        dims=["lat", "lon"],
        coords={"lat": da_era5_z.lat, "lon": da_era5_z.lon},
    )
    _X_mean = da_fc_z.where(_mask_da).mean(["lat", "lon"])  # (year, model)
    _y_mean = da_era5_z.where(_mask_da).mean(["lat", "lon"])  # (year,)

    # Single Lasso fit via internal helper
    _coefs, _alpha, _loo = regression._fit_pixel_lasso(
        _X_mean.values, _y_mean.values, regression.DEFAULT_LASSO_ALPHAS
    )

    _valid = np.isfinite(_y_mean.values) & np.isfinite(_loo)
    aoi_lasso_r_scalar, _ = _pearsonr(_loo[_valid], _y_mean.values[_valid])

    aoi_lasso_coefs = xr.DataArray(
        _coefs, dims=["model"], coords={"model": da_fc_z.model}
    )
    aoi_lasso_loo = xr.DataArray(
        _loo, dims=["year"], coords={"year": _y_mean.year}
    )
    aoi_lasso_obs = _y_mean

    print(f"AOI Lasso — alpha: {_alpha:.4f}, LOO r: {aoi_lasso_r_scalar:.3f}")
    print("Coefs:", dict(zip(da_fc_z.model.values, _coefs.round(3))))
    return aoi_lasso_coefs, aoi_lasso_loo, aoi_lasso_obs, aoi_lasso_r_scalar


@app.cell
def _(
    aoi_lasso_coefs,
    aoi_lasso_loo,
    aoi_lasso_obs,
    aoi_lasso_r_scalar,
    np,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: LOO predicted vs observed scatter for AOI mean
    _ax = _axes[0]
    _valid = np.isfinite(aoi_lasso_obs.values) & np.isfinite(
        aoi_lasso_loo.values
    )
    _ax.scatter(
        aoi_lasso_loo.values[_valid],
        aoi_lasso_obs.values[_valid],
        color="steelblue",
        zorder=3,
    )
    _lim = 2.5
    _ax.axline((0, 0), slope=1, color="k", lw=0.8, ls="--")
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.axvline(0, color="gray", lw=0.5)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel("LOO predicted z-score")
    _ax.set_ylabel("ERA5 AOI-mean z-score")
    _ax.set_title(f"AOI Lasso LOO skill  r = {aoi_lasso_r_scalar:.3f}")

    # Right: coefficient bar chart (Lasso → sparse)
    _ax = _axes[1]
    _models = aoi_lasso_coefs.model.values
    _coefs = aoi_lasso_coefs.values
    _colors = [
        "steelblue" if c > 1e-6 else "lightgray" for c in np.abs(_coefs)
    ]
    _ax.barh(_models, _coefs, color=_colors)
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set_xlabel("Lasso coefficient β")
    _ax.set_title("Model weights (gray = zeroed out)")

    plt.suptitle("AOI-mean Lasso — JAS Apr-issued", y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(da_era5_z, da_fc_z, skill):
    mom = skill.model_of_models_tercile(da_fc_z, da_era5_z)
    mom_r = mom["r"]
    mom_bss = mom["bss"]
    frac_below = mom["frac_below"]
    obs_below = mom["obs_below"]
    return frac_below, mom_bss, mom_r, obs_below


@app.cell
def _(add_boundaries, ccrs, mme_r, mom_bss, mom_r, plt):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for _ax, _da, _title, _vmin, _vmax in zip(
        _axes,
        [mom_r, mom_bss, mme_r],
        [
            "MoM r (prob vs binary)",
            "MoM Brier skill score",
            "Ridge LOO r (comparison)",
        ],
        [-0.8, -0.5, -0.8],
        [0.8, 0.5, 0.8],
    ):
        add_boundaries(_ax)
        _im = _ax.pcolormesh(
            _da.lon,
            _da.lat,
            _da.values,
            cmap="RdBu",
            vmin=_vmin,
            vmax=_vmax,
            transform=ccrs.PlateCarree(),
        )
        _ax.set_title(_title)
        plt.colorbar(_im, ax=_ax, shrink=0.7)

    plt.suptitle(
        "Model-of-models tercile probability — JAS Apr-issued", y=1.02
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(frac_below, np, obs_below, plt):
    _n_models = frac_below.shape[1] if hasattr(frac_below, "model") else 9
    # frac_below values are multiples of 1/9
    _fc_flat = frac_below.values.ravel()
    _ob_flat = obs_below.values.ravel()
    _mask = np.isfinite(_fc_flat) & np.isfinite(_ob_flat)
    _fc_flat, _ob_flat = _fc_flat[_mask], _ob_flat[_mask]

    # Group by forecast probability bin
    _bins = np.linspace(0, 1, 11)  # 0, 0.1, ..., 1.0
    _bin_centers = 0.5 * (_bins[:-1] + _bins[1:])
    _obs_freq = []
    _counts = []
    for _lo, _hi in zip(_bins[:-1], _bins[1:]):
        _sel = (_fc_flat >= _lo) & (_fc_flat < _hi)
        if _lo == _bins[-2]:  # include right edge for last bin
            _sel = (_fc_flat >= _lo) & (_fc_flat <= _hi)
        _counts.append(_sel.sum())
        _obs_freq.append(_ob_flat[_sel].mean() if _sel.sum() > 0 else np.nan)
    _obs_freq = np.array(_obs_freq)
    _counts = np.array(_counts)

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _width = 0.08
    _bars = _ax.bar(
        _bin_centers,
        _obs_freq,
        width=_width,
        color="steelblue",
        alpha=0.7,
        label="Observed frequency",
    )
    _ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect reliability")
    _ax.axhline(1 / 3, color="gray", lw=0.8, ls=":", label="Climatology (1/3)")

    for _bc, _of, _cnt in zip(_bin_centers, _obs_freq, _counts):
        if np.isfinite(_of) and _cnt > 0:
            _ax.annotate(
                f"n={_cnt}",
                xy=(_bc, _of),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    _ax.set_xlabel("Forecast probability (fraction of models below tercile)")
    _ax.set_ylabel("Observed frequency of below-tercile")
    _ax.set_xlim(-0.05, 1.05)
    _ax.set_ylim(-0.05, 1.05)
    _ax.legend(fontsize=9)
    _ax.set_title("Reliability — fraction of models below tercile")
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(PROJECT_PREFIX, da_fc_z, skill):
    import os as _os
    import tempfile as _tempfile

    from src.utils import blob_utils as _blob_utils

    _ew = skill.equal_weight_ensemble(da_fc_z)  # (year, lat, lon)
    _blob_name = (
        f"{PROJECT_PREFIX}/processed/ensemble/equal_weight_forecast.nc"
    )

    _tmp = _tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    _tmp.close()
    try:
        _ew.to_netcdf(_tmp.name)
        with open(_tmp.name, "rb") as _f:
            _blob_utils._upload_blob_data(_f, _blob_name, stage="dev")
    finally:
        _os.unlink(_tmp.name)
    print(f"Saved equal-weight forecast → {_blob_name}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
