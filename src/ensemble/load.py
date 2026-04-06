"""
Load multi-model ensemble hindcasts from Azure Blob Storage.

Files are GRIB format at:
    pa-aa-bfa-drought/raw/cds/apr_issued/{centre}.grib

Each file contains all ensemble members, all leadtimes (1–6), all hindcast
years for that centre.  This module takes the ensemble mean, selects the JAS
leadtimes, and returns a seasonal-mean DataArray per centre.

Usage
-----
    from src.ensemble.load import load_forecasts

    ds = load_forecasts()
    # ds.precip has dims (year, model, lat, lon)
    # centres may have different year ranges; missing years are NaN
"""

import logging
import tempfile

import cfgrib
import numpy as np
import ocha_stratus as stratus
import xarray as xr

logger = logging.getLogger(__name__)

BLOB_PREFIX = "pa-aa-bfa-drought/raw/cds/apr_issued"

CENTRES = ["ecmwf", "bom", "jma", "meteofrance"]

JAS_MONTHS = [7, 8, 9]

# Leadtime indices (0-based) for JAS from an April issue: May=0, Jun=1,
# Jul=2, Aug=3, Sep=4, Oct=5  →  JAS = indices 2, 3, 4
_JAS_LEADTIME_INDICES = [2, 3, 4]

# Known precipitation variable names across C3S centres
_PRECIP_VARS = ("tprate", "tp", "total_precipitation", "pr", "prate")


def _blob_name(centre: str) -> str:
    return f"{BLOB_PREFIX}/{centre}.grib"


def _load_single_centre(
    centre: str,
    valid_months: list[int] = JAS_MONTHS,
    stage: str = "dev",
) -> xr.DataArray:
    """
    Load one centre's GRIB hindcast from blob.

    Returns a DataArray with dims (year, lat, lon) containing the JAS-mean
    ensemble-mean total precipitation.
    """
    blob_name = _blob_name(centre)
    logger.info("Loading %s from blob …", blob_name)
    raw = stratus.load_blob_data(blob_name, stage=stage)

    with tempfile.NamedTemporaryFile(suffix=".grib") as tmp:
        tmp.write(raw)
        tmp.flush()
        # cfgrib reads lazily, so .load() must be called before the temp file
        # is deleted at the end of this block
        datasets = [ds.load() for ds in cfgrib.open_datasets(tmp.name)]

    # Pick the dataset that contains a precipitation variable
    ds = None
    var = None
    for candidate_ds in datasets:
        for v in candidate_ds.data_vars:
            if v in _PRECIP_VARS:
                ds, var = candidate_ds, v
                break
        if ds is not None:
            break
    if ds is None:
        # Fall back to first dataset / first variable
        ds = datasets[0]
        var = list(ds.data_vars)[0]
        logger.warning(
            "%s: no known precip variable found, using '%s'", centre, var
        )

    da = ds[var]

    # --- Ensemble mean ---
    member_dims = [
        d for d in da.dims if d in ("number", "realization", "member")
    ]
    if member_dims:
        da = da.mean(dim=member_dims[0])
    else:
        logger.debug("%s: no ensemble member dimension found", centre)

    # --- Select JAS lead times and reduce to (year, lat, lon) ---
    #
    # GRIB files come in two structures:
    #   A) time (forecast ref year) × step (lead offset) — classic C3S layout
    #   B) time encodes every valid month directly (e.g. 1993-07, 1993-08, …)
    #
    step_dim = next(
        (d for d in da.dims if d in ("step", "leadtime_month")), None
    )
    if step_dim is None:
        step_dim = next(
            (d for d in da.dims if "step" in d.lower() or "lead" in d.lower()),
            None,
        )
    time_dim = next(
        (d for d in da.dims if "time" in d.lower() and d != step_dim), None
    )

    if step_dim is not None:
        # Structure A: select JAS steps then average
        if "valid_time" in da.coords:
            vt = da.valid_time
            if (
                step_dim in vt.dims
                and time_dim is not None
                and time_dim in vt.dims
            ):
                vt = vt.isel({time_dim: 0})
            jas_idx = [
                i
                for i, m in enumerate(vt.dt.month.values)
                if m in valid_months
            ]
            if jas_idx:
                da = da.isel({step_dim: jas_idx}).mean(dim=step_dim)
            else:
                logger.warning(
                    "%s: valid_time month filter found no JAS steps; "
                    "falling back to positional indices %s",
                    centre,
                    _JAS_LEADTIME_INDICES,
                )
                da = da.isel({step_dim: _JAS_LEADTIME_INDICES}).mean(
                    dim=step_dim
                )
        else:
            da = da.isel({step_dim: _JAS_LEADTIME_INDICES}).mean(dim=step_dim)

        # Rename the reference-time dim to year
        if time_dim is not None:
            da = da.rename({time_dim: "year"})
            da["year"] = da["year"].dt.year.astype(int)

    elif time_dim is not None:
        # Structure B: time holds all valid months — filter to JAS then
        # group by year and average
        da = da.sel({time_dim: da[time_dim].dt.month.isin(valid_months)})
        years = da[time_dim].dt.year.astype(int).values
        da = da.assign_coords({time_dim: years}).rename({time_dim: "year"})
        da = da.groupby("year").mean()

    # --- Standardise spatial dim names ---
    renames = {}
    for d in da.dims:
        if d.lower() == "latitude":
            renames[d] = "lat"
        elif d.lower() in ("longitude", "long"):
            renames[d] = "lon"
    if renames:
        da = da.rename(renames)

    da.name = "precip"
    return da


def load_forecasts(
    centres: list[str] = None,
    valid_months: list[int] = None,
    stage: str = "dev",
    target_lat: np.ndarray = None,
    target_lon: np.ndarray = None,
) -> xr.Dataset:
    """
    Load all centre hindcasts and align them onto a common grid.

    Different centres may cover different year ranges.  The returned Dataset
    uses an outer join on the year dimension, so years missing for a given
    centre are NaN.  Callers should drop all-NaN years before fitting.

    Parameters
    ----------
    centres : list of str, optional
        Centres to load.  Defaults to all in CENTRES.
    valid_months : list of int, optional
        Valid season months (default [7, 8, 9] = JAS).
    stage : str
        Blob storage stage ('dev' or 'prod').
    target_lat, target_lon : array-like, optional
        Target 1-D coordinate arrays for regridding.  If omitted, the grid
        of the first loaded centre is used.

    Returns
    -------
    xr.Dataset
        Dataset with a ``precip`` variable of dims (year, model, lat, lon).
    """
    if centres is None:
        centres = CENTRES
    if valid_months is None:
        valid_months = JAS_MONTHS

    das = []
    for centre in centres:
        da = _load_single_centre(
            centre, valid_months=valid_months, stage=stage
        )
        das.append((centre, da))

    if target_lat is None or target_lon is None:
        target_lat = das[0][1].lat.values
        target_lon = das[0][1].lon.values

    aligned = []
    for centre, da in das:
        da_interp = da.interp(lat=target_lat, lon=target_lon, method="linear")
        da_interp = da_interp.assign_coords(model=centre).expand_dims("model")
        aligned.append(da_interp)

    # outer join so all year values are preserved; missing → NaN
    combined = xr.concat(aligned, dim="model", join="outer")
    return combined.to_dataset(name="precip")
