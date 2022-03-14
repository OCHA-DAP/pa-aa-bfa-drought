"""
Trigger monitoring script.

This script computes the trigger as defined by OCHA's Anticipatory Action
Framework on drought in Burkina Faso.
The trigger is based on IRI's seasonal tercile precipitation forecast.
The definition of the trigger is:
1. 40% or more probability of below average rainfall AND
2. The probability of below average rainfall should be 5 percentage points
higher than that of above average rainfall on at least 10% of the zone
(Boucle de Mouhoun, Centre Nord, Sahel, and Nord)

This is evaluated in mid-March for the June-August period, i.e. a leadtime
of 3 months, and in mid-July for the August-October period,
i.e. a leadtime of 1 month

"""
import itertools
from pathlib import Path

import aatoolbox.utils.raster  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from aatoolbox import CodAB, GeoBoundingBox, IriForecastProb
from dateutil.relativedelta import relativedelta
from geopandas import GeoDataFrame
from rasterio.enums import Resampling

import constants
import utils


def compute_trigger_bfa():
    """
    Compute the trigger for Burkina Faso.

    Compute whether the trigger based on the below average tercile of IRI's
    seasonal precipitation forecast has been met.
    The trigger needs to be defined as:
    >= `perc_area`% of `gdf_aoi` meets the condition.
    Where the condition is defined at cell level as:
    >=`threshold_bavg`% probability of below average rainfall and optionally
    (%bavg-%above average) >= `threshold_diff`

    For the computation of the statistics, an approximate mask of the
    area in `gdf_aoi` is done. This is done by upsampling the raster forecast
    data to a 0.05 degrees resolution (original resolution is 1 degree),
    after which all upsampled cells with their  centre within `gdf_aoi` are
    selected.
    """
    gdf_aoi, iri_prob = setup()
    iri_prob.download(clobber=True)
    iri_prob.process(clobber=True)
    # Load the data that contains the probability per tercile
    # this allows us to solely look at the below-average tercile
    # C indicates the tercile (below-average, normal, or above-average).
    # F indicates the publication month, and L the leadtime
    ds_iri = iri_prob.load()
    ds_iri_mask = approx_mask_raster(ds_iri)
    # only include the area of interest
    ds_iri_mask = ds_iri_mask.rio.clip(gdf_aoi["geometry"], all_touched=False)

    # compute the statistics on the area
    df_stats_aoi_bavg = compute_stats_iri(
        da=ds_iri_mask.prob,
        gdf_aoi=gdf_aoi,
        adm0_col="ADM0_FR",
        pcode0_col="ADM0_PCODE",
        perc_area=constants.perc_area,
        threshold_bavg=constants.threshold_bavg,
        threshold_diff=constants.threshold_diff,
    )

    # select the months and leadtimes included in the trigger
    df_stats_aoi_bavg_trig_mom = df_stats_aoi_bavg[
        df_stats_aoi_bavg[["pub_month", "L"]]
        .apply(tuple, axis=1)
        .isin(constants.trig_mom)
    ]

    # TODO: would it make more sense to have one file per prediction
    #  date? And possibly even only save the prediction dates that are
    #  included in the trigger?
    df_stats_aoi_bavg.to_csv(
        get_trigger_output_filename(base_dir=iri_prob._processed_base_dir)
    )

    df_stats_aoi_bavg_trig_mom.to_csv(
        get_trigger_output_filename(
            base_dir=iri_prob._processed_base_dir, only_trig_mom=True
        )
    )


def setup():
    codab = CodAB(country_config=constants.country_config)
    gdf_adm1 = codab.load(admin_level=1)
    gdf_aoi = gdf_adm1[gdf_adm1["ADM1_FR"].isin(constants.adm_sel)]
    geobb = GeoBoundingBox.from_shape(gdf_adm1)
    iri_prob = IriForecastProb(
        country_config=constants.country_config, geo_bounding_box=geobb
    )
    return gdf_aoi, iri_prob


# TODO: move this to toolbox utils?
def approx_mask_raster(
    ds: xr.Dataset,
    resolution: float = 0.05,
) -> xr.Dataset:
    """
    Resample raster data to given resolution.

    Uses as resample method nearest neighbour, i.e. aims to keep the values
    the same as the original data. Mainly used to create an approximate mask
    over an area

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to resample.
    resolution: float, default = 0.05
        Resolution in degrees to resample to

    Returns
    -------
        Upsampled dataset
    """
    upsample_list = []
    # can only do reproject on 3D array so
    # loop over all +3D dimensions
    list_dim = [
        d for d in ds.dims if (d != ds.aat.x_dim) & (d != ds.aat.y_dim)
    ]
    # select from second element of list_dim since can loop over 3D
    # loop over all combs of dims
    for i in itertools.product(*[ds[d].values for d in list_dim[1:]]):
        ds_sel = ds.sel({d: i[k] for k, d in enumerate(list_dim[1:])})
        ds_sel_upsample = ds_sel.rio.reproject(
            ds_sel.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        upsample_list.append(
            ds_sel_upsample.expand_dims(
                {d: [i[k]] for k, d in enumerate(list_dim[1:])}
            )
        )
    ds_upsample = xr.combine_by_coords(upsample_list)
    # reproject changes spatial dims names to x and y
    # so change back here
    ds_upsample = ds_upsample.rename({"x": ds.aat.x_dim, "y": ds.aat.y_dim})
    return ds_upsample


def compute_stats_iri(
    da: xr.DataArray,
    gdf_aoi: GeoDataFrame,
    adm0_col: str,
    pcode0_col: str,
    perc_area: float,
    threshold_bavg: float,
    threshold_diff: float,
) -> pd.DataFrame:
    """
    Compute statistics on the IRI tercile probability forecast for the AOI.

    Parameters
    ----------
    da: xr.DataArray
        DataArray containing the IRI forecast
    gdf_aoi: gpd.GeoDataFrame
        GeoDataFrame of which the outer bounds are the area of interest
    # TODO: figure out reqs dissolve?
    adm0_col
    pcode0_col
    threshold_bavg
        minimum probability of below average precipitation for the raster
        cell to be reaching the threshold
    threshold_diff
        minimum difference of the below average and above average
        probability for the raster cell to be reaching the threshold
    perc_area
        minimum percentage of the area within `gdf_aoi` to reach the
        threshold(s) for the trigger to be met

    Returns
    -------
    DataFrame containing the statistics on the AOI.
    """
    # dissolve the region to one polygon
    # i.e. we compute the statistics over the whole region
    gdf_aoi_dissolved = gdf_aoi.dissolve(by=adm0_col)
    gdf_aoi_dissolved = gdf_aoi_dissolved[[pcode0_col, "geometry"]]

    # sel below average tercile
    da_bavg = da.sel(C=0)

    # compute the stats on all data
    df_stats_aoi_bavg = da_bavg.aat.compute_raster_stats(
        gdf=gdf_aoi_dissolved,
        feature_col=pcode0_col,
        percentile_list=[100 - perc_area],
    )

    # compute the stats on data above threshold
    da_bavg_thresh = da_bavg.where(da_bavg >= threshold_bavg)
    df_stats_aoi_bavg_thresh = da_bavg_thresh.aat.compute_raster_stats(
        gdf=gdf_aoi_dissolved, feature_col=pcode0_col
    )
    # compute perc of region above bavg threshold
    df_stats_aoi_bavg["perc_threshold_bavg"] = (
        df_stats_aoi_bavg_thresh[f"count_{pcode0_col}"]
        / df_stats_aoi_bavg[f"count_{pcode0_col}"]
        * 100
    )

    # compute the stats on data that is above the bavg threshold and has more
    # than `threshold_diff` difference between bavg and abv avg
    if threshold_diff is not None:
        # diff between below average (C=0) and above average (C=2) tercile
        da_diff_bel_abv = da.sel(C=0) - da.sel(C=2)
        # sel where above threshold prob bavg and threshold difference
        da_threshold_diff = da_bavg.where(
            (da_bavg >= threshold_bavg) & (da_diff_bel_abv >= threshold_diff)
        )
        df_stats_aoi_bavg_diff_thresh = (
            da_threshold_diff.aat.compute_raster_stats(
                gdf=gdf_aoi_dissolved, feature_col=pcode0_col
            )
        )
        # perc of area meeting both thresholds
        df_stats_aoi_bavg["perc_thresh"] = (
            df_stats_aoi_bavg_diff_thresh[f"count_{pcode0_col}"]
            / df_stats_aoi_bavg[f"count_{pcode0_col}"]
            * 100
        )

    # add info publication and prediction month of forecast
    df_stats_aoi_bavg["F"] = pd.to_datetime(
        df_stats_aoi_bavg["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
    )
    df_stats_aoi_bavg["pub_month"] = df_stats_aoi_bavg.F.dt.month
    df_stats_aoi_bavg["pred_seas"] = df_stats_aoi_bavg.apply(
        lambda x: utils.get_month_season_mapping()[
            int((x["F"] + relativedelta(months=int(x["L"]))).strftime("%-m"))
        ],
        axis=1,
    )

    trigger_col = (
        "perc_thresh" if threshold_diff is not None else "perc_threshold_bavg"
    )
    df_stats_aoi_bavg["trigger_met"] = np.where(
        df_stats_aoi_bavg[trigger_col] >= perc_area, True, False
    )

    return df_stats_aoi_bavg


def get_trigger_output_filename(base_dir, only_trig_mom=False) -> Path:
    # TODO: does it make sense to just save it in the iri dir or should
    #  we have a separate "trigger" dir?
    if only_trig_mom:
        filename_substr = "trigger_moments"
    else:
        filename_substr = "all_dates"
    filename = f"{constants.iso3}_statistics_{filename_substr}.csv"
    return base_dir / filename


if __name__ == "__main__":
    compute_trigger_bfa()
