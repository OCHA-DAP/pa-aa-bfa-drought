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
from calendar import month_name
from pathlib import Path
from typing import List, Optional, Tuple

import aatoolbox.utils.raster  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from aatoolbox import CodAB, GeoBoundingBox, IriForecastProb
from aatoolbox.config.countryconfig import CountryConfig
from dateutil.relativedelta import relativedelta
from geopandas import GeoDataFrame
from rasterio.enums import Resampling

from src import constants

# number of months that is considered a season
SEAS_LEN = 3
# map month number to str of abbreviations of month names
# month number refers to the last month of the season
END_MONTH_SEASON_MAPPING = {
    m: "".join(
        [month_name[(m - i) % 12 + 1][0] for i in range(SEAS_LEN, 0, -1)]
    )
    for m in range(1, 13)
}


def compute_trigger_bfa():
    """
    Compute the trigger for Burkina Faso.
    """
    # question: does it make sense to define all vars here or should they
    # be global constants?
    codab = CodAB(country_config=constants.country_config)
    codab.download()
    gdf_adm1 = codab.load(admin_level=1)
    adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]
    gdf_aoi = gdf_adm1[gdf_adm1["ADM1_FR"].isin(adm_sel)]
    df_all, df_trig_mom = compute_trigger_bavg_iri(
        country_config=constants.country_config,
        gdf_aoi=gdf_aoi,
        threshold_bavg=40,
        threshold_diff=5,
        perc_area=10,
        trig_mom=[(3, 3), (7, 1)],
        adm0_col_name="ADM0_FR",
    )
    # question: would it make more sense to have one file per prediction
    # date? And possibly even only save the prediction dates that are
    # included in the trigger?
    df_all.to_csv(
        get_trigger_output_filename(
            iso3=constants.iso3,
            country_config=constants.country_config,
            gdf_aoi=gdf_aoi,
        )
    )

    df_trig_mom.to_csv(
        get_trigger_output_filename(
            iso3=constants.iso3,
            country_config=constants.country_config,
            gdf_aoi=gdf_aoi,
            only_trig_mom=True,
        )
    )


def get_trigger_output_filename(
    iso3, country_config, gdf_aoi, only_trig_mom=False
) -> Path:
    # question: does it make sense to just save it in the iri dir or should
    # we have a separate "trigger" dir?
    iri_prob = _define_iri_class(country_config, gdf_aoi)
    if only_trig_mom:
        filename_substr = "trigger_moments"
    else:
        filename_substr = "all_dates"
    filename = f"{iso3}_statistics_{filename_substr}.csv"
    return iri_prob._processed_base_dir / filename


def compute_trigger_bavg_iri(
    country_config: CountryConfig,
    gdf_aoi: GeoDataFrame,
    threshold_bavg: float,
    perc_area: float,
    threshold_diff: Optional[float] = None,
    trig_mom: Optional[List[Tuple]] = None,
    adm0_col_name: str = "ADM0_FR",
    pcode0_col_name: str = "ADM0_PCODE",
) -> (pd.DataFrame, pd.DataFrame):
    """
    Trigger computation.

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

    #TODO: is there a way to automatically fill the types in the parameters
    section?
    Parameters
    ----------
    country_config
        aa-toolbox country configuration object
    gdf_aoi
        GeoDataFrame of which the outer bounds are the area of interest (AOI)
    threshold_bavg
        minimum probability of below average precipitation for the raster
        cell to be reaching the threshold
    perc_area
        minimum percentage of the area within `gdf_aoi` to reach the
        threshold(s) for the trigger to be met
    threshold_diff
        minimum difference of the below average and above average
        probability for the raster cell to be reaching the threshold
    trig_mom
        Moments at which to evaluate the trigger. Each tuple indicates a
        moment, where the first entry is the publication month, and the
        second the leadtime
    #TODO: maybe doesnt necessarily have to be adm0. Figure out what is
    needed for the dissolve
    adm0_col_name:
        column indicating adm0 col in gdf_aoi
    pcode0_col_name

    Returns
    -------
    Two dataframes indicating if the trigger has been met and corresponding
    stats. One dataframe contains all dates, the other only the publication
    date-leadtime combinations that are included in the trigger.

    """
    # Load the data that contains the probability per tercile
    # this allows us to solely look at the below-average tercile
    # C indicates the tercile (below-average, normal, or above-average).
    # F indicates the publication month, and L the leadtime
    ds_iri = load_iri_tercile_probability_forecast(
        country_config=country_config, gdf_aoi=gdf_aoi
    )
    ds_iri_mask = approx_mask_raster(ds_iri, gdf_aoi)
    # only select cells of upsampled ds that have their centre within the aoi
    ds_iri_mask = ds_iri_mask.rio.clip(gdf_aoi["geometry"], all_touched=False)
    df_stats_aoi_bavg = compute_stats_iri(
        da=ds_iri_mask.prob,
        gdf_aoi=gdf_aoi,
        adm0_col=adm0_col_name,
        pcode0_col=pcode0_col_name,
        perc_area=perc_area,
        threshold_bavg=threshold_bavg,
        threshold_diff=threshold_diff,
    )
    trigger_col = (
        "perc_thresh" if threshold_diff is not None else "perc_threshold_bavg"
    )
    df_stats_aoi_bavg["trigger_met"] = np.where(
        df_stats_aoi_bavg[trigger_col] >= perc_area, True, False
    )
    # select the months and leadtimes included in the trigger
    df_stats_aoi_bavg_trig_mom = df_stats_aoi_bavg[
        df_stats_aoi_bavg[["pub_month", "L"]]
        .apply(tuple, axis=1)
        .isin(trig_mom)
    ]
    return df_stats_aoi_bavg, df_stats_aoi_bavg_trig_mom


def _define_iri_class(country_config, gdf_aoi):
    geobb = GeoBoundingBox.from_shape(gdf_aoi)
    iri_prob = IriForecastProb(country_config, geo_bounding_box=geobb)
    return iri_prob


def load_iri_tercile_probability_forecast(
    country_config: CountryConfig,
    gdf_aoi: GeoDataFrame,
) -> xr.Dataset:
    """
    Load the IRI seasonal tercile precipitation forecast.

    Contains a probability per tercile.

    Parameters
    ----------
    country_config
        aa-toolbox country configuration object
    gdf_aoi
        GeoDataFrame of which the outer bounds are the area of interest

    Returns
    -------
    Dataset containing the IRI seasonal precipitation forecast for the AOI
    for all available dates and leadtimes.
    """
    iri_prob = _define_iri_class(country_config, gdf_aoi)
    iri_prob.download()
    iri_prob.process()
    ds_iri = iri_prob.load()
    return ds_iri


def approx_mask_raster(
    ds: xr.Dataset,
    gdf: GeoDataFrame,
    resolution: float = 0.05,
) -> xr.Dataset:
    """
    Resample raster data to given resolution.

    Uses as resample method nearest neighbour, i.e. aims to keep the values
    the same as the original data. Mainly used to create an approximate mask
    over an area

    Parameters
    ----------
    ds
        Dataset to resample.
    resolution
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
    da
        DataArray containing the IRI forecast
    gdf_aoi
        GeoDataFrame of which the outer bounds are the area of interest
    #TODO: figure out reqs dissolve
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
    # compute perc of region above threshold
    df_stats_aoi_bavg["perc_threshold_bavg"] = (
        df_stats_aoi_bavg_thresh[f"count_{pcode0_col}"]
        / df_stats_aoi_bavg[f"count_{pcode0_col}"]
        * 100
    )

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
        lambda x: END_MONTH_SEASON_MAPPING[
            int((x["F"] + relativedelta(months=int(x["L"]))).strftime("%-m"))
        ],
        axis=1,
    )
    return df_stats_aoi_bavg


if __name__ == "__main__":
    compute_trigger_bfa()
