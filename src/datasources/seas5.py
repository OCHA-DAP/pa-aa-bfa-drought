from typing import List, Literal

import ocha_stratus as stratus
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from src.constants import ORIGINAL_MO_LT_COMBOS, ORIGINAL_Q
from src.datasources import codab
from src.utils import blob_utils
from src.utils.raster import upsample_dataarray

START_YEAR = 1981
END_YEAR = 2025


def open_seas5_cog(issued_date_str: str, lt: int):
    blob_name = (
        f"seas5/monthly/processed/precip_em_i{issued_date_str}_lt{lt}.tif"
    )
    return stratus.open_blob_cog(
        blob_name, stage="prod", container_name="raster"
    )


def open_seas5_rasters(
    mo_lt_combos: List[dict] = None, years: List[int] = None
):
    if mo_lt_combos is None:
        # set to original IRI framework combinations
        mo_lt_combos = ORIGINAL_MO_LT_COMBOS
    if years is None:
        years = range(START_YEAR, END_YEAR + 1)
    das = []
    for year in tqdm(years):
        for mo_lt_combo in mo_lt_combos:
            mo = mo_lt_combo["mo"]
            for lt in mo_lt_combo["lts"]:
                issued_date_str = f"{year}-{str(mo).zfill(2)}-01"
                da_in = open_seas5_cog(issued_date_str, lt)
                da_in = da_in.squeeze(drop=True)
                da_in["lt"] = lt
                da_in["year"] = year
                da_in["issued_month"] = mo
                da_in = da_in.expand_dims(["year", "issued_month", "lt"])
                das.append(da_in)
    da_out = xr.combine_by_coords(das, combine_attrs="drop_conflicts")
    return da_out


def process_seas5_rasters():
    """Process SEAS5 rasters with specific quantile raster stat."""
    adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
    da_seas5 = open_seas5_rasters()
    da_seas5_tri = da_seas5.mean(dim="lt")
    da_seas5_up = upsample_dataarray(da_seas5_tri)
    da_seas5_clip = da_seas5_up.rio.clip(adm1.geometry)
    da_seas5_q = da_seas5_clip.quantile(q=ORIGINAL_Q, dim=["x", "y"])
    with ProgressBar():
        da_seas5_q_computed = da_seas5_q.compute()
    df_seas5 = da_seas5_q_computed.to_dataframe("q")["q"].reset_index()
    blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_original_trigger_raster_stats.parquet"  # noqa
    blob_utils.upload_parquet_to_blob(df_seas5, blob_name)


def load_seas5_stats(variable: Literal["zscore", "abs", "anomaly"] = "abs"):
    if variable == "abs":
        blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_original_trigger_raster_stats.parquet"  # noqa
    else:
        blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_{variable}_q10.parquet"  # noqa
    df_seas5 = blob_utils.load_parquet_from_blob(blob_name)
    return df_seas5
