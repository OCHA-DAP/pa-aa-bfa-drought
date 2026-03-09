from typing import List

import xarray as xr
from tqdm.auto import tqdm

from src.utils import blob_utils

START_YEAR = 1981
END_YEAR = 2024


def open_seas5_cog(issued_date_str: str):
    blob_name = (
        f"era5/monthly/processed/precip_reanalysis_v{issued_date_str}.tif"
    )
    return blob_utils.open_blob_cog(
        blob_name, stage="prod", container_name="raster"
    )


def open_era5_rasters(months: List[dict] = None):
    if months is None:
        # set to original IRI framework combinations
        months = [6, 7, 8, 9]
    das = []
    for year in tqdm(range(START_YEAR, END_YEAR + 1)):
        for month in months:
            issued_date_str = f"{year}-{str(month).zfill(2)}-01"
            da_in = open_seas5_cog(issued_date_str)
            da_in = da_in.squeeze(drop=True)
            da_in["year"] = year
            da_in["issued_month"] = month
            da_in = da_in.expand_dims(["year", "issued_month"])
            das.append(da_in)
    da_out = xr.combine_by_coords(das, combine_attrs="drop_conflicts")
    return da_out
