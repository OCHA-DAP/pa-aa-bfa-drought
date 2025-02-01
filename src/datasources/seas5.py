from typing import List

import xarray as xr
from tqdm.auto import tqdm

from src.constants import ORIGINAL_MO_LT_COMBOS
from src.utils import blob_utils

START_YEAR = 1981
END_YEAR = 2024


def open_seas5_cog(issued_date_str: str, lt: int):
    blob_name = (
        f"seas5/monthly/processed/precip_em_i{issued_date_str}_lt{lt}.tif"
    )
    return blob_utils.open_blob_cog(
        blob_name, stage="prod", container_name="raster"
    )


def open_seas5_rasters(mo_lt_combos: List[dict] = None):
    if mo_lt_combos is None:
        # set to original IRI framework combinations
        mo_lt_combos = ORIGINAL_MO_LT_COMBOS
    das = []
    for year in tqdm(range(START_YEAR, END_YEAR + 1)):
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
