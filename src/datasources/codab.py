import os
from pathlib import Path

import geopandas as gpd
import requests

from src.constants import AOI_ADM1_PCODES, ISO3
from src.utils import blob_utils

DATA_DIR = Path(os.environ["AA_DATA_DIR_NEW"])
OLD_DATA_DIR = Path(os.environ["AA_DATA_DIR"])

CODAB_RAW_DIR = (
    OLD_DATA_DIR
    / "public"
    / "raw"
    / "bfa"
    / "cod_ab"
    / "bfa_adm_igb_20200323_shp"
)


def load_codab():
    filename = "bfa_admbnda_adm2_igb_20200323.shp"
    return gpd.read_file(CODAB_RAW_DIR / filename)


def get_blob_name(iso3: str = ISO3):
    iso3 = iso3.lower()
    return f"{blob_utils.PROJECT_PREFIX}/raw/codab/{iso3}.shp.zip"


def download_codab_to_blob(iso3: str = ISO3):
    url = f"https://data.fieldmaps.io/cod/originals/{iso3}.shp.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    blob_name = get_blob_name(iso3)
    blob_utils._upload_blob_data(response.content, blob_name, stage="dev")


def load_codab_from_blob(
    iso3: str = ISO3, admin_level: int = 0, aoi_only: bool = False
):
    iso3 = iso3.lower()
    shapefile = f"{iso3}_adm{admin_level}.shp"
    gdf = blob_utils.load_shp_from_blob(
        blob_name=get_blob_name(iso3),
        shapefile=shapefile,
        stage="dev",
    )
    if admin_level > 0 & aoi_only:
        gdf = gdf[gdf["ADM1_PCODE"].isin(AOI_ADM1_PCODES)]
    return gdf
