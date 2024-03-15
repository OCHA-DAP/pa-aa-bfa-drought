import os
from pathlib import Path

import geopandas as gpd

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
