import datetime
import pandas as pd
import xarray as xr
import numpy as np
import itertools
from rasterio.enums import Resampling

iso3 = "bfa"
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
)

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SPI_PROC_DIR = AA_DATA_DIR / "public/processed/bfa/cmorph_spi/new"


def compute_spi_trigger():
    date = datetime.date(2023, 7, 26)
    filename = f"bfa-NOAA_CPC_CMORPH-spi-1mo_{date}.nc"
    ds = xr.open_dataset(SPI_PROC_DIR / filename)
    country_config = create_country_config(iso3=iso3)
    codab = CodAB(country_config=country_config)
    gdf_adm1 = codab.load(admin_level=1)
    gdf_aoi = gdf_adm1[gdf_adm1["ADM1_FR"].isin(adm_sel)]
    geobb = GeoBoundingBox.from_shape(gdf_adm1)
    print(ds)
    return


if __name__ == "__main__":
    compute_spi_trigger()