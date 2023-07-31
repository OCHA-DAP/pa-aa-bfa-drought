import os
from pathlib import Path

import pandas as pd
import rioxarray as rxr
from dotenv import load_dotenv
from ochanticipy import CodAB, create_country_config

load_dotenv()

NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SPI_PROC_DIR = AA_DATA_DIR / "public/processed/bfa/cmorph_spi/new"
RAW_DIR = AA_DATA_DIR / "public/raw/bfa"


def process_spi():
    country_config = create_country_config(iso3="bfa")
    codab = CodAB(country_config=country_config)
    gdf_adm0 = codab.load()

    filenames = [
        file for file in os.listdir(NEW_CMORPH_DIR) if file.endswith(".tif")
    ]
    for filename in filenames:
        datestr = filename.removeprefix(
            "GLOBAL-NOAA_CPC_CMORPH-spi-1mo_"
        ).removesuffix(".tif")
        date = pd.to_datetime(datestr, format="%Y-%m-%d")
        da = rxr.open_rasterio(NEW_CMORPH_DIR / filename)
        da = da.rio.clip(gdf_adm0["geometry"], all_touched=True)
        da = da.expand_dims(dim={"date": [date]})
        da = da.sel(band=1).drop("band")
        ds = da.to_dataset(name="spi_1")
        savename = (
            f"bfa{filename.removeprefix('GLOBAL').removesuffix('.tif')}.nc"
        )
        ds.to_netcdf(SPI_PROC_DIR / savename)


if __name__ == "__main__":
    process_spi()
