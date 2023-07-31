import datetime
import os
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv

# from ochanticipy import CodAB, GeoBoundingBox, create_country_config

iso3 = "bfa"
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

load_dotenv()

NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SPI_PROC_DIR = AA_DATA_DIR / "public/processed/bfa/cmorph_spi/new"


def compute_spi_trigger():
    date = datetime.date(2023, 7, 26)
    filename = f"bfa-NOAA_CPC_CMORPH-spi-1mo_{date}.nc"
    ds = xr.open_dataset(SPI_PROC_DIR / filename)
    # country_config = create_country_config(iso3=iso3)
    # codab = CodAB(country_config=country_config)
    # gdf_adm1 = codab.load(admin_level=1)
    # gdf_aoi = gdf_adm1[gdf_adm1["ADM1_FR"].isin(adm_sel)]
    # geobb = GeoBoundingBox.from_shape(gdf_adm1)
    print(ds)
    return


def compute_wrsi_trigger():
    return


if __name__ == "__main__":
    compute_spi_trigger()
    compute_wrsi_trigger()
