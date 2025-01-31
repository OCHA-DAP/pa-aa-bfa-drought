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


def process_spi(verbose: bool = False):
    country_config = create_country_config(iso3="bfa")
    codab = CodAB(country_config=country_config)
    gdf_adm0 = codab.load()

    filenames = [
        file for file in os.listdir(NEW_CMORPH_DIR) if file.endswith(".tif")
    ]
    for filename in filenames:
        print(filename)
        # note: they keep changing the filename and directory structure,
        # so this is a bit messy for now
        clean_filename = filename.removeprefix(
            "datasets_cog_v1_2023_2023-09-30_"
        )
        datestr = clean_filename.removeprefix(
            "GLOBAL-NOAA_CPC_CMORPH-spi-1mo"
        ).removesuffix(".tif")[1:]
        output_filename = f"bfa-spi1-{datestr}.nc"
        output_filepath = SPI_PROC_DIR / output_filename
        if output_filepath.exists():
            print(f"already processed {output_filename}")
            continue
        date = pd.to_datetime(datestr, format="%Y-%m-%d")
        da = rxr.open_rasterio(NEW_CMORPH_DIR / filename)
        if verbose:
            print(da)
        da = da.rio.clip(gdf_adm0["geometry"], all_touched=True)
        da = da.expand_dims(dim={"date": [date]})
        da = da.sel(band=1).drop("band")
        ds = da.to_dataset(name="spi_1")
        if output_filepath.exists():
            output_filepath.unlink()
        ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    process_spi()
