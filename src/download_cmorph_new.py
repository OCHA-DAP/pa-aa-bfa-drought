import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import datetime
import urllib.request

load_dotenv()
NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
SPI1_CURRENT_URL = "https://storage.googleapis.com/noaa-nidis-drought-gov-data/current-conditions/tile/v1/ce-GLOBAL-NOAA_CPC_CMORPH-spi-1mo/"
current_data_name = "GLOBAL-NOAA_CPC_CMORPH-spi-1mo.tif"
DATA_URL = SPI1_CURRENT_URL + current_data_name
current_metadata_name = "info.json"
METADATA_URL = SPI1_CURRENT_URL + current_metadata_name


def download_current_spi():
    # get metadata
    with urllib.request.urlopen(METADATA_URL) as url:
        metadata = json.load(url)

    date = metadata.get("date")
    print(f"Metadata date: {date}")
    filename = f"GLOBAL-NOAA_CPC_CMORPH-spi-1mo_{date}.tif"

    # get data
    try:
        savepath = NEW_CMORPH_DIR / filename
        urllib.request.urlretrieve(DATA_URL, savepath)
        print(f"Downloaded to {savepath}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    download_current_spi()