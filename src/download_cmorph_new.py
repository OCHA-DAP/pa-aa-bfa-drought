import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
SPI1_CURRENT_URL = (
    "https://storage.googleapis.com/noaa-nidis-drought-gov-data"
    "/current-conditions/tile/v1/ce-GLOBAL-NOAA_CPC_CMORPH-spi-1mo/"
)
SPI_HISTORICAL_BASE_URL = (
    "https://storage.googleapis.com/noaa-nidis-drought-gov-data/"
    "datasets/cog/v1/2024/{date_str}/"
    "GLOBAL-NOAA_CPC_CMORPH-spi-1mo-{date_str}.tif"
)
current_data_name = "GLOBAL-NOAA_CPC_CMORPH-spi-1mo.tif"
DATA_URL = SPI1_CURRENT_URL + current_data_name
current_metadata_name = "info.json"
METADATA_URL = SPI1_CURRENT_URL + current_metadata_name


def download_historical_spi(date_str):
    url = SPI_HISTORICAL_BASE_URL.format(date_str=date_str)
    filename = f"GLOBAL-NOAA_CPC_CMORPH-spi-1mo_{date_str}.tif"
    savepath = NEW_CMORPH_DIR / filename
    try:
        urllib.request.urlretrieve(url, savepath)
        print(f"Downloaded to {savepath}")
    except Exception as e:
        print(e)


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
