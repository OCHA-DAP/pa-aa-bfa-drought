from dotenv import load_dotenv
from pathlib import Path
import os
import datetime
import urllib.request
import pandas as pd

load_dotenv()
DATA_DIR = Path(os.environ["AA_DATA_DIR"])
BASE_URL = "https://edcftp.cr.usgs.gov/project/fews/dekadal/africa_west/"
HISTORICAL_URL = BASE_URL + "historical/west/"
WRSI_RAW_DIR = Path(os.environ["WRSI_RAW_DIR"])

# historical

start_date = datetime.date(2001, 1, 1)
end_date = datetime.date(2023, 1, 1)
dates = pd.date_range(start_date, end_date, freq="10D")

dekads = range(13, 34)
years = range(1, 23)

for year in years:
    for dekad in dekads:
        dayofyear = (dekad - 1) * 10
        eff_date = datetime.datetime(year + 2000, 1, 1) + datetime.timedelta(
            dayofyear - 1
        )
        filename = f"w{year:02d}{dekad:02d}wa.zip"
        url = HISTORICAL_URL + filename
        save_path = WRSI_RAW_DIR / filename
        print(filename)
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(e)
