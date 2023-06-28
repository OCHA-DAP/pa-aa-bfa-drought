import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import datetime
import urllib.request

load_dotenv()
CMORPH_DIR = Path(os.environ["CMORPH_DIR"])
BASE_URL = "https://www.ncei.noaa.gov/pub/data/nidis/test/cmorph/"

start_date = datetime.date(2020, 5, 1)
end_date = datetime.date(2020, 8, 1)

dates = pd.date_range(start_date, end_date, freq="M")

for date in dates:
    filename = f"cmorph_spi_gamma_30_day_{date:%Y-%m-%d}.nc"
    url = BASE_URL + filename
    save_path = CMORPH_DIR / filename
    urllib.request.urlretrieve(url, save_path)