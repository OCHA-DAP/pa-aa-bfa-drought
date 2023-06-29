from dotenv import load_dotenv
from pathlib import Path
import os
import datetime
import urllib.request
import pandas as pd

load_dotenv()
DATA_DIR = Path(os.environ["AA_DATA_DIR"])
BASE_URL = "https://edcftp.cr.usgs.gov/project/fews/dekadal/africa_west/"

start_date = datetime.date(2020, 5, 1)
end_date = datetime.date(2023, 1, 1)
dates = pd.date_range(start_date, end_date, freq="W")