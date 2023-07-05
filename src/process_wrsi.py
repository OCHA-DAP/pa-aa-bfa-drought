import zipfile
from dotenv import load_dotenv
from pathlib import Path
import os
import datetime

load_dotenv()

WRSI_RAW_DIR = Path(os.environ["WRSI_RAW_DIR"])


dekads = range(13, 34)
years = range(1, 23)

for year in years:
    for dekad in dekads:
        dayofyear = (dekad - 1) * 10
        eff_date = datetime.datetime(year + 2000, 1, 1) + datetime.timedelta(
            dayofyear - 1
        )
        filename = f"w{year:02d}{dekad:02d}wa.zip"
        extract_folder = filename.removesuffix(".zip")
        zip_path = WRSI_RAW_DIR / filename
        extract_path = WRSI_RAW_DIR / extract_folder
        print(filename)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            print(e)