---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: venv
    language: python
    name: venv
---

```python
%load_ext jupyter_black
```

```python
from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
    ChirpsMonthly,
    IriForecastDominant,
    IriForecastProb,
)
from dotenv import load_dotenv
import datetime
import os
from pathlib import Path
import pandas as pd
import rioxarray
import matplotlib.pyplot as plt
import xarray as xr
```

```python
adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

load_dotenv()
country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm2 = codab.load(admin_level=2)
geobb = GeoBoundingBox.from_shape(gdf_adm2)
gdf_aoi = gdf_adm2.loc[gdf_adm2.ADM1_FR.isin(adm1_sel)]

RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/bfa"
WFP_FILE_PATH = RAW_DIR / "chirps/bfa-rainfall-adm2-full.csv"
```

```python
# download most recent
start_date = datetime.date(2020, 1, 1)
new_chirps_monthly = ChirpsMonthly(
    country_config=country_config,
    geo_bounding_box=geobb,
    start_date=start_date,
)
new_chirps_monthly.download()
```

```python
start_date = datetime.date(2007, 11, 1)
end_date = datetime.date(2021, 1, 1)
chirps_monthly = ChirpsMonthly(
    country_config=country_config,
    geo_bounding_box=geobb,
    start_date=start_date,
    end_date=end_date,
)
chirps_monthly.process()
chirps_monthly_data = chirps_monthly.load()
```

```python
df = pd.read_csv(WFP_FILE_PATH, skiprows=[1])
```

```python
fn = "/Users/tdowning/OCHA/data/bfa/iri_spi_africa.nc"
xr = rioxarray.open_rasterio(fn, decode_times=False)
```

```python
xr[1, :, :].plot()
```

```python

```
