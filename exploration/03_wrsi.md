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
)
from dotenv import load_dotenv
import datetime
import os
from pathlib import Path
import rioxarray as rxr
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from colour import Color
import pandas as pd
```

```python
adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

load_dotenv()
country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1.loc[gdf_adm1.ADM1_FR.isin(adm1_sel)]

RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/bfa"
WRSI_DIR = RAW_DIR / "fewsnet/wrsi"
gdf_aoi.plot()
```

```python
start_date = datetime.date(2023, 5, 1)
end_date = datetime.date(2023, 7, 1)
dates = pd.date_range(start_date, end_date, freq="10D")
for date in dates:
    dekad = date.dayofyear // 10
    print(dekad)
```

```python
# wa is croplands, w1 is rangelands

start_date = datetime.date(2023, 5, 1)
end_date = datetime.date(2023, 7, 1)
dates = pd.date_range(start_date, end_date, freq="W")

dss = []

for date in dates:
    dekad = date.dayofyear // 10
    year = date.year - 2000
    zip_filename = "zip+file://" / WRSI_DIR / "w2317wa.zip!/w2317dd.tif"
    filename = WRSI_DIR / f"w{year}{dekad}wa/w{year}{dekad}dd.tif"
    try:
        ds = rxr.open_rasterio(filename)
    except Exception as error:
        print(error)
        continue
    ds = ds.rio.reproject("EPSG:4326")
    ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
    dss.append(ds)

    print(dss)
ds = xr.combine_by_coords(dss)
df = ds.to_dataframe(name="WRSI").reset_index()
df = df.drop(columns=["spatial_ref", "band"])
df = df[~(df["WRSI"] == 255)]
year = 2023
dekad = 17
dayofyear = (dekad - 1) * 10
date = datetime.datetime(2023, 1, 1) + datetime.timedelta(dayofyear - 1)
df["date"] = date
print(df)
```

```python
# bin WRSI

bins = [0, 50, 60, 80, 95, 100]
bin_labels = []
bin_colors = [
    x.get_hex() for x in Color("red").range_to(Color("white"), len(bins))
]
for x in range(len(bins) - 1):
    bin_labels.append(f"{bins[x]} to {bins[x+1]}")
df_agg = (
    df.groupby(["date", pd.cut(df["WRSI"], bins, labels=bin_labels)])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
print(df_agg)
```

```python

```
