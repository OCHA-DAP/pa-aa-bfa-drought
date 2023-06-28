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
import pandas as pd
import rioxarray
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
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
CMORPH_DIR = Path(os.environ["CMORPH_DIR"])
```

```python
# read CMORPH data

start_date = datetime.date(2020, 5, 1)
end_date = datetime.date(2020, 8, 1)
dates = pd.date_range(start_date, end_date, freq="M")

ds = xr.open_mfdataset(os.path.join(CMORPH_DIR, "*.nc"))
ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
ds = ds.sortby(ds.lon)
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
ds.rio.write_crs("EPSG:3857", inplace=True)
ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
```

```python
ds["spi_gamma_30_day"][0, :, :].plot()
```

```python

```
