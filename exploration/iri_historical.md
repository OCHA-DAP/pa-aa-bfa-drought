---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
---

# IRI historical

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
    IriForecastProb,
)
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
SAH_IRI_PATH = (
    DATA_DIR
    / "private"
    / "processed"
    / "sah"
    / "iri"
    / "sah_iri_forecast_seasonal_precipitation_tercile_prob_Np24Sp7Ep24Wm6.nc"
)
BFA_PROC_DIR = DATA_DIR / "private" / "processed" / "bfa" / "iri"
```

```python
def upsample_dataarray(
    da: xr.DataArray, resolution: float = 0.1
) -> xr.DataArray:
    new_lat = np.arange(da.latitude.min(), da.latitude.max(), resolution)
    new_lon = np.arange(da.longitude.min(), da.longitude.max(), resolution)
    return da.interp(latitude=new_lat, longitude=new_lon, method="nearest")
```

```python
adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1.loc[gdf_adm1.ADM1_FR.isin(adm1_sel)]
```

```python
iri = xr.load_dataset(SAH_IRI_PATH)
iri = iri.rename({"X": "longitude", "Y": "latitude"})
iri = iri.rio.write_crs(4326)
iri_up = upsample_dataarray(iri, resolution=0.05)
iri_up_clip = iri_up.rio.clip(gdf_aoi.geometry, all_touched=True)
```

```python
fig, ax = plt.subplots()
iri_up_clip.isel(F=0, L=0, C=0)["prob"].plot(ax=ax)
gdf_aoi.boundary.plot(ax=ax)
```

```python
df = iri_up_clip.to_dataframe()["prob"].dropna().reset_index()
df["F"] = pd.to_datetime(df["F"].apply(lambda x: x.strftime("%Y-%m-%d")))
```

```python
df["year"] = df["F"].dt.year
df["f_month"] = df["F"].dt.month
df["C"] = df["C"].replace({0: "lower", 1: "average", 2: "higher"})
df = df.pivot(
    columns="C",
    values="prob",
    index=["L", "year", "f_month", "latitude", "longitude"],
).reset_index()
df
```

```python
df["high_low_diff"] = df["lower"] - df["higher"]
```

```python
area_frac = 0.1

dicts = []

for year in df["year"].unique():
    df_year = df[df["year"] == year]
    # March forecast for JJA
    dff = df_year[(df_year["f_month"] == 3) & (df_year["L"] == 3)]
    march_lower, march_diff = dff.quantile(1 - area_frac)[
        ["lower", "high_low_diff"]
    ]
    # July forecast for ASO
    dff = df_year[(df_year["f_month"] == 7) & (df_year["L"] == 1)]
    july_lower, july_diff = dff.quantile(1 - area_frac)[
        ["lower", "high_low_diff"]
    ]
    dicts.append(
        {
            "year": year,
            "march_lower": march_lower,
            "march_diff": march_diff,
            "july_lower": july_lower,
            "july_diff": july_diff,
        }
    )

results = pd.DataFrame(dicts)
results["march_act"] = (results["march_lower"] > 40) & (
    results["march_diff"] > 5
)
results["july_act"] = (results["july_lower"] > 40) & (results["july_diff"] > 5)
results["either_act"] = results["march_act"] | results["july_act"]
```

```python
results
filename = "historical_forecast_activations.csv"
results.to_csv(BFA_PROC_DIR / filename, index=False)
```
