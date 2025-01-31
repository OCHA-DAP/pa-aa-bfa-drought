---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
---

# Oct 2024 monitoring

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import pandas as pd
import xarray as xr

from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
)

from src.download_cmorph_new import download_historical_spi
from src.process_spi import process_spi
```

```python
NEW_CMORPH_DIR = Path(os.environ["NEW_CMORPH_DIR"])
AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
SPI_PROC_DIR = AA_DATA_DIR / "public/processed/bfa/cmorph_spi/new"
```

```python
iso3 = "bfa"
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

country_config = create_country_config(iso3=iso3)
codab = CodAB(country_config=country_config)
gdf_adm1 = codab.load(admin_level=1)
gdf_aoi = gdf_adm1[gdf_adm1["ADM1_FR"].isin(adm_sel)]
```

```python
gdf_aoi.plot()
```

```python
dates = pd.date_range("2024-09-01", "2024-10-01")
```

```python
for date in dates:
    download_historical_spi(str(date.date()))
```

```python
process_spi(verbose=True)
```

```python
f"{dates[0].date()}"
```

```python
verbose = False
das = []
for date in dates:
    filename = f"bfa-spi1-{date.date()}.nc"
    try:
        da_in = xr.open_dataset(SPI_PROC_DIR / filename)["spi_1"]
    except Exception as e:
        if verbose:
            print(e)
        continue
    das.append(da_in)

da = xr.concat(das, dim="date")
```

```python
da.mean(dim=["x", "y"])
```

```python
da.quantile(0.3, dim=["x", "y"]).plot()
```

```python

```
