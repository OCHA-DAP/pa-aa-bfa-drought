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

# SPI current exploration

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import geopandas as gpd
import xarray as xr
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
CPC_GLOBAL_DIR = DATA_DIR / "public" / "raw" / "glb" / "cpc"
CODAB_DIR = DATA_DIR / "public" / "raw" / "bfa" / "cod_ab"
```

```python
filename = "bfa_cod_ab.shp/bfa_admbnda_adm1_igb_20200323.shp"
codab = gpd.read_file(CODAB_DIR / filename)
```

```python
codab.plot()
```

```python
filename = "GLOBAL-NOAA_CPC_CMORPH-spi-1mo_downloaded_2024-02-28.tif"
da = xr.load_dataarray(CPC_GLOBAL_DIR / filename)
da_clip = da.rio.clip(codab.geometry, all_touched=True)
```

```python
mask_val = da_clip.isel(y=0, x=20).values[0]
```

```python
df = da.to_dataframe()
```

```python
df["band_data"].value_counts().reset_index().sort_values("band_data")
```

```python
df_clip = da_clip.to_dataframe()
```

```python
df_clip["band_data"].value_counts()
```

```python
da_clip.where(da_clip == mask_val).plot()
```

```python
da.where(da == mask_val).plot()
```

```python

```
