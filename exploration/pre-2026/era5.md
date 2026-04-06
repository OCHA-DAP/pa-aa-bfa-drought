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

# ERA5

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from dask.diagnostics import ProgressBar

from src.datasources import era5, codab
from src.utils.raster import upsample_dataarray
from src.constants import *
```

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

```python
da_era5 = era5.open_era5_rasters()
```

```python
da_era5
```

```python
da_era5_season = da_era5.mean(dim="issued_month")
```

```python
da_era5_up = upsample_dataarray(da_era5_season)
```

```python
da_era5_clip = da_era5_up.rio.clip(adm1.geometry)
```

```python
da_era5_clip
```

```python
da_era5_clip_yearchunk = da_era5_clip.chunk({"year": -1})
```

```python
da_era5_rank = da_era5_clip_yearchunk.rank(dim="year", pct=True)
```

```python
with ProgressBar():
    da_era5_rank_computed = da_era5_rank.compute()
```

```python
da_era5_rank_q = da_era5_rank.quantile(q=ORIGINAL_Q, dim=["x", "y"])
```

```python
with ProgressBar():
    da_era5_rank_q_computed = da_era5_rank_q.compute()
```

```python
df_era5_rank_q = da_era5_rank_q_computed.to_dataframe("q")["q"].reset_index()
```

```python
df_era5_rank_q.plot(x="year", y="q")
```

```python
df_era5_rank_q[df_era5_rank_q["year"] >= 2001].sort_values("q")
```
