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

# SEAS5 monitoring 2025

<!-- markdownlint-disable MD013 -->

Monitoring SEAS5 forecast for 2025 updated framework.

Trigger if AOI has at least 10% of area with seasonal rainfall in lowest quintile.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from src.datasources import seas5, codab
from src.constants import *
from src.utils.raster import upsample_dataarray
```

## Load data

### CODAB

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

### SEAS5 current

Set `window` based on monitoring window (1 is March, 2 is July).

```python
window = 1
year = 2024
```

```python
mo_lt_combos = ORIGINAL_MO_LT_COMBOS[:window]
```

```python
mo_lt_combos
```

```python
da_seas5 = seas5.open_seas5_rasters(mo_lt_combos=mo_lt_combos, years=[year])
```

```python
# squeeze to remove issued_month
da_seas5_tri = da_seas5.mean(dim="lt").squeeze(drop=True)
da_seas5_up = upsample_dataarray(da_seas5_tri)
da_seas5_clip = da_seas5_up.rio.clip(adm1.geometry)
```

```python
da_seas5_clip
```

### SEAS5 historical

```python
da_seas5_historical = seas5.open_seas5_rasters(mo_lt_combos=mo_lt_combos)
```

```python
# squeeze to remove issued_month
da_seas5_tri_h = da_seas5_historical.mean(dim="lt").squeeze(drop=True)
da_seas5_up_h = upsample_dataarray(da_seas5_tri_h)
da_seas5_clip_h = da_seas5_up_h.rio.clip(adm1.geometry)
# rechunk because otherwise percentile calc doesn't work
da_seas5_clip_rechunk_h = da_seas5_clip_h.chunk({"year": -1})
```

```python
da_seas5_clip_rechunk_h
```

```python
da_seas5_clip_h
```

## Evaluate trigger

### Calculate pixel-wise percentile

Note that this excludes the current year from the percentile calculation, so the distribution doesn't change between monitoring years.

```python
def percentile_rank(historical_values, current_value):
    # Calculate the percentile rank of the current value in the historical values
    return np.percentile(historical_values, current_value)
```

```python
percentile_ranks = xr.apply_ufunc(
    percentile_rank,
    da_seas5_clip_rechunk_h,  # historical data array (year, x, y)
    da_seas5_clip,  # current data array (x, y)
    input_core_dims=[["year", "x", "y"], ["x", "y"]],
    output_core_dims=[["x", "y"]],
    vectorize=True,
    dask="parallelized",
)
```

```python
percentile_ranks
```

```python
with ProgressBar():
    percentile_ranks_computed = percentile_ranks.compute()
```
