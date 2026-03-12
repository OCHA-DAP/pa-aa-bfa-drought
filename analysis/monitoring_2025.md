---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
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
import calendar

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from scipy.stats import percentileofscore
from dask.diagnostics import ProgressBar

from src.datasources import seas5, codab
from src.constants import *
from src.utils.raster import upsample_dataarray
```

```python
THRESH = 0.2
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
year = 2026
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
with ProgressBar():
    da_seas5_current_computed = da_seas5_clip.compute()
```

```python
da_seas5_current_computed
```

### SEAS5 historical

```python
# only open the rasters for this window
da_seas5_historical = seas5.open_seas5_rasters(mo_lt_combos=mo_lt_combos)
```

```python
# squeeze to remove issued_month
da_seas5_tri_h = da_seas5_historical.mean(dim="lt").squeeze(drop=True)
da_seas5_up_h = upsample_dataarray(da_seas5_tri_h)
da_seas5_clip_h = da_seas5_up_h.rio.clip(adm1.geometry)
```

```python
with ProgressBar():
    da_seas5_historical_computed = da_seas5_clip_h.compute()
```

```python
da_seas5_historical_computed
```

## Evaluate trigger

### Calculate pixel-wise percentile

Note that this excludes the current year from the percentile calculation (provided that the current year is not within the reference period 1981-2024), so the distribution doesn't change between monitoring years.

```python
# count the years where value is less than or equal to current value
# then divide by total number of years - this is the percentile rank

# taken together, these two steps are equivalent to taking the mean of the
# boolean of the historical years under/over current value, per pixel

# note that we need to mask again by the NaN areas, since the boolean
# will return False for NaN pixels, instead of NaN

da_current_percentile = (
    (da_seas5_historical_computed <= da_seas5_current_computed)
    .mean(dim="year")
    .where(~da_seas5_current_computed.isnull())
)
```

```python
da_current_percentile.max()
```

```python
# just check that it looks sensible
da_current_percentile.plot(vmin=0, vmax=1)
```

### Calculate spatial quantile

Take the 10th spatial quantile. It it's below the theshold (20%), trigger.

```python
da_current_percentile.quantile(ORIGINAL_Q, dim=["x", "y"])
```

Conversely we can look at the fraction of the area under the threshold. If it's above 10%, trigger.

```python
da_triggering = (da_current_percentile < THRESH).where(
    ~da_current_percentile.isnull()
)
```

```python
frac_area_triggering = float(da_triggering.mean())
frac_area_triggering
```

## Plotting

```python
mo = mo_lt_combos[0].get("mo")
```

```python
mo_fr = FRENCH_MONTHS.get(calendar.month_abbr[mo])
```

```python
v_mo_fr = ", ".join(
    [
        FRENCH_MONTHS.get(calendar.month_abbr[x + mo])
        for x in mo_lt_combos[0].get("lts")
    ]
)
```

```python
boundaries = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = [
    "crimson",
    "gold",
    "lightgrey",
    "lightskyblue",
    "dodgerblue",
]  # specify your desired colors

# Create a custom colormap
cmap = mcolors.ListedColormap(colors)

# Normalize the values to the specified boundaries
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# Plot the data
fig, ax = plt.subplots(dpi=200, figsize=(12, 5))
da_current_percentile.plot(cmap=cmap, norm=norm, ax=ax, add_colorbar=False)
ax.axis("off")

adm1.boundary.plot(ax=ax, color="k", linewidth=0.5)

ax.set_title(
    f"Prévisions SEAS5 publiées en {mo_fr} {year} pour {v_mo_fr},\n"
    "centile historique des précipitations totales (années références 1981-2024)"
)

# bottom_text = (
#     f"Fraction de avec centile < {THRESH*100:.0f} = {frac_area_triggering:.1f}"
# )

bottom_text = (
    f"Pourcentage de superficie avec centile historique < {THRESH*100:.0f}e = {frac_area_triggering*100:.1f}%\n"
    "(Seuil : au moins 10 %)"
)

# Add the bottom text with smaller font, italicized
ax.text(
    0.5,
    -0.1,
    bottom_text,
    ha="center",
    va="bottom",
    transform=ax.transAxes,
    fontsize=10,
    style="italic",
    color="black",
)


cbar = plt.colorbar(
    ax.collections[0], ax=ax, norm=norm, cmap=cmap, boundaries=boundaries
)
cbar.set_label("Centile historique")
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}e"))
```

## Testing

We can verify that this method of percentile rank calculation correponds to that for the threshold calculation by running the monitoring for the year 2024 and comparing the results

```python
# calculate rank using method in thresholds
da_seas5_historical_ranks = da_seas5_historical_computed.rank(
    dim="year", pct=True
)
```

```python
da_check = da_seas5_historical_ranks.sel(year=year) == da_current_percentile
```

```python
# check by plotting- values match everywhere
da_check.plot()
```

```python

```
