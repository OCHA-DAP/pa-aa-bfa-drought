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

# C3S multimodel anomaly and rank

Include all models.

Using only April for JJASO, as specified in 2026 framework revision

<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import calendar

import ocha_stratus as stratus
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
import statsmodels.api as sm
from dask.diagnostics import ProgressBar
from scipy.stats import skewnorm

from src.datasources import seas5, iri, codab, era5
from src.utils.raster import upsample_dataarray
from src.utils.rp_calc import calculate_groups_rp, calculate_one_group_rp
from src.utils import blob_utils
from src.utils.blob_utils import PROJECT_PREFIX
from src.constants import *
```

```python
adm2 = codab.load_codab_from_blob(admin_level=2)
```

```python
adm2_aoi = adm2[adm2["ADM2_PCODE"].isin(AOI_ADM2_PCODES_2026)]
```

```python
adm2_aoi.plot()
```

```python
adm2_aoi.to_crs(3857).area.sum()
```

```python
adm2_aoi_old = adm2[adm2["ADM1_PCODE"].isin(AOI_ADM1_PCODES)]
```

```python
adm2_aoi_old.plot()
```

```python
adm2_aoi_old.to_crs(3857).area.sum() / adm2_aoi.to_crs(3857).area.sum()
```

## Process anomaly

```python
new_mo_lt_combos = [{"mo": 4, "lts": [2, 3, 4, 5, 6]}]
new_years = range(1981, 2025 + 1)
```

```python
da_seas5 = seas5.open_seas5_rasters(
    mo_lt_combos=new_mo_lt_combos, years=new_years
)
```

```python
da_seas5
```

```python
da_seas5_tri = da_seas5.mean(dim="lt")
da_seas5_up = upsample_dataarray(da_seas5_tri)
da_seas5_clip = da_seas5_up.rio.clip(adm2_aoi.geometry)
```

```python
da_seas5_clip
```

```python
da_seas5_clip
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
da_seas5_clip.isel(year=-1).plot(ax=ax, cmap="RdBu")
adm2_aoi.boundary.plot(ax=ax, color="k")
ax.axis("off")
```

## Process rank/percentile

```python
da_seas5_clip_yearchunk = da_seas5_clip.chunk({"year": -1})
```

```python
da_seas5_rank = da_seas5_clip_yearchunk.rank(dim="year", pct=True)
```

```python
with ProgressBar():
    da_seas5_rank_computed = da_seas5_rank.compute()
```

```python
da_seas5_rank_computed.isel(x=20, y=1, issued_month=0)
```

```python
vmin, vmax = 0, 1
```

```python
da_seas5_rank_computed.sel(year=2015, issued_month=4).plot(
    cmap="RdBu", vmin=vmin, vmax=vmax
)
```

```python
da_seas5_rank_q = da_seas5_rank_computed.quantile(q=ORIGINAL_Q, dim=["x", "y"])
```

```python
with ProgressBar():
    da_seas5_rank_q_computed = da_seas5_rank_q.compute()
```

```python
df_seas5_rank_q = da_seas5_rank_q_computed.to_dataframe("q")["q"].reset_index()
```

```python
df_seas5_rank_q["q"].hist()
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_rank_q10_2026.parquet"  # noqa
blob_utils.upload_parquet_to_blob(df_seas5_rank_q, blob_name)
```

### C3S

```python
blob_name = f"{PROJECT_PREFIX}/processed/ensemble/equal_weight_forecast.nc"
data = stratus.load_blob_data(blob_name)

with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
    f.write(data)
    tmp_path = f.name
try:
    ew = xr.open_dataarray(tmp_path)
finally:
    os.unlink(tmp_path)
```

```python
ew
```

```python
ew.isel(year=0).plot()
```

```python
ew.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
```

```python
ew_up = upsample_dataarray(ew, x_var="lon", y_var="lat")
ew_up = ew_up.rename({"lon": "x", "lat": "y"})
ew_clip = ew_up.rio.clip(adm2_aoi.geometry)
```

```python
ew_clip.isel(year=0).plot()
```

```python
ew_clip_yearchunk = ew_clip.chunk({"year": -1})
```

```python
ew_clip_rank = ew_clip_yearchunk.rank(dim="year", pct=True)
```

```python
with ProgressBar():
    ew_clip_rank_computed = ew_clip_rank.compute()
```

```python
ew_clip_rank_computed.isel(x=20, y=1)
```

```python
vmin, vmax = 0, 1
```

```python
ew_clip_rank_computed.sel(year=2015).plot(cmap="RdBu", vmin=vmin, vmax=vmax)
```

```python
ew_clip_rank_q = ew_clip_rank_computed.quantile(q=ORIGINAL_Q, dim=["x", "y"])
```

```python
ew_clip_rank_computed.mean(dim=["x", "y"])
```

```python
with ProgressBar():
    ew_clip_rank_q_computed = ew_clip_rank_q.compute()
```

```python
df_ew_rank_q = ew_clip_rank_q_computed.to_dataframe("q")["q"].reset_index()
```

```python
df_ew_rank_q
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/c3s/c3s_ew_rank_q10_2026.parquet"  # noqa
blob_utils.upload_parquet_to_blob(df_ew_rank_q, blob_name)
```

### For ERA5

```python
da_era5 = era5.open_era5_rasters(months=[6, 7, 8, 9, 10])
```

```python
da_era5
```

```python
da_era5_tri = da_era5.mean(dim="issued_month")
da_era5_up = upsample_dataarray(da_era5_tri)
da_era5_clip = da_era5_up.rio.clip(adm2_aoi.geometry)
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
da_era5_rank_computed.isel(year=0).plot()
```

```python
da_era5_rank_q_computed = da_era5_rank_computed.quantile(
    q=ORIGINAL_Q, dim=["x", "y"]
)
```

```python
df_era5_rank_q = da_era5_rank_q_computed.to_dataframe("q")["q"].reset_index()
```

```python
df_era5_rank_q
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/era5/era5_rank_q10_2026.parquet"  # noqa
blob_utils.upload_parquet_to_blob(df_era5_rank_q, blob_name)
```

## SEAS5 thresholds

### Loading and processing

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_rank_q10_2026.parquet"  # noqa
df_seas5_rank_q = stratus.load_parquet_from_blob(blob_name)
```

```python
df_seas5_rank_q = calculate_one_group_rp(df_seas5_rank_q, col_name="q")
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/era5/era5_rank_q10_2026.parquet"  # noqa
df_era5_rank_q = stratus.load_parquet_from_blob(blob_name)
```

```python
df_era5_rank_q = calculate_one_group_rp(df_era5_rank_q, col_name="q")
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/c3s/c3s_ew_rank_q10_2026.parquet"  # noqa
df_ew_rank_q = stratus.load_parquet_from_blob(blob_name)
```

```python
df_ew_rank_q = calculate_one_group_rp(df_ew_rank_q, col_name="q")
```

```python
df_seas5_rank_q[df_seas5_rank_q["year"] >= 2000].sort_values("q_rank")
```

```python
df_era5_rank_q[df_era5_rank_q["year"] >= 2000].sort_values("q_rank")
```

```python
df_compare = df_era5_rank_q.merge(
    df_seas5_rank_q, on="year", suffixes=("_e", "_s")
).merge(df_ew_rank_q, on="year")
```

```python
df_compare
```

```python
df_compare.corr()["q_e"]
```

```python
df_compare.plot(x="q_s", y=["q_e"], marker=".", linewidth=0)
```

```python
df_compare.plot(x="q", y=["q_e"], marker=".", linewidth=0)
```

```python
df_compare.corr()
```

```python
df_compare.plot(x="year", y=["q_e", "q_s", "q"])
```

```python
df_seas5_rank_q_recent = df_seas5_rank_q[
    df_seas5_rank_q["year"] >= 2000
].copy()
```

```python
df_seas5_rank_q_recent = calculate_one_group_rp(
    df_seas5_rank_q_recent, col_name="q"
)
```

```python
df_seas5_rank_q_recent.sort_values("q_rank")
```

```python
df_seas5_median = (
    da_seas5_rank_computed.median(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("median_s")["median_s"]
    .reset_index()
)
df_era5_median = (
    da_era5_rank_computed.median(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("median_e")["median_e"]
    .reset_index()
)
df_ew_median = (
    ew_clip_rank_computed.median(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("median_c")["median_c"]
    .reset_index()
)
```

```python
df_comp_median = df_seas5_median.merge(df_era5_median).merge(df_ew_median)
```

```python
df_comp_median.corr()
```

```python
df_comp_median.plot(x="median_e", y=["median_c", "median_s"])
```

```python
df_comp_median.sort_values("median_c")
```

```python
(len(df_comp_median) + 1) / 3
```

```python
df_seas5_mean = (
    da_seas5_rank_computed.mean(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("mean_s")["mean_s"]
    .reset_index()
)
df_era5_mean = (
    da_era5_rank_computed.mean(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("mean_e")["mean_e"]
    .reset_index()
)
df_ew_mean = (
    ew_clip_rank_computed.mean(dim=["x", "y"])
    .squeeze(drop=True)
    .to_dataframe("mean_c")["mean_c"]
    .reset_index()
)
```

```python
df_comp_mean = df_seas5_mean.merge(df_era5_mean).merge(df_ew_mean)
```

```python
df_comp_mean.corr()
```

```python
(2025-2001+1+1)/4
```

```python
4/(2025-2001+1+1)
```

```python
5/(2025-2001+1+1)
```

```python
(2025-2001+1+1)/5
```

```python
(2025-2001+1+1)/6
```

```python
(2025-2001+1+1)/7
```

```python
(2025 - 2001 + 1 + 1) / 8
```

```python
8 / (2025 - 2001 + 1 + 1)
```

```python
8 / (2025 - 2001 + 1 + 1) * 8
```

```python

```

Quick comparison of the three metrics shows high correlation as expected.

Seems like (and I have no clue how significant this is) Z-score is a bit closer to anomaly than rank.
And both rank and anomaly are closer to Z-score than to each other.

```python
for mo, group in df_seas5_compare.groupby("issued_month"):
    display(group[[x for x in df_seas5_compare.columns if "q_" in x]].corr())
```

```python
df_seas5_compare.plot(x="q_anomaly", y="q_zscore")
```

```python
x_var, y_var = "q_zscore", "q_anomaly"
for mo, group in df_seas5_compare.groupby("issued_month"):
    fig, ax = plt.subplots()
    df_seas5_compare.plot(x=x_var, y=y_var, ax=ax, linewidth=0, legend=False)
    for year, row in group.set_index("year").iterrows():
        ax.annotate(year, (row[x_var], row[y_var]), fontsize=6)
    ax.set_ylabel(y_var)
    ax.set_title(f"issued {mo}")
```

```python
x_var, y_var = "q_zscore", "q_rank"
for mo, group in df_seas5_compare.groupby("issued_month"):
    fig, ax = plt.subplots()
    df_seas5_compare.plot(x=x_var, y=y_var, ax=ax, linewidth=0, legend=False)
    for year, row in group.set_index("year").iterrows():
        ax.annotate(year, (row[x_var], row[y_var]), fontsize=6)
    ax.set_ylabel(y_var)
    ax.set_title(f"issued {mo}")
```

```python
x_var, y_var = "q_rank", "q_anomaly"
for mo, group in df_seas5_compare.groupby("issued_month"):
    fig, ax = plt.subplots()
    df_seas5_compare.plot(x=x_var, y=y_var, ax=ax, linewidth=0, legend=False)
    for year, row in group.set_index("year").iterrows():
        ax.annotate(year, (row[x_var], row[y_var]), fontsize=6)
    ax.set_ylabel(y_var)
    ax.set_title(f"issued {mo}")
```

Next cell proceeds with rest of calculations using rank.

This is selected purely because it can argued that it's closest in meaning to the original trigger (tercile forecast), since the output is a percentile.
I think, this also fits best with the overall intent of the framework- to capture the worst historical years. The cleanest way to do that in my mind is to just take the historical rank (in percentile form), and avoids the issues of having to make any assumptions about the distribution.

```python
df_seas5 = df_seas5_rank.copy()
```

```python
# just check the histogram to see that it's sensible
for issued_month, group in df_seas5.groupby("issued_month"):
    group["q"].hist(alpha=0.3)
```

```python
df_seas5 = calculate_groups_rp(df_seas5, ["issued_month"])
```

```python
df_pivot_rps = df_seas5.pivot(
    index="year", columns="issued_month", values="q_rp"
).reset_index()
df_pivot_rps = df_pivot_rps.rename(columns={x: f"issued_{x}" for x in [3, 7]})
```

```python
dicts = []
min_individual_rp = 2

rp_list = df_seas5["q_rp"].unique()
rp_list = rp_list[rp_list >= min_individual_rp]
for rp_3 in rp_list:
    for rp_7 in rp_list:
        dff = df_pivot_rps[
            (df_pivot_rps["issued_3"] >= rp_3)
            | (df_pivot_rps["issued_7"] >= rp_7)
        ]
        dicts.append(
            {
                "rp_3": rp_3,
                "rp_7": rp_7,
                "rp_overall": (df_seas5["year"].nunique() + 1)
                / dff["year"].nunique(),
            }
        )
df_rps = pd.DataFrame(dicts)
```

### Check trend

```python
df_pivot = df_seas5.pivot(
    index="year", columns="issued_month", values="q"
).reset_index()
df_pivot = df_pivot.rename(columns={x: f"issued_{x}" for x in [3, 7]})
```

```python
df_pivot.plot(x="year", y=["issued_3", "issued_7"])
```

```python
for issued_month in [3, 7]:
    X = sm.add_constant(df_pivot.index)
    model = sm.OLS(df_pivot[f"issued_{issued_month}"], X).fit()
    print(f"issued month {issued_month}")
    print(model.summary())
```

Same as for absolute values -
as a crude check, we see that the confidence intervals for the slope are
positive. So we can try to filter to more recent years to hopefully
make it trendless.

```python
min_year = 2001
df_pivot_recent = df_pivot[df_pivot["year"] >= min_year]
```

```python
df_pivot_recent.plot(x="year", y=["issued_3", "issued_7"])
```

```python
for issued_month in [3, 7]:
    X = sm.add_constant(df_pivot_recent.index)
    model = sm.OLS(df_pivot_recent[f"issued_{issued_month}"], X).fit()
    print(f"issued month {issued_month}")
    print(model.summary())
```

Confidence intervals of slope span 0 now, so we're good.

```python
df_pivot_recent
```

### Plot historical activations

```python
rp_individual_seas5 = 9

thresh_3 = df_pivot_recent["issued_3"].quantile(1 / rp_individual_seas5)
thresh_7 = df_pivot_recent["issued_7"].quantile(1 / rp_individual_seas5)

rp_overall = (len(df_pivot_recent) + 1) / df_pivot_recent[
    (df_pivot_recent["issued_3"] <= thresh_3)
    | (df_pivot_recent["issued_7"] <= thresh_7)
]["year"].nunique()

fig, ax = plt.subplots(dpi=200, figsize=(6, 6))

# min_val = -0.19
# max_val = 0.11
min_val, max_val = 0, 1
xmin = min_val
xmax = max_val
ymin = min_val
ymax = max_val

alpha = 0.1

color_3 = "darkorange"
ax.axvline(thresh_3, color=color_3)
ax.axvspan(xmin=xmin, xmax=thresh_3, facecolor=color_3, alpha=alpha)
ax.annotate(
    f" Seuil PR {rp_individual_seas5}-ans = {thresh_3:.2f}",
    (thresh_3, ymin),
    rotation=90,
    ha="right",
    va="bottom",
    fontsize=8,
    color=color_3,
)

color_7 = "rebeccapurple"
ax.axhline(thresh_7, color=color_7)
ax.axhspan(ymin=ymin, ymax=thresh_7, facecolor=color_7, alpha=alpha)
ax.annotate(
    f" Seuil PR {rp_individual_seas5}-ans = {thresh_7:.2f}",
    (xmin, thresh_7),
    ha="left",
    va="bottom",
    fontsize=8,
    color=color_7,
)

for year, row in df_pivot_recent.set_index("year").iterrows():
    ax.annotate(
        year,
        (row["issued_3"], row["issued_7"]),
        va="center",
        ha="center",
        fontsize=6,
        fontweight="bold",
    )

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

ax.set_xlabel(
    "Prévision de mars : z-score précipitations JJA,\n"
    "10e centile sur la zone d'intérêt"
)
ax.set_ylabel(
    "Prévision de juillet : z-score précipitations JJA,\n"
    "10e centile sur la zone d'intérêt"
)
ax.set_title(
    f"Déclenchements historiques des prévisions SEAS, depuis {min_year}\n"
    f"(période de retour combinée = {rp_overall:.2f} ans)"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

Fixing thresholds based on proposed values for framework. We are fixing the same value for both months for a couple reasons:

- We can't really say whether the spatial quantile values we're plotting have a different distribution from issue month to issue month, so it doesn't really make sense to fix the threshold independently for each one.
- Having the same threshold for each month is just easier to remember and easier to explain.
- Also, conveniently, the threshold that corresponds to the requested forecast RP (around 6 years combined), is 20th, which is the lower quintile

```python
# thresh_3 = -0.9
# thresh_7 = -0.66
thresh_3 = 0.2
thresh_7 = thresh_3


rp_overall = (len(df_pivot_recent) + 1) / df_pivot_recent[
    (df_pivot_recent["issued_3"] <= thresh_3)
    | (df_pivot_recent["issued_7"] <= thresh_7)
]["year"].nunique()

fig, ax = plt.subplots(dpi=200, figsize=(6, 6))

# min_val = -2.2
# max_val = 1.5
min_val, max_val = 0, 1
xmin = min_val
xmax = max_val
ymin = min_val
ymax = max_val

alpha = 0.2

color_3 = "darkorange"
color_7 = "rebeccapurple"

ax.axvline(thresh_3, color=color_3)
ax.axvspan(xmin=xmin, xmax=thresh_3, facecolor=color_3, alpha=alpha)
ax.annotate(
    f" Seuil = {thresh_3* 100:.0f}e",
    (thresh_3, ymin),
    rotation=90,
    ha="right",
    va="bottom",
    fontsize=8,
    color=color_3,
)

ax.axhline(thresh_7, color=color_7)
ax.axhspan(ymin=ymin, ymax=thresh_7, facecolor=color_7, alpha=alpha)
ax.annotate(
    f" Seuil = {thresh_7* 100:.0f}e",
    (xmin, thresh_7),
    ha="left",
    va="bottom",
    fontsize=8,
    color=color_7,
)

for year, row in df_pivot_recent.set_index("year").iterrows():
    if (row["issued_3"] < thresh_3) & (row["issued_7"] < thresh_7):
        color = "black"
    elif row["issued_3"] < thresh_3:
        color = color_3
    elif row["issued_7"] < thresh_7:
        color = color_7
    else:
        color = "grey"
    ax.annotate(
        year,
        (row["issued_3"], row["issued_7"]),
        va="center",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=color,
    )

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))


# Custom formatter for the ticks
def custom_percentage_formatter(x, pos):
    return f"{x * 100:.0f}e"  # Multiplies by 100 and appends 'e'


ax.xaxis.set_major_formatter(FuncFormatter(custom_percentage_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(custom_percentage_formatter))

ax.set_xlabel(
    "Prévision de mars : centile historique des précipitations JJA,\n"
    "10e centile sur la zone d'intérêt"
)
ax.set_ylabel(
    "Prévision de juillet : centile historique des précipitations ASO,\n"
    "10e centile sur la zone d'intérêt"
)
ax.set_title(
    f"Déclenchements historiques des prévisions SEAS, depuis {min_year}\n"
    f"(période de retour combinée = {rp_overall:.2f} ans)".replace(".", ",")
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

```python
(2024 - 2001 + 1) * 2
```

```python
((2024 - 2001 + 1) * 2 + 1) / 5
```

```python

```
