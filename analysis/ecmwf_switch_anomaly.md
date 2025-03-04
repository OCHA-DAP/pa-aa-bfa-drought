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

# ECMWF z-score

Doing the same thing as in `ecmwf_switch` but with Z-score

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import calendar

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import statsmodels.api as sm
from dask.diagnostics import ProgressBar
from scipy.stats import skewnorm

from src.datasources import seas5, iri, codab
from src.utils.raster import upsample_dataarray
from src.utils.rp_calc import calculate_groups_rp
from src.utils import blob_utils
from src.constants import *
```

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

## Process Z-score

```python
da_seas5 = seas5.open_seas5_rasters()
```

```python
da_seas5_tri = da_seas5.mean(dim="lt")
da_seas5_up = upsample_dataarray(da_seas5_tri)
da_seas5_clip = da_seas5_up.rio.clip(adm1.geometry)
```

```python
da_seas5_clip
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
da_seas5_clip.isel(year=-1, issued_month=1).plot(ax=ax, cmap="RdBu")
adm1.boundary.plot(ax=ax, color="k")
ax.axis("off")
```

### Calculate mean

Computing and plotting things as I go just to ensure they look sensible.

```python
da_seas5_mean = da_seas5_clip.mean(dim="year")
```

```python
with ProgressBar():
    da_seas5_mean_computed = da_seas5_mean.compute()
```

```python
da_seas5_mean_computed.isel(issued_month=0).plot()
```

### Calculate anomaly

```python
da_seas5_anomaly = (
    da_seas5_clip - da_seas5_mean_computed
) / da_seas5_mean_computed
```

```python
with ProgressBar():
    da_seas5_anomaly_computed = da_seas5_anomaly.compute()
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
da_seas5_anomaly_computed.isel(year=-1, issued_month=1).plot(
    ax=ax, cmap="RdBu"
)
adm1.boundary.plot(ax=ax, color="k")
ax.axis("off")
```

```python
da_seas5_anomaly_q = da_seas5_anomaly_computed.quantile(
    q=ORIGINAL_Q, dim=["x", "y"]
)
```

```python
with ProgressBar():
    da_seas5_anomaly_q_computed = da_seas5_anomaly_q.compute()
```

```python
da_seas5_anomaly_q_computed.isel(issued_month=0).plot()
```

```python
vmin = da_seas5_anomaly_computed.sel(year=2015, issued_month=3).min()
vmax = -vmin
```

```python
da_seas5_anomaly_computed.sel(year=2015, issued_month=3).plot(
    vmin=vmin, vmax=vmax, cmap="RdBu"
)
```

```python
da_seas5_anomaly_computed.sel(year=2019, issued_month=3).plot(
    vmin=vmin, vmax=vmax, cmap="RdBu"
)
```

### Write to `df` and save to blob

```python
df_seas5_anomaly_q = da_seas5_anomaly_q_computed.to_dataframe("q")[
    "q"
].reset_index()
```

```python
df_seas5_anomaly_q["q"].hist()
```

```python
df_seas5_anomaly_q["q"].quantile(1 / 3)
```

```python
df_seas5_anomaly_q
```

<!-- markdownlint-disable MD013 -->

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_anomaly_q10.parquet"  # noqa
blob_utils.upload_parquet_to_blob(df_seas5_anomaly_q, blob_name)
```

### Calculate percentile

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
da_seas5_rank_computed.sel(year=2015, issued_month=3).plot(
    cmap="RdBu", vmin=vmin, vmax=vmax
)
```

```python
da_seas5_rank_computed.sel(year=2019, issued_month=3).plot(
    cmap="RdBu", vmin=vmin, vmax=vmax
)
```

```python
da_seas5_rank_q = da_seas5_rank.quantile(q=ORIGINAL_Q, dim=["x", "y"])
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
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_rank_q10.parquet"  # noqa
blob_utils.upload_parquet_to_blob(df_seas5_rank_q, blob_name)
```

## SEAS5

### Loading and processing

```python
df_seas5_anomaly = seas5.load_seas5_stats(variable="anomaly")
```

```python
df_seas5_zscore = seas5.load_seas5_stats(variable="zscore")
```

```python
df_seas5_rank = seas5.load_seas5_stats(variable="rank")
```

```python
df_seas5_compare = df_seas5_anomaly.merge(
    df_seas5_zscore,
    suffixes=("_anomaly", "_zscore"),
    on=["issued_month", "year"],
).merge(
    df_seas5_rank.rename(columns={"q": "q_rank"}), on=["issued_month", "year"]
)
df_seas5_compare = df_seas5_compare[df_seas5_compare["year"] >= 2000]
```

```python
df_seas5_compare
```

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

Fixing thresholds based on modeled RP (values calculated a few cells down)

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

alpha = 0.1

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
    fontsize=10,
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
    "Prévision de juillet : centile historique des précipitations JJA,\n"
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

```
