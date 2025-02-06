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

# ECMWF switch

Notebook to develop SEAS5 trigger to replaces IRI seasonal forecast.
SEAS5 trigger mechanism intended to match behaviour of IRI, keeping the same:

- issue months
- valid months
- geographic aggregation method (10% of area meets condition)

Note that return period cannot really be matched because return period of IRI
trigger was not estimated.

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
import numpy as np
import statsmodels.api as sm

from src.datasources import seas5, iri, codab
from src.utils.raster import upsample_dataarray
from src.utils.rp_calc import calculate_groups_rp
from src.utils import blob_utils
from src.constants import *
```

## SEAS5

### Loading and processing

```python
# if needed, process SEAS5 rasters (takes a few minutes)
# seas5.process_seas5_rasters()
```

```python
df_seas5 = seas5.load_seas5_stats()
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
df_seas5
```

### Checking combined RP

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

```python
heatmap_data = df_rps.pivot(index="rp_7", columns="rp_3", values="rp_overall")
```

```python
bounds = [1, 2, 2.5, 3, 4, 5, 6, 10, 1000]
tick_bounds = bounds[1:-1]
cmap = plt.cm.Spectral_r
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(dpi=200, figsize=(8, 8))

sns.heatmap(
    heatmap_data,
    annot=False,
    cmap=cmap,
    norm=norm,
    alpha=0.8,
    cbar_kws={
        "label": "Période de retour combinée",
        "ticks": tick_bounds,
        "shrink": 0.8,
    },
    ax=ax,
)
ax.invert_yaxis()
ax.set_aspect("equal", adjustable="box")
ax.set_title(
    "Périodes de retour individuelles vs. période de retour combinée,\n"
    f"prévisions saisonnières SEAS5, depuis {df_seas5['year'].min()}"
)
ax.set_xlabel("Mars : période de retour des prévisions")
ax.set_ylabel("Juillet : période de retour des prévisions")

tick_positions = np.interp(
    tick_bounds, heatmap_data.columns, range(len(heatmap_data.columns))
)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_bounds, rotation=90)
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_bounds)

for x in tick_positions:
    ax.axvline(x, color="k", alpha=0.5, linewidth=0.5)
    ax.axhline(x, color="k", alpha=0.5, linewidth=0.5)

plt.show()
```

Looks like taking 5 years for each month would yield a
3-4 year combined RP.

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

As a crude check, we see that the confidence intervals for the slope are
positive. So we can try to filter to more recent years to hopefully
make it trendless.

```python
min_year = 2000
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
rp_individual_seas5 = 5

thresh_3 = df_pivot_recent["issued_3"].quantile(1 / rp_individual_seas5)
thresh_7 = df_pivot_recent["issued_7"].quantile(1 / rp_individual_seas5)

rp_overall = (len(df_pivot_recent) + 1) / df_pivot_recent[
    (df_pivot_recent["issued_3"] <= thresh_3)
    | (df_pivot_recent["issued_7"] <= thresh_7)
]["year"].nunique()

fig, ax = plt.subplots(dpi=200, figsize=(6, 6))

# uncomment to plot actual points
# df_pivot.plot(
#     x="issued_3",
#     y="issued_7",
#     ax=ax,
#     marker=".",
#     linewidth=0,
#     legend=False,
#     color="k",
# )

xmin = df_pivot_recent["issued_3"].min() * 0.95
xmax = df_pivot_recent["issued_3"].max() * 1.02
ymin = df_pivot_recent["issued_7"].min() * 0.95
ymax = df_pivot_recent["issued_7"].max() * 1.02

alpha = 0.1

color_3 = "darkorange"
ax.axvline(thresh_3, color=color_3)
ax.axvspan(xmin=xmin, xmax=thresh_3, facecolor=color_3, alpha=alpha)
ax.annotate(
    f" Seuil PR {rp_individual_seas5}-ans = {thresh_3:.2f} mm",
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
    f" Seuil PR {rp_individual_seas5}-ans = {thresh_7:.2f} mm",
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
    "Prévision de mars : précipitations moyennes JJA,\n"
    "10e centile sur la zone d'intérêt (mm / jour)"
)
ax.set_ylabel(
    "Prévision de juillet : précipitations moyennes JJA,\n"
    "10e centile sur la zone d'intérêt (mm / jour)"
)
ax.set_title(
    f"Déclenchements historiques des prévisions SEAS, depuis {min_year}\n"
    f"(période de retour combinée = {rp_overall:.2f} ans)"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

## IRI

### Load and process

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

Do same processing with IRI, although this is faster since it's reading a
local file

```python
ds_iri = iri.load_raw_iri()
```

```python
da_iri = ds_iri.isel(C=0)["prob"]
da_iri = da_iri.rio.write_crs(4326)
```

```python
da_iri_clip_low = da_iri.rio.clip(adm1.geometry, all_touched=True)
```

```python
da_iri_up = upsample_dataarray(da_iri_clip_low, x_var="X", y_var="Y")
```

```python
da_iri_clip = da_iri_up.rio.clip(adm1.geometry)
```

```python
da_iri_q = da_iri_clip.quantile(1 - ORIGINAL_Q, dim=["X", "Y"])
```

```python
da_iri_q
```

```python
df_iri = da_iri_q.to_dataframe("q")["q"].reset_index()
```

```python
df_iri["issued_date"] = pd.to_datetime(df_iri["F"].astype(str))
df_iri["issued_year"] = df_iri["issued_date"].dt.year
df_iri["issued_month"] = df_iri["issued_date"].dt.month
```

```python
df_iri_triggers = df_iri[
    ((df_iri["issued_month"] == 3) & (df_iri["L"] == 3))
    | ((df_iri["issued_month"] == 7) & (df_iri["L"] == 1))
][["issued_year", "issued_month", "q"]].rename(columns={"issued_year": "year"})
df_iri_triggers
```

### Plot trend

```python
df_iri_triggers.pivot(index="year", columns="issued_month", values="q").plot()
```

```python
df_iri_triggers
```

```python
for issued_month in [3, 7]:
    dff = df_iri_triggers[df_iri_triggers["issued_month"] == issued_month]
    X = sm.add_constant(dff.index)
    model = sm.OLS(dff["q"], X).fit()
    print(f"issued month {issued_month}")
    print(model.summary())
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/iri/iri_original_trigger_raster_stats.parquet"
blob_utils.upload_parquet_to_blob(df_iri_triggers, blob_name)
```

## Comparison

### Plot comparison plots

```python
df_compare = df_seas5.merge(
    df_iri_triggers, on=["year", "issued_month"], suffixes=["_seas5", "_iri"]
)
```

```python
x_var = "q_iri"
y_var = "q_seas5"
x_color = "darkblue"
y_color = "green"
xmin, xmax = 20, 45
alpha = 0.1

for issued_month, seas5_thresh, trimester in zip(
    [3, 7], [thresh_3, thresh_7], ["JJA", "ASO"]
):
    fig, ax = plt.subplots(dpi=200)
    dff = df_compare[df_compare["issued_month"] == issued_month]

    for year, row in dff.set_index("year").iterrows():
        ax.annotate(
            year,
            (row[x_var], row[y_var]),
            va="center",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    y_buffer = (dff[y_var].max() - dff[y_var].min()) * 0.1
    ymin, ymax = (
        dff[y_var].min() - y_buffer,
        dff[y_var].max() + y_buffer,
    )

    ax.axvline(ORIGINAL_IRI_THRESH, color=x_color)
    ax.axvspan(ORIGINAL_IRI_THRESH, xmax, facecolor=x_color, alpha=alpha)
    ax.annotate(
        " Seuil actuel du cadre",
        (ORIGINAL_IRI_THRESH, ymin),
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color=x_color,
    )

    ax.axhline(seas5_thresh, color=y_color)
    ax.axhspan(ymin, seas5_thresh, facecolor=y_color, alpha=alpha)
    ax.annotate(
        f" Seuil PR {rp_individual_seas5}-ans = {seas5_thresh:.2f} mm",
        (xmin, seas5_thresh),
        va="bottom",
        ha="left",
        fontsize=8,
        color=y_color,
    )

    ax.set_title(
        "Comparaison des prévisions de "
        f"{FRENCH_MONTHS.get(calendar.month_abbr[issued_month])} "
        f"pour {trimester}"
    )
    ax.set_ylabel(
        "Précipitations moyennes sur trimestre,\n"
        "10e centile sur zone d'intérêt (mm / jour) [SEAS5]"
    )
    ax.set_xlabel(
        f"Probabilité de précipitations inférieures à normale,\n"
        "90e centile sur zone d'intérêt (%) [IRI]"
    )

    ax.set_xlim((20, xmax))
    ax.set_ylim((ymin, ymax))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
```

```python

```
