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

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import calendar
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from dask.diagnostics import ProgressBar

from src.datasources import codab, seas5, iri
from src.utils.raster import upsample_dataarray
from src.utils import blob_utils
from src.constants import *
```

```python
codab.download_codab_to_blob()
```

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

```python
adm1.plot()
```

```python
da_seas5 = seas5.open_seas5_rasters()
```

```python
da_seas5
```

```python
da_seas5_tri = da_seas5.mean(dim="lt")
```

```python
da_seas5_up = upsample_dataarray(da_seas5_tri)
```

```python
da_seas5_clip = da_seas5_up.rio.clip(adm1.geometry)
```

```python
da_seas5_q = da_seas5_clip.quantile(q=ORIGINAL_Q, dim=["x", "y"])
```

```python
with ProgressBar():
    da_seas5_q_computed = da_seas5_q.compute()
```

```python
da_seas5_q_computed
```

```python
df_seas5 = da_seas5_q_computed.to_dataframe("q")["q"].reset_index()
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/seas5/seas5_original_trigger_raster_stats.parquet"
blob_utils.upload_parquet_to_blob(df_seas5, blob_name)
```

```python
for issued_month, group in df_seas5.groupby("issued_month"):
    group["q"].hist(alpha=0.3)
```

```python
def calculate_groups_rp(df: pd.DataFrame, by: List):
    def calculate_rp(group, col_name: str = "q", ascending: bool = True):
        group[f"{col_name}_rank"] = group[col_name].rank(ascending=ascending)
        group[f"{col_name}_rp"] = (len(group) + 1) / group[f"{col_name}_rank"]
        return group

    return (
        df.groupby(by)
        .apply(calculate_rp, include_groups=False)
        .reset_index()
        .drop(columns=f"level_{len(by)}")
    )
```

```python
df_seas5 = calculate_groups_rp(df_seas5, ["issued_month"])
```

```python
dicts = []
min_individual_rp = 2

rp_list = df_seas5["q_rp"].unique()
rp_list = rp_list[rp_list >= min_individual_rp]
for rp_3 in rp_list:
    for rp_7 in rp_list:
        dff = df_pivot[(df_pivot[3] >= rp_3) | (df_pivot[7] >= rp_7)]
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
norm = mcolors.BoundaryNorm(bounds, cmap.N)
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
        "label": "Période de retour globale",
        "ticks": tick_bounds,
        "shrink": 0.8,
    },
    ax=ax,
)
ax.invert_yaxis()
ax.set_aspect("equal", adjustable="box")
ax.set_title(
    "Périodes de retour individuelles vs. période de retour globale,\n"
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

```python
df_pivot = df_seas5.pivot(
    index="year", columns="issued_month", values="q"
).reset_index()
df_pivot = df_pivot.rename(columns={x: f"issued_{x}" for x in [3, 7]})
```

```python
df_pivot
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

```python
min_year = 2000
df_pivot_recent = df_pivot[df_pivot["year"] >= min_year]
for issued_month in [3, 7]:
    X = sm.add_constant(df_pivot_recent.index)
    model = sm.OLS(df_pivot_recent[f"issued_{issued_month}"], X).fit()
    print(f"issued month {issued_month}")
    print(model.summary())
```

```python
df_pivot_recent = df_
```

```python
rp_individual_seas5 = 5

fig, ax = plt.subplots(dpi=200, figsize=(6, 6))

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

alpha = 0.15

color_3 = "darkorange"
thresh_3 = df_pivot_recent["issued_3"].quantile(1 / rp_individual_seas5)
ax.axvline(thresh_3, color=color_3)
ax.axvspan(xmin=xmin, xmax=thresh_3, facecolor=color_3, alpha=alpha)
ax.annotate(
    f" Seuil PR {rp_individual_seas5}-ans",
    (thresh_3, ymin),
    rotation=90,
    ha="right",
    va="bottom",
    fontsize=8,
    color=color_3,
)

color_7 = "rebeccapurple"
thresh_7 = df_pivot_recent["issued_7"].quantile(1 / rp_individual_seas5)
ax.axhline(thresh_7, color=color_7)
ax.axhspan(ymin=ymin, ymax=thresh_7, facecolor=color_7, alpha=alpha)
ax.annotate(
    f" Seuil PR {rp_individual_seas5}-ans",
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

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

```python
print("overall rp:")
print(
    (len(df_pivot_recent) + 1)
    / df_pivot_recent[
        (df_pivot_recent["issued_3"] <= thresh_3)
        | (df_pivot_recent["issued_7"] <= thresh_7)
    ]["year"].nunique()
)
```

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

```python
df_iri_triggers.pivot(index="year", columns="issued_month", values="q").plot()
```

```python
blob_name = f"{blob_utils.PROJECT_PREFIX}/processed/iri/iri_original_trigger_raster_stats.parquet"
blob_utils.upload_parquet_to_blob(df_iri_triggers, blob_name)
```

```python
df_compare = df_seas5.merge(
    df_iri_triggers, on=["year", "issued_month"], suffixes=["_seas5", "_iri"]
)
```

```python
FRENCH_MONTHS.get(calendar.month_abbr[1])
```

```python
x_var = "q_iri"
y_var = "q_seas5"
x_color = "darkblue"
y_color = "green"
xmin, xmax = 20, 45
alpha = 0.2

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
        f" Seuil PR {rp_individual_seas5}-ans",
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
