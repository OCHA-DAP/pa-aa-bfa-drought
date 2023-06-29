---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: venv
    language: python
    name: venv
---

```python
%load_ext jupyter_black
```

```python
from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
)
from dotenv import load_dotenv
import datetime
import os
from pathlib import Path
import pandas as pd
import rioxarray
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
from plotly.subplots import make_subplots
from colour import Color

pyo.init_notebook_mode()
```

```python
adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

load_dotenv()
country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1.loc[gdf_adm1.ADM1_FR.isin(adm1_sel)]

RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/bfa"
CMORPH_DIR = Path(os.environ["CMORPH_DIR"])
```

```python
# read CMORPH data in bulk

ds = xr.open_mfdataset(
    os.path.join(CMORPH_DIR, "*.nc"),
    combine="nested",
)
ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
ds = ds.sortby(ds.lon)
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
ds.rio.write_crs("EPSG:3857", inplace=True)
# TODO: set up more precise clipping with downcasting
ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
df = ds.to_dataframe().reset_index().drop(columns=["spatial_ref"])
```

```python
# read CMORPH data one by one

start_date = datetime.date(2020, 5, 1)
end_date = datetime.date(2023, 1, 1)
dates = pd.date_range(start_date, end_date, freq="W")

dss = []

for date in dates:
    print(date)
    path = CMORPH_DIR / f"daily/cmorph_spi_gamma_30_day_{date:%Y-%m-%d}.nc"
    try:
        ds = xr.open_dataset(path)
    except FileNotFoundError as error:
        print(error)
        continue
    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds.rio.write_crs("EPSG:4326", inplace=True)
    try:
        ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
    except Exception as error:
        print(error)
        continue
    dss.append(ds)

ds = xr.merge(dss)
df = ds.to_dataframe().reset_index().drop(columns=["spatial_ref"])
```

```python
# Bin SPI

bins = [-float("inf"), -3, -2.5, -2, -1.5, -1, -0.5, 0, float("inf")]
bin_labels = []
bin_colors = [
    x.get_hex() for x in Color("red").range_to(Color("white"), len(bins))
]
for x in range(len(bins) - 1):
    bin_labels.append(f"{bins[x]} to {bins[x+1]}")
df_agg = (
    df.groupby(
        ["time", pd.cut(df["spi_gamma_30_day"], bins, labels=bin_labels)]
    )
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)

CELLS_CUTOFF = 150

# set dates to drop based on not enough cells
total_cells = df_agg.groupby("time").sum()
drop_dates = total_cells[total_cells["count"] < CELLS_CUTOFF].reset_index()[
    "time"
]

# set dates to drop based on dates with erroneous measurements (too many with SPI <-3)
drop_dates = pd.concat(
    [
        drop_dates,
        df_agg[
            (df_agg["spi_gamma_30_day"] == bin_labels[0])
            & (df_agg["count"] > CELLS_CUTOFF)
        ]["time"],
    ]
)

df_agg = df_agg[~df_agg["time"].isin(drop_dates)]
```

```python
# Plot SPI for all times

fig = go.Figure(
    layout=dict(
        template="simple_white",
        title="SPI distribution over time",
        xaxis_title="Date",
        yaxis_title="Percentage of AOI",
        legend_title="SPI range",
    ),
)

for bin_label, bin_color in zip(bin_labels, bin_colors):
    dff = df_agg.loc[df_agg["spi_gamma_30_day"] == bin_label]
    fig.add_trace(
        go.Scatter(
            x=dff["time"],
            y=dff["count"],
            mode="lines",
            stackgroup="one",
            groupnorm="percent",
            line=dict(width=0, color=bin_color),
            name=bin_label,
        )
    )

pyo.iplot(fig)
```

```python
# Plot SPI for months of interest

months_of_interest = [6, 7, 8, 9]
dff = df_agg[df_agg["time"].dt.month.isin(months_of_interest)]
dff["plot_date"] = dff["time"].apply(
    lambda x: datetime.datetime(1900, x.month, x.day)
)

years = dff["time"].dt.year.unique()
cols = 2
rows = len(years) // cols + 1
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[str(year) for year in years],
    shared_xaxes="all",
)
fig.update_layout(
    template="simple_white",
    title="SPI distribution over time",
)
fig.update_xaxes(tickformat="%b")

for index, year in enumerate(years):
    row = index // 2 + 1
    col = index - (row - 1) * 2 + 1
    dfff = dff[dff["time"].dt.year == year]
    for bin_label, bin_color in zip(bin_labels, bin_colors):
        dffff = dfff.loc[dfff["spi_gamma_30_day"] == bin_label]
        fig.add_trace(
            go.Scatter(
                x=dffff["plot_date"],
                y=dffff["count"],
                mode="lines",
                stackgroup=str(year),
                groupnorm="percent",
                line=dict(width=0, color=bin_color),
                name=bin_label,
                showlegend=False if index > 0 else True,
            ),
            row=row,
            col=col,
        )

pyo.iplot(fig)
```

```python
spi_thesholds = [-3, -2.5, -2, -1.5, -1, -0.5]
percent_thresholds = [0.1, 0.2, 0.3]

dff = df[
    (df["time"].dt.month.isin(months_of_interest))
    & ~(df["time"].isin(drop_dates))
]

df_recur = pd.DataFrame()

for spi_t in spi_thesholds:
    for per_t in percent_thresholds:
        for year in years:
            df_year = dff[dff["time"].dt.year == year]
            for date in df_year["time"].unique():
                df_date = df_year[df_year["time"] == date]
                df_date = df_date.dropna()
                frac = len(df_date[df_date["spi_gamma_30_day"] < spi_t]) / len(
                    df_date
                )
                if frac > per_t:
                    df_count = pd.DataFrame(
                        [[date, frac, spi_t, per_t]],
                        columns=["date", "frac", "spi_t", "per_t"],
                    )
                    df_recur = pd.concat([df_recur, df_count])


df_recur["year"] = df_recur["date"].dt.year
df_recur = df_recur.groupby(["year", "spi_t", "per_t"]).count().reset_index()

# print(df_recur)

df_recur = df_recur.pivot_table(
    index="spi_t",
    columns="per_t",
    values="frac",
    aggfunc="count",
)

df_recur = len(years) / df_recur
df_recur = df_recur.fillna("inf")

df_recur.columns = df_recur.columns.astype(str)
df_recur = df_recur.reindex(sorted(df_recur.columns, reverse=True), axis=1)
df_recur.index = df_recur.index.astype(str)

print(df_recur)

fig = px.imshow(df_recur, text_auto=True)
fig.update_layout(coloraxis_colorbar_title="Recurrence (years)")
fig.update_xaxes(side="top", title_text="Percent of AOI Threshold")
fig.update_yaxes(title="SPI Threshold")

pyo.iplot(fig)
```

```python

```
