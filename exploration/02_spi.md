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
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
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
import numpy as np
import rioxarray
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
from plotly.subplots import make_subplots
from colour import Color
import netCDF4
import time
import statsmodels.api as sm

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
PROCESSED_DIR = (
    Path(os.environ["AA_DATA_DIR"]) / "public/processed/bfa/cmorph_spi"
)
EXP_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/bfa"
```

```python jupyter={"outputs_hidden": true}
# read and process historical CMORPH file (roughly 8 seconds per saved file)

filename = CMORPH_DIR / "cmorph_spi_gamma_30_day.nc"

ds_hist = xr.open_dataarray(filename)
ds_hist.coords["lon"] = (ds_hist.coords["lon"] + 180) % 360 - 180
ds_hist = ds_hist.sortby(ds_hist.lon)
ds_hist = ds_hist.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
ds_hist.rio.write_crs("EPSG:4326", inplace=True)


start_date = pd.to_datetime(ds_hist["time"])[0]
end_date = pd.to_datetime(ds_hist["time"])[-1]
dates = pd.date_range(start_date, end_date, freq="M").to_series()
dates = dates[dates.dt.month.isin([7, 8, 9])]
print(dates)


for date in dates:
    print(f"{date:%Y-%m-%d}")
    ds = ds_hist.loc[date, :, :]
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
    ds.to_netcdf(
        CMORPH_DIR
        / f"processed/cmorph_spi_gamma_30_day_{date:%Y-%m-%d}_bfa.nc",
        engine="h5netcdf",
    )
```

```python
# process recent CMORPH files

start_date = datetime.date(2020, 5, 1)
end_date = datetime.date(2023, 3, 1)
dates = pd.date_range(start_date, end_date, freq="M").to_series()
dates = dates[dates.dt.month.isin([7, 8, 9])]

for date in dates:
    print(date)
    path = CMORPH_DIR / f"daily/cmorph_spi_gamma_30_day_{date:%Y-%m-%d}.nc"
    try:
        ds = xr.open_dataset(path)
    except FileNotFoundError as error:
        print(f"couldn't read {date}")
        print(error)
        continue
    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
    ds.to_netcdf(
        CMORPH_DIR
        / f"processed/cmorph_spi_gamma_30_day_{date:%Y-%m-%d}_bfa.nc",
        engine="h5netcdf",
    )
```

```python
# read processed files (bfa clip)

# drop dates with erroneous measurement
DROP_DATES = ["2021-09-30"]

start_date = datetime.date(1998, 7, 1)
end_date = datetime.date(2023, 3, 1)
dates = pd.date_range(start_date, end_date, freq="M").to_series()
dates = dates[dates.dt.month.isin([7, 8, 9])]
dates = dates.drop(DROP_DATES)

dss = []

for date in dates:
    path = PROCESSED_DIR / f"cmorph_spi_gamma_30_day_{date:%Y-%m-%d}_bfa.nc"
    try:
        ds = xr.open_dataset(path)
    except FileNotFoundError as error:
        print(f"couldn't read {date}")
        continue

    dss.append(ds)

da = xr.concat(dss, dim="time")["spi_gamma_30_day"]
df = da.to_dataframe().reset_index()
```

```python
# read yield data
YIELD_PATH = (
    RAW_DIR / "fao/Sorghum_Yield_Historical_FAOSTAT_data_en_7-7-2023.csv"
)
df_yield = pd.read_csv(YIELD_PATH)
df_yield = df_yield.rename(columns={"Value": "Yield"})
```

```python
# compare yield and spi

months_of_interest = [7, 8, 9]

df_moi = df[df["time"].dt.month.isin(months_of_interest)].dropna()
df_moi = df_moi.rename(columns={"spi_gamma_30_day": "SPI"})
df_moi["Year"] = df_moi["time"].dt.year
for col in ["Year"]:
    df_moi[f"{col} norm"] = (df_moi[col] - df_moi[col].mean()) / df_moi[
        col
    ].std()

for month in months_of_interest:
    df_month = df_moi[df_moi["time"].dt.month == month]
    df_month = df_month.groupby("Year", as_index=False).mean()
    df_month = pd.merge(df_month, df_yield[["Year", "Yield"]], on="Year")
    fig = go.Figure(
        layout=dict(
            template="simple_white", title=f"BFA Sorghum Yield vs. SPI"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_month["SPI"],
            y=df_month["Yield"],
            name="",
            mode="markers",
            customdata=df_month["Year"],
            hovertemplate="Year: %{customdata}<br>SPI: %{x}<br>Yield: %{y}",
        )
    )
    fig.update_xaxes(title=f"SPI {month}")
    fig.update_yaxes(title="Yield (100 g/ha)")
    fig.show()

    results = sm.OLS(
        df_month["Yield"], sm.add_constant(df_month[[f"SPI", "Year norm"]])
    ).fit()
    print(results.summary())


df_agg = df_moi.groupby("Year", as_index=False).mean()
df_agg = pd.merge(df_agg, df_yield[["Year", "Yield"]], on="Year")

fig = go.Figure(
    layout=dict(template="simple_white", title=f"BFA Sorghum Yield vs. SPI")
)
fig.add_trace(
    go.Scatter(
        x=df_agg["SPI"],
        y=df_agg["Yield"],
        name="",
        mode="markers",
        customdata=df_agg["Year"],
        hovertemplate="Year: %{customdata}<br>SPI: %{x}<br>Yield: %{y}",
    )
)
fig.update_xaxes(title="SPI")
fig.update_yaxes(title="Yield (100 g/ha)")
fig.show()

results = sm.OLS(
    df_agg["Yield"], sm.add_constant(df_agg[[f"SPI", "Year norm"]])
).fit()
print(results.summary())
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

print(drop_dates)

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

fig.update_layout(height=3000)
pyo.iplot(fig)
```

```python
# determine historical triggers based on single threshold

spi_thesholds = np.round(np.arange(-2, -0.75 + 0.01, 0.25), 2)
percent_thresholds = np.round(np.arange(0.05, 0.4 + 0.01, 0.05), 2)

dff = df[
    (df["time"].dt.month.isin(months_of_interest))
    & ~(df["time"].isin(drop_dates))
]

years = df["time"].dt.year.unique()

df_events = pd.DataFrame()

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
                    df_events = pd.concat([df_events, df_count])

df_events["Year"] = df_events["date"].dt.year
print(df_events)
```

```python
# plot triggers based on single threshold

pd.options.mode.chained_assignment = None


def years_to_string(cell):
    if isinstance(cell, float):
        return "None"
    else:
        return "<br>".join(map(str, cell))


month_ranges = [[7], [8], [9], [7, 8, 9]]
month_range_names = ["July", "August", "September", "ANY MONTH"]
range_color = [1, 5]

for month_range, month_range_name in zip(month_ranges, month_range_names):
    dff = df_events[df_events["date"].dt.month.isin(month_range)]
    dff["year"] = dff["date"].dt.year
    dff = (
        dff.groupby(["year", "spi_t", "per_t"])["date"]
        .apply(list)
        .reset_index()
    )
    df_freq = dff.pivot_table(
        index="spi_t",
        columns="per_t",
        values="date",
        aggfunc="count",
    )

    df_freq = len(years) / df_freq
    df_freq = df_freq.fillna("inf")
    df_freq = df_freq.astype(float).round(1)

    df_freq.columns = df_freq.columns.astype(str)
    df_freq = df_freq.reindex(sorted(df_freq.columns, reverse=True), axis=1)
    df_freq.index = df_freq.index.astype(str)

    df_records = dff.pivot_table(
        index="spi_t",
        columns="per_t",
        values="year",
        aggfunc=lambda x: list(x),
    )

    df_records = df_records.applymap(years_to_string)
    df_records.columns = df_records.columns.astype(str)
    df_records = df_records.reindex(
        sorted(df_freq.columns, reverse=True), axis=1
    )
    df_records.index = df_records.index.astype(str)

    fig = px.imshow(
        df_freq,
        text_auto=True,
        range_color=range_color,
        color_continuous_scale="Reds",
    )
    fig.update(
        data=[
            {
                "customdata": df_records,
                "hovertemplate": "Years activated:<br>%{customdata}",
            }
        ]
    )
    fig.update_traces(name="")
    fig.update_layout(
        coloraxis_colorbar_title="Recurrence (years)",
        template="simple_white",
        coloraxis_colorbar_outlinewidth=0,
        coloraxis_colorbar_tickvals=range_color,
        title=month_range_name,
    )
    fig.update_xaxes(
        side="top",
        title_text="Percent of AOI Threshold",
        mirror=True,
        showline=False,
    )
    fig.update_yaxes(
        title="SPI Threshold",
        mirror=True,
        showline=False,
        fixedrange=True,
    )

    pyo.iplot(fig)
```

```python
# write triggered years to file

filename = "bfa_years_triggered.csv"
filepath = EXP_DIR / "obsv_trigger" / filename

keep = [
    "aoi=0.1 spi=-1.75",
    "aoi=0.1 spi=-2.0",
    "aoi=0.15 spi=-1.75",
    "aoi=0.2 spi=-1.5",
    "aoi=0.25 spi=-1.0",
    "aoi=0.3 spi=-1.25",
    "aoi=0.3 spi=-1.5",
    "aoi=0.4 spi=-1.25",
]

df_log = df_events.copy()
df_log["aoi spi"] = (
    "aoi="
    + df_log["per_t"].astype(str)
    + " spi="
    + df_log["spi_t"].astype(str)
)

df_log = df_log[df_log["aoi spi"].isin(keep)]

df_log = df_log.pivot_table(
    index="Year", columns="aoi spi", values="date", aggfunc="first"
)
df_log = df_log.reindex(years)
df_log = df_log.fillna("-")
df_existing = pd.read_csv(filepath, index_col="Year")
df_log = df_log.combine_first(df_existing)
df_log = df_log.sort_values("Year", ascending=False)
df_log.to_csv(filepath, date_format="%Y-%m-%d")
```

```python jupyter={"outputs_hidden": true}
# determine historical triggers with different thresholds for different months

aois = [
    [0.2, 0.15, 0.15],
    [0.2, 0.2, 0.15],
    [0.15, 0.15, 0.15],
    [0.2, 0.2, 0.2],
]

years = df["time"].dt.year.unique()

df_events = pd.DataFrame()

for spi_t in [-1, -1.25, -1.5]:
    for aoi_t in aois:
        month2thresholds = {
            7: {"spi": spi_t, "aoi": aoi_t[0]},
            8: {"spi": spi_t, "aoi": aoi_t[1]},
            9: {"spi": spi_t, "aoi": aoi_t[2]},
        }
        df_events_t = pd.DataFrame()
        for year in years:
            for month in month2thresholds.keys():
                dff = pd.DataFrame()
                dff = df[
                    (df["time"].dt.year == year)
                    & (df["time"].dt.month == month)
                ].dropna()
                if dff.empty:
                    continue
                date = dff["time"].iloc[-1]
                frac = len(
                    dff[
                        dff["spi_gamma_30_day"]
                        < month2thresholds.get(month).get("spi")
                    ]
                ) / len(dff)
                aoi_str = ", ".join([str(x) for x in aoi_t])
                if frac > month2thresholds.get(month).get("aoi"):
                    df_count = pd.DataFrame(
                        [[date, frac, spi_t, aoi_str]],
                        columns=["date", "frac", "spi_t", "per_t"],
                    )
                    df_events = pd.concat([df_events, df_count])
                    df_events_t = pd.concat([df_events_t, df_count])

        years_triggered = df_events_t["date"].dt.year.unique()
        return_period = len(years) / len(years_triggered)
        print(
            f"with SPI {spi_t} and AOI {aoi_str}: triggered {len(years_triggered)} out of {len(years)} (return period {return_period:.3} years)"
        )
        print(years_triggered)
        print()
```

```python
# plot triggers with different thresholds for different months

dff = df_events.copy()
dff["year"] = dff["date"].dt.year
dff = dff.groupby(["year", "spi_t", "per_t"])["date"].apply(list).reset_index()
df_freq = dff.pivot_table(
    index="spi_t",
    columns="per_t",
    values="date",
    aggfunc="count",
)

df_freq = len(years) / df_freq
df_freq = df_freq.fillna("inf")
df_freq = df_freq.astype(float).round(1)

print(df_freq)

df_freq.columns = df_freq.columns.astype(str)
df_freq = df_freq.reindex(sorted(df_freq.columns, reverse=True), axis=1)
df_freq.index = df_freq.index.astype(str)

df_records = dff.pivot_table(
    index="spi_t",
    columns="per_t",
    values="year",
    aggfunc=lambda x: list(x),
)

df_records = df_records.applymap(years_to_string)
df_records.columns = df_records.columns.astype(str)
df_records = df_records.reindex(sorted(df_freq.columns, reverse=True), axis=1)
df_records.index = df_records.index.astype(str)

range_color = [1, 5]
fig = px.imshow(
    df_freq,
    text_auto=True,
    range_color=range_color,
    color_continuous_scale="Reds",
)
fig.update(
    data=[
        {
            "customdata": df_records,
            "hovertemplate": "Years activated:<br>%{customdata}",
        }
    ]
)
fig.update_traces(name="")
fig.update_layout(
    coloraxis_colorbar_title="Recurrence (years)",
    template="simple_white",
    coloraxis_colorbar_outlinewidth=0,
    coloraxis_colorbar_tickvals=range_color,
)
fig.update_xaxes(
    side="top",
    title_text="AOI thresholds [Jul, Aug, Sep]",
    mirror=True,
    showline=False,
)
fig.update_yaxes(
    title="SPI Threshold",
    mirror=True,
    showline=False,
    fixedrange=True,
)

pyo.iplot(fig)
```

```python
# plot SPI for specific time

date = datetime.datetime(2000, 9, 30)

fig, ax = plt.subplots()
ax.axis("off")
ax.set_title(f"Date: {date:%Y-%m-%d}")

gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="grey")
da.sel(time=date).plot(ax=ax, cmap="Reds_r", vmin=-2, vmax=0)
```

```python
# SPI animation

range_color = [-2, 0]
fig = px.imshow(
    da,
    animation_frame="time",
    origin="lower",
    range_color=range_color,
    color_continuous_scale="Reds_r",
    aspect="equal",
)
fig.update_layout(template="simple_white")
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)


pyo.iplot(fig, auto_play=False)
```

```python

```
