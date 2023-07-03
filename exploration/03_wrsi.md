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
import rioxarray as rxr
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from colour import Color
import pandas as pd
import xarray as xr
import numpy as np
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
WRSI_DIR = RAW_DIR / "fewsnet/wrsi"
```

```python
# wa is croplands, w1 is rangelands

start_date = datetime.date(2023, 5, 1)
end_date = datetime.date(2023, 7, 1)
dates = pd.date_range(start_date, end_date, freq="10D")

das = []

for date in dates:
    dekad = date.dayofyear // 10
    year = date.year - 2000
    dayofyear = (dekad - 1) * 10
    eff_date = datetime.datetime(date.year, 1, 1) + datetime.timedelta(
        dayofyear - 1
    )
    print(eff_date)
    zip_filename = "zip+file://" / WRSI_DIR / "w2317wa.zip!/w2317dd.tif"
    filename = WRSI_DIR / f"w{year}{dekad}wa/w{year}{dekad}dd.tif"
    try:
        da = rxr.open_rasterio(filename)
    except Exception as error:
        print(error)
        continue
    da = da.rio.reproject("EPSG:4326")
    da = da.rio.clip(gdf_aoi["geometry"], all_touched=True)
    da = da.expand_dims(time=[eff_date])
    das.append(da)

da = xr.concat(das, dim="time")
da = da.where(da != 255).sel(band=1)
df = da.to_dataframe(name="WRSI").reset_index()
df = df.drop(columns=["spatial_ref", "band"])
df = df.dropna()

print(df)
```

```python
# bin WRSI

bins = [0, 50, 60, 80, 95, 100]
bin_labels = []
bin_colors = [
    x.get_hex() for x in Color("red").range_to(Color("white"), len(bins))
]
for x in range(len(bins) - 1):
    bin_labels.append(f"{bins[x]} to {bins[x+1]}")
df_agg = (
    df.groupby(["time", pd.cut(df["WRSI"], bins, labels=bin_labels)])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
print(df_agg)
```

```python
wrsi_thesholds = np.round(np.arange(95, 100 + 0.01, 1), 2)
percent_thresholds = np.round(np.arange(0.05, 0.3 + 0.01, 0.05), 2)

years = df["time"].dt.year.unique()

dff = df.copy()

df_events = pd.DataFrame()

for wrsi_threshold in wrsi_thesholds:
    for per_t in percent_thresholds:
        for year in years:
            df_year = dff[dff["time"].dt.year == year]
            for date in df_year["time"].unique():
                df_date = df_year[df_year["time"] == date]
                df_date = df_date.dropna()
                frac = len(df_date[df_date["WRSI"] < wrsi_threshold]) / len(
                    df_date
                )
                if frac > per_t:
                    df_count = pd.DataFrame(
                        [[date, frac, wrsi_threshold, per_t]],
                        columns=["date", "frac", "spi_t", "per_t"],
                    )
                    df_events = pd.concat([df_events, df_count])

print(df_events)
```

```python
# plot threshold frontier

pd.options.mode.chained_assignment = None


def years_to_string(cell):
    if isinstance(cell, float):
        return "None"
    else:
        return "<br>".join(map(str, cell))


month_ranges = [range(1, 13)]
month_range_names = ["ANY MONTH"]
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

    fig = px.imshow(df_freq, text_auto=True, range_color=range_color)
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
# WRSI animation

# range_color = [-2, 0]
fig = px.imshow(
    da,
    animation_frame="time",
    origin="lower",
    #     range_color=range_color,
    color_continuous_scale="Reds_r",
)
fig.update_layout(template="simple_white")
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)


pyo.iplot(fig, auto_play=False)
```

```python

```