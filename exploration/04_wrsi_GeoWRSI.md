---
jupyter:
  jupytext:
    formats: md,ipynb
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

# GeoWRSI

```python
%load_ext jupyter_black
```

```python
from ochanticipy import (
    create_country_config,
    CodAB,
    GeoBoundingBox,
)
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import chart_studio.plotly as py
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.colors
import chart_studio
import statsmodels.api as sm
import datetime
import numpy as np
from rasterio.enums import Resampling

pyo.init_notebook_mode()
load_dotenv()
```

```python
CS_KEY = os.environ["CHARTSTUDIO_APIKEY"]
chart_studio.tools.set_credentials_file(
    username="tristandowning", api_key=CS_KEY
)
```

```python
adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1.loc[gdf_adm1.ADM1_FR.isin(adm1_sel)]

RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/bfa"
PROCESSED_DIR = (
    Path(os.environ["AA_DATA_DIR"]) / "public/processed/bfa/geowrsi"
)
GEOWRSI_OUTPUT_DIR = Path(os.environ["GEOWRSI_OUTPUT_DIR"])
SAVE_DIR = Path("/Users/tdowning/OCHA/data/bfa")
EXP_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/bfa"
```

```python
GEOWRSI_OUTPUT_DIR
```

## Process specific date

```python
# process raw GeoWRSI output

year = 2024
dekad = 24

filestem = f"WRSI{year}{dekad}"
# filename = "West Sahel Africa_WRSI_Index_d22_2023_bfa.nc"

dayofyear = (dekad - 1) * 10
eff_date = datetime.datetime(year, 1, 1) + datetime.timedelta(dayofyear - 1)
da = rxr.open_rasterio(GEOWRSI_OUTPUT_DIR / f"{filestem}.bil")[0]
da = da.rio.write_crs("EPSG:4326")
da = da.rio.clip(gdf_adm1["geometry"], all_touched=True)
da["date"] = eff_date
ds = da.to_dataset(name="WRSI")
ds.to_netcdf(PROCESSED_DIR / f"{filestem}_bfa.nc", engine="h5netcdf")
```

```python
month = ((dekad - 1) // 3) + 1
dekad_of_month = ((dekad - 1) % 3) + 1
```

```python
month, dekad_of_month
```

```python
month_fr = {7: "juill.", 8: "août", 9: "sept.", 10: "oct."}.get(month)
```

```python
month_fr
```

```python
ds = xr.open_dataset(PROCESSED_DIR / f"{filestem}_bfa.nc")
da = ds["WRSI"]
da.rio.write_crs("EPSG:4326", inplace=True)
display(da)
da.plot()
resolution = 0.01
da = da.rio.reproject(
    dst_crs="EPSG:4326",
    resolution=resolution,
    resampling=Resampling.nearest,
)
da = da.rio.clip(gdf_aoi["geometry"], all_touched=True)
da.attrs["_FillValue"] = np.nan

df = da.to_dataframe()
```

```python
da = da.rio.clip(gdf_aoi["geometry"], all_touched=True)
df = da.to_dataframe()["WRSI"].reset_index().dropna()
```

```python
thresholds = [60, 75]
cutoffs = [10, 30]

bounds = [0, 50, 60, 80, 95, 99, 100]
colors = np.array(
    [
        [237, 108, 55, 255],
        [201, 169, 76, 255],
        [249, 255, 202, 255],
        [203, 253, 92, 255],
        [111, 219, 65, 255],
        [77, 170, 75, 255],
    ]
).astype(float)
colors /= 255
cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))

for threshold, cutoff in zip(thresholds, cutoffs):
    percent = len(df[df["WRSI"] < threshold]) / len(df) * 100
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.axis("off")

    gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="grey")
    da.plot(ax=ax, cmap=cmap, norm=norm, alpha=0.7)
    da.plot.contour(
        levels=[0, threshold - 0.0001, 101],
        ax=ax,
        cmap="Reds",
        linewidths=1.5,
    )
    percent_fr = f"{percent:.1f}".replace(".", ",")
    ax.set_title(
        f"WRSI actuel à fin de {dekad_of_month}e décade de {month_fr}\n"
        f"Frac. zone avec WRSI < {threshold} : {percent_fr}% (seuil {cutoff}%)"
    )
```

## Process all files

```python
# load FAO yield data

YIELD_PATH = (
    RAW_DIR / "fao/Sorghum_Yield_Historical_FAOSTAT_data_en_7-7-2023.csv"
)
df_yield = pd.read_csv(YIELD_PATH)
```

```python
# load EOS WRSI data outputted from GeoWRSI

years = range(2001, 2023, 1)

das = []

for year in years:
    wrsi_eos_path = (
        GEOWRSI_OUTPUT_DIR / f"West Sahel Africa_WRSI_Index_EOS_{year}.bil"
    )
    try:
        da = rxr.open_rasterio(wrsi_eos_path)[0]
    except Exception as e:
        print(e)
        continue
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.clip(gdf_adm1["geometry"], all_touched=True)

    da["Year"] = year
    das.append(da)

da = xr.concat(das, dim="Year")
df = da.to_dataframe(name="WRSI")
df = df.reset_index()
df = df.drop(columns=["band", "spatial_ref"])
df = df[(df["WRSI"] != 254) & (df["WRSI"] != 0)]
```

```python
# process WRSI data from GeoWRSI and store in processed dir

years = range(2001, 2024, 1)
dekads = range(1, 37, 1)

not_found = []

for year in years:
    for dekad in dekads:
        dayofyear = (dekad - 1) * 10
        eff_date = datetime.datetime(year, 1, 1) + datetime.timedelta(
            dayofyear - 1
        )
        filename = f"West Sahel Africa_WRSI_Index_d{dekad:02d}_{year}"
        try:
            da = rxr.open_rasterio(GEOWRSI_OUTPUT_DIR / f"{filename}.bil")[0]
        except Exception as e:
            not_found.append(f"d{dekad:02d}_{year}")
            continue
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.clip(gdf_adm1["geometry"], all_touched=True)
        da["date"] = eff_date
        ds = da.to_dataset(name="WRSI")
        ds.to_netcdf(PROCESSED_DIR / f"{filename}_bfa.nc", engine="h5netcdf")
```

```python
# read processed data

ds = xr.open_mfdataset(
    os.path.join(PROCESSED_DIR, "*.nc"), concat_dim="date", combine="nested"
)
ds = ds.rio.write_crs("EPSG:4326")
ds = ds.rio.clip(gdf_aoi["geometry"], all_touched=True)
da = ds["WRSI"]
da = da.sortby("date")

df = da.to_dataframe()
df = df.reset_index()
df = df.drop(columns=["band", "spatial_ref"])
df = df.dropna()
df = df[(df["WRSI"] != 254) & (df["WRSI"] != 0)]
```

## Historical analysis

```python
# determine trigger events

mois = [7, 8, 9]

dff = df[df["date"].dt.month.isin(mois)]
dff = dff.sort_values("date")
dff["month"] = dff["date"].dt.month
dff["year"] = dff["date"].dt.year

# take only last measurement of each month
dff = dff.groupby(["y", "x", "year", "month"]).last().reset_index()

t_aois = np.round(np.arange(0.05, 0.4 + 0.01, 0.05), 2)
t_wrsis = [50, 55, 60, 65, 70, 75, 80]
years = dff["date"].dt.year.unique()

maximum_cells = dff.groupby("date").size().max()

df_events = pd.DataFrame()

for t_aoi in t_aois:
    for t_wrsi in t_wrsis:
        for year in years:
            df_year = dff[dff["date"].dt.year == year]
            for date in df_year["date"].unique():
                df_date = df_year[df_year["date"] == date]
                frac = len(df_date[df_date["WRSI"] < t_wrsi]) / maximum_cells
                if frac > t_aoi:
                    df_count = pd.DataFrame(
                        [[date, t_aoi, t_wrsi]],
                        columns=["Date", "t_aoi", "t_wrsi"],
                    )
                    df_events = pd.concat([df_events, df_count])

df_events["Year"] = df_events["Date"].dt.year
```

```python
df_events
```

```python
# determine only trigger event for finalized trigger
df_events[(df_events["t_aoi"] == 0.3) & (df_events["t_wrsi"] == 75)]
```

```python
# plot trigger matrix

df_plot = df_events.groupby(["Year", "t_aoi", "t_wrsi"]).count().reset_index()
df_plot = df_plot.sort_values("Year", ascending=False)

df_text = df_plot.pivot_table(
    index="t_wrsi",
    columns="t_aoi",
    values="Year",
    aggfunc=lambda x: list(x),
)


def years_to_string(cell):
    if isinstance(cell, float):
        return "None"
    else:
        return "<br>".join(map(str, cell))


df_text = df_text.applymap(years_to_string)
df_text = df_text.reindex(sorted(df_text.columns, reverse=True), axis=1)
df_text.columns = df_text.columns.astype(str)
df_text.index = df_text.index.astype(str)

df_plot = df_plot.pivot_table(
    index="t_wrsi", columns="t_aoi", values="Date", aggfunc="count"
)
df_plot = len(years) / df_plot
df_plot = df_plot.astype(float).round(1)

df_plot = df_plot.reindex(sorted(df_plot.columns, reverse=True), axis=1)
df_plot.columns = df_plot.columns.astype(str)
df_plot.index = df_plot.index.astype(str)
print(df_plot)


range_color = [1, 5]

fig = px.imshow(
    df_plot,
    text_auto=True,
    range_color=range_color,
    color_continuous_scale="Reds",
    aspect="equal",
)
fig.update(
    data=[
        {
            "customdata": df_text,
            "hovertemplate": "Years activated:<br>%{customdata}",
        }
    ]
)
fig.update_traces(name="")
fig.update_layout(
    coloraxis_colorbar_title="Return period (years)",
    template="simple_white",
    coloraxis_colorbar_outlinewidth=0,
    coloraxis_colorbar_tickvals=range_color,
    title="",
)
fig.update_xaxes(
    side="top",
    title_text="Fraction of AOI Threshold",
    mirror=True,
    showline=False,
)
fig.update_yaxes(
    title="WRSI Threshold",
    mirror=True,
    showline=False,
    fixedrange=True,
)

fig.show()
```

```python
# write triggered years to file

filename = "bfa_years_triggered.csv"
filepath = EXP_DIR / "obsv_trigger" / filename

keep = [
    "aoi=0.1 wrsi=60",
    "aoi=0.1 wrsi=55",
    "aoi=0.2 wrsi=70",
    "aoi=0.3 wrsi=75",
    "aoi=0.3 wrsi=70",
    "aoi=0.2 wrsi=65",
    "aoi=0.4 wrsi=80",
    "aoi=0.4 wrsi=75",
]

df_log = df_events.copy()
df_log["aoi wrsi"] = (
    "aoi="
    + df_log["t_aoi"].astype(str)
    + " wrsi="
    + df_log["t_wrsi"].astype(str)
)

df_log = df_log[df_log["aoi wrsi"].isin(keep)]

df_log = df_log.pivot_table(
    index="Year", columns="aoi wrsi", values="Date", aggfunc="first"
)
df_log = df_log.reindex(years)
df_log = df_log.fillna("-")
df_existing = pd.read_csv(filepath, index_col="Year")
df_log = df_log.combine_first(df_existing)
df_log = df_log.sort_values("Year", ascending=False)
df_log.to_csv(filepath, date_format="%Y-%m-%d")
```

```python
# check correlation between bad years and triggered years

filename = "bfa_years_triggered.csv"
filepath = EXP_DIR / "obsv_trigger" / filename

df_corr = pd.read_csv(filepath, index_col="Year")

df_corr = df_corr.applymap(lambda x: False if x == "-" else x)
df_corr = df_corr.applymap(lambda x: True if x == "Bad" else x)
df_corr["Reported bad years v2"] = df_corr["Reported bad years v2"].apply(
    lambda x: True if float(x) < 12 else x
)
df_corr["Reported bad years v2"] = df_corr["Reported bad years v2"].apply(
    lambda x: False if float(x) > 11 else x
)
df_corr = df_corr.applymap(lambda x: True if isinstance(x, str) else x)
df_corr = df_corr.astype(bool)

df_corr = df_corr.corr()
df_corr = df_corr.round(2)

for col1 in df_corr.columns:
    for col2 in df_corr.columns:
        if col1 == col2:
            break
        df_corr.loc[col2, col1] = None


fig = px.imshow(df_corr, text_auto=True)
fig.update_layout(
    template="simple_white", title="Correlation between triggers"
)

fig.show()
```

```python
# clip da if needed

da_aoi = da.rio.clip(gdf_aoi["geometry"], all_touched=True)
df_aoi = da_aoi.to_dataframe(name="WRSI")
df_aoi = df_aoi.reset_index()
df_aoi = df_aoi.drop(columns=["band", "spatial_ref"])
df_aoi = df_aoi[(df_aoi["WRSI"] != 254) & (df_aoi["WRSI"] != 0)]
```

```python
# aggregate for correlation with yield

df_agg = df.groupby("Year", as_index=False)["WRSI"].mean()
df_agg_aoi = df_aoi.groupby("Year", as_index=False)["WRSI"].mean()
df_agg_aoi = df_agg_aoi.rename(columns={"WRSI": "WRSI AOI"})
df_agg = pd.merge(
    df_agg,
    df_agg_aoi,
    on="Year",
)
df_agg = pd.merge(df_agg, df_yield, on="Year")

for col in ["Year", "WRSI", "WRSI AOI"]:
    df_agg[f"{col} norm"] = (df_agg[col] - df_agg[col].mean()) / df_agg[
        col
    ].std()

df_agg = df_agg.rename(columns={"Value": "Yield"})

px.scatter(df_agg, y="WRSI", x="Year").show()
px.scatter(df_agg, y="Yield", x="Year").show()
```

```python
# plot correlation with yield

for wrsi in ["WRSI", "WRSI AOI"]:
    fig = go.Figure(
        layout=dict(
            template="simple_white", title=f"BFA Sorghum Yield vs. {wrsi}"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_agg[wrsi],
            y=df_agg["Yield"],
            name="",
            mode="markers",
            customdata=df_agg["Year"],
            hovertemplate="Year: %{customdata}<br>WRSI: %{x}<br>Yield: %{y}",
        )
    )
    fig.update_xaxes(title="End of Season WRSI")
    fig.update_yaxes(title="Yield (100 g/ha)")
    fig.show()

    results = sm.OLS(
        df_agg["Yield"], sm.add_constant(df_agg[[f"{wrsi} norm", "Year norm"]])
    ).fit()
    print(results.summary())
```

```python
# WRSI animation

da_plot = da.where(da != 0).where(da != 254)

range_color = [0, 100]
fig = px.imshow(
    da_plot,
    animation_frame="date",
    origin="lower",
    range_color=range_color,
    color_continuous_scale="Reds_r",
    aspect="equal",
)
fig.update_layout(
    template="simple_white", title_text="End of Season WRSI (Sorghum)"
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

pyo.iplot(fig, auto_play=False)

filename = "bfa_wrsi_eos_sorghum.html"

f = open(SAVE_DIR / filename, "w")
f.close()
with open(SAVE_DIR / filename, "a") as f:
    f.write(
        fig.to_html(full_html=True, include_plotlyjs="cdn", auto_play=False)
    )
f.close()
```

```python
# WRSI animation

da_plot = da_aoi.where(da_aoi != 0).where(da_aoi != 254)

range_color = [0, 100]
fig = px.imshow(
    da_plot,
    animation_frame="Year",
    origin="lower",
    range_color=range_color,
    color_continuous_scale="Reds_r",
    aspect="equal",
)
fig.update_layout(
    template="simple_white", title_text="End of Season WRSI (Sorghum)"
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

pyo.iplot(fig, auto_play=False)

filename = "bfa_aoi_wrsi_eos_sorghum.html"

f = open(SAVE_DIR / filename, "w")
f.close()
with open(SAVE_DIR / filename, "a") as f:
    f.write(
        fig.to_html(full_html=True, include_plotlyjs="cdn", auto_play=False)
    )
f.close()
```

```python
py.plot(fig, filename="BFA Sorghum Yield", auto_open=True)
```

```python

```
