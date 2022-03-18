# Explore plotting

Short notebook to explore nice plotting of the forecast.

We explore plotting the dominant tercile.
The struggle here is that it somewhat would be nice to use IRI's colors, but
they define the bin boundaries at e.g. 37.5 and 42.5 while our threshold is 40,
so this can cause confusion.

Therefore a better option might be to only show the below average tercile values,
which is shown next.

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import rioxarray
import cftime

from aatoolbox import (
    CodAB,
    IriForecastProb,
    IriForecastDominant,
    GeoBoundingBox,
)

from src import utils, constants
```

## Set variables

```python
# Get geometry parameters
codab = CodAB(country_config=constants.country_config)
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1[gdf_adm1.ADM1_FR.isin(constants.adm_sel)]
```

## Plot dominant tercile

```python
iri_dom = IriForecastDominant(constants.country_config, geo_bounding_box=geobb)
ds_iri_dom = iri_dom.load()
da_iri_dom = ds_iri_dom.dominant
# F indicates the publication month, and L the leadtime.
# A leadtime of 1 means a forecast published in May is forecasting JJA
da_iri_dom_clip = da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
# iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
# rounded bins for easier interpretability
plt_levels = [-100, -70, -60, -50, -45, -40, 40, 45, 50, 60, 70, 100]
plt_colors = [
    "#783200",
    "#ab461e",
    "#d18132",
    "#e8b832",
    "#fafa02",
    "#ffffff",
    "#d1f8cc",
    "#acf8a0",
    "#73bb6e",
    "#3a82b3",
    "#0e3bf4",
]
```

```python
# pub_date = cftime.Datetime360Day(2022, 2, 16, 0, 0, 0, 0)
pub_date = cftime.Datetime360Day(2021, 8, 16, 0, 0, 0, 0)
leadtime = 1
month_season_mapping = utils.get_month_season_mapping()
for_seas = month_season_mapping[(pub_date.month + leadtime + 1) % 12 + 1]

da_date = da_iri_dom_clip.sel(F=pub_date, L=leadtime)

g = da_date.where(da_date <= -40).plot(
    levels=plt_levels[:6],
    colors=plt_colors[:6],
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "ticks": plt_levels[:6],
        "label": "probability below average",
    },
    figsize=(15, 7),
)

da_date.where(da_date >= 40).plot(
    levels=plt_levels[6:],
    colors=plt_colors[6:],
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "ticks": plt_levels[6:],
        "label": "probability above average",
    },
    # figsize=(15, 7),
    ax=g.axes,
)

gdf_adm1.boundary.plot(linewidth=1, ax=g.axes, color="grey")
gdf_aoi.dissolve().boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off")
g.axes.set_title((
    f"Forecast published in {pub_date.strftime('%b %Y')} \n"
    "Predicting {for_seas} (lt={leadtime} months)")
);
```

### Plot probability below average

```python
# Load the data that contains the probability per tercile
# this allows us to solely look at the below-average tercile
# C indicates the tercile (below-average, normal, or above-average).
# F indicates the publication month, and L the leadtime
iri_prob = IriForecastProb(constants.country_config, geo_bounding_box=geobb)
ds_iri = iri_prob.load()
da_iri = ds_iri.prob
```

```python
plt_levels_bavg = [
    30,
    35,
    40,
    45,
    50,
    55,
    60,
]

plt_colors_bavg = [
    "#8c510a",
    "#bf812d",
    "#dfc27d",
    "#f6e8c3",
    "#c7eae5",
    "#35978f",
    "#01665e",
][::-1]
```

```python
# TODO: the color levels are still getting messed up when several
# levels are not found in the data
pub_date = cftime.Datetime360Day(2021, 8, 16, 0, 0, 0, 0)
leadtime = 1
for_seas = month_season_mapping[(pub_date.month + leadtime + 1) % 12 + 1]

da_date = da_iri.sel(F=pub_date, L=leadtime)

g = da_date.sel(C=0).plot(
    levels=plt_levels_bavg,
    colors=plt_colors_bavg,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "ticks": plt_levels_bavg,
        "label": "probability of below average (%)",
        "extend": "both",
    },
    figsize=(15, 7),
)

gdf_adm1.boundary.plot(linewidth=1, ax=g.axes, color="grey")
gdf_aoi.dissolve().boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off")
g.axes.set_title((
    f"Forecast published in {pub_date.strftime('%b %Y')} \n"
    "Predicting {for_seas} (lt={leadtime} months)")
)
cmap = g.get_cmap()
cmap.set_over("#543005")
cmap.set_under("#01665e")

g.set_cmap(cmap)
```

```python

```
