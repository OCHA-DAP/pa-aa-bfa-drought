# Evaluate trigger status during 2022 season

During 2022 the trigger was not reached. As evaluation the country team asked
to provide some figures which are computed here. They were then added to
[this presentation](https://docs.google.com/presentation/d/1DSFSVfV4JCnB43vQP155omYS2cf082nHmk3O38WBYWg/edit?usp=sharing)
which was shared with the wider team.

The trigger is evaluated at two points in time:

- Trigger I: In March, if the trigger is met for risk  of below average
precipitation in the June, July, August period5,
funds  are  released  for  the  commencement  of  activities  by  all
selected  sectors,  including  food  security  and
livelihoods, nutrition, protection, as well as water, hygiene, and sanitation.
- Trigger II: In July, if the trigger is met for risk of below average
precipitation in the August, September, and October
period, funds are released for the implementation of activities by the
food security and livelihoods sector.

The trigger makes use of IRI's seasonal tercile precipitation forecast and
is defined as:

1. 40% or more probability of below average rainfall AND
2. The probability of below average rainfall should be 5 percentage points
higher than that of above average rainfall.

Finally,  activation  is  also  contingent  on  at  least  10%  of  the  zone
(Boucle  de  Mouhoun,  Centre  Nord,  Sahel,  and  Nord)
meeting these criteria. This criterion aims to capture through geographical
spread a more severe e shock,  in line with the
humanitarian objectives of the pilot. Overall, this framework aims to
target 1 in 4-year type of event

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import geopandas as gpd
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import cftime
import calendar
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
import hvplot.xarray
import altair as alt
import matplotlib.pyplot as plt

import aatoolbox.utils.raster
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

```python
trig_year = 2022
```

## Inspect forecasts

We load the iri data indicating the dominant tercile.
The negative values indicate forecasted below average rainfall,
and the positive values above average. We use rounded bins (e.g. -40 to 40).
The IRI website bins are slightly different,
where values between -37.5 and 37.5 are assigned to the normal tercile.

We plot the forecast raster data for the periods and leadtimes of interest.
The red areas are the admin1's we are focussing on.
These figures are the same as
[the figure on the IRI Maproom](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html)
, except for the bin boundaries.

```python
iri_dom = IriForecastDominant(constants.country_config, geo_bounding_box=geobb)
```

```python
# iri_dom.download(clobber=True)
# iri_dom.process(clobber=True)
```

```python
ds_iri_dom = iri_dom.load()
```

```python
da_iri_dom = ds_iri_dom.dominant
```

```python
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
def plt_single_date_leadtime(da_single):
    g = da_single.plot(
        levels=plt_levels,
        colors=plt_colors,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
            "ticks": plt_levels,
        },
        figsize=(15, 7),
    )

    gdf_adm1.boundary.plot(linewidth=1, ax=g.axes, color="grey")
    gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
    g.axes.axis("off")
    pub_mon = int(da_single.F.dt.month)
    lt = int(da_single.L)
    for_seas = utils.get_month_season_mapping()[(pub_mon + lt + 1) % 12 + 1]
    g.axes.set_title(
        (
            f"Forecast published in {calendar.month_name[pub_mon]} predicting"
            f" {for_seas} (leadtime of {lt} months)"
        ),
        y=1.1,
    )
    plt.savefig(
        f"forecast_{str(da_mar_2022.F.dt.strftime('%b_%Y').values)}_lt_{lt}.png",
        dpi=300,
    )
```

Note that for the
[presentation](https://docs.google.com/presentation/d/1DSFSVfV4JCnB43vQP155omYS2cf082nHmk3O38WBYWg/edit?usp=sharing)
the legend from IRI's website was added instead of this one as it more clearly
shows what the numbers mean. Be aware that
[IRI's website legend](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/)
shows normal values to be grey whereas they are in our graphics
(and actually also in IRI's) white.

```python
da_mar_2022 = da_iri_dom_clip.sel(
    F=cftime.Datetime360Day(2022, 3, 16, 0, 0, 0, 0), L=3
)
plt_single_date_leadtime(da_mar_2022)
```

```python
da_mar_2022 = da_iri_dom_clip.sel(
    F=cftime.Datetime360Day(2022, 7, 16, 0, 0, 0, 0), L=1
)
plt_single_date_leadtime(da_mar_2022)
```

### Selecting the data

Before we looked at the dominant tercile.
From here on we use a dataset that contains the probability per tercile,
such that we can compute the range of values of the below average tercile,
regardless of the values of the other terciles.

We load the data and apply an approximate mask to our region of interest.

```python
# Load the data that contains the probability per tercile
# this allows us to solely look at the below-average tercile
# C indicates the tercile (below-average, normal, or above-average).
# F indicates the publication month, and L the leadtime
iri_prob = IriForecastProb(constants.country_config, geo_bounding_box=geobb)
# iri_prob.download(clobber=True)
# iri_prob.process(clobber=True)
ds_iri = iri_prob.load()
da_iri = ds_iri.prob
```

```python
# select all cells touching the region
da_iri_allt = da_iri.rio.clip(gdf_aoi["geometry"], all_touched=True)
# C=0 indicates the below average tercile
da_iri_allt_bavg = da_iri_allt.sel(C=0)
```

```python
# upsample the resolution in order to create a mask of our aoi
resolution = 0.05
mask_list = []
for terc in da_iri_allt.C.values:
    for lt in da_iri_allt.L.values:
        da_terc_lt = da_iri_allt.sel(C=terc, L=lt)
        da_terc_lt_mask = da_terc_lt.rio.reproject(
            da_terc_lt.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        mask_list.append(da_terc_lt_mask.expand_dims({"C": [terc], "L": [lt]}))
da_iri_mask = (
    xr.combine_by_coords(mask_list)
    .rio.clip(gdf_aoi["geometry"], all_touched=False)
    .prob
)
# reproject changes longitude and latitude name to x and y
# so change back here
da_iri_mask = da_iri_mask.rename({"x": "longitude", "y": "latitude"})
da_iri_mask_bavg = da_iri_mask.sel(C=0)
```

### Compute stats

We can now compute the statistics of the region of interest.
We use the approximate mask to define the cells included
for the computation of the statistics.

We have to set two parameters: the minimum probability of below average,
and the method of aggregation.

Based on the above plots we set the minimum probability of below average to 40%.
We have limited data to base this on but the reasoning is
that 40% is relatively rare while not unreachable.

How rare it is, also depends on how the cells within the region of interest
are aggregated.
We chose to look at the percentage of cells reaching the threshold.
Since the max would be too sensitive, while the mean wouldn't be sensitive enough.

While we experimented with different percentages,
we set the required percentage of the region reaching the threshold to 10%.
This is relatively low, and could maybe be increased in the future.
But it was partly-based on the data of 2017 where we did see a drought
and the forecast had around 10% of the area with a probability of 40% of
below average.

```python
#% probability of bavg
threshold = 40
# percent point diff bavg minus above avg
threshold_diff = 5
# min percentage of the area that needs to reach the threshold
perc_area = 10
```

```python
adm0_col = "ADM0_FR"
pcode0_col = "ADM0_PCODE"
```

```python
# compute stats
# dissolve the region to one polygon
gdf_aoi_dissolved = gdf_aoi.dissolve(by=adm0_col)
gdf_aoi_dissolved = gdf_aoi_dissolved[[pcode0_col, "geometry"]]
df_stats_aoi_bavg = da_iri_mask_bavg.aat.compute_raster_stats(
    gdf=gdf_aoi_dissolved,
    feature_col=pcode0_col,
    percentile_list=[100 - perc_area],
)
da_iri_mask_thresh = da_iri_mask_bavg.where(da_iri_mask_bavg >= threshold)
df_stats_aoi_bavg_thresh = da_iri_mask_thresh.aat.compute_raster_stats(
    gdf=gdf_aoi_dissolved, feature_col=pcode0_col, stats_list=["count"]
)
df_stats_aoi_bavg["perc_thresh_bavg"] = (
    df_stats_aoi_bavg_thresh[f"count_{pcode0_col}"]
    / df_stats_aoi_bavg[f"count_{pcode0_col}"]
    * 100
)
da_diff_bel_abv = da_iri_mask.sel(C=0) - da_iri_mask.sel(C=2)
da_iri_mask_thresh_diff = da_iri_mask_bavg.where(
    (da_iri_mask_bavg >= threshold) & (da_diff_bel_abv >= threshold_diff)
)
df_stats_aoi_bavg_diff_thresh = (
    da_iri_mask_thresh_diff.aat.compute_raster_stats(
        gdf=gdf_aoi_dissolved, feature_col=pcode0_col, stats_list=["count"]
    )
)
df_stats_aoi_bavg["perc_thresh"] = (
    df_stats_aoi_bavg_diff_thresh[f"count_{pcode0_col}"]
    / df_stats_aoi_bavg[f"count_{pcode0_col}"]
    * 100
)
df_stats_aoi_bavg["F"] = pd.to_datetime(
    df_stats_aoi_bavg["F"].apply(lambda x: x.strftime("%Y-%m-%d"))
)
df_stats_aoi_bavg["month"] = df_stats_aoi_bavg.F.dt.month
```

NaN values indicate that the whole region is covered by a dry mask at that point.
See
[here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/)
for more information

```python
df_stats_aoi_bavg = df_stats_aoi_bavg[
    (~df_stats_aoi_bavg.perc_thresh.isnull())
]
```

```python
df_stats_aoi_bavg = df_stats_aoi_bavg.sort_values(
    "perc_thresh", ascending=False
)
```

## Analyze statistics probability below average

Compute stats of below average in 2022. As we can see from the table,
nowhere close to triggering

```python
# select the months and leadtimes included in the trigger
df_stats_aoi_bavg_trig_mom = df_stats_aoi_bavg[
    df_stats_aoi_bavg[["month", "L"]]
    .apply(tuple, axis=1)
    .isin(constants.trig_mom)
]
```

```python
df_stats_aoi_bavg_trig_mom["pred_month"] = df_stats_aoi_bavg_trig_mom.apply(
    lambda x: x["F"] + relativedelta(months=int(x["L"])), axis=1
)
```

```python
df_stats_aoi_bavg_trig_mom[
    df_stats_aoi_bavg_trig_mom.F.dt.year == trig_year
].sort_values(["pred_month", "L"])
```

As we can see during the two trigger moments in 2022, there was no percentage
of the area meeting the 40% below average criterium and the max was a
probability of 29.4%

### Dominant tercile

While we require there to be at least 40% probability of below average,
we also require the probability of below average rainfall to be 5 percentage
points higher than that of above average rainfall
Check which moments reached this full trigger and if 2022 is included, which
it isn't

```python
ds_iri_mask = da_iri_mask.to_dataset(name="prob")
```

```python
print(
    f"Moments >={perc_area}% of the area >={threshold}% prob of below average"
    f"and below average >={threshold_diff} percent points higher than above average."
)
display(
    df_stats_aoi_bavg_trig_mom[
        df_stats_aoi_bavg_trig_mom["perc_thresh"] >= perc_area
    ]
)
```
