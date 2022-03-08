# IRI forecast as a trigger for drought in Burkina Faso

This notebook entails the analysis that was done for analyzing the
IRI seasonal forecast as part of drought-related trigger in Burkina Faso.
An initial proposal from in-country partners was:

- Trigger #1 in March covering June-July-August. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%.
- Targeted Admin1s: Boucle de Mounhoun, Centre Nord, Sahel, Nord.

This notebook explores if and when these triggers would be reached.
As part of this exploration methods for aggregation from raster level
to the percentage of the area are discussed.

These are the main conclusions:

1) Due to the limited data availability it is almost impossible to set an
educated threshold. There is only data since 2017.
We did see a CERF allocation in 2017 but other than that
we don't have drought-events in BFA since 2017.
2) The general skill (GROC) over BFA is decent, especially compared to other
parts of the world.
3) A threshold of 50% is rather high.
4) We instead recommend a threshold of 40% over 10% of the area as this
is expected to be reached from time to time but not too often.
However, this threshold couldn't be validated extensively, and thus a
threshold of e.g. 45% might also have worked.
5) We recommend a 5% difference between the probability of below and above
average. We didn't observe this in the limited data but we still
include it for certainty.
6) Forecasted patterns change with differening leadtimes.

We didn't do an elaborate analysis of how they change and
how we should take these changing patterns into account.

## Skill

Before diving into any code, lets analyze the skill as produced by IRI.
The GROC is shown below where grey indicates no skill, and white a dry mask.
As can be seen from the images, over most parts of Burkina Faso the forecasts
show a positive skill for both leadtimes.

There are some differences between the leadtimes.
However the differences between leadtimes are small and
thus should be interpreted with caution.

<!-- markdownlint-disable MD013 MD033 -->
<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_jja_Ld3.gif" alt="drawing" width="700"/>
<img src="https://iri.columbia.edu/climate/verification/images/NAskillmaps/pcp/PR1_groc_aso_Ld1.gif" alt="drawing" width="700"/>
<!-- markdownlint-enable MD013 MD033 -->

## Load libraries and set global constants

```python
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
from shapely.geometry import mapping
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

from aatoolbox.pipeline import Pipeline
import aatoolbox.utils.raster
```

```python
hdx_blue="#007ce0"
```

```python
#month number refers to the last month of the season
month_season_mapping={
    1:"NDJ",2:"DJF",3:"JFM",4:"FMA",5:"MAM",6:"AMJ",
    7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND"
}
```

```python
# leadtime_mar=3
# leadtime_jul=1
```

```python
iso3="bfa"
```

```python
pipeline=Pipeline(iso3)
```

```python
gdf_adm1=pipeline.load_codab(admin_level=1)
```

```python
geobb=pipeline.load_geoboundingbox_gdf(gdf_adm1)
```

## Set variables

```python
#list of months and leadtimes that could be part of the trigger
#first entry refers to the publication month, second to the leadtime
trig_mom=[(3,3),(7,1)]
```

```python
adm_sel=["Boucle du Mouhoun","Nord","Centre-Nord","Sahel"]
# adm_sel_str=re.sub(r"[ -]", "", "".join(adm_sel)).lower()
# threshold_mar=40
# threshold_jul=50
```

```python
gdf_aoi=gdf_adm1[gdf_adm1.ADM1_FR.isin(adm_sel)]
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
ds_iri_dom=pipeline.load_iri_dominant_tercile_seasonal_forecast(
        geo_bounding_box=geobb
    )
```

```python
da_iri_dom=ds_iri_dom.dominant
```

```python
#F indicates the publication month, and L the leadtime.
#A leadtime of 1 means a forecast published in May is forecasting JJA
da_iri_dom_clip=da_iri_dom.rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
def plt_raster_iri(da_iri_dom_clip,
                   pub_mon,
                   lt,
                   plt_levels,
                   plt_colors,
                  ):
    for_seas=month_season_mapping[(pub_mon+lt+1)%12+1]
    g=da_iri_dom_clip.where(da_iri_dom_clip.F.dt.month.isin([pub_mon]),
                            drop=True).sel(L=lt).plot(
    col="F",
    col_wrap=5,
    levels=plt_levels,
    colors=plt_colors,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        'ticks': plt_levels,
    },
    figsize=(25,7)
    )
    for ax in g.axes.flat:
        gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
        gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    g.fig.suptitle(
        (f"Forecasts published in {calendar.month_abbr[pub_mon]} predicting"
    f"{for_seas} (lt={lt}) \n The subtitles indicate the publishing date"),
    y=1.1);
```

```python
#iri website bins
# plt_levels=[-100,-67.5,-57.5,-47.5,-42.5,-37.5,37.5,42.5,47.5,57.5,67.5,100]
#rounded bins for easier interpretability
plt_levels=[-100,-70,-60,-50,-45,-40,40,45,50,60,70,100]
plt_colors=(['#783200','#ab461e','#d18132','#e8b832','#fafa02','#ffffff',
             '#d1f8cc','#acf8a0','#73bb6e','#3a82b3','#0e3bf4'])
```

```python
plt_raster_iri(
    da_iri_dom_clip,pub_mon=3,lt=3,plt_levels=plt_levels,plt_colors=plt_colors
)
```

```python
plt_raster_iri(
    da_iri_dom_clip,pub_mon=7,lt=1,plt_levels=plt_levels,plt_colors=plt_colors
)
```

From the above plots we can see that since 2017 for our months and leadtimes
of interest, most forecasts were forecasting around normal conditions
(between -40 and 40%). In some years the forecasts leaned slightly more to
above average and in March 2017 to below average.

In 2017 a drought was observed in Burkina Faso, so this could possibly be a
point in time that we would want to trigger.

Just for comparison we plot below the forecast for JJA but published in May.
So with a one month leadtime instead of 3.
We can see here that the patterns quite heavily changed,
which serves as a reminder that predictions change over time

```python
plt_raster_iri(
    da_iri_dom_clip,pub_mon=5,lt=1,plt_levels=plt_levels,plt_colors=plt_colors
)
```

Below we plot a few examples of "tricky" forecasts.
For each of the three we could wonder whether the trigger should be met.
Is 40% a high-enough threshold? How much of the area should be above a certain
threshold? And what if we see opposite patterns within the region?

These figures are to guide the discussion on which forecasts
we would have wanted to trigger and for which we wouldn't

```python
g=da_iri_dom_clip.where(
    da_iri_dom_clip.F.isin([cftime.Datetime360Day(2021, 8, 16, 0, 0, 0, 0),
                            cftime.Datetime360Day(2020, 3, 16, 0, 0, 0, 0),
                            cftime.Datetime360Day(2021, 9, 16, 0, 0, 0, 0),]),
    drop=True
).sel(L=1).plot(
    col="F",
    col_wrap=4,
    levels=plt_levels,
    colors=plt_colors,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        'ticks': plt_levels,
    },
    figsize=(15,7)
)
for ax in g.axes.flat:
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    gdf_aoi.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

### Which cells to include for aggregation?

For the trigger we have to aggregate a selection of raster cells to one number.
Before we can do this, we have to decide which cells to include for the aggregation.
We inspect 3 different methods: including all cells with their centre in the region,
all cells touching the region, and an approximate mask.

After discussion we concluded that the approximate
mask is a valid method and thus use this further on.

```python
#sel random values to enable easy plotting of included cells (so values are irrelevant)
da_iri_dom_blue=da_iri_dom.sel(F="2020-05-16",L=1).squeeze()
```

```python
da_iri_dom_blue_centre=da_iri_dom_blue.rio.clip(gdf_aoi["geometry"], all_touched=False)
g=da_iri_dom_blue_centre.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(
    "Included area with cell centres:"
    f"{da_iri_dom_blue_centre.count().values} cells included"
)
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
da_iri_dom_blue_touched=da_iri_dom_blue.rio.clip(gdf_aoi["geometry"], all_touched=True)
g=da_iri_dom_blue_touched.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(
    f"Included area with all cells touching:"
    "{da_iri_dom_blue_touched.count().values} cells included"
)
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

```python
#approximate of a mask
da_iri_dom_blue_res = da_iri_dom_blue.rio.reproject(
    da_iri_dom_blue.rio.crs,
    #resolution it will be changed to, original is 1
    resolution=0.05,
    #use nearest so cell values stay the same, only cut
    #into smaller pieces
    resampling=Resampling.nearest,
    nodata=np.nan,
).rio.clip(gdf_aoi["geometry"], all_touched=False)
```

```python
g=da_iri_dom_blue_res.plot.imshow(cmap=ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Included area with approx mask")
gdf_aoi.boundary.plot(linewidth=1, ax=g.axes, color="red")
g.axes.axis("off");
```

#### Selecting the data

Before we looked at the dominant tercile.
From here on we use a dataset that contains the probability per tercile,
such that we can the range of values of the below average tercile,
regardless of the values of the other terciles.

We load the data and apply an approximate mask to our region of interest.

```python
#Load the data that contains the probability per tercile
#this allows us to solely look at the below-average tercile
#C indicates the tercile (below-average, normal, or above-average).
#F indicates the publication month, and L the leadtime
ds_iri=pipeline.load_iri_all_terciles_seasonal_forecast(
        geo_bounding_box=geobb
    )
da_iri = ds_iri.prob
```

```python
#select all cells touching the region
da_iri_allt=da_iri.rio.clip(gdf_aoi["geometry"], all_touched=True)
#C=0 indicates the below average tercile
da_iri_allt_bavg=da_iri_allt.sel(C=0)
```

```python
#upsample the resolution in order to create a mask of our aoi
resolution = 0.05
mask_list=[]
for terc in da_iri_allt.C.values:
    for lt in da_iri_allt.L.values:
        da_terc_lt = da_iri_allt.sel(C=terc,L=lt)
        da_terc_lt_mask = da_terc_lt.rio.reproject(
            da_terc_lt.rio.crs,
            resolution=resolution,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )
        mask_list.append(da_terc_lt_mask.expand_dims({"C":[terc],"L":[lt]}))
da_iri_mask=xr.combine_by_coords(mask_list).rio.clip(gdf_aoi["geometry"],all_touched=False).prob
# reproject changes longitude and latitude name to x and y
# so change back here
da_iri_mask = da_iri_mask.rename(
    {"x": "longitude", "y": "latitude"}
)
da_iri_mask_bavg=da_iri_mask.sel(C=0)
```

```python
#check that masking is done correctly
g=da_iri_mask.sel(F="2021-03-16",L=2,C=0).squeeze().plot.imshow(cmap=ListedColormap([hdx_blue]),add_colorbar=False)
gdf_adm1.boundary.plot(ax=g.axes);
```

### Threshold

Next we analyze the probabilities of below average to try and set a threshold.

The proposed thresholds from in-country partners were 40% and 50%.
The first plot shows all values across all raster cells in the world,
across all seasons and leadtimes. We can see that the median is at 34.
Values above 60 are never occured and above 50 are already exreme.

The second plot shows the values of only the raster cells that touch
the region but across all seasons.
We can see that the median is slightly lower, around 33, depending on the leadtime.
The distributions per leadtime differ slightly,
seeing a broader distribution for 3 and 4 months leadtime.
Values higher than 50 didn't occur.
We should be aware though that we only have 5 years of data.

Moreover, the pattern might be very different depending on the season.
The third plot show the distribution when we only select the seasons and
leadtimes that might be part of the trigger.
We see that the values are slighlty lower,
but there are only 10 datapoints in this plot,
so we shouldn't attach much value to it.

We should also be aware that these plots show the values at raster cell level.
If we thereafter require 10% of the area meeting the probability threshold,
this is even less likely to occur.

Due to the limited data availability it is very hard to determine the threshold objectively.
However, a threshold anywhere between 40 and 50 could be reasonable.
We experimented with these different thresholds.

Based on this experimentation we propose a threshold of 40%.
This because we estimate it to be already quite rare,
in combination with the 20% of the area requirement,
but at the same time we estimate it to be possible to occur.

```python
#TODO: also change this to violin plot
da_iri.sel(C=0).hvplot.box('prob',alpha=0.5).opts(ylabel="Probability below average",
title="Forecasted probabilities of below average"
      "\n at raster level in the whole world across all seasons and leadtimes, 2017-2021")
```

```python
#TODO: would want to replace `box` by `violin`
#but for some reason violin is not working in this env
#on my `antact` env it is working so has to do smth with the env
#but cannot figure out what
#already matched versions of hvplot, bokeh, and holoviews
da_iri_mask_bavg.hvplot.box(
    'prob',by="L",color='L', cmap='Category20'
).opts(
    ylabel="Probability below average",
    xlabel="leadtime",
title="Observed probabilities of bavg at raster level in the region of interest")
```

```python
#transform data such that we can select
# by combination of publication month (F) and leadtime (L)
da_plt=da_iri_mask_bavg.assign_coords(F=da_iri_mask_bavg.F.dt.month)
da_plt=da_plt.stack(comb=["F","L"])
#only select data that is selected for trigger
da_iri_mask_trig_mom=xr.concat([da_plt.sel(comb=m) for m in trig_mom],dim="comb")
```

```python
da_iri_mask_trig_mom.hvplot.box(
    'prob'
).opts(ylabel="Probability below average",
       title="observed probabilities of bavg for the month and leadtime"
             "combinations \n included in the triger"
       )
```

#### Compute stats

We can now compute the statistics of the region of interest.
We use the approximate mask to define the cells included
for the computation of the statistics.

We have to set two parameters: the minimum probability of below average,
and the method of aggregation.

Based on the above plots we set the minimum probability of below average to 40%.
We have limited data to base this on but the reasoning is
that 40% is relatively rare while not unreachable.

How rare it is, also depends on how the cells within the region of interest are aggregated.
We chose to look at the percentage of cells reaching the threshold.
Since the max would be too sensitive, while the mean wouldn't be sensitive enough.

While we experimented with different percentages,
we set the required percentage of the region reaching the threshold to 10%.
This is relatively low, and could maybe be increased in the future.
But it was partly-based on the data of 2017 where we did see a drought
and the forecast had around 10% of the area with a probability of 40% of below average.

```python
#% probability of bavg
threshold=40
#min percentage of the area that needs to reach the threshold
perc_area=10
```

```python
adm0_col="ADM0_FR"
pcode0_col="ADM0_PCODE"
```

```python
#compute stats
#dissolve the region to one polygon
gdf_aoi_dissolved=gdf_aoi.dissolve(by=adm0_col)
gdf_aoi_dissolved=gdf_aoi_dissolved[[pcode0_col,"geometry"]]
df_stats_aoi_bavg=da_iri_mask_bavg.aat.compute_raster_stats(
    gdf=gdf_aoi_dissolved,feature_col=pcode0_col)
da_iri_mask_thresh=da_iri_mask_bavg.where(da_iri_mask_bavg>=threshold)
df_stats_aoi_bavg_thresh=da_iri_mask_thresh.aat.compute_raster_stats(
    gdf=gdf_aoi_dissolved,feature_col=pcode0_col)
df_stats_aoi_bavg["perc_thresh"] = df_stats_aoi_bavg_thresh[f"count_{pcode0_col}"]/df_stats_aoi_bavg[f"count_{pcode0_col}"]*100
df_stats_aoi_bavg["F"]=pd.to_datetime(df_stats_aoi_bavg["F"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_aoi_bavg["month"]=df_stats_aoi_bavg.F.dt.month
```

NaN values indicate that the whole region is covered by a dry mask at that point.
See [here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/)
for more information

```python
df_stats_aoi_bavg=df_stats_aoi_bavg[(~df_stats_aoi_bavg.perc_thresh.isnull())]
```

```python
df_stats_aoi_bavg=df_stats_aoi_bavg.sort_values("perc_thresh",ascending=False)
```

## Analyze statistics probability below average

We plot the occurrences of the probability of below average being above
the given threshold and given minimum percentage of the area.

```python
print(f"{round(len(df_stats_aoi_bavg[df_stats_aoi_bavg['perc_thresh']>=perc_area])/len(df_stats_aoi_bavg)*100)}%"
      f"({round(len(df_stats_aoi_bavg[df_stats_aoi_bavg['perc_thresh']>=perc_area]))}/{len(df_stats_aoi_bavg)})"
      " of forecasts across all seasons and leadtimes"
      f" predicted >={perc_area}% of the area >={threshold}% prob of below average")
```

```python
histo=alt.Chart(df_stats_aoi_bavg).mark_bar().encode(
    alt.X("perc_thresh:Q", bin=alt.Bin(step=1),
          title=f"% of region with >={threshold} probability of bavg"),
    y='count()',
).properties(
    title=[f"Occurence of the percentage of the region with >={threshold}"
           "probability of bavg",
                    "Across all seasons and leadtimes",
                    "Red line indicates the threshold on the % of the area"])
line = alt.Chart(pd.DataFrame({'x': [perc_area]})).mark_rule(color="red").encode(x='x')
histo+line
```

```python
#select the months and leadtimes included in the trigger
df_stats_aoi_bavg_trig_mom=df_stats_aoi_bavg[df_stats_aoi_bavg[['month', 'L']].apply(
    tuple, axis=1).isin(trig_mom)]
```

```python
print(f"{round(len(df_stats_aoi_bavg_trig_mom[df_stats_aoi_bavg_trig_mom['perc_thresh']>=perc_area])/len(df_stats_aoi_bavg_trig_mom)*100)}%"
      f"({round(len(df_stats_aoi_bavg_trig_mom[df_stats_aoi_bavg_trig_mom['perc_thresh']>=perc_area]))}/{len(df_stats_aoi_bavg_trig_mom)})"
      "of forecasts of moments that would be included in the trigger"
      f" predicted >={perc_area}% of the area >={threshold}% prob of below average")
```

```python
histo=alt.Chart(df_stats_aoi_bavg_trig_mom).mark_bar().encode(
    alt.X("perc_thresh:Q", bin=alt.Bin(step=1),
          title=f"% of region with >={threshold} probability of bavg"),
    y='count()',
).properties(
    title=[f"Occurence of the percentage of the region"
           "with >={threshold} probability of bavg",
                    "For the publication months and leadtimes included in the trigger",
                    "Red line indicates the threshold on the % of the area"])
line = alt.Chart(pd.DataFrame({'x': [perc_area]})).mark_rule(color="red").encode(x='x')
histo+line
```

```python
df_stats_aoi_bavg_trig_mom["pred_month"]=df_stats_aoi_bavg_trig_mom.apply(
    lambda x: x["F"]+relativedelta(months=int(x["L"])),axis=1)
```

```python
df_stats_aoi_bavg_trig_mom.sort_values(["pred_month","L"]).head()
```

### Dominant tercile

While we require there to be at least 40% probability of below average,
we also want to be sure that this probability is higher than that of above average.
The plots below show that this chance is small that this occurs when there is a
40% probability of below average.
Nevertheless, we did decide to build a mechanism for this in the trigger
to ensure in such a situation the trigger isn't activated. I.e. we require that
probability below average >= (probability above average + x%)

We determine this at the pixel level. Based on experiments we set x to 5%

```python
ds_iri_mask=da_iri_mask.to_dataset(name="prob")
```

```python
#cannot get hvplot violin to show up somehow
# da_iri_mask_diff=da_iri_mask.sel(C=0)-da_iri_mask.sel(C=2)
# da_iri_mask_diff.hvplot.violin(
# 'prob', by='L', color='L', cmap='Category20').opts(ylabel="%bavg - %abv avg")
```

```python
(da_iri.sel(C=0)-da_iri.sel(C=2)).hvplot.hist(
    'prob',alpha=0.5,
    title="difference between below and above average tercile"
)
```

```python
da_iri.where((da_iri.sel(C=0)-da_iri.sel(C=2)<=5),drop=True).sel(C=0).hvplot.hist(
    'prob',alpha=0.5,
    title="below average probability when there"
          "is less than 5% diff between bel avg and abv avg"
)
```

### Conclusion

Based on the analysis the proposed threshold is >=40% probability
of below average + at least 5% higher probability on below than above average rainfall.
This is determined at the pixel level.
We then require at least 10% of the pixels in the region of interest
to meet this threshold during JJA or ASO.

We are aware of several limitations

- The threshold of 40% couldn't be validated thoroughly due to limited data availability
- The same goes for the 5% difference requirement.
We didn't observe this in the limited data but we still include it for certainty.
- Forecasted patterns change with differening leadtimes.
We didn't do an elaborate analysis of how they change and
how we should take these changing patterns into account.
