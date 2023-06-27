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
    ChirpsMonthly,
)
from dotenv import load_dotenv
import datetime

adm1_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]

load_dotenv()
country_config = create_country_config(iso3="bfa")
codab = CodAB(country_config=country_config)
# codab.download()
gdf_adm1 = codab.load(admin_level=1)
geobb = GeoBoundingBox.from_shape(gdf_adm1)
gdf_aoi = gdf_adm1[gdf_adm1.ADM1_FR.isin(adm1_sel)]
```

```python
# download most recent

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 5, 1)

new_chirps_monthly = ChirpsMonthly(
    country_config=country_config,
    geo_bounding_box=geobb,
    start_date=start_date,
    end_date=end_date,
)

new_chirps_monthly.download()
```

```python
start_date = datetime.date(2008, 1, 1)
end_date = datetime.date(2020, 3, 1)

chirps_monthly = ChirpsMonthly(
    country_config=country_config,
    geo_bounding_box=geobb,
    start_date=start_date,
    end_date=end_date,
)

chirps_monthly_data = chirps_monthly.load()
```

```python
print(chirps_monthly_data)
```

```python

```
