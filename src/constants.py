from aatoolbox import create_country_config

iso3 = "bfa"
country_config = create_country_config(iso3)

# Geographic parameters
seas_len = 3  # number of months that is considered a season
adm_sel = ["Boucle du Mouhoun", "Nord", "Centre-Nord", "Sahel"]
# list of months and leadtimes that are part of the trigger
# first entry refers to the publication month, second to the leadtime
trig_mom = [(3, 3), (7, 1)]

# Trigger parameters (derived from analysis)
threshold_bavg = 40
threshold_diff = 5
perc_area = 10
