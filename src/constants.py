from ochanticipy import create_country_config

iso3 = "bfa"
# adding in all caps for clearer importing
ISO3 = iso3
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

# set ADM1_PCODEs for relevant regions
BOUCLEDUMOUHOUN1 = "BF46"
NORD1 = "BF54"
CENTRENORD1 = "BF49"
SAHEL1 = "BF56"

AOI_ADM1_PCODES = [BOUCLEDUMOUHOUN1, NORD1, CENTRENORD1, SAHEL1]

CERF_YEARS = [2008, 2012]
