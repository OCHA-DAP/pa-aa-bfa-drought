# Burkina Faso Anticipatory Action: drought

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

## Background information

The pilot on Anticipatory Action in Burkina Faso is on seasonal drought.
To determine the likelihood of such a drought, we make use of precipitation
as meteorological indicator.
More specifically we look at the probability of below-average precipitation
according to the seasonal forecast
by the International Research Institute for Climate and Society (IRI).
The pilot covers four ADMIN1 regions,
namely Boucle du Mouhoun, Nord, Centre-Nord, and Sahel.

## Overview of analysis

We first explored data on humanitarian risk and vulnerabilities to help define
the geographic scope and shock for the pilot.
This exploration is summarized in
[this document](https://ocha-dap.github.io/pa-anticipatory-action/analyses/bfa/notebooks/bfa_risk_overview.html)
, the code for which can be found in  `docs/bfa_risk_overview.Rmd`
and `drought_trigger/01_data_summary.R`.

Thereafter we analyzed observed and forecasted precipitation patterns.
This code can be found in the `.md` files in the `drought_trigger/` directory.

## Data description

The forecast source for the trigger is
[IRI's seasonal forecast](https://iridl.ldeo.columbia.edu/maproom/Global/Forecasts/NMME_Seasonal_Forecasts/Precipitation_ELR.html).
This forecast is issued on the 15th of each month and has 1 to 4 months
leadtime.
The data can be downloaded after the creation of an account.

To validate the forecast, we use observational precipitation data by
[CHIRPS](https://www.chc.ucsb.edu/data/chirps).
This data is publicly available and is updated every month.

We also make use of the administrative boundaries from the
Common Operational Datasets (CODs) on HDX.

### TO be added

- Where does the data come from? Are there any licensing or usage restrictions?
- How can the data be accessed?
- Why were these datasets selected?
- Are there any limitations with these datasets that one should be aware
    of when running the analysis and interpreting results?

## Reproducing this analysis

Create a directory where you would like the data to be stored,
and point to it using an environment variable called
`AA_DATA_DIR`.

Next create a new virtual environment and install the requirements with:

```shell
pip install --index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple aa-toolbox==0.4.0.dev9
pip install -r requirements.txt
```

Install `src` as a package so that its contents can be used by the notebooks:

```shell
pip install -e .
```

To run the pipeline that downloads and processes the data, execute:

```shell
python src/main.py
```

To see runtime options, execute:

```shell
python src/main.py -h
```

If you would like to instead receive the processed data from our team, please
[contact us](mailto:centrehumdata@un.org).

### Monitoring

To monitor the trigger, run:

```shell
python src/monitor.py
```

## Development

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

The `markdownlint` hook will require
[Ruby](https://www.ruby-lang.org/en/documentation/installation/)
to be installed on your computer.

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```
