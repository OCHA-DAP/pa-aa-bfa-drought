# Burkina Faso Anticipatory Action: drought

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

## Background information

Provide a basic overview of the context of anticipatory action in this country.
Link to the GDrive Trigger Card document for greater context and details.

## Overview of analysis

What is the basic process of the analysis contained within this repository?

## Data description

- Where does the data come from? Are there any licensing or usage restrictions?
- How can the data be accessed?
- Why were these datasets selected?
- Are there any limitations with these datasets that one should be aware
    of when running the analysis and interpreting results?

## Directory structure

The code in this repository is organized as follows:

```shell

├── analysis      # Main repository of analytical work for the AA pilot
├── docs          # .Rmd files or other relevant documentation
├── exploration   # Experimental work not intended to be replicated
├── src           # Code to run any relevant data acquisition/processing pipelines
|
├── .gitignore
├── README.md
└── requirements.txt

```

## Reproducing this analysis

Create a new virtual environment and install the requirements with

```shell
pip install -r requirements.txt
pip install cchardet
pip install --index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple aa-toolbox==0.4.0.dev9
pip install jupyterlab
jupyter labextension install jupyterlab-jupytext@1.2.2
```

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

The `markdownlint` hook will require [Ruby](https://www.ruby-lang.org/en/documentation/installation/)
to be installed on your computer.

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```

It is also **strongly** recommended to use `jupytext`
to convert all Jupyter notebooks (`.ipynb`) to Markdown files (`.md`)
before committing them into version control. This will make for
cleaner diffs (and thus easier code reviews) and will ensure that cell outputs aren't
committed to the repo (which might be problematic if working with sensitive data).

For baseline datasets, you might also need to sync [this](https://drive.google.com/drive/u/3/folders/1RVpnCUpxHQ-jokV_27xLRqOs6qR_8mqQ)
directory from Google drive to your local machine.
Create an environment variable called `AA_DATA_DIR` that points to this directory.
