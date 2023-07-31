"""Class to download and load CMORPH SPI data."""
import logging
import ssl
from abc import abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlopen

import pandas as pd
import requests
import xarray as xr
from ochanticipy.config.countryconfig import CountryConfig
from ochanticipy.datasources.datasource import DataSource
from ochanticipy.utils.check_file_existence import check_file_existence
from ochanticipy.utils.dates import get_date_from_user_input
from ochanticipy.utils.geoboundingbox import GeoBoundingBox

logger = logging.getLogger(__name__)

# check this
_FIRST_AVAILABLE_DATE = date(year=1981, month=1, day=1)

_BASE_URL = (
    "https://console.cloud.google.com/storage/browser/"
    "noaa-nidis-drought-gov-data"
)


class _SPI(DataSource):
    """
    Base class object to retrieve SPI data.

    Parameters
    ----------
    """

    @abstractmethod
    def __init__(
        self,
        country_config: CountryConfig,
        geo_bounding_box: GeoBoundingBox,
        date_range_freq: str,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ):
        super().__init__(
            country_config=country_config,
            datasource_base_dir="cmorph",
            is_public=True,
        )

        self._date_range_freq = date_range_freq

        if start_date is None:
            start_date = _FIRST_AVAILABLE_DATE
        if end_date is None:
            end_date = self._get_last_available_date()

        self._start_date = get_date_from_user_input(start_date)
        self._end_date = get_date_from_user_input(end_date)

        self._check_dates_validity()

    def download(
        self,
        clobber: bool = False,
    ):
        """
        Download the SPI data as GeoTIFF.

        Parameters
        ----------
        clobber: bool, default = False
            If True, overwrites existing raw files

        Returns
        -------
        The folder where the data is downloaded.
        """
        # Create a list of date tuples
        date_list = self._create_date_list(logging_level=logging.INFO)

        # Data download
        for d in date_list:
            last_filepath = self._download_prep(d=d, clobber=clobber)

        return last_filepath.parents[0]

    def process(self, clobber: bool = False):
        """
        Process the SPI data.

        Should only be called after data has been downloaded.

        Parameters
        ----------
        clobber : bool, default = False
            If True, overwrites existing processed files.

        Returns
        -------
        The folder where the data is processed.
        """
        # Create a list with all raw data downloaded
        filepath_list = self._get_to_be_processed_path_list()

        for filepath in filepath_list:
            try:
                ds = xr.open_dataset(filepath, decode_times=False)
            except ValueError as err:
                raise ValueError(
                    f"The dataset {filepath} is not a valid netcdf file: "
                    "something probbly went wrong during the download. "
                    "Try downloading the file again."
                ) from err
            processed_file_path = self._get_processed_path(filepath)
            processed_file_path.parent.mkdir(parents=True, exist_ok=True)
            last_filepath = self._process(
                filepath=processed_file_path, ds=ds, clobber=clobber
            )

        return last_filepath.parents[0]

    def load(self) -> xr.Dataset:
        """
        Load the SPI data.

        Should only be called after the data has been downloaded and processed.

        Returns
        -------
        The processed SPI dataset.
        """
        # Get list of filepaths of files to be loaded

        filepath_list = self._get_to_be_loaded_path_list()

        # Merge all files in one dataset
        if not filepath_list:
            raise FileNotFoundError(
                "Cannot find any netcdf file for the chosen combination "
                "of frequency, resolution and area. Make sure "
                "sure that you have already called the 'process' method."
            )

        try:
            ds = xr.open_mfdataset(
                filepath_list,
            )
            # include the names of all files that are included in the ds
            ds.attrs["included_files"] = [f.stem for f in filepath_list]
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "Cannot find one or more netcdf files corresponding "
                "to the selected range. Make sure that you already "
                "downloaded and processed those data."
            ) from err

        return ds.rio.write_crs("EPSG:4326", inplace=True)

    def _download_prep(self, d, clobber):
        # Preparatory steps for the actual download
        year = str(d.year)
        month = f"{d.month:02d}"
        day = f"{d.day:02d}"
        output_filepath = self._get_raw_path(year=year, month=month, day=day)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        url = self._get_url(year=year, month=month, day=day)
        # Actual download
        return self._download(
            filepath=output_filepath,
            url=url,
            clobber=clobber,
        )

    def _create_date_list(self, logging_level=logging.DEBUG):
        """Create list of tuples containing the range of dates of interest."""
        date_list = pd.date_range(
            self._start_date, self._end_date, freq=self._date_range_freq
        ).tolist()

        # Create a message containing information on the downloaded data
        msg = (
            f"{self._frequency.capitalize()} "
            "data will be downloaded, starting from "
            f"{date_list[0]} to {date_list[-1]}."
        )

        logging.log(logging_level, msg)

        return date_list

    def _check_dates_validity(self):
        """Check dates vailidity."""
        end_avail_date = self._get_last_available_date()

        if (
            not _FIRST_AVAILABLE_DATE
            <= self._start_date
            <= self._end_date
            <= end_avail_date
        ):
            raise ValueError(
                "Make sure that the input dates are ordered in the following "
                f"way: {_FIRST_AVAILABLE_DATE} <= {self._start_date} <= "
                f"{self._end_date} <= {end_avail_date}. The two dates above "
                "indicate the range for which CHIRPS data are "
                "currently available."
            )

    def _get_file_name_base(
        self,
        year: str,
        month: str,
    ) -> str:
        if len(month) == 1:
            month = f"0{month}"

        file_name_base = (
            f"{self._country_config.iso3}_cmorph_"
            f"{self._frequency}_{year}_{month}_"
        )

        return file_name_base

    @abstractmethod
    def _get_file_name(
        self,
        year: str,
        month: str,
        day: str,
    ) -> str:
        pass

    def _get_raw_path(self, year: str, month: str, day: str) -> Path:
        return self._raw_base_dir / self._get_file_name(
            year=year, month=month, day=day
        )

    def _get_processed_path(self, raw_path: Path) -> Path:
        return self._processed_base_dir / raw_path.parts[-1]

    def _get_location_url(self):
        pass

    @abstractmethod
    def _get_url(self, year: str, month: str, day: str):
        pass

    @abstractmethod
    def _get_last_available_date(self):
        pass

    def _get_to_be_processed_path_list(self):
        """Get list of filepaths of files to be processed."""
        date_list = self._create_date_list()

        filepath_list = [
            self._get_raw_path(
                year=f"{d.year}", month=f"{d.month:02d}", day=f"{d.day:02d}"
            )
            for d in date_list
        ]
        filepath_list.sort()

        return filepath_list

    def _get_to_be_loaded_path_list(self):
        """Get list of filepaths of files to be loaded."""
        date_list = self._create_date_list()

        filepath_list = [
            self._get_processed_path(
                self._get_raw_path(
                    year=f"{d.year}",
                    month=f"{d.month:02d}",
                    day=f"{d.day:02d}",
                )
            )
            for d in date_list
        ]
        filepath_list.sort()

        return filepath_list

    @staticmethod
    def _read_csv_from_url(url):
        context = ssl.create_default_context()
        context.set_ciphers("DEFAULT")
        result = urlopen(url, context=context)

        return pd.read_csv(result)

    @check_file_existence
    def _download(self, filepath: Path, url: str, clobber: bool) -> Path:
        logger.info("Downloading CMORPH GeoTIFF file.")
        response = requests.get(
            url,
        )
        with open(filepath, "wb") as out_file:
            out_file.write(response.content)
        return filepath

    @check_file_existence
    def _process(self, filepath: Path, ds, clobber: bool) -> Path:
        pass


class SPI1(_SPI):
    """
    Class object to retrieve SPI-1 data.
    """

    def __init__(
        self,
        country_config: CountryConfig,
        geo_bounding_box: GeoBoundingBox,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ):
        super().__init__(
            country_config=country_config,
            geo_bounding_box=geo_bounding_box,
            date_range_freq="SMS",
            start_date=start_date,
            end_date=end_date,
        )

    def _get_file_name(
        self,
        year: str,
        month: str,
        day: str,
    ) -> str:
        file_name_base = self._get_file_name_base(
            year=year,
            month=month,
        )

        file_name = f"{file_name_base}"
        return file_name

    def _get_last_available_date(self):
        """Get the most recent date for which data is available."""

        # TODO: implement correctly

        datetime_object = datetime(2023, 7, 29)

        return datetime_object.date()

    def _get_url(self, year: str, month: str, day: str) -> str:
        # Convert month from month number (in string format) to
        # three-letter name

        url = (
            f"{_BASE_URL}/"
            f"datasets/cog/v1/{year}-{month}-{day}/"
            f"GLOBAL-NOAA_CPC_CMORPH-spi-1mo-{year}-{month}-{day}.tif"
        )

        return url

    @check_file_existence
    def _process(self, filepath: Path, ds, clobber: bool) -> Path:
        # fix dates
        ds.oap.correct_calendar(inplace=True)
        ds = xr.decode_cf(ds)
        if "prcp" in list(ds.keys()):
            ds = ds.rename({"prcp": "precipitation"})
        xr.Dataset.to_netcdf(ds, path=filepath)
        return filepath
