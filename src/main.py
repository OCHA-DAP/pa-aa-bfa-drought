import argparse
import logging

from aatoolbox import (
    CodAB,
    GeoBoundingBox,
    IriForecastDominant,
    IriForecastProb,
)

from src import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clobber", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    return parser.parse_args()


def run_pipeline(clobber: bool = False):
    cod_ab = CodAB(country_config=constants.country_config)
    cod_ab.download(clobber=clobber)
    gdf_adm0 = cod_ab.load(admin_level=0)

    geobb = GeoBoundingBox.from_shape(gdf_adm0)
    iri_dom = IriForecastDominant(
        constants.country_config, geo_bounding_box=geobb
    )
    iri_dom.download(clobber=clobber)
    iri_dom.process(clobber=clobber)

    iri_all_terciles = IriForecastProb(
        constants.country_config, geo_bounding_box=geobb
    )
    iri_all_terciles.download(clobber=clobber)
    iri_all_terciles.process(clobber=clobber)


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    run_pipeline(clobber=args.clobber)
