"""Download ERA5 CAPE + wind-shear fields to fill the cape/shear channels
that SEVIR doesn't provide (see preprocess.py's module docstring).

Unlike download_sevir.py, this CANNOT be fully verified without your own
Copernicus CDS account/token — there's no anonymous access to gate-test
against. The API endpoint/format below was verified live (checked
cds.climate.copernicus.eu/how-to-api on 2026-09-29: current URL is
https://cds.climate.copernicus.eu/api, auth via a personal access token in
~/.cdsapirc), but the actual data request has not been run end-to-end.
Treat this as a documented starting point, not a guaranteed-working script
the way download_sevir.py/preprocess.py are.

Setup (one-time, manual — cannot be scripted):
  1. Create a free account at https://cds.climate.copernicus.eu
  2. Get your personal access token from your profile page.
  3. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: <YOUR-PERSONAL-ACCESS-TOKEN>
  4. pip install cdsapi (already in hpc_train/requirements.txt)

What this pulls:
  - CAPE directly (`convective_available_potential_energy`, single-levels).
  - u/v wind components at 1000hPa and 500hPa (pressure-levels) to compute
    bulk wind shear magnitude as a post-processing step — ERA5 has no
    single "shear" variable.

This covers the CONUS bounding box for the 2018 Sep-Dec period matching the
SEVIR subset selected by download_sevir.py (edit AREA/YEAR/MONTHS below if
you change that selection). Matching each processed SEVIR event to the
nearest ERA5 grid cell/hour is NOT done here — that join happens in
preprocess.py if/when you wire this in (currently preprocess.py fills
cape/shear with a constant 0.5 placeholder; replacing that with a real
ERA5 lookup is documented as future work in hpc_train/README.md).

Usage:
    python data_prep/download_era5.py --out-dir data/raw/era5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Matches the SEVIR subset download_sevir.py's default selection covers
# (CONUS storms, Sep-Dec 2018) — edit if you change that selection.
AREA = [50, -125, 24, -66]  # N, W, S, E (CONUS bounding box)
YEAR = "2018"
MONTHS = ["09", "10", "11", "12"]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]  # every 3h — plenty for 10-min-cadence interpolation context


def download_cape(out_dir: Path) -> None:
    import cdsapi

    client = cdsapi.Client()
    dest = out_dir / "era5_cape.nc"
    if dest.exists():
        logger.info("already have %s, skipping", dest.name)
        return
    logger.info("Requesting CAPE from CDS (this can take minutes to hours to queue on their end)...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "convective_available_potential_energy",
            "year": YEAR,
            "month": MONTHS,
            "day": DAYS,
            "time": TIMES,
            "area": AREA,
            "format": "netcdf",
        },
        str(dest),
    )
    logger.info("Saved %s", dest)


def download_wind_for_shear(out_dir: Path) -> None:
    import cdsapi

    client = cdsapi.Client()
    dest = out_dir / "era5_wind_levels.nc"
    if dest.exists():
        logger.info("already have %s, skipping", dest.name)
        return
    logger.info("Requesting u/v wind at 1000/500 hPa from CDS for shear computation...")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": ["1000", "500"],
            "year": YEAR,
            "month": MONTHS,
            "day": DAYS,
            "time": TIMES,
            "area": AREA,
            "format": "netcdf",
        },
        str(dest),
    )
    logger.info("Saved %s", dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/raw/era5")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_cape(out_dir)
    download_wind_for_shear(out_dir)
    logger.info(
        "Done. NOTE: computing bulk shear magnitude from era5_wind_levels.nc and "
        "joining both fields into preprocess.py's per-event cape/shear channels "
        "is not yet implemented — see hpc_train/README.md."
    )
