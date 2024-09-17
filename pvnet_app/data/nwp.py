import numpy as np
import xarray as xr
import xesmf as xe
import logging
import os
import fsspec

from pvnet_app.consts import nwp_ukv_path, nwp_ecmwf_path

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.abspath(__file__))


def _download_nwp_data(source, destination):

    logger.info(f"Downloading NWP data from {source} to {destination}")

    fs = fsspec.open(source).fs
    fs.get(source, destination, recursive=True)


def download_all_nwp_data():
    """Download the NWP data"""
    _download_nwp_data(os.environ["NWP_UKV_ZARR_PATH"], nwp_ukv_path)
    _download_nwp_data(os.environ["NWP_ECMWF_ZARR_PATH"], nwp_ecmwf_path)


def regrid_nwp_data(nwp_zarr, target_coords_path, method):
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.
    """

    logger.info(f"Regridding NWP data {nwp_zarr} to expected grid to {target_coords_path}")

    ds_raw = xr.open_zarr(nwp_zarr)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude)
        and ds_raw.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No NWP regridding required for {nwp_zarr} - skipping this step")
        return

    logger.info(f"Regridding NWP {nwp_zarr} to expected grid")

    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()

    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regrid_chunk_dict = {
        "step": 1,
        "latitude": -1,
        "longitude": -1,
        "x": -1,
        "y": -1,
    }

    regridder = xe.Regridder(ds_raw, ds_target_coords, method=method)
    ds_regridded = regridder(
        ds_raw.chunk(
            {k: regrid_chunk_dict[k] for k in list(ds_raw.xindexes) if k in regrid_chunk_dict}
        )
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    os.system(f"rm -rf {nwp_zarr}")
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)

    # Rechunk to these dimensions when saving
    save_chunk_dict = {
        "step": 5,
        "latitude": 100,
        "longitude": 100,
        "x": 100,
        "y": 100,
    }

    ds_regridded.chunk(
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict}
    ).to_zarr(nwp_zarr)


def fix_ecmwf_data():

    ds = xr.open_zarr(nwp_ecmwf_path).compute()
    ds["variable"] = ds["variable"].astype(str)

    name_sub = {"t": "t2m", "clt": "tcc"}

    if any(v in name_sub for v in ds["variable"].values):
        logger.info(f"Renaming the ECMWF variables")
        ds["variable"] = np.array(
            [name_sub[v] if v in name_sub else v for v in ds["variable"].values]
        )
    else:
        logger.info(f"No ECMWF renaming required - skipping this step")

    logger.info(f"Extending the ECMWF data to reach the shetlands")
    # Thw data must be extended to reach the shetlands. This will fill missing lats with NaNs
    # and reflects what the model saw in training
    ds = ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))

    # Re-save inplace
    os.system(f"rm -rf {nwp_ecmwf_path}")
    ds.to_zarr(nwp_ecmwf_path)


def fix_ukv_data():
    """Extra steps to align UKV production data with training

    - In training the UKV data is float16. This causes it to overflow into inf values which are then
      clipped.
    """

    ds = xr.open_zarr(nwp_ukv_path).compute()
    ds = ds.astype(np.float16)

    ds["variable"] = ds["variable"].astype(str)

    # Re-save inplace
    os.system(f"rm -rf {nwp_ukv_path}")
    ds.to_zarr(nwp_ukv_path)


def preprocess_nwp_data():

    # Regrid the UKV data
    regrid_nwp_data(
        nwp_zarr=nwp_ukv_path,
        target_coords_path=f"{this_dir}/../../data/nwp_ukv_target_coords.nc",
        method="bilinear",
    )

    # Regrid the ECMWF data
    regrid_nwp_data(
        nwp_zarr=nwp_ecmwf_path,
        target_coords_path=f"{this_dir}/../../data/nwp_ecmwf_target_coords.nc",
        method="conservative",  # this is needed to avoid zeros around edges of ECMWF data
    )

    # UKV data must be float16 to allow overflow to inf like in training
    fix_ukv_data()

    # Names need to be aligned between training and prod, and we need to infill the shetlands
    fix_ecmwf_data()
