"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_UKV_ZARR_PATH
    - NWP_ECMWF_ZARR_PATH
    - SATELLITE_ZARR_PATH
"""

import logging
import os
import tempfile
import warnings
from datetime import timedelta


import numpy as np
import pandas as pd
import torch
import typer
import xarray as xr
import dask
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from nowcasting_datamodel.models.base import Base_Forecast
from ocf_datapipes.load import OpenGSPFromDatabase
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from ocf_datapipes.batch import BatchKey, stack_np_examples_into_batch
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

import pvnet
from pvnet.data.utils import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.utils import GSPLocationLookup

import pvnet_app
from pvnet_app.utils import (
    worker_init_fn, populate_data_config_sources, convert_dataarray_to_forecasts, preds_to_dataarray
)
from pvnet_app.data import (
    download_all_sat_data, download_all_nwp_data, preprocess_sat_data, preprocess_nwp_data,
)

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If the solar elevation is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0

# Forecast made for these GSP IDs and summed to national with ID=>0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# Huggingfacehub model repo and commit for PVNet (GSP-level model)
default_model_name = "openclimatefix/pvnet_v2"
default_model_version = "5ed2b179974993d8804a1e60fdc850dc547e9025"

# Huggingfacehub model repo and commit for PVNet summation (GSP sum to national model)
# If summation_model_name is set to None, a simple sum is computed instead
default_summation_model_name = "openclimatefix/pvnet_v2_summation"
default_summation_model_version = "22a264a55babcc2f1363b3985cede088a6b08977"

model_name_ocf_db = "pvnet_v2"
use_adjuster = os.getenv("USE_ADJUSTER", "True").lower() == "true"

# If environmental variable is true, the sum-of-GSPs will be computed and saved under a different
# model name. This can be useful to compare against the summation model and therefore monitor its
# performance in production
save_gsp_sum = os.getenv("SAVE_GSP_SUM", "False").lower() == "true"
gsp_sum_model_name_ocf_db = "pvnet_gsp_sum"

# ---------------------------------------------------------------------------
# LOGGER

class SQLAlchemyFilter(logging.Filter):

    def filter(self, record):
        return "sqlalchemy" not in record.pathname

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Get rid of the verbose sqlalchemy logs
stream_handler.addFilter(SQLAlchemyFilter())
sql_logger = logging.getLogger("sqlalchemy.engine.Engine")
sql_logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# APP MAIN

def app(
    t0=None,
    apply_adjuster: bool = use_adjuster,
    gsp_ids: list[int] = all_gsp_ids,
    write_predictions: bool = True,
    num_workers: int = -1,
):
    """Inference function for production

    This app expects these evironmental variables to be available:
        - DB_URL
        - NWP_UKV_ZARR_PATH
        - NWP_ECMWF_ZARR_PATH
        - SATELLITE_ZARR_PATH
    Args:
        t0 (datetime): Datetime at which forecast is made
        apply_adjuster (bool): Whether to apply the adjuster when saving forecast
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
        num_workers (int): Number of workers to use to load batches of data. When set to default
            value of -1, it will use one less than the number of CPU cores workers.
    """

    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    if num_workers>0:
        # Without this line the dataloader will hang if multiple workers are used
        dask.config.set(scheduler='single-threaded')

    logger.info(f"Using `pvnet` library version: {pvnet.__version__}")
    logger.info(f"Using {num_workers} workers")
    logger.info(f"Using adjduster: {use_adjuster}")
    logger.info(f"Saving GSP sum: {save_gsp_sum}")

    # Allow environment overwrite of model
    model_name = os.getenv("APP_MODEL", default=default_model_name)
    model_version = os.getenv("APP_MODEL_VERSION", default=default_model_version)
    summation_model_name = os.getenv("APP_SUMMATION_MODEL", default=default_summation_model_name)
    summation_model_version = os.getenv(
        "APP_SUMMATION_MODEL", default=default_summation_model_version
    )

    # ---------------------------------------------------------------------------
    # 0. If inference datetime is None, round down to last 30 minutes
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor(timedelta(minutes=30))
    else:
        t0 = pd.to_datetime(t0).floor(timedelta(minutes=30))

    if len(gsp_ids) == 0:
        gsp_ids = all_gsp_ids

    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Making forecast for GSP IDs: {gsp_ids}")

    # ---------------------------------------------------------------------------
    # 1. Prepare data sources

    # Make pands Series of most recent GSP effective capacities

    logger.info("Loading GSP metadata")

    ds_gsp = next(iter(OpenGSPFromDatabase()))
    
    # Get capacities from the database
    url = os.getenv("DB_URL")
    db_connection = DatabaseConnection(url=url, base=Base_Forecast, echo=False)
    with db_connection.get_session() as session:
        #  Pandas series of most recent GSP capacities
        gsp_capacities = get_latest_gsp_capacities(session, gsp_ids)
        
        # National capacity is needed if using summation model
        national_capacity = get_latest_gsp_capacities(session, [0])[0]

    # Set up ID location query object
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)

    # Download satellite data
    logger.info("Downloading satellite data")
    download_all_sat_data()

    # Process the 5/15 minutely satellite data
    preprocess_sat_data(t0)
    
    # Download NWP data
    logger.info("Downloading NWP data")
    download_all_nwp_data()
    
    # Preprocess the NWP data
    preprocess_nwp_data()
        
    # ---------------------------------------------------------------------------
    # 2. Set up data loader
    logger.info("Creating DataLoader")
    
    # Pull the data config from huggingface
    data_config_filename = PVNetBaseModel.get_data_config(
        model_name,
        revision=model_version,
    )
        
    # Populate the data config with production data paths
    temp_dir = tempfile.TemporaryDirectory()
    populated_data_config_filename = f"{temp_dir.name}/data_config.yaml"
    
    populate_data_config_sources(data_config_filename, populated_data_config_filename)

    # Location and time datapipes
    location_pipe = IterableWrapper([gsp_id_to_loc(gsp_id) for gsp_id in gsp_ids])
    t0_datapipe = IterableWrapper([t0]).repeat(len(location_pipe))

    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # Batch datapipe
    batch_datapipe = (
        construct_sliced_data_pipeline(
            config_filename=populated_data_config_filename,
            location_pipe=location_pipe,
            t0_datapipe=t0_datapipe,
            production=True,
            check_satellite_no_zeros=True,
        )
        .batch(batch_size)
        .map(stack_np_examples_into_batch)
    )

    # Set up dataloader for parallel loading
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,  # batched in datapipe step
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )
    
    dataloader = DataLoader(batch_datapipe, **dataloader_kwargs)

    # ---------------------------------------------------------------------------
    # 3. set up model
    logger.info(f"Loading model: {model_name} - {model_version}")

    model = PVNetBaseModel.from_pretrained(
        model_name,
        revision=model_version,
    ).to(device)

    if summation_model_name is not None:
        summation_model = SummationBaseModel.from_pretrained(
            summation_model_name,
            revision=summation_model_version,
        ).to(device)

        if (
            summation_model.pvnet_model_name != model_name
            or summation_model.pvnet_model_version != model_version
        ):
            warnings.warn(
                f"The PVNet version running in this app is {model_name}/{model_version}. "
                "The summation model running in this app was trained on outputs from PVNet version "
                f"{summation_model.pvnet_model_name}/{summation_model.pvnet_model_version}. "
                "Combining these models may lead to an error if the shape of PVNet output doesn't "
                "match the expected shape of the summation model. Combining may lead to unreliable "
                "results even if the shapes match."
            )

    # 4. Make prediction
    logger.info("Processing batches")
    normed_preds = []
    gsp_ids_each_batch = []
    sun_down_masks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting for batch: {i}")

            # Store GSP IDs for this batch for reordering later
            these_gsp_ids = batch[BatchKey.gsp_id]
            gsp_ids_each_batch += [these_gsp_ids]

            # Run batch through model
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            preds = model(device_batch).detach().cpu().numpy()

            # Calculate unnormalised elevation and sun-dowm mask
            logger.info("Zeroing predictions after sundown")
            elevation = batch[BatchKey.gsp_solar_elevation] * ELEVATION_STD + ELEVATION_MEAN
            # We only need elevation mask for forecasted values, not history
            elevation = elevation[:, -preds.shape[1] :]
            sun_down_mask = elevation.min(dim="time") < MIN_DAY_ELEVATION

            # Store predictions
            normed_preds += [preds]
            sun_down_masks += [sun_down_mask]

            # log max prediction
            logger.info(f"GSP IDs: {these_gsp_ids}")
            logger.info(f"Max prediction: {np.max(preds, axis=1)}")
            logger.info(f"Completed batch: {i}")

    normed_preds = np.concatenate(normed_preds)
    sun_down_masks = np.concatenate(sun_down_masks)

    gsp_ids_all_batches = np.concatenate(gsp_ids_each_batch).squeeze()
    
    n_times = normed_preds.shape[1]
    
    valid_times = pd.to_datetime([t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)])

    # Reorder GSP order which ends up shuffled if multiprocessing is used
    inds = gsp_ids_all_batches.argsort()

    normed_preds = normed_preds[inds]
    sun_down_masks = sun_down_masks[inds]
    gsp_ids_all_batches = gsp_ids_all_batches[inds]

    logger.info(f"{gsp_ids_all_batches.shape}")

    # ---------------------------------------------------------------------------
    # 5. Merge batch results to xarray DataArray
    logger.info("Processing raw predictions to DataArray")

    da_normed = preds_to_dataarray(normed_preds, model, valid_times, gsp_ids_all_batches)

    da_sundown_mask = xr.DataArray(
        data=sun_down_masks,
        dims=["gsp_id", "target_datetime_utc"],
        coords=dict(
            gsp_id=gsp_ids_all_batches,
            target_datetime_utc=valid_times,
        ),
    )

    # Multiply normalised forecasts by capacities and clip negatives
    logger.info(f"Converting to absolute MW using {gsp_capacities}")
    da_abs = da_normed.clip(0, None) * gsp_capacities.values[:, None, None]
    max_preds = da_abs.sel(output_label="forecast_mw").max(dim="target_datetime_utc")
    logger.info(f"Maximum predictions: {max_preds}")

    # Apply sundown mask
    da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)

    # ---------------------------------------------------------------------------
    # 6. Make national total
    logger.info("Summing to national forecast")

    if summation_model_name is not None:
        logger.info("Using summation model to produce national forecast")

        # Make national predictions using summation model
        inputs = {
            "pvnet_outputs": torch.Tensor(normed_preds[np.newaxis]).to(device),
            "effective_capacity": (
                torch.Tensor(gsp_capacities.values / national_capacity)
                .to(device)
                .unsqueeze(0)
                .unsqueeze(-1)
            ),
        }
        normed_national = summation_model(inputs).detach().squeeze().cpu().numpy()

        # Convert national predictions to DataArray
        da_normed_national = preds_to_dataarray(
            normed_national[np.newaxis], 
            summation_model, 
            valid_times, 
            gsp_ids=[0]
        )

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs_national = da_normed_national.clip(0, None) * national_capacity

        # Apply sundown mask - All GSPs must be masked to mask national
        da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")

    else:
        logger.info("Summing across GSPs to produce national forecast")
        da_abs_national = (
            da_abs.sum(dim="gsp_id").expand_dims(dim="gsp_id", axis=0).assign_coords(gsp_id=[0])
        )
        da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")

    logger.info(
        f"National forecast is {da_abs_all.sel(gsp_id=0, output_label='forecast_mw').values}"
    )
        
    if save_gsp_sum:
        # Compute the sum if we are logging the sume of GSPs independently
        logger.info("Summing across GSPs to for independent sum-of-GSP saving")
        da_abs_sum_gsps = (
            da_abs.sum(dim="gsp_id")
            # Only select the central forecast for the GSP sum. The sums of different p-levels 
            # are not a meaningful qauntities
            .sel(output_label=["forecast_mw"])
            .expand_dims(dim="gsp_id", axis=0)
            .assign_coords(gsp_id=[0])
        )

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return da_abs_all

    # ---------------------------------------------------------------------------
    # 7. Write predictions to database
    logger.info("Writing to database")

    connection = DatabaseConnection(url=os.environ["DB_URL"])
    with connection.get_session() as session:
        sql_forecasts = convert_dataarray_to_forecasts(
            da_abs_all, session, model_name=model_name_ocf_db, version=pvnet_app.__version__
        )

        save_sql_forecasts(
            forecasts=sql_forecasts,
            session=session,
            update_national=True,
            update_gsp=True,
            apply_adjuster=apply_adjuster,
        )
        
        if save_gsp_sum:
            # Save the sum of GSPs independently - mainly for summation model monitoring
            sql_forecasts = convert_dataarray_to_forecasts(
                da_abs_sum_gsps, 
                session, 
                model_name=gsp_sum_model_name_ocf_db, 
                version=pvnet_app.__version__
            )

            save_sql_forecasts(
                forecasts=sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=False,
                apply_adjuster=False,
            )
            
    temp_dir.cleanup()
    logger.info("Finished forecast")


if __name__ == "__main__":
    typer.run(app)
