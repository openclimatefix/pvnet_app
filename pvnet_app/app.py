"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_UKV_ZARR_PATH
    - NWP_ECMWF_ZARR_PATH
    - SATELLITE_ZARR_PATH
    - RUN_EXTRA_MODELS
    - USE_ADJUSTER
    - SAVE_GSP_SUM
"""

import logging
import warnings
import os
import tempfile
from datetime import timedelta


import numpy as np
import pandas as pd
import torch
import typer
import dask
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from nowcasting_datamodel.models.base import Base_Forecast
from ocf_datapipes.load import OpenGSPFromDatabase
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.batch import stack_np_examples_into_batch, batch_to_tensor, copy_batch_to_device

from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

import pvnet
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.utils import GSPLocationLookup

import pvnet_app
from pvnet_app.utils import (
    worker_init_fn,
    populate_data_config_sources,
    convert_dataarray_to_forecasts,
    find_min_satellite_delay_config,
    save_yaml_config,
)
from pvnet_app.data import (
    download_all_sat_data,
    download_all_nwp_data,
    preprocess_sat_data,
    preprocess_nwp_data,
    check_model_inputs_available,
)
from pvnet_app.forecast_compiler import ForecastCompiler

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Forecast made for these GSP IDs and summed to national with ID=>0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# Dictionary of all models to run
# - The dictionary key will be used as the model name when saving to the database
# - The key "pvnet_v2" must be included
# - Batches are prepared only once, so the extra models must be able to run on the batches created
#   to run the pvnet_v2 model
models_dict = {
    "pvnet_v2": {
        # Huggingfacehub model repo and commit for PVNet (GSP-level model)
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "9989666ae3792a576dbc16872e152985c950a42e",
        },
        # Huggingfacehub model repo and commit for PVNet summation (GSP sum to national model)
        # If summation_model_name is set to None, a simple sum is computed instead
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "22a264a55babcc2f1363b3985cede088a6b08977",
        },
        # Whether to use the adjuster for this model - for pvnet_v2 is set by environmental variable
        "use_adjuster": os.getenv("USE_ADJUSTER", "true").lower() == "true",
        # Whether to save the GSP sum for this model - for pvnet_v2 is set by environmental variable
        "save_gsp_sum": os.getenv("SAVE_GSP_SUM", "false").lower() == "true",
        # Where to log information through prediction steps for this model
        "verbose": True,
    },
    # Extra models which will be run on dev only
    "pvnet_v2-sat0min-v8-batches": {
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "849f19b0c774a1a3fe10e20f901e225131f5645b",
        },
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "22a264a55babcc2f1363b3985cede088a6b08977",
        },
        "use_adjuster": False,
        "save_gsp_sum": False,
        "verbose": False,
    },
}

day_ahead_model_dict = {
    "pvnet_day_ahead": {
        # Huggingfacehub model repo and commit for PVNet day ahead (GSP-level model)
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region_day_ahead",
            "version": "d87565731692a6003e43caac4feaed0f69e79272",
        },
        "summation": {
            "name": None,
            "version": None,
        },
        "use_adjuster": False,
        # Since no summation model the sum of GSPs is already calculated
        "save_gsp_sum": False,
        "verbose": True,
    },
}

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
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
        num_workers (int): Number of workers to use to load batches of data. When set to default
            value of -1, it will use one less than the number of CPU cores workers.
    """

    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    if num_workers > 0:
        # Without this line the dataloader will hang if multiple workers are used
        dask.config.set(scheduler="single-threaded")

    day_ahead_model_used = os.getenv("DAY_AHEAD_MODEL", "false").lower() == "true"

    if day_ahead_model_used:
        logger.info(f"Using day ahead PVNet model")

    logger.info(f"Using `pvnet` library version: {pvnet.__version__}")
    logger.info(f"Using `pvnet_app` library version: {pvnet_app.__version__}")
    logger.info(f"Using {num_workers} workers")

    if day_ahead_model_used:
        logger.info(f"Using adjduster: {day_ahead_model_dict['pvnet_day_ahead']['use_adjuster']}")
        logger.info(f"Saving GSP sum: {day_ahead_model_dict['pvnet_day_ahead']['save_gsp_sum']}")

    else:
        logger.info(f"Using adjduster: {models_dict['pvnet_v2']['use_adjuster']}")
        logger.info(f"Saving GSP sum: {models_dict['pvnet_v2']['save_gsp_sum']}")

    # Used for temporarily storing things
    temp_dir = tempfile.TemporaryDirectory()

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
    db_connection = DatabaseConnection(url=os.getenv("DB_URL"), base=Base_Forecast, echo=False)
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

    # Preprocess the satellite data and record the delay of the most recent non-nan timestep
    sat_delay_mins = preprocess_sat_data(t0)

    # Download NWP data
    logger.info("Downloading NWP data")
    download_all_nwp_data()

    # Preprocess the NWP data
    preprocess_nwp_data()

    # ---------------------------------------------------------------------------
    # 2. Set up models

    if day_ahead_model_used:
        model_to_run_dict = {"pvnet_day_ahead": day_ahead_model_dict["pvnet_day_ahead"]}
    # Remove extra models if not configured to run them
    elif os.getenv("RUN_EXTRA_MODELS", "false").lower() == "false":
        model_to_run_dict = {"pvnet_v2": models_dict["pvnet_v2"]}
    else:
        model_to_run_dict = models_dict

    # Prepare all the models which can be run
    forecast_compilers = {}
    data_config_filenames = []
    for model_name, model_config in model_to_run_dict.items():
        # First load the data config
        data_config_filename = PVNetBaseModel.get_data_config(
            model_config["pvnet"]["name"],
            revision=model_config["pvnet"]["version"],
        )

        # Check if the data available will allow the model to run
        model_can_run = check_model_inputs_available(data_config_filename, sat_delay_mins)

        if model_can_run:
            # Set up a forecast compiler for the model
            forecast_compilers[model_name] = ForecastCompiler(
                model_name=model_config["pvnet"]["name"],
                model_version=model_config["pvnet"]["version"],
                summation_name=model_config["summation"]["name"],
                summation_version=model_config["summation"]["version"],
                device=device,
                t0=t0,
                gsp_capacities=gsp_capacities,
                national_capacity=national_capacity,
                verbose=model_config["verbose"],
            )

            # Store the config filename so we can create batches suitable for all models
            data_config_filenames.append(data_config_filename)
        else:
            warnings.warn(f"The model {model_name} cannot be run with input data available")

    if len(forecast_compilers) == 0:
        raise Exception(
            f"No models were compatible with the available input data. Sat delay {sat_delay_mins} mins"
        )

    # Find the config with satellite delay suitable for all models running
    common_config = find_min_satellite_delay_config(data_config_filenames)

    # Save the commmon config
    common_config_path = f"{temp_dir.name}/common_config_path.yaml"
    save_yaml_config(common_config, common_config_path)

    # ---------------------------------------------------------------------------
    # Set up data loader
    logger.info("Creating DataLoader")

    # Populate the data config with production data paths
    populated_data_config_filename = f"{temp_dir.name}/data_config.yaml"

    populate_data_config_sources(common_config_path, populated_data_config_filename)

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
    # Make predictions
    logger.info("Processing batches")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting for batch: {i}")

            for forecast_compiler in forecast_compilers.values():
                # need to do copy the batch for each model, as a model might change the batch
                device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
                forecast_compiler.predict_batch(device_batch)

    # ---------------------------------------------------------------------------
    # Merge batch results to xarray DataArray
    logger.info("Processing raw predictions to DataArray")

    for forecast_compiler in forecast_compilers.values():
        forecast_compiler.compile_forecasts()

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        temp_dir.cleanup()
        return forecast_compilers["pvnet_v2"].da_abs_all

    # ---------------------------------------------------------------------------
    # Write predictions to database
    logger.info("Writing to database")

    with db_connection.get_session() as session:
        for model_name, forecast_compiler in forecast_compilers.items():
            sql_forecasts = convert_dataarray_to_forecasts(
                forecast_compiler.da_abs_all,
                session,
                model_name=model_name,
                version=pvnet_app.__version__,
            )
            save_sql_forecasts(
                forecasts=sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=True,
                apply_adjuster=model_to_run_dict[model_name]["use_adjuster"],
            )

            if model_to_run_dict[model_name]["save_gsp_sum"]:
                # Compute the sum if we are logging the sume of GSPs independently
                da_abs_sum_gsps = (
                    forecast_compiler.da_abs_all.sel(gsp_id=slice(1, 317))
                    .sum(dim="gsp_id")
                    # Only select the central forecast for the GSP sum. The sums of different p-levels
                    # are not a meaningful qauntities
                    .sel(output_label=["forecast_mw"])
                    .expand_dims(dim="gsp_id", axis=0)
                    .assign_coords(gsp_id=[0])
                )

                # Save the sum of GSPs independently - mainly for summation model monitoring
                sql_forecasts = convert_dataarray_to_forecasts(
                    da_abs_sum_gsps,
                    session,
                    model_name=f"{model_name}_gsp_sum",
                    version=pvnet_app.__version__,
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
