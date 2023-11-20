"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_ZARR_PATH
    - SATELLITE_ZARR_PATH
"""

import logging
import os
import yaml
import tempfile
import warnings
from datetime import datetime, timedelta, timezone

import fsspec
import numpy as np
import pandas as pd
import torch
import typer
import xarray as xr
import xesmf as xe
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import (
    ForecastSQL,
    ForecastValue,
)
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
    get_model,
)
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from ocf_datapipes.load import OpenGSPFromDatabase
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.transform.numpy.batch.sun_position import ELEVATION_MEAN, ELEVATION_STD
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.utils import stack_np_examples_into_batch
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from sqlalchemy.orm import Session
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper

import pvnet
from pvnet.data.datamodule import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.utils import GSPLocationLookup

import pvnet_app

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# TODO: Host data config alongside model?
this_dir = os.path.dirname(os.path.abspath(__file__))

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
default_model_version = "805ca9b2ee3120592b0b70b7c75a454e2b4e4bec"

# Huggingfacehub model repo and commit for PVNet summation (GSP sum to national model)
# If summation_model_name is set to None, a simple sum is computed instead
default_summation_model_name = "openclimatefix/pvnet_v2_summation"
default_summation_model_version = "01393d6e4a036103f9c7111cba6f03d5c19beb54"

model_name_ocf_db = "pvnet_v2"
use_adjuster = os.getenv("USE_ADJUSTER", "True").lower() == "true"

# If environmental variable is true, the sum-of-GSPs will be computed and saved under a different
# model name. This can be useful to compare against the summation model and therefore monitor its
# performance in production
save_gsp_sum = os.getenv("SAVE_GSP_SUM", "False").lower() == "true"
gsp_sum_model_name_ocf_db = "pvnet_gsp_sum"

# ---------------------------------------------------------------------------
# LOGGER
formatter = logging.Formatter(
    fmt="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, os.getenv("LOGLEVEL", "INFO")))
logger.addHandler(stream_handler)

# Get rid of these verbose logs
sql_logger = logging.getLogger("sqlalchemy.engine.Engine")
sql_logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS

def regrid_nwp_data(nwp_path):
    """This function loads the NWP data, then regrids and saves it back out if the data is not on
    the same grid as expected. The data is resaved in-place.
    """
    ds_raw = xr.open_zarr(nwp_path)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(f"{this_dir}/../data/nwp_target_coords.nc")
    
    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude) and
         ds_raw.longitude.equals(ds_target_coords.longitude)
        
    )
    
    if not needs_regridding:
        logger.info("No NWP regridding required - skipping this step")
        return
    
    logger.info("Regridding NWP to expected grid")
    
    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()
    
    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regridder = xe.Regridder(ds_raw, ds_target_coords, method="bilinear")
    ds_regridded = regridder(
        ds_raw.chunk(dict(x=-1, y=-1, step=1))
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    os.system(f"rm -fr {nwp_path}")
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)
    ds_regridded.chunk(dict(step=12, x=100, y=100)).to_zarr(nwp_path)
    
    return


def sat_data_qualtiy_control(sat_path):
    """This function loads the satellite data, removes timestamps which don't pass some quality
    checks and saves it back out. The data is resaved in-place.
    """
    # Need to set this high enough not to catch the zeros in the 2 visible spectrum channels at 
    # night which is valid. 2 out of a total of 11 channels + 10% of 9 others will trigger
    zero_frac_limit = (2 + 0.1*9)/11
    
    # Pull the raw data into RAM
    ds = xr.open_zarr(sat_path).compute()
    
    # Check fraction of zeros at each time step
    frac_zeros = (ds.data==0).mean(dim=("y_geostationary", "x_geostationary", "variable"))
    
    logger.info(
        f"Found zeros fractions in each timestamp:\n"
        f"{frac_zeros.to_dataframe().rename({'data':'zero_fraction'}, axis=1)}"
    )
    
    if (frac_zeros<=zero_frac_limit).all():
        logger.info("No sat quality issues - skipping this step")
    
    else:
        bad_timestamp_mask = frac_zeros>zero_frac_limit
        bad_timestamps = frac_zeros.where(bad_timestamp_mask, drop=True).time.values
        logger.info(f"Removing timestamps: {bad_timestamps}")
        ds = ds.where(~bad_timestamp_mask, drop=True)
        os.system(f"rm -fr {sat_path}")
        ds.to_zarr(sat_path)
    
    

def populate_data_config_sources(input_path, output_path):
    """Resave the data config and replace the source filepaths

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
    """
    with open(input_path) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
        
    production_paths = {
        "gsp": os.environ["DB_URL"],
        "nwp": "nwp.zarr",
        "satellite": "sat.zarr.zip",
        # TODO: include hrvsatellite
    }        
    
    # Replace data sources
    for source in ["gsp", "nwp", "satellite", "hrvsatellite"]:
        if source in config["input_data"]:
            # If not empty - i.e. if used
            if config["input_data"][source][f"{source}_zarr_path"]!="":
                assert source in production_paths, f"Missing production path: {source}"
                config["input_data"][source][f"{source}_zarr_path"] = production_paths[source]

    # We do not need to set PV path right now. This currently done through datapipes
    # TODO - Move the PV path to here
    
    with open(output_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def convert_dataarray_to_forecasts(
    forecast_values_dataarray: xr.DataArray, session: Session, model_name: str, version: str
) -> list[ForecastSQL]:
    """
    Make a ForecastSQL object from a DataArray.

    Args:
        forecast_values_dataarray: Dataarray of forecasted values. Must have `target_datetime_utc`
            `gsp_id`, and `output_label` coords. The `output_label` coords must have `"forecast_mw"`
            as an element.
        session: database session
        model_name: the name of the model
        version: the version of the model
    Return:
        List of ForecastSQL objects
    """
    logger.debug("Converting DataArray to list of ForecastSQL")

    assert "target_datetime_utc" in forecast_values_dataarray.coords
    assert "gsp_id" in forecast_values_dataarray.coords
    assert "forecast_mw" in forecast_values_dataarray.output_label

    # get last input data
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    # get model name
    model = get_model(name=model_name, version=version, session=session)

    forecasts = []

    for gsp_id in forecast_values_dataarray.gsp_id.values:
        gsp_id = int(gsp_id)
        # make forecast values
        forecast_values = []

        # get location
        location = get_location(session=session, gsp_id=gsp_id)

        gsp_forecast_values_da = forecast_values_dataarray.sel(gsp_id=gsp_id)

        for target_time in pd.to_datetime(gsp_forecast_values_da.target_datetime_utc.values):
            # add timezone
            target_time_utc = target_time.replace(tzinfo=timezone.utc)
            this_da = gsp_forecast_values_da.sel(target_datetime_utc=target_time)

            forecast_value_sql = ForecastValue(
                target_time=target_time_utc,
                expected_power_generation_megawatts=(
                    this_da.sel(output_label="forecast_mw").item()
                ),
            ).to_orm()

            forecast_value_sql.adjust_mw = 0.0

            properties = {}

            if "forecast_mw_plevel_10" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_10").item()
                # `val` can be NaN if PVNet has probabilistic outputs and PVNet_summation doesn't,
                # or if PVNet_summation has probabilistic outputs and PVNet doesn't.
                # Do not log the value if NaN
                if not np.isnan(val):
                    properties["10"] = val

            if "forecast_mw_plevel_90" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_90").item()

                if not np.isnan(val):
                    properties["90"] = val
                    
            if len(properties)>0:
                forecast_value_sql.properties = properties

            forecast_values.append(forecast_value_sql)

        # make forecast object
        forecast = ForecastSQL(
            model=model,
            forecast_creation_time=datetime.now(tz=timezone.utc),
            location=location,
            input_data_last_updated=input_data_last_updated,
            forecast_values=forecast_values,
            historic=False,
        )

        forecasts.append(forecast)

    return forecasts


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
        - NWP_ZARR_PATH
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

    # ------------ GSP
    
    logger.info("Loading GSP metadata")
    ds_gsp = next(iter(OpenGSPFromDatabase()))
    
    # Get capacities from the database
    url = os.getenv("DB_URL")
    db_connection = DatabaseConnection(url=url, base=Base_Forecast)
    with db_connection.get_session() as session:
        #  Pandas series of most recent GSP capacities
        gsp_capacities = get_latest_gsp_capacities(session, gsp_ids)
        
        # National capacity is needed if using summation model
        national_capacity = get_latest_gsp_capacities(session, [0])[0]

    # Set up ID location query object
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)
    
    # ------------ SATELLITE
    
    # Download satellite data
    logger.info("Downloading zipped satellite data")
    fs = fsspec.open(os.environ["SATELLITE_ZARR_PATH"]).fs
    fs.get(os.environ["SATELLITE_ZARR_PATH"], "sat.zarr.zip")
    
    # Satellite data quality control step
    sat_data_qualtiy_control("sat.zarr.zip")

    # Also download 15-minute satellite if it exists
    sat_latest_15 = os.environ["SATELLITE_ZARR_PATH"].replace(".zarr.zip", "_15.zarr.zip")
    if fs.exists(sat_latest_15):
        logger.info("Downloading 15-minute satellite data")
        fs.get(sat_latest_15, "sat_15.zarr.zip")
        
        # 15-min satellite data quality control step
        sat_data_qualtiy_control("sat_15.zarr.zip")
    
    # ------------ NWP
    
    # Download NWP data
    logger.info("Downloading nwp data")
    fs = fsspec.open(os.environ["NWP_ZARR_PATH"]).fs
    fs.get(os.environ["NWP_ZARR_PATH"], "nwp.zarr", recursive=True)
    
    # Regrid the NWP data if needed
    regrid_nwp_data("nwp.zarr")
    
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
    rs = MultiProcessingReadingService(
        num_workers=num_workers,
        multiprocessing_context="spawn",
        worker_prefetch_cnt=0 if num_workers == 0 else 2,
    )
    dataloader = DataLoader2(batch_datapipe, reading_service=rs)

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
            sun_down_mask = elevation < MIN_DAY_ELEVATION

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

    # Reorder GSP order which ends up shuffled if multiprocessing is used
    inds = gsp_ids_all_batches.argsort()

    normed_preds = normed_preds[inds]
    sun_down_masks = sun_down_masks[inds]
    gsp_ids_all_batches = gsp_ids_all_batches[inds]

    logger.info(f"{gsp_ids_all_batches.shape}")

    # ---------------------------------------------------------------------------
    # 5. Merge batch results to xarray DataArray
    logger.info("Processing raw predictions to DataArray")

    n_times = normed_preds.shape[1]

    if model.use_quantile_regression:
        output_labels = model.output_quantiles
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        normed_preds = normed_preds[..., np.newaxis]

    da_normed = xr.DataArray(
        data=normed_preds,
        dims=["gsp_id", "target_datetime_utc", "output_label"],
        coords=dict(
            gsp_id=gsp_ids_all_batches,
            target_datetime_utc=pd.to_datetime(
                [t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)],
            ),
            output_label=output_labels,
        ),
    )

    da_sundown_mask = xr.DataArray(
        data=sun_down_masks,
        dims=["gsp_id", "target_datetime_utc"],
        coords=dict(
            gsp_id=gsp_ids_all_batches,
            target_datetime_utc=pd.to_datetime(
                [t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)],
            ),
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
        if summation_model.use_quantile_regression:
            sum_output_labels = summation_model.output_quantiles
            sum_output_labels = [
                f"forecast_mw_plevel_{int(q*100):02}" for q in summation_model.output_quantiles
            ]
            sum_output_labels[sum_output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
        else:
            sum_output_labels = ["forecast_mw"]

        da_normed_national = xr.DataArray(
            data=normed_national[np.newaxis],
            dims=["gsp_id", "target_datetime_utc", "output_label"],
            coords=dict(
                gsp_id=[0],
                target_datetime_utc=da_abs.target_datetime_utc,
                output_label=sum_output_labels,
            ),
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
            f"National forecast is {da_abs.sel(gsp_id=0, output_label='forecast_mw').values}"
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