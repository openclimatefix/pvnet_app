from datetime import timezone, datetime
import warnings
import logging
import torch

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD

import pvnet
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from sqlalchemy.orm import Session

from nowcasting_datamodel.models import ForecastSQL, ForecastValue
from nowcasting_datamodel.read.read import get_latest_input_data_last_updated, get_location
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save as save_sql_forecasts

import pvnet_app

logger = logging.getLogger(__name__)

# If the solar elevation (in degrees) is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0


_model_mismatch_msg = (
    "The PVNet version running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet version {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)


class ForecastCompiler:
    """Class for making and compiling solar forecasts from for all GB GSPsn and national total"""
    def __init__(
        self, 
        model_tag: str,
        model_name: str, 
        model_version: str, 
        summation_name: str | None, 
        summation_version: str | None,
        device: torch.device, 
        t0: pd.Timestamp, 
        gsp_capacities: xr.DataArray, 
        national_capacity: float, 
        apply_adjuster: bool,
        save_gsp_sum: bool,
        save_gsp_to_recent: bool,
        verbose: bool = False,
        use_legacy: bool = False,
    ):
        """Class for making and compiling solar forecasts from for all GB GSPsn and national total
        
        Args:
            model_tag: The name the model results will be saved to the database under
            model_name: Name of the huggingface repo where the PVNet model is stored
            model_version: Version of the PVNet model to run within the huggingface repo
            summation_name: Name of the huggingface repo where the summation model is stored
            summation_version: Version of the summation model to run within the huggingface repo
            device: Device to run the model on
            t0: The t0 time used to compile the results to numpy array
            gsp_capacities: DataArray of the solar capacities for all regional GSPs at t0
            national_capacity: The national solar capacity at t0
            apply_adjuster: Whether to apply the adjuster when saving to database
            save_gsp_sum: Whether to save the GSP sum
            save_gsp_to_recent: Whether to save the GSP results to the 
                forecast_value_last_seven_days table
            verbose: Whether to log all messages throughout prediction and compilation
            legacy: Whether to run legacy dataloader
        """
        
        logger.info(f"Loading model: {model_name} - {model_version}")
        
        
        # Store settings
        self.model_tag = model_tag
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.gsp_capacities = gsp_capacities
        self.national_capacity = national_capacity
        self.apply_adjuster = apply_adjuster
        self.save_gsp_sum = save_gsp_sum
        self.save_gsp_to_recent = save_gsp_to_recent
        self.verbose = verbose
        self.use_legacy = use_legacy
        
        # Create stores for the predictions
        self.normed_preds = []
        self.gsp_ids_each_batch = []
        self.sun_down_masks = []
        
        # Load the GSP and summation models
        self.model, self.summation_model = self.load_model(
            model_name, 
            model_version, 
            summation_name, 
            summation_version,
            device,
        )
        
        # These are the valid times this forecast will predict for
        self.valid_times = (
            t0 + pd.timedelta_range(start='30min', freq='30min', periods=self.model.forecast_len)
        )
        
    @staticmethod
    def load_model(
        model_name: str, 
        model_version: str, 
        summation_name: str | None, 
        summation_version: str | None,
        device: torch.device, 
    ):
        """Load the GSP and summation models"""

        # Load the GSP level model
        model = PVNetBaseModel.from_pretrained(
            model_id=model_name,
            revision=model_version,
        ).to(device)
        
        # Load the summation model
        if summation_name is None:
            sum_model = None
        else:
            sum_model = SummationBaseModel.from_pretrained(
                model_id=summation_name,
                revision=summation_version,
            ).to(device)
            
            # Compare the current GSP model with the one the summation model was trained on
            this_gsp_model = (model_name, model_version)
            sum_expected_gsp_model = (sum_model.pvnet_model_name, sum_model.pvnet_model_version)
            
            if sum_expected_gsp_model!=this_gsp_model:
                warnings.warn(_model_mismatch_msg.format(*this_gsp_model, *sum_expected_gsp_model))
        
        return model, sum_model
        
    
    def log_info(self, message: str) -> None:
        """Maybe log message depending on verbosity"""
        if self.verbose:
            logger.info(message)
    
    
    def predict_batch(self, batch: NumpyBatch) -> None:
        """Make predictions for a batch and store results internally"""
        
        self.log_info(f"Predicting for model: {self.model_name}-{self.model_version}")
        # Store GSP IDs for this batch for reordering later
        these_gsp_ids = batch[BatchKey.gsp_id].cpu().numpy()
        self.gsp_ids_each_batch += [these_gsp_ids]
        
        # TODO: This change should be moved inside PVNet
        batch[BatchKey.gsp_id] = batch[BatchKey.gsp_id].unsqueeze(1)

        # Run batch through model
        preds = self.model(batch).detach().cpu().numpy()

        # Calculate unnormalised elevation and sun-dowm mask
        self.log_info("Computing sundown mask")
        if self.use_legacy:
            # The old dataloader standardises the data
            elevation = (
                batch[BatchKey.gsp_solar_elevation].cpu().numpy() * ELEVATION_STD + ELEVATION_MEAN
            )
        else:
            # The new dataloader normalises the data to [0, 1]
            elevation = (batch[BatchKey.gsp_solar_elevation].cpu().numpy() - 0.5) * 180
        
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, -preds.shape[1] :]
        sun_down_mask = elevation < MIN_DAY_ELEVATION

        # Store predictions internally
        self.normed_preds += [preds]
        self.sun_down_masks += [sun_down_mask]

        # Log max prediction
        self.log_info(f"GSP IDs: {these_gsp_ids}")
        self.log_info(f"Max prediction: {np.max(preds, axis=1)}")
        
    
    def compile_forecasts(self) -> None:
        """Compile all forecasts internally in a single DataArray
        
        Steps:
        - Compile all the GSP level forecasts
        - Make national forecast
        - Compile all forecasts into a DataArray stored inside the object as `da_abs_all`
        """
        
        # Complie results from all batches
        normed_preds = np.concatenate(self.normed_preds)
        sun_down_masks = np.concatenate(self.sun_down_masks)
        gsp_ids_all_batches = np.concatenate(self.gsp_ids_each_batch).squeeze()
        
        # Reorder GSPs which can end up shuffled if multiprocessing is used
        inds = gsp_ids_all_batches.argsort()

        normed_preds = normed_preds[inds]
        sun_down_masks = sun_down_masks[inds]
        gsp_ids_all_batches = gsp_ids_all_batches[inds]
        
        # Merge batch results to xarray DataArray
        da_normed = self.preds_to_dataarray(
            normed_preds, 
            self.model.output_quantiles, 
            gsp_ids_all_batches
        )
        
        da_sundown_mask = xr.DataArray(
            data=sun_down_masks,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=gsp_ids_all_batches,
                target_datetime_utc=self.valid_times,
            ),
        )

        # Multiply normalised forecasts by capacities and clip negatives
        self.log_info(f"Converting to absolute MW using {self.gsp_capacities}")
        da_abs = da_normed.clip(0, None) * self.gsp_capacities.values[:, None, None]
        max_preds = da_abs.sel(output_label="forecast_mw").max(dim="target_datetime_utc")
        self.log_info(f"Maximum predictions: {max_preds}")

        # Apply sundown mask
        da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)
        
        if self.summation_model is None:
            self.log_info("Summing across GSPs to produce national forecast")
            da_abs_national = (
                da_abs.sum(dim="gsp_id").expand_dims(dim="gsp_id", axis=0).assign_coords(gsp_id=[0])
            )
        else:
            self.log_info("Using summation model to produce national forecast")

            # Make national predictions using summation model
            inputs = {
                "pvnet_outputs": torch.Tensor(normed_preds[np.newaxis]).to(self.device),
                "effective_capacity": (
                    torch.Tensor(self.gsp_capacities.values / self.national_capacity)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                ),
            }
            normed_national = self.summation_model(inputs).detach().squeeze().cpu().numpy()

            # Convert national predictions to DataArray
            da_normed_national = self.preds_to_dataarray(
                normed_national[np.newaxis], 
                self.summation_model.output_quantiles, 
                gsp_ids=[0],
            )

            # Multiply normalised forecasts by capacities and clip negatives
            da_abs_national = da_normed_national.clip(0, None) * self.national_capacity

            # Apply sundown mask - All GSPs must be masked to mask national
            da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        self.log_info(
            f"National forecast is {da_abs_national.sel(output_label='forecast_mw').values}"
        )
        
        # Store the compiled predictions internally
        self.da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")


    def preds_to_dataarray(
        self, 
        preds: np.ndarray, 
        output_quantiles: list[float] | None, 
        gsp_ids: list[int],
    ) -> xr.DataArray:
        """Put numpy array of predictions into a dataarray"""

        if output_quantiles is not None:
            output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in output_quantiles]
            output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
        else:
            output_labels = ["forecast_mw"]
            preds = preds[..., np.newaxis]

        da = xr.DataArray(
            data=preds,
            dims=["gsp_id", "target_datetime_utc", "output_label"],
            coords=dict(
                gsp_id=gsp_ids,
                target_datetime_utc=self.valid_times,
                output_label=output_labels,
            ),
        )
        return da
    
        
    def log_forecast_to_database(self, session: Session) -> None:
        """Log the compiled forecast to the database"""
        
        self.log_info("Converting DataArray to list of ForecastSQL")

        sql_forecasts = self.convert_dataarray_to_forecasts(
            self.da_abs_all,
            session,
            model_tag=self.model_tag,
            version=pvnet_app.__version__,
        )
        
        self.log_info("Saving ForecastSQL to database")

        if self.save_gsp_to_recent:
            
            # Save all forecasts and save to last_seven_days table
            save_sql_forecasts(
                forecasts=sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=True,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=True,
            )
        else:
            # Save national and save to last_seven_days table
            save_sql_forecasts(
                forecasts=sql_forecasts[0:1],
                session=session,
                update_national=True,
                update_gsp=False,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=True,
            )
            
            # Save GSP results but not to last_seven_dats table
            save_sql_forecasts(
                forecasts=sql_forecasts[1:],
                session=session,
                update_national=False,
                update_gsp=True,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=False,
            )

        if self.save_gsp_sum:
            # Compute the sum if we are logging the sum of GSPs independently
            da_abs_sum_gsps = (
                self.da_abs_all.sel(gsp_id=slice(1, 317))
                .sum(dim="gsp_id")
                # Only select the central forecast for the GSP sum. The sums of different p-levels
                # are not a meaningful qauntities
                .sel(output_label=["forecast_mw"])
                .expand_dims(dim="gsp_id", axis=0)
                .assign_coords(gsp_id=[0])
            )

            # Save the sum of GSPs independently - mainly for summation model monitoring
            gsp_sum_sql_forecasts = self.convert_dataarray_to_forecasts(
                da_abs_sum_gsps,
                session,
                model_tag=f"{self.model_tag}_gsp_sum",
                version=pvnet_app.__version__,
            )

            save_sql_forecasts(
                forecasts=gsp_sum_sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=False,
                apply_adjuster=False,
                save_to_last_seven_days=True,
            )
            
            

    @staticmethod
    def convert_dataarray_to_forecasts(
        da_preds: xr.DataArray, session: Session, model_tag: str, version: str
    ) -> list[ForecastSQL]:
        """
        Make a ForecastSQL object from a DataArray.

        Args:
            da_preds: DataArray of forecasted values
            session: Database session
            model_key: the name of the model to saved to the database
            version: The version of the model
        Return:
            List of ForecastSQL objects
        """

        assert "target_datetime_utc" in da_preds.coords
        assert "gsp_id" in da_preds.coords
        assert "forecast_mw" in da_preds.output_label

        # get last input data
        input_data_last_updated = get_latest_input_data_last_updated(session=session)

        # get model name
        model = get_model(name=model_tag, version=version, session=session)

        forecasts = []

        for gsp_id in da_preds.gsp_id.values:

            # make forecast values
            forecast_values = []

            location = get_location(session=session, gsp_id=int(gsp_id))

            da_gsp = da_preds.sel(gsp_id=gsp_id)

            for target_time in pd.to_datetime(da_gsp.target_datetime_utc.values):

                da_gsp_time = da_gsp.sel(target_datetime_utc=target_time)

                forecast_value_sql = ForecastValue(
                    target_time=target_time.replace(tzinfo=timezone.utc),
                    expected_power_generation_megawatts=(
                        da_gsp_time.sel(output_label="forecast_mw").item()
                    ),
                ).to_orm()

                properties = {}

                if "forecast_mw_plevel_10" in da_gsp_time.output_label:
                    p10 = da_gsp_time.sel(output_label="forecast_mw_plevel_10").item()
                    # `p10` can be NaN if PVNet has probabilistic outputs and PVNet_summation 
                    # doesn't, or vice versa. Do not log the value if NaN
                    if not np.isnan(p10):
                        properties["10"] = p10

                if "forecast_mw_plevel_90" in da_gsp_time.output_label:
                    p90 = da_gsp_time.sel(output_label="forecast_mw_plevel_90").item()

                    if not np.isnan(p90):
                        properties["90"] = p90

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