""" A pydantic model for the ML models"""
import os
import logging

from typing import List, Optional

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class ModelHF(BaseModel):
    repo: str = Field(..., title="Repo name", description="The HF Repo")
    version: str = Field(..., title="Repo version", description="The HF version")


class Model(BaseModel):
    """One ML Model"""

    name: str = Field(..., title="Model Name", description="The name of the model")
    pvnet: ModelHF = Field(..., title="PVNet", description="The PVNet model")
    summation: ModelHF = Field(..., title="Summation", description="The Summation model")

    use_adjuster: Optional[bool] = Field(
        False, title="Use Adjuster", description="Whether to use the adjuster model"
    )
    save_gsp_sum: Optional[bool] = Field(
        False, title="Save GSP Sum", description="Whether to save the GSP sum"
    )
    verbose: Optional[bool] = Field(
        False, title="Verbose", description="Whether to print verbose output"
    )
    save_gsp_to_recent: Optional[bool] = Field(
        False,
        title="Save GSP to Forecast Value Last Seven Days",
        description="Whether to save the GSP to Forecast Value Last Seven Days",
    )
    day_ahead: Optional[bool] = Field(
        False, title="Day Ahead", description="If this model is day ahead or not"
    )

    ecmwf_only: Optional[bool] = Field(
        False, title="ECMWF ONly", description="If this model is only using ecmwf data"
    )

    uses_satellite_data:    Optional[bool] = Field(
        True, title="Uses Satellite Data", description="If this model uses satellite data"
    )


class Models(BaseModel):
    """A group of ml models"""

    models: List[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast"
    )

    @field_validator("models")
    @classmethod
    def name_must_be_unique(cls, v: List[Model]) -> List[Model]:
        """Ensure that all model names are unique"""
        names = [model.name for model in v]
        unique_names = set(names)

        if len(names) != len(unique_names):
            raise Exception(f"Model names must be unique, names are {names}")
        return v


def get_all_models(
    get_ecmwf_only: Optional[bool] = False,
    get_day_ahead_only: Optional[bool] = False,
    run_extra_models: Optional[bool] = False,
) -> List[Model]:
    """
    Returns all the models for a given client

    Args:
        get_ecmwf_only: If only the ECMWF model should be returned
        get_day_ahead_only: If only the day ahead model should be returned
        run_extra_models: If extra models should be run
    """

    # load models from yaml file
    filename = os.path.dirname(os.path.abspath(__file__)) + "/all_models.yaml"

    with fsspec.open(filename, mode="r") as stream:
        models = parse_config(data=stream)
        models = Models(**models)

    models = config_pvnet_v2_model(models)

    if get_ecmwf_only:
        log.info("Using ECMWF model only")
        models.models = [model for model in models.models if model.ecmwf_only]

    elif get_day_ahead_only:
        log.info("Using Day Ahead model only")
        models.models = [model for model in models.models if model.day_ahead]

    elif not run_extra_models:
        log.info("Not running extra models")
        models.models = [model for model in models.models if model.name == "pvnet_v2"]

    return models.models


def config_pvnet_v2_model(models):
    """Function to adjust pvnet model"""
    # special case for environment variables
    use_adjuster = os.getenv("USE_ADJUSTER", "true").lower() == "true"
    save_gsp_sum = os.getenv("SAVE_GSP_SUM", "false").lower() == "true"
    # find index where name=pvnet_v2
    pvnet_v2_index = 0
    models.models[pvnet_v2_index].use_adjuster = use_adjuster
    models.models[pvnet_v2_index].save_gsp_sum = save_gsp_sum

    return models
