"""A script to download default models from huggingface.

Downloading these model files in the build means we do not need to download them each time the app
is run.
"""

import typer
# This is needed other we get an error. # TODO find out why
import pvnet_app.app # noqa: F401

from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.app import models_dict


def main():
    """Download model from Huggingface and save it to cache."""
    # Model will be downloaded and saved to cache on disk
    PVNetBaseModel.from_pretrained(
        models_dict["pvnet_v2"]["pvnet"]["name"],
        revision=models_dict["pvnet_v2"]["pvnet"]["version"],
    )

    # Model will be downloaded and saved to cache on disk
    SummationBaseModel.from_pretrained(
        models_dict["pvnet_v2"]["summation"]["name"],
        revision=models_dict["pvnet_v2"]["summation"]["version"],
    )


if __name__ == "__main__":
    typer.run(main)
