FROM continuumio/miniconda3

ARG TESTING=0

SHELL ["/bin/bash", "-l", "-c"]

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install unzip g++ gcc libgeos++-dev libproj-dev proj-data proj-bin -y

# Copy files
COPY setup.py app/setup.py
COPY README.md app/README.md
COPY requirements.txt app/requirements.txt
COPY pvnet_app/ app/pvnet_app/
COPY tests/ app/tests/
COPY scripts/ app/scripts/
COPY data/ app/data/

# Install requirements
RUN conda install python=3.10
RUN conda install -c conda-forge xesmf esmpy h5py -y
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install git+https://github.com/SheffieldSolar/PV_Live-API#pvlive_api

# Install CPU torch
RUN pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# Change to app folder
WORKDIR /app

# Install library
RUN pip install -e .

# Download models so app can used cached versions instead of pulling from huggingface
RUN python scripts/cache_default_models.py

RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

CMD ["python", "-u","pvnet_app/app.py"]
