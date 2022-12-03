# R builds
FROM r-base:4.2.2

# PRETTY_NAME="Debian GNU/Linux bookworm/sid"
# NAME="Debian GNU/Linux"
# VERSION_CODENAME=bookworm
# ID=debian
# HOME_URL="https://www.debian.org/"
# SUPPORT_URL="https://www.debian.org/support"
# BUG_REPORT_URL="https://bugs.debian.org/"
# root@c114db8c9ccf:/app# 


RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    gdal-bin \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libgdal-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    libpng-dev \
    libproj-dev \
    libssh2-1-dev \
    libssl-dev \
    libtiff5-dev \
    libudunits2-dev \
    libxml2-dev \
    proj-bin \
    zlib1g-dev

RUN R -e "install.packages(c('devtools', 'plyr', 'tidyverse', 'raster', 'celestial', 'caret', 'fastICA', 'SOAR', 'RStoolbox', 'jsonlite', 'data.table', 'spdep'))"
RUN R -e "devtools::install_github('OpenDroneMap/FIELDimageR')"

# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# RUN apt-get update && apt-get install -y software-properties-common
# RUN apt-get update && apt-get install -y && \
#     pip install --upgrade pip
    
# COPY requirements.txt /tmp/requirements-docker.txt

# ENV INPUT_DATA="./data/input"
# ENV OUTPUT_DATA="./data/output"

# RUN pip install -r /tmp/requirements-docker.txt && \
#     rm /tmp/requirements-docker.txt

RUN mkdir -p /app
WORKDIR /app

COPY . .

CMD [ "bash", "run_model.sh" ]
