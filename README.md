# Image Enhancement

## Setup

### Using Docker

```sh
# prerequisites: docker-compose should be installed
.docker/compose.cu11 up -d
```

### Install manually

```sh
# prerequisites: conda should be installed
conda env create -f .docker/environments.cu11.yml
conda activate imtools
conda develop .
```

## Build CUDA extensions

```bash
# prerequisites: CUDA toolkit should be installed
find /usr/local -maxdepth 1 -name 'cuda'
# change environment variables
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
# check nvcc is available
nvcc --version
# compile sources
cd extensions
python setup.py install
```
