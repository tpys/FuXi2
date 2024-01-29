## FuXi2


## Installation
The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── sample
│   │    ├── input
│   │         ├── geopotential.nc
│   │         ├── temperature.nc
│   │         ├── ......
│   │         ├── total_precipitation.nc
|   |
│   ├── model
│   |    ├── short
│   |    ├── short.onnx
│   |    ├── medium
│   |    ├── medium.onnx
|   |   
│   ├── inference.py
│   ├── data_util.py

```

1. Install xarray 

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
```

2. Install pytorch

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Inference with fuxi onnx model

```python 
python inference.py \
    --model_dir model/onnx \
    --input sample/input.nc \
    --total_step 40 \
    --save_dir sample/output;
```


## Input preparation 

### fuxi-c79
The `input.nc` file contains preprocessed data from the origin ERA5 files. It will be generated automatically based on the downloaded data located under the sample/input directory. The file has a shape of (2, 79, 721, 1440), where the first dimension represents two time steps. The second dimension represents all variable and level combinations, which are named in the following exact order:
```python
['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
'v700', 'v850', 'v925', 'v1000', 'q50', 'q100', 'q150', 'q200',
'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850', 'q925',
'q1000', 't2m', 'd2m', 'sst', 'u10m', 'v10m', 'u100m', 'v100m',
'msl', 'sp', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp']
```

The last 14 variables: ('t2m', 'd2m', 'sst', 'u10m', 'v10m', 'u100m', 'v100m',
'msl', 'sp', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp') are surface variables, The remaining variables represent atmosphere variables with numbers denoting pressure levels. The accumulated variables ('ssr', 'ssrd', 'fdir', 'ttr', 'tp') are not needed for input and can be set to zero. Their unit is 6-hourly average.

### fuxi-c92
The c92 `input.nc` is similar to c79, but it includes an additional pressure level variable called specific cloud liquid water content. As a result, the input shape is (2, 92, 721, 1440), and the channel order is as follows:
```python
['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
'v700', 'v850', 'v925', 'v1000', 'q50', 'q100', 'q150', 'q200',
'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850', 'q925',
'q1000', 'clwc50', 'clwc100', 'clwc150', 'clwc200', 'clwc250',
'clwc300', 'clwc400', 'clwc500', 'clwc600', 'clwc700', 'clwc850',
'clwc925', 'clwc1000', 't2m', 'd2m', 'sst', 'u10m', 'v10m',
'u100m', 'v100m', 'msl', 'sp', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp']
```


