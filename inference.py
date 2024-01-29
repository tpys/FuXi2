import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from copy import deepcopy
from data_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help="FuXi onnx model dir")
parser.add_argument('--input', type=str, required=True, help="The input data file, store in netcdf format")
parser.add_argument('--device', type=str, default="cuda", help="The device to run FuXi model")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--total_step', type=int, default=40)
args = parser.parse_args()

stages = ['short', 'medium']


def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)

    if save_name.endswith("nc"):
        obj = ds.to_netcdf(save_name, compute=False)
    elif save_name.endswith("zarr"):
        obj = ds.to_zarr(save_name, compute=False)

    with ProgressBar():
        obj.compute()


def save_like(output, input, lead_time, save_dir=""):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        init_time = pd.to_datetime(input.time.values[-1])

        ds = xr.DataArray(
            data=output,
            dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                channel=input.channel,
                lat=input.lat.values,
                lon=input.lon.values,
            )
        ).astype(np.float32)
        print_dataarray(ds)
        save_name = os.path.join(save_dir, f'{lead_time:03d}.nc')
        save_with_progress(ds, save_name)


def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t*freq) for t in [i-1, i, i+1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year/366, p.hour/24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)



def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption

    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = 24
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,  
        sess_options=options, 
        providers=providers
    )
    return session


def run_inference(models, input, total_step, save_dir=""):
    hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(hours=6)

    lat = input.lat.values 
    lon = input.lon.values 
    assert lat[0] == 90 and lat[-1] == -90
    batch = input.values[None]
    print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')
    print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")

    print(f'Inference ...')
    start = time.perf_counter()
    for step in range(total_step):
        lead_time = (step + 1) * 6
        valid_time = init_time + pd.Timedelta(hours=step * 6)

        stage = stages[min(len(models)-1, step // 20)]
        model = models[stage]

        input_names = [x.name for x in model.get_inputs()]
        inputs = {'input': batch}        
        
        if "step" in input_names:
            inputs['step'] = np.array([step], dtype=np.float32)

        if "hour" in input_names:
            hour = valid_time.hour/24 
            inputs['hour'] = np.array([hour], dtype=np.float32)

        t0 = time.perf_counter()
        new_input, = model.run(None, inputs)
        output = deepcopy(new_input[:, -1:])
        step_time = time.perf_counter() - t0
        print(f"stage: {stage}, lead_time: {lead_time:03d} h, step_time: {step_time:.3f} sec")

        save_like(output, input, lead_time, save_dir)
        batch = new_input

    run_time = time.perf_counter() - start
    print(f'Inference done take {run_time:.2f}')


if __name__ == "__main__":
    if os.path.exists(args.input):
        input = xr.open_dataarray(args.input)
    else:
        input = make_sample("sample/input")
        input.to_netcdf("sample/input.nc")
        print_dataarray(input, "input")

    models = {}
    for stage in stages:
        model_path = os.path.join(args.model_dir, f"{stage}.onnx")
        if os.path.exists(model_path):
            start = time.perf_counter()
            print(f'Load FuXi {stage} ...')       
            model = load_model(model_path, args.device)            
            models[stage] = model
            print(f'Load FuXi {stage} take {time.perf_counter() - start:.2f} sec')
            
    run_inference(models, input, args.total_step, save_dir=args.save_dir)
