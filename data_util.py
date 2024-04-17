import os
import numpy as np
import xarray as xr

__all__ = ["make_sample", "print_dataarray"]

levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


pl_names = dict(
    z='geopotential',
    t='temperature',
    u='u_component_of_wind',
    v='v_component_of_wind',
    q='specific_humidity',
    clwc='specific_cloud_liquid_water_content',
)

sfc_names = dict(
    t2m='2m_temperature',
    d2m='2m_dewpoint_temperature',
    sst='sea_surface_temperature',
    u10m='10m_u_component_of_wind',
    v10m='10m_v_component_of_wind',
    u100m='100m_u_component_of_wind',
    v100m='100m_v_component_of_wind',
    msl='mean_sea_level_pressure',
    sp='surface_pressure',
    lcc='low_cloud_cover',
    mcc='medium_cloud_cover',
    hcc='high_cloud_cover',
    tcc='total_cloud_cover',
    mdts='mean_direction_of_total_swell',
    mdww='mean_direction_of_wind_waves',
    mpts='mean_period_of_total_swell',
    mpww='mean_period_of_wind_waves',
)

accum_names = dict(
    ssr='surface_net_solar_radiation',
    ssrd='surface_solar_radiation_downwards',
    fdir='total_sky_direct_solar_radiation_at_surface',
    ttr='top_net_thermal_radiation',
    tp='total_precipitation',
)

input_names = dict(
    c75=['z','t','u','v','q','t2m','u10m','v10m','msl','sp','ssr','ssrd','fdir','ttr','tp'],
    c79=['z','t','u','v','q','t2m','d2m','sst','u10m','v10m','u100m','v100m','msl','sp','ssr','ssrd','fdir','ttr','tp'],
    c92=['z','t','u','v','q','clwc','t2m','d2m','sst','u10m','v10m','u100m','v100m','msl','sp','ssr','ssrd','fdir','ttr','tp'],
    c88=['z','t','u','v','q','t2m','d2m','sst','u10m','v10m','u100m','v100m','msl', 'lcc', 'mcc', 'hcc', 'tcc', 'mdts', 'mdww', 'mpts', 'mpww', 'ssr','ssrd','fdir','ttr','tp'],
)

def level_to_channel(ds, short_name):
    if "level" not in ds.dims:
        ds = ds.expand_dims({'level': [1]}, axis=1)        
    if len(ds.level) == 1:
        channel = [short_name]
    else:
        channel = [f'{short_name}{lvl}' for lvl in ds.level.data]
    ds.attrs = {}     
    ds = ds.rename({short_name: 'data', 'level': 'channel'})
    ds = ds.assign_coords(channel=channel)  
    return ds


def convert_unit(v, short_name):
    if short_name in ["ciwc", "clwc", "q"]:
        print(f"Convert {short_name} to g/kg")
        v = v * 1000

    if short_name in ['ttr', 'ssr', 'ssrd', 'ssrdc', 'fdir']:
        print(f"Convert {short_name} to wat")
        v = v / 3600

    if short_name == "tp":
        print(f"Convert {short_name} to mm")
        v = v * 1000
        print(f"Apply log transform for {short_name}")
        v = np.log(1 + v.clip(0, 1000))
    return v


def make_sample(data_dir, version="c79"):
    ds = []
    for short_name in input_names[version]:
        if short_name in pl_names:
            file_name = os.path.join(data_dir, f"{pl_names[short_name]}.nc")
            v = xr.open_dataarray(file_name)            
        elif short_name in sfc_names:
            file_name = os.path.join(data_dir, f"{sfc_names[short_name]}.nc")
            v = xr.open_dataarray(file_name)    
        elif short_name in accum_names:
            v = ds[-1] * 0
        v = convert_unit(v, short_name)
        v = level_to_channel(v, short_name)
        print(f"{short_name}: {v.shape}, {v.min():.3f} ~ {v.max():.3f}")
        v.name = "data"
        v.attrs = {}        
        ds.append(v)
    
    ds = xr.concat(ds, 'channel')
    return ds


def print_dataarray(
    ds, msg='', 
    names=["z500", "t850", "q700", "t2m", "msl"]
):
    v = ds.isel(time=0)
    msg += f"shape: {v.shape}"
    
    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"   

    if "level" in v.dims and len(v.level) > 1:
        for lvl in np.intersect1d(names, v.level.data):
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        for ch in  np.intersect1d(names, v.channel.data):
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)



def compare_dataarray(x1, x2):
    channel = np.intersect1d(x1.channel, x2.channel)
    for ch in channel:
        v1 = x1.sel(channel=ch)
        v2 = x2.sel(channel=ch)
        diff = v2 - v1
        print(f"name: {ch}: v1: {v1.max().item():.3f}, v2: {v2.max().item():.3f}, diff: {diff.max().item():.3f}")

