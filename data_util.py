import os
import numpy as np
import xarray as xr

__all__ = ["make_sample", "print_dataarray"]

levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


pl_names = [
    ('geopotential', 'z'),
    ('temperature', 't'),
    ('u_component_of_wind', 'u'),
    ('v_component_of_wind', 'v'),
    ('specific_humidity', 'q'),
]

sfc_names = [
    ('2m_temperature', 't2m'),
    ('2m_dewpoint_temperature', 'd2m'),
    ('sea_surface_temperature', 'sst'), 
    ('10m_u_component_of_wind', 'u10m'),
    ('10m_v_component_of_wind', 'v10m'),               
    ('100m_u_component_of_wind', 'u100m'),
    ('100m_v_component_of_wind', 'v100m'),   
    ('mean_sea_level_pressure', 'msl'),
    ('surface_pressure', 'sp'),   
]

avg_names = [
    ('surface_net_solar_radiation', 'ssr'),
    ('surface_solar_radiation_downwards', 'ssrd'),
    ('total_sky_direct_solar_radiation_at_surface', 'fdir'),
    ('top_net_thermal_radiation','ttr'),
    ('total_precipitation', 'tp'),     
]

def is_pressure(short_name):
    return short_name in ["z", "t", "u", "v", "q"]


def get_channel(pl_names):
    channel = []
    for (_, short_name) in pl_names:
        channel += [f'{short_name}{l}' for l in levels]
    for (_, short_name) in sfc_names + avg_names:
        channel += [short_name]
    return channel


def make_sample(data_dir, version="c79"):

    new_pl_names = pl_names
    if version == "c92":
        new_pl_names += [('specific_cloud_liquid_water_content', 'clwc')]

    ds = []
    for (long_name, short_name) in new_pl_names + sfc_names:   
        file_name = os.path.join(data_dir, f"{long_name}.nc")
        v = xr.open_dataarray(file_name)
        if is_pressure(short_name) and v.level.values[0] != 50:
            v = v.reindex(level=v.level[::-1])

        if short_name in ["q", "clwc"]:
            print(f"Convert {short_name} to g/kg")
            v = v * 1000
        
        v.name = "data"
        v.attrs = {}        
        ds.append(v)

    for (long_name, short_name) in avg_names:
        zero = v * 0
        print(zero)
        print(f"zero: {zero.min():.3f} ~ {zero.max():.3f}")
        ds.append(zero)

    ds = xr.concat(ds, 'level').rename({"level": "channel"})
    ds = ds.assign_coords(channel=get_channel(new_pl_names))
    return ds


def print_dataarray(
    ds, msg='', 
    names=["z500", "t850", "q700", "t2m", "d2m", "sst", "msl", "tp"]
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
        names = np.intersect1d(names, v.level.data)
        for lvl in names:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        names = np.intersect1d(names, v.channel.data)
        for ch in names:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)
