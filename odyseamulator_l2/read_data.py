import datetime
import numpy
import xarray
import logging
from typing import Optional
import odyseamulator_l2.metadata as metadata


logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)

ATTR_VARS = metadata.VARIABLES
ATTR_GEO = metadata.GEOMETRY

def load_model(path_model: str, start: datetime.datetime,
               end: datetime.datetime, dic_coord: Optional[dict] = {},
               dic_var: Optional[dict] = {},
               arakawa:Optional[bool] = False,
               reshape_coord: Optional[bool] = True) -> xarray.Dataset:

    model = xarray.open_mfdataset(path_model, combine='by_coords',
                                  data_vars='different', coords='different',
                                  engine="netcdf4")
    #model.time.values.astype(float)
    if 'time_units' in dic_coord.keys():
        attrs = {'units': dic_coord['time_units']} #'days since 1950-01-01'}
        ntime = 'time'
        for key, value in dic_coord.items():
            if 'time' in value:
                ntime = key
        time = xarray.Dataset({ntime: (ntime, model[ntime].values, attrs)})
        time = xarray.decode_cf(time)
        model[ntime] = time[ntime].astype('datetime64[ns]')
        _ = dic_coord.pop('time_units')
    if len(dic_coord.keys()) > 0:
        list_dism = []
        for key, value in dic_coord.items():
            if 'lon' in value:
                model.coords[key] = (model.coords[key] + 180) % 360 - 180
        model = model.rename(name_dict=dic_coord)
    #for key in ('nav_lon_u', 'nav_lon_v', 'nav_lat_u', 'nav_lat_v'):
    #    model.drop_vars(key)
    strstart = datetime.datetime.strftime(start, '%Y-%m-%d')
    strend = datetime.datetime.strftime(end, '%Y-%m-%d')
    model = model.sel(time=slice(strstart, strend))
    logger.info(f'simulation for [{strstart}, {strend}] period')
    logger.debug(f'model starts at f{model["time"][0]}')
    logger.debug(f'model ends at f{model["time"][-1]}')
    return model


def load_model2(path_model: str, start: datetime.datetime,
                end: datetime.datetime, dic_coord: Optional[dict] = {},
                dic_var: Optional[dict] = {'u': 'u', 'v': 'v'},
                arakawa:Optional[bool] = False,
                reshape_coord: Optional[bool] = True) -> xarray.Dataset:

    model = xarray.open_mfdataset(path_model, combine='by_coords',
                                  data_vars='different', coords='different',
                                  engine="netcdf4")
    #model.time.values.astype(float)
    if 'time_units' in dic_coord.keys():
        attrs = {'units': dic_coord['time_units']} #'days since 1950-01-01'}
        ntime = 'time'
        for key, value in dic_coord.items():
            if 'time' in value:
                ntime = key
        time = xarray.Dataset({ntime: (ntime, model[ntime].values, attrs)})
        time = xarray.decode_cf(time)
        model[ntime] = time[ntime].astype('datetime64[ns]')
        _ = dic_coord.pop('time_units')
    if len(dic_coord.keys()) > 0:
        list_dism = []
        for key, value in dic_coord.items():
            if 'lon' in value:
                model.coords[key] = (model.coords[key] + 180) % 360 - 180
            if (reshape_coord is True) and (len(model[key].shape) > 1):
                dic_dim = {'lat_u': 'y_u', 'lon_u': 'x_u', 'lat_v': 'y_v',
                           'lon_v': 'x_v', 'lon': 'x', 'lat': 'y'}
                if 'lat' in key:
                     ds = xarray.Dataset({value: (dic_dim[value],
                                          model[key].values[:, -1], {})})
                elif 'lon' in key:
                     ds = xarray.Dataset({value: (dic_dim[value],
                                          model[key].values[0, :], {})})
                model.coords[value] = ds[value]
                list_dism.append(key)
        for key in list_dism:
            _ = dic_coord.pop(key)
        model = model.rename(name_dict=dic_coord)
        if 'nav_lat_u' in list_dism:
            dic_dim = {'y_u': 'lat_u', 'y_v': 'lat_v', 'x_u': 'lon_u',
                         'x_v': 'lon_v'} #, 'time_counter': 'time'}
        else:
            dic_dim = {'x': 'lon', 'y': 'lat'} #,'time_counter': 'time'}
        model = model.rename(name_dict=dic_dim)
    #for key in ('nav_lon_u', 'nav_lon_v', 'nav_lat_u', 'nav_lat_v'):
    #    model.drop_vars(key)
    model2 = xarray.Dataset()
    if arakawa is True:
        model2 = model2.assign_coords(coords={'time': (['time'], model['time'].data, attrs),
                                             'lon_u': (['lon_u'], model['lon_u'].data, {}),
                                             'lon_v': (['lon_v'], model['lon_v'].data, {}),
                                             'lat_u': (['lat_u'], model['lat_u'].data, {}),
                                             'lat_v': (['lat_v'], model['lat_v'].data, {}),
                                            })
        model2 = model2.assign({dic_var['u']: (['time', 'lat_u', 'lon_u'], model[dic_var['u']].data, {}),
                               dic_var['v']: (['time', 'lat_v', 'lon_v'], model[dic_var['v']].data, {}),
                               })
    else:
        model2 = model2.assign_coords(coords={'time': (['time'], model['time'].data, attrs),
                                             'lon': (['lon'], model['lon'].data, {}),
                                             'lat': (['lat'], model['lat'].data, {}),
                                            })
        model2 = model2.assign({dic_var['u']: (['time', 'lat', 'lon'], model[dic_var['u']].data, {}),
                               dic_var['v']: (['time', 'lat', 'lon'], model[dic_var['v']].data, {}),
                              })
    #model = model.drop_duplicates('time')
    if arakawa is True:
        model2 = model2.sortby(model2.lon_u)
    else:
        model2 = model2.sortby(model2.lon)
    strstart = datetime.datetime.strftime(start, '%Y-%m-%d')
    strend = datetime.datetime.strftime(end, '%Y-%m-%d')
    model2 = model2.sel(time=slice(strstart, strend))
    logger.info(f'simulation for [{strstart}, {strend}] period')
    logger.debug(f'model starts at f{model2["time"][0]}')
    return model2


def convert_wind(model: xarray.Dataset, varu: Optional[str] = 'u_model',
                 varv: Optional[str] = 'v_model'):
    model['norm'] = numpy.sqrt(model[varu]**2 + model[varv]**2)
    model['direction'] = numpy.arctan2(model[varu]/model['norm'],
                                     model[varv]/model['norm'])
    model['direction'] = numpy.rad2deg(model['direction'])
    return model


def colocateSwathCurrents(model: xarray.Dataset, orbit: xarray.Dataset,
                          varu: str, varv:str, arakawa: Optional[bool] = False,
                          method: Optional[str] = 'linear') -> xarray.Dataset:

    """
    Colocate model current data to a swath (2d continuous array)
    Coordinates in model data should be named lon, lat, time for non arakawa
    grid and lon_u, lat_u, lon_v, lat_v, time for arakawa grid

    Args:
        orbit (object): xarray dataset orbit generated via the
         orbit.getOrbit() call.
    Returns:
       original orbit containing model data linearly interpolated to
        the orbit swath. new data is contained in u_model, v_model

    """
    lats = orbit['lat'].values.flatten()
    lons = orbit['lon'].values.flatten()
    times = orbit['sample_time'].values.flatten()
    if arakawa is True:
        ds_u = model[varu].interp(time=xarray.DataArray(times, dims='z'),
                                  lat_u=xarray.DataArray(lats, dims='z'),
                                  lon_u=xarray.DataArray(lons, dims='z'),
                                  method=method)
        ds_v = model[varv].interp(time=xarray.DataArray(times, dims='z'),
                                  lat_v=xarray.DataArray(lats, dims='z'),
                                  lon_v=xarray.DataArray(lons, dims='z'),
                                  method=method)
    else:
        ds_u = model[varu].interp(time=xarray.DataArray(times, dims='z'),
                                  lat=xarray.DataArray(lats, dims='z'),
                                  lon=xarray.DataArray(lons, dims='z'),
                                  method='linear')
        ds_v = model[varv].interp(time=xarray.DataArray(times, dims='z'),
                                  lat=xarray.DataArray(lats, dims='z'),
                                  lon=xarray.DataArray(lons, dims='z'),
                                  method='linear')

    u_interp = numpy.reshape(ds_u.values, numpy.shape(orbit['lat'].values))
    v_interp = numpy.reshape(ds_v.values, numpy.shape(orbit['lat'].values))

    orbit = orbit.assign({'u_model': (['along_track', 'cross_track'],
                                      u_interp, ATTR_VARS['u_model']),
                          'v_model': (['along_track', 'cross_track'],
                                      v_interp, ATTR_VARS['v_model'])})

    return orbit
