# vim: ts=4:sts=4:sw=4
#
# @author <lucile.gaultier@oceandatalab.com>
# @date 2024-01-10
#
# Copyright (C) 2024 OceanDataLab

"""
Create Odysea L2 like data from netcdf files and a python parameter file
"""


from odyseamulator_l2.swath_sampling import OdyseaSwath
import odyseamulator_l2.metadata as metadata
import odyseamulator_l2.read_data as read_data

import numpy
import os
import sys
import tqdm
import yaml
import datetime
import itertools
import xarray
import logging
from scipy.interpolate import RegularGridInterpolator
from typing import Optional, Tuple

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)

ATTR_VARS = metadata.VARIABLES
ATTR_GEO = metadata.GEOMETRY


def load_python_file(file_path: str):
    """Load a file and parse it as a Python module."""
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    init_parameters(module)
    return module


def init_parameters(params):
    params.wind_path = getattr(params, 'wind_path', None)
    params.wind_speed = getattr(params, 'wind_speed', 7)
    params.wind_dir = getattr(params, 'wind_dir', 0)
    params.var_wind = getattr(params, 'var_wind', ('geo5_u10m', 'geo5_v10m'))
    params.var_current = getattr(params, 'var_current', ('SSU', 'SSV'))
    params.dic_coord = getattr(params, 'dic_coord', {})
    params.dic_coord_wind = getattr(params, 'dic_coord_wind', {})
    params.ecef = getattr(params, 'ecef', True)
    params.altitude = getattr(params, 'a;titude', 650000)
    return None


def vradialSTDLookup(wind_speed: numpy.ndarray, wind_dir: numpy.ndarray,
                     encoder_angle: numpy.ndarray, azimuth: numpy.ndarray,
                     vradial_interpolator) -> numpy.ndarray:
    relative_azimuth = numpy.mod((numpy.mod(wind_dir + 180, 360) - 180
                                  - numpy.mod(azimuth +360, 360)) + 180, 360) - 180
    encoder_norm = numpy.mod(encoder_angle + 180, 360) - 180
    wind_speed[wind_speed>19] = 19
    interp = vradial_interpolator((wind_speed.flatten(),
                                   relative_azimuth.flatten(),
                                   encoder_norm.flatten()))
    return numpy.reshape(interp, numpy.shape(encoder_norm))


def generate_interpolator(lut_fn: str, key: Optional[str] = 'sigma_vr'
                          ) -> RegularGridInterpolator:
    lut = numpy.load(lut_fn)
    if 'wind_speed_range' in lut.files:
        wind_speed_range = lut['wind_speed_range']
    else:
        wind_speed_range = numpy.arange(0, 20, .1)
    if 'wind_dir_range' in lut.files:
        wind_dir_range = lut['wind_dir_range']
    else:
        wind_dir_range = numpy.arange(-195, 195, 5)
    if 'encoder_angle_range' in lut.files:
        encoder_angle_range = lut['encoder_angle_range']
    else:
        encoder_angle_range = numpy.arange(-190, 190, 5)
    vradial_lut = lut[key]
    vradial_interpolator = RegularGridInterpolator((wind_speed_range,
                                                    wind_dir_range,
                                                    encoder_angle_range),
                                                    vradial_lut,
                                                    bounds_error=False,
                                                    fill_value=0)
    return vradial_interpolator




def load_orbit(orbit_fname: str, config_fname: str,
               start_time: datetime.datetime, end_time: datetime.datetime,
               year_ref: Optional[int] = 2020,
               bounding_box: Optional[list] = [-180, 180, -90, 90]
               ) -> xarray.Dataset:
    odysea = OdyseaSwath(orbit_fname=orbit_fname, config_fname=config_fname,
                         year_ref=year_ref)
    orbits = odysea.getOrbits(start_time=start_time, end_time=end_time,
                              bounding_box=bounding_box)
    return orbits



def job_odysea(o, model, i, c, bb, pattern_out, asc):
    oa_out = interp_model(o, model, bb, asc=True)
    if oa_out is not None:
        oa_out.to_netcdf(f'{pattern_out}_c_{c:02d}_p{i:03d}.nc', 'w')



def run(parameter_file:str, first_cycle: int, last_cycle: int):
    logger.debug(f'Load parameter file {parameter_file}')
    params = load_python_file(parameter_file)
    logger.debug('load orbit')
    save_file = params.pickle_swath
    logger.debug(f'load configuration{params.config_file}')
    with open(params.config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    npass = cfg['NPASS']
    if not os.path.exists(save_file):
        yorbits = load_orbit(params.orbit_file, params.config_file,
                              params.start_time, params.end_time,
                              year_ref=params.year_ref, ecef=params.ecef,
                              alt=params.altitude,
                              bounding_box=params.bounding_box)
        #list_orbit = [o for o in itertools.islice(yorbits, 0, 500)]
        list_orbit = [o for o in itertools.islice(yorbits, 0, 50000)]

        logger.debug(f'generate noise interpolator')
        vradial_interpolator = generate_interpolator(params.lut_fn,
                                                     key=params.sigma_vr)
        import pickle
        dic_save = {'orbits': list_orbit,
                    'vradial_interpolator': vradial_interpolator}
        logger.info(f'saving in file {save_file}')
        with open(f'{save_file}', 'wb') as f:
            pickle.dump(dic_save, f)
    else:
        logger.info(f'orbit file {save_file} exists, delete it to regenerate')
