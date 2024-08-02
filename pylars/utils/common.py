import base64
import hashlib
import itertools
import json
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import pylars
from scipy.signal import find_peaks
import scipy.ndimage


def Gaussian(x, A, mu, sigma):
    y = A * np.exp(-((x - mu) / sigma)**2 / 2) / np.sqrt(2 * np.pi * sigma**2)
    return y


def func_linear(x, a, b):
    return a * x + b


def get_deterministic_hash(id: str) -> str:
    """Return an hash with 7 characters from a string.

    In detail, returns a base32 lowercase string of length determined
    from hashing the configs. Based on
    https://github.com/AxFoundation/strax/blob/
    156254287c2037876a7040460b3551d590bf5589/strax/utils.py#L303

    Args:
        id (str): thing to hash

    Returns:
        str: hashed version of the thing
    """
    jsonned = json.dumps(id)
    digest = hashlib.sha1(jsonned.encode('ascii')).digest()
    readable_hash = base64.b32encode(digest)[:7].decode('ascii').lower()
    return readable_hash


def load_ADC_config(model: str, F_amp: float) -> Dict[str, Union[int, float]]:
    """Load the ADC related quantities depending on the model.

    Args:
        model (str): model of the digitizer
        F_amp (float): signal amplification from the sensor (pre-amp *
            external amplification on the rack).

    Raises:
        NotImplementedError: Raised if the requested model is not yet
            implemented

    Returns:
        dict: Python dictionary with the digitizer-related configs.
    """

    available_model_configs = ['v1724', 'v1730']

    if model == 'v1724':
        """ More info at https://www.caen.it/products/v1724/"""

        ADC_config = {'ADC_range': 2.25,  # V
                      'ADC_impedance': 50,  # ohm
                      'F_amp': F_amp,  # external amplification
                      'ADC_res': 2**14,  # bit-wise resolution
                      'q_e': 1.602176634e-19,  # electron charge
                      'dt': 10e-9}  # sampling time length

    elif model == 'v1730':
        """More info at https://www.caen.it/products/v1730/"""

        ADC_config = {'ADC_range': 2.00,  # V
                      'ADC_impedance': 50,  # ohm
                      'F_amp': F_amp,  # external amplification
                      'ADC_res': 2**14,  # bit-wise resolution
                      'q_e': 1.602176634e-19,  # electron charge
                      'dt': 2e-9}  # sampling time length
    else:
        raise NotImplementedError(f'''The requested model ({model}) is not
            implemented. Choose from {available_model_configs}.''')

    return ADC_config


def get_gain(F_amp: float,
             spe_area: float,
             ADC_range: float = 2.25,
             ADC_impedance: float = 50,
             ADC_res: float = 2**16,
             q_e: float = 1.602176634e-19,
             ) -> float:
    """Compute the gain given the value of the SPE area and the ADC
        paramenters.

    Args:
        F_amp (float): Total signal amplification factor.
        spe_area (float): mean area of spe (in ADC bins x ns).
        ADC_range (float, optional): Dynamic range of the ADC. Defaults
            to 2.25.
        ADC_impedance (float, optional): Impedance of the ADC. Defaults to 50.
        ADC_res (float, optional): bit.wise resolution of the ADC. Defaults
            to 2**16.
        q_e (float, optional): electron charge [C]. Defaults to 1.602176634e-19.

    Returns:
        float: the calculated gain.
    """

    gain = (ADC_range * spe_area * 1e-9 / ADC_impedance / F_amp /
            ADC_res / q_e)

    return gain


def find_minmax(array: np.ndarray) -> List[np.ndarray]:
    """Return local peaks and valeys of an 1d array.

    Args:
        array (np.ndarray): 1d array to compute peaks and valeys

    Returns:
        List[np.ndarray]: res[0] is an array with the indexes of where peaks
            were identified, valeys at res[1].
    """
    peaks = np.where(
        (array[1:-1] > array[0:-2]) * (array[1:-1] > array[2:]))[0] + 1
    valeys = np.where(
        (array[1:-1] < array[0:-2]) * (array[1:-1] < array[2:]))[0] + 1
    return [peaks, valeys]


def get_channel_list(process) -> List[Tuple[int, str]]:
    """Fetch the channels available for a dataset by reading the original
    ROOT file.

    Args:
        process (run_processor): initiated `run_processor` object.

    Returns:
        List[Tuple[int,str]]: list with the available channels of a given
            dataset in the format [(mod, ch)_i]
    """
    _datasets = process.datasets_df
    modules = np.unique(_datasets['module'])
    ch_list = []
    for mod in modules:
        raw = pylars.utils.input.raw_data(
            _datasets[_datasets['module'] == mod].sample(1)['path'].values[0],
            V=123,
            T=123,
            module=mod)

        raw.load_root()
        raw.get_available_channels()
        channels = raw.channels
        ch_list += list(itertools.product([mod], channels))
    return ch_list


def wstd(array: np.ndarray, waverage: float, weights: np.ndarray) -> float:
    """Compute weighted standard deviation.

    Args:
        array (np.ndarray): 1D array to compute wstd of
        waverage (float): weighted average value
        weights (np.ndarray): weights to consider

    Returns:
        float: value of the weighted standard deviation
    """

    N = np.count_nonzero(weights)

    wvar = N * np.sum(weights * (array - waverage)**2) / \
        (N - 1) / np.sum(weights)
    wstd = np.sqrt(wvar)

    return wstd


def get_peak_rough_positions(area_array: np.ndarray,
                             cut_mask,
                             bins: Union[int, np.ndarray, list] = 1000,
                             filt=scipy.ndimage.gaussian_filter1d,
                             filter_options=3,
                             plot: Union[bool, str] = False) -> tuple:
    '''Takes the area histogram (fingerplot) and looks for peaks
    and valeys. SPE should be the 2nd peak on most cases. Higher
    PE values might be properly unrecognized
    Returns two lists: list of the x value of the peaks, list of
    the x value of the valeys.
    Optional plot feature:
    - False: no plot
    - True: displays plot in notebook
    - string - sufix on the name of the file'''

    area_hist = np.histogram(
        area_array[cut_mask], bins=bins)

    area_x = area_hist[1]
    area_x = (area_x + (area_x[1] - area_x[0]) / 2)[:-1]
    area_y = area_hist[0]

    area_filt = filt(area_y, filter_options)
    area_peaks_x, peak_properties = find_peaks(area_filt,
                                               prominence=20,
                                               distance=50)

    if plot != False:
        from pylars.plotting.plotanalysis import plot_found_area_peaks
        plot_found_area_peaks(
            area_x, area_y, area_filt, area_peaks_x)

    return area_x[area_peaks_x], peak_properties


def apply_tile_labels(df: pd.DataFrame, label_map: dict):
    """Adds a column with the tile. Requires a label map of the form:
    {'mod#' : {'wf#' : [tile]}.

    Args:
        label_map (dict): tile label map

    Returns:
        pd.dataframe: the dataframe with a column for the tile
    """

    def map_label(row, label_map=label_map):
        return label_map[f"mod{row['module']}"][row['channel']]

    df['tile'] = df.apply(map_label, axis=1)
    df = df.sort_values('tile', ignore_index=True)
    return df


def get_summary_info(summary_path):
    with open(summary_path, 'r') as _summ_file:
        _summary = _summ_file.readlines()
    _t_stop = np.datetime64(int(_summary[0].strip().split(' ')[-1]), 's')
    _duration = np.timedelta64(int(_summary[1].strip().split(' ')[3]), 's')
    _n_events = int(_summary[2].strip().split(' ')[-1])

    return _t_stop, _duration, _n_events
