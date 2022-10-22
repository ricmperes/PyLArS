import base64
import hashlib
import json
import numpy as np

def Gaussean(x, A, mu, sigma):
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

def load_ADC_config(model: str, F_amp: float) -> dict:
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

    if model not in available_model_configs:
        raise NotImplementedError(f'''The requested model ({model}) is not 
            implemented. Choose from {available_model_configs}.''')

    if model == 'v1724':
        """ More info at https://www.caen.it/products/v1724/"""

        ADC_config = {'ADC_range': 2.25, #V
                      'ADC_impedance': 50, #ohm
                      'F_amp': F_amp, #external amplification
                      'ADC_res': 2**14, #bit-wise resolution
                      'q_e': 1.602176634e-19, # electron charge
                      'dt':10e-9} # sampling time length
    
    elif model == 'v1730':
        """More info at https://www.caen.it/products/v1730/"""

        ADC_config = {'ADC_range': 2.00, #V
                      'ADC_impedance': 50, #ohm
                      'F_amp': F_amp, #external amplification
                      'ADC_res': 2**14, #bit-wise resolution
                      'q_e': 1.602176634e-19, # electron charge
                      'dt':2e-9} # sampling time length
    
    return ADC_config
    
