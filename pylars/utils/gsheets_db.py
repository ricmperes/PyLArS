from glob import glob
from typing import Tuple
import json

import numpy as np
import pandas as pd
import uproot
from pylars.utils.common import load_ADC_config

import gspread
import os

class xenoscope_db():
    """Main xenoscope database class with methods to fetch data from Google
    Sheets and ROOT files.
    """

    def __init__(self):
        self.gs_config = self.fetch_gs_config()

        self.gs_servive = gspread.service_account( filename = 
            os.path.join(self.gs_config_path, self.gs_config['credsfileName'])) # type: ignore
            
        self.gs_file = self.gs_servive.open(self.gs_config['gsheetName'])
        self.gs_run_db = self.gs_file.worksheet('Runs')


    def fetch_gs_config(self):
        """Fetch the Google Sheets config file.
        """
        home_dir = os.path.expanduser("~")
        default_gs_config = '.xenoscope-db/xenoscope-db-config.json'
        config_path = os.path.join(home_dir, default_gs_config)

        if os.path.exists(config_path):
            self.gs_config_path = os.path.join(home_dir, '.xenoscope-db')
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError('Could not find config file for '
                                    'Google Sheets.\n'
                                    'Please create it and try again.')
    
    def count_rows(self):
        """Count the number of rows in the Google Sheet.
        """
        return self.gs_run_db.row_count
    
    def get_header(self):
        """Get the header of the Google Sheet.
        """
        return self.gs_run_db.row_values(1)
    
    def get_run_types(self):
        """Get all the run types existing in the run db.
        """
        _run_types = self.gs_run_db.col_values(2)
        _run_types.pop(0)
        return set(_run_types)
    
    def get_run_db_df(self):
        """Get the run database as a pandas DataFrame.
        """
        _run_db = self.gs_run_db.get_all_records()
        return pd.DataFrame(_run_db)

