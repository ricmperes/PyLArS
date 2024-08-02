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

        self.gs_servive = gspread.service_account(
            filename=os.path.join(
                self.gs_config_path,
                self.gs_config['credsfileName']))  # type: ignore

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

    def get_run_dict(self, run_number):
        row_number = self.get_db_row_number(run_number)

        row_values = self.gs_run_db.get_values(f'{row_number}:{row_number}')[0]
        header = self.get_header()
        run_dict = {}
        for i, key in enumerate(header):
            run_dict[key] = row_values[i]
        return run_dict

    def get_db_row_number(self, run_number):
        """Get the row number of the run in the database.
        """

        # Use self.db.gs_run_db.find(in_column=run_number_col) instead?
        db_header = self.get_header()
        run_number_col = db_header.index(
            'Run number') + 1  # GShhet is 1-indexed
        all_run_numbers = self.gs_run_db.col_values(run_number_col)
        return all_run_numbers.index(str(run_number)) + 1
