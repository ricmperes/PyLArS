from typing import Tuple, Union

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import glob 
import pylars

class LED_window():
    '''Class to hold analysis of LED ON data with integrating window. 
    Use independently of BV datasets.
    '''

    def __init__(self,
                 led_window: Tuple[int, int],
                 led_data_path: str) -> None:
        self.led_window = led_window
        self.led_data_path = led_data_path
        self.baseline_samples = 50

        self.files_df = self.find_files()

    def find_files(self) -> pd.DataFrame:
        '''Find the raw data file and the LED data file.
        '''

        files = glob.glob(self.led_data_path + '**/*.root', recursive=True)
        _dfs = []
        for f in tqdm(files, desc='Finding LED data files: '):
            try:
                _module = int(f.split('/')[-1].split('_')[4])
                _width, _voltage = self.get_LED_width_voltage(f)
                _dfs.append({'LEDwidth': _width, 
                             'LEDvoltage': _voltage, 
                             'module': _module,
                             'path': f})
            except:
                print(f'Failed to get info for file {f}')
                continue
        files_df = pd.DataFrame(_dfs)
        files_df.sort_values(by=['LEDwidth', 'LEDvoltage', 'module'], 
                     ignore_index = True, inplace=True)
        del _dfs, files

        return files_df

    @staticmethod
    def get_LED_width_voltage(path):
        root_name_list = path.split('/')[-1].split('_')
        pulser_width = int(root_name_list[0][:-2])
        voltage = float(root_name_list[1] + '.' + root_name_list[2][0])
        return pulser_width, voltage

    def process_dataset(self, data_path: str, module: int = 0) -> pd.DataFrame:
        '''Process a dataset with fixwindowprocessor. Requires a raw data file
        and a LED window specified.
        '''

        processor = pylars.processing.fixwindowprocessor.window_processor(
            baseline_samples=self.baseline_samples, 
            led_window=(self.led_window[0],self.led_window[1]))
        
        processor.load_raw_data(path_to_raw=data_path, module=module)
        df_processed = processor.process_all_channels()
        return df_processed
    


    def process_all_datasets(self) -> pd.DataFrame:
        '''Process all datasets in the LED data path.
        '''

        _dfs = []
        for i, row in tqdm(self.files_df.iterrows(), 
                           total=len(self.files_df), 
                           desc='Processing LED data: '):
            _df = self.process_dataset(row['path'], module=row['module'])
            _df['LEDwidth'] = row['LEDwidth']
            _df['LEDvoltage'] = row['LEDvoltage']
            _df['module'] = row['module']
            _dfs.append(_df)
        df_processed = pd.concat(_dfs)
        del _dfs
        
        self.df_processed = df_processed

        return df_processed