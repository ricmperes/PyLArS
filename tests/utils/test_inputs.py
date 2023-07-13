from random import sample
from pylars.utils.input import *
from hypothesis import given, strategies as st
import pytest


class test_raw_data():
        
    @given(st.text(), st.floats(), st.floats(), st.integers())
    def test_rawdata_creation(path, voltage, temperature, module):
        raw_data_to_test = raw_data(path, voltage, temperature, module)

        assert raw_data_to_test.bias_voltage == voltage
        assert raw_data_to_test.temperature == temperature
        assert raw_data_to_test.module == module
        
        # to use pytest fixture
        assert raw_data_to_test.charge_factor == standard_charge_factor


    def test_rawdata_sample(sample_data_raw_data):
        sample_data_raw_data = raw_data(raw_path = '../data-sample/', 
            V = ,
            T = ,
            module = )
        channels = sample_data_raw_data.channels
        sample_data_channels = sample_data_raw_data.get_channel_data('wf3')
        sample_channel_data.load_root()
        assert channels == ['wf0','wf2','wf3','wf4','wf5','wf6']
        assert isinstance(sample_data_channels, np.ndarray)
        assert sample_channel_data[0] == 12345


class test_run():

    @given(st.integers(), st.text())
    def test_run_creation(run_number, path):
        run_to_test = run(run_number, path)

        assert run_to_test.run_number == run_number
        assert run_to_test.main_run_path == (
            run_to_test.main_data_path +
            f'run{run_to_test.run_number}/data/')
        assert isinstance(run_to_test.root_files, list)
        assert isinstance(run_to_test.datasets, list)
    
    def test_run_sample(sample_data_run):
        df = sample_data_run.get_run_df()

        assert sample_data_run.run_number == 0
        assert sample_data_run.main_data_path == '../sample_data/'
        assert sample_data_run.main_run_path == '../sample_data/run0/data/'
        assert sample_data_run.root_files == ['fill me with the correct paths']
        assert isinstance(sample_data_run.datasets, list)
        assert isinstance(sample_data_run.datasets[0], dataset)
        assert len(sample_data_run.datasets) == 2
        assert isinstance(df, pd.DataFrame)
        #assert test some dataset values here


class test_dataset():

    @given(st.text(), st.text(), st.integers(), st.floats(), st.floats())
    def test_dataset_creation(path, kind, module, temperature, voltage):
        dataset_to_test = dataset(path, kind, module, temperature, voltage)

        assert dataset_to_test.path == path
        assert dataset_to_test.kind == kind
        assert dataset_to_test.module == module
        assert dataset_to_test.temp == temperature
        assert dataset_to_test.vbias == voltage

        assert isinstance(dataset_to_test.print_config(), str)

    def test_dataset_sample(sample_data_dataset):
        #raw_path, voltage, temperature, module = get_sample_config()

        sample_data_dataset.path == ('../sample_data/run0/data/BV_170K_50_50V/'
                                     'Module0/BV_170K_50_50V_Module_0_0.root')
        sample_data_dataset.kind == 'BV'
        sample_data_dataset.module == 0
        sample_data_dataset.temp == 170.
        sample_data_dataset.vbias == 50.50

