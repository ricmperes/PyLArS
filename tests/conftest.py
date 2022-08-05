from pylars.utils.input import raw_data, run, dataset
import pytest

@pytest.fixture
def standard_charge_factor():
    return 1234

def get_sample_config():
    raw_path = ('../sample_data/run0/data/BV_170K_50_50V/Module0'
                '/BV_170K_50_50V_Module_0_0.root')
    voltage = 50.50
    temperature = 170.00
    module = 0
    return raw_path, voltage, temperature, module


@pytest.fixture(scope = "package")
def sample_data_raw_data():
    raw_path, voltage, temperature, module = get_sample_config()
    sample_raw_data = raw_data(raw_path, voltage, temperature, module)
    sample_raw_data.load_root()
    sample_raw_data.get_available_channels()
    return sample_raw_data

@pytest.fixture(scope = "package")
def sample_data_run():
    path = ('../sample_data/')
    run_number = 0
    
    sample_run = run(run_number=run_number, main_data_path=path)
    return sample_run

@pytest.fixture(scope="package")
def sample_data_dataset():
    raw_path, voltage, temperature, module = get_sample_config()
    kind = 'BV'
    sample_dataset = dataset(path = raw_path, 
                             kind = kind,
                             module = module,
                             temp = temperature,
                             vbias = voltage)
    return sample_dataset