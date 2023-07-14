from pylars.utils.input import *

# Most of the tests are for now almost meaningless appart
# from creating the differnent data types. For proper
# testing a small set of data must be available.

path = 'beep'
kind = 'boop'
voltage = 2
temperature = 3
module = 4
run_number = 5


def test_rawdata():
    raw_data_to_test = raw_data(path, voltage, temperature, module)

    assert raw_data_to_test.bias_voltage == voltage
    assert raw_data_to_test.temperature == temperature
    assert raw_data_to_test.module == module


def test_run():
    run_to_test = run(run_number, path)

    assert run_to_test.run_number == run_number
    assert run_to_test.main_run_path == (
        run_to_test.main_data_path +
        f'run{run_to_test.run_number}/data/')
    assert isinstance(run_to_test.root_files, list)
    assert isinstance(run_to_test.datasets, list)


def test_dataset():
    dataset_to_test = dataset(path, kind, module, temperature, voltage)

    assert dataset_to_test.path == path
    assert dataset_to_test.kind == kind
    assert dataset_to_test.module == module
    assert dataset_to_test.temp == temperature
    assert dataset_to_test.vbias == voltage

    assert isinstance(dataset_to_test.print_config(), str)
