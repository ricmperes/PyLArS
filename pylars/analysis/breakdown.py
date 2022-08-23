import pylars
import numpy as np

class BV_dataset():
    """Object class to hold breakdown voltage related instances and
    methods. Collects all the data and properties of a single MMPC, 
    meaning the pair (module, channel) for all the available voltages at
    a certain temperature.
    """

    def __init__(self, run: pylars.utils.input.run, temperature: float,
        module: int, channel: str, 
        processor: pylars.processing.rawprocessor.run_processor):

        self.run = run
        self.temp = temperature
        self.module = module
        self.channel = channel
        self.process = processor
        self.voltages = self.get_voltages_available()

    def get_voltages_available(self) -> np.array:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.array: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'BV'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def load_processed_data(self, force_processing: bool = False) -> dict:
        self.data = {}
        for _voltage in self.voltages:
            processed_data = pylars.utils.output.processed_dataset(
                run=self.run,
                kind='BV',
                vbias=_voltage,
                temp=self.temp,
                path_processed=('/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
                                'processed_data/'),
                process_hash=self.process.hash)
            processed_data.load_data(force = force_processing)

            _df = processed_data.data
            mask = ((_df['module'] == self.module) &
                    (_df['channel'] == self.channel))
            
            self.data[_voltage] = _df[mask].copy()
        
        return self.data #not needed but doesn't harm and can be useful 
        

        

    