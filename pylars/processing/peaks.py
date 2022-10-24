import numpy as np
import numba as nb

class peak_processing():
    """All the things peaks. Peaks are sums of pulses found in waveforms.

    This class, by definition, is a collection of class methods related 
    to peak processing to be used in `peakprocessor`, where a processor 
    object is then constructed.
    """       

    @classmethod
    def apply_ADCcounts_to_e(cls, waveforms_subtracted: np.ndarray, 
                             ADC_config: dict) -> np.ndarray:
        """Convert ADC counts/sample to charge.

        Applies the charge converting factor to waveforms to turn ADC counts 
        (which are integrated over 1 sample) to charge. `waveforms_subtracted` 
        can be one or more channels.

        Args:
            waveforms_subtracted (np.ndarray): value of ADC counts per sample 
                above calculate local baseline
            ADC_config (dict): dictionary with the ADC config

        Raises:
            ValueError: If the parsed ADC_config dictionary does not have the 
                required keys.

        Returns:
            np.ndarray: waveforms in charge.
        """

        try:
            ADC_range = ADC_config['ADC_range']
            ADC_impedance = ADC_config['ADC_impedance']
            F_amp = ADC_config['F_amp']
            ADC_res = ADC_config['ADC_res']
            q_e = ADC_config['q_e']
            dt = ADC_config['dt']
        except:
            raise ValueError('The ADC_config dictionary is probably missing ' +
                             'something.')
        
        to_e_constant = (ADC_range * dt / ADC_impedance / F_amp / 
                         ADC_res / q_e)

        waveforms_charge = waveforms_subtracted * to_e_constant

        return waveforms_charge

    @classmethod
    def apply_e_to_pe(cls,waveforms_charge: np.ndarray,
                      gains: np.ndarray) -> np.ndarray: 
        """Transform waveforms from charge to pe with gain per channel.

        Takes waveforms already converted to charge from ADC counts and an 
        array with the gains for each channel in units of [e/pe]. The ammount 
        of rows in the waveform array needs to be the same as the length of 
        the gains array.

        The gains array is assumed to be on the correct order in respect to 
        the order of channels in waveforms_charge.

        Args:
            waveforms_charge (_type_): _description_

        Returns:
            np.ndarray: 
        """

        assert len(gains) == np.shape(waveforms_charge)[0], ('''Size of 
        gains and channels in waveforms array do not match.''')

        waveforms_pe = (waveforms_charge.T / gains).T

        return waveforms_pe

    @classmethod
    def apply_baseline_subtract(cls, waveforms: np.ndarray, 
                                baselines:np.ndarray) -> np.ndarray:
        """Apply baseline subtracting and flipping from negative to positive 
        pulses.

        Args:
             waveforms (np.ndarray): waveforms, all channels stacked by rows.
            baselines (np.ndarray): computed baselines, all channels stacked 
                by rows.

        Returns:
            np.ndarray: waveforms flipped and where 0 is local baseline.
        """

        assert len(baselines) == np.shape(waveforms)[0], ('''Size of 
        baseines and channels in waveforms array do not match.''')

        waveforms_subtracted = (baselines - waveforms.T).T  # type: ignore

        return waveforms_subtracted

    @classmethod
    def apply_waveforms_transform(cls, waveforms: np.ndarray,
                                  baselines: np.ndarray,
                                  gains: np.ndarray,
                                  ADC_config: dict) -> np.ndarray:
        """Converts waveforms from ADC counts/sample to pe/s.
        
        Takes the initials waveforms stacked for all channels and returns 
        the waveforms in converted pe/s space.

        Args:
            waveforms (np.ndarray): waveforms, all channels stacked by rows.
            baselines (np.ndarray): computed baselines, all channels stacked 
                by rows.
            gains (np.ndarray): gains, all channels stacked by rows.
            ADC_config (dict): dictionary with the specific digitizer configs.

        Returns:
            np.ndarray: transformed waveforms
        """

        waveforms_subtracted = cls.apply_baseline_subtract(
            waveforms, baselines)
        waveforms_charge = cls.apply_ADCcounts_to_e(
            waveforms_subtracted, ADC_config)
        waveforms_pe = cls.apply_e_to_pe(waveforms_charge, gains)

        return waveforms_pe

    @classmethod
    def get_sum_waveform(cls, waveforms_pe: np.ndarray) -> np.ndarray:
        """Sums the (transformed to pe/s) waveforms of all channels.

        Args:
            waveforms_pe (np.ndarray): array with waveforms from all the 
                channels.

        Returns:
            np.ndarray: Summed waveform.
        """

        summed_waveform = np.sum(waveforms_pe, axis = 0)

        return summed_waveform
