from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle


def plot_sensor_layout(layout: np.ndarray,
                       r_tpc: Optional[float] = None,
                       labels: Optional[List[str]] = None,
                       ax=None):
    """Generate the rectangles of where sensors are, the basis of a
    hitpattern.

    Args:
        layout (np.ndarray): layout of array, each row a sensor.
        r_tpc (Optional[float], optional): _description_. Defaults to None.
        ax (_type_, optional): Axes. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for i, _sensor in enumerate(layout):
        xy = (_sensor[0], _sensor[2])
        width = _sensor[1] - _sensor[0]
        height = _sensor[3] - _sensor[2]
        ax.add_patch(Rectangle(xy, width, height, fill=False,
                               color='k', zorder=10))
        if labels is not None:
            ax.text(xy[0] + width / 2, xy[1] + height / 2, labels[i],
                    ha='center', va='center', zorder=10)

    if r_tpc is not None:
        ax.add_patch(Circle((0, 0), r_tpc, color='r', fill=False,
                            label='TPC edge'))

    return ax


def plot_hitpattern(hitpattern: Union[np.ndarray, List[float]],
                    layout: np.ndarray,
                    labels: Optional[List[str]] = None,
                    r_tpc: Optional[float] = None,
                    cmap: str = 'gnuplot',
                    log: bool = False,
                    ax=None):
    """Plot a beautiful hitpattern.

    Args:
        hitpattern (Union[np.ndarray, List[float]]): array with the are per
            sensor.
        layout (np.ndarray): layout of the sensor array (x1,x2,y1,y2) corners.
        labels (Optional[List[str]], optional): ordered labels to put in the
            center of each sensor. Defaults to None.
        r_tpc (Optional[float], optional): plot a line at the tpc edge.
            Defaults to None.
        cmap (str, optional): name of colormap to use. Defaults to 'gnuplot'.
        log (bool, optional): plot the log10 of pe instead of pe. Defaults
            to False.
        ax (_type_, optional): axis where to draw the hitpattern. Defaults
            to None.

    Returns:
        (axis, mappable): axis with the hitpattern drawned and the mappable
            for a colorbar.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    cm = plt.get_cmap(cmap)  # type: ignore

    if log == True:
        hitpattern = np.log10(hitpattern)

    color_max = max(hitpattern)
    color_min = min(hitpattern)

    for i, _sensor in enumerate(layout):
        pe = hitpattern[i]

        xy = (_sensor[0], _sensor[2])
        width = _sensor[1] - _sensor[0]
        height = _sensor[3] - _sensor[2]
        ax.add_patch(Rectangle(xy, width, height, fill=True,
                               edgecolor='k',
                               facecolor=cm((pe - color_min) /
                                            (color_max - color_min))))
        if labels is not None:
            ax.text(xy[0] + width / 2, xy[1] + height / 2, labels[i],
                    ha='center', va='center', zorder=10)

    if r_tpc is not None:
        ax.add_patch(Circle((0, 0), r_tpc, color='r', fill=False,
                            label='TPC edge'))

    norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)

    mappable = matplotlib.cm.ScalarMappable(  # type: ignore
        norm=norm, cmap=cmap)

    return (ax, mappable)


def plot_identified_peaks_individual_single(_area: float,
                                            _length: int,
                                            _position: int,
                                            _amplitude: float,
                                            sum_waveform: np.ndarray,
                                            x_unit: str = 'sample',
                                            detail_text: bool = True,
                                            figax: Optional[tuple] = None) -> Optional[tuple]:
    """Make a plot of an identified peak in a waveform.

    Args:
        _area (float): area of the peak in PE
        _length (int): length of the peak in samples
        _position (int): position of the peak in samples
        _amplitude (float): amplitude of the peak in PE/ns
        sum_waveform (np.ndarray): array with the sum waveform PEs
            over samples
        x_unit (str, optional): units to use on the x axis: sample or time.
            Defaults to 'sample'.
    """

    if figax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig, ax = figax

    _start = _position - 200 if _position - 200 > 0 else 0
    _end = _position + _length + 200 if _position + _length + \
        200 < len(sum_waveform) else len(sum_waveform)

    ax.plot(np.arange(_start, _end),
            sum_waveform[_start: _end], label='Sum waveform')

    
    # ax.fill_between(y1 = 0,
    #                 y2 = sum_waveform[_position: _position+_length],
    #                 x=np.arange(_position, _position+_length),
    #                 color='C4',
    #                 alpha = 0.3,
    #                 linestyle='--')
    #ax.axvline(x=_position, color='C1', linestyle='--', alpha = 0.4,
    #          label='Peak boundaries')
    #ax.axvline(x=_position + _length, color='C1', linestyle='--', alpha = 0.4)

    ax.set_ylabel('Amplitude [PE/ns]')
    ax.set_xlabel('# Sample')
    if x_unit == 'time':
        ax.set_xticks(ax.get_xticks(), ax.get_xticks() * 10 / 1000)
        ax.set_xlabel('Time [us]')
        ax.set_xlim(_start, _end)

    if detail_text:
        text_for_box = (f'Area: {_area:.2f} PE\nLength: {_length:d} samples\n'
                        f'Position: {_position:d} samples\n'
                        f'Amplitude: {_amplitude:.2f} PE/ns')
        ax.text(0.95, 0.95, text_for_box, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.1))

    if figax is None:
        plt.show()
    else:
        return fig, ax


def plot_identified_peaks_individual_all(lengths: list,
                                         positions: list,
                                         amplitudes: list,
                                         sum_waveform: np.ndarray,
                                         x_unit: str = 'sample',
                                         figax: Optional[tuple] = None):
    """Highlight all the identified peaks in a waveform.

    Args:
        lengths (list): list of lengths resulting from wf processing
        positions (list): list of positions resulting from wf processing
        amplitudes (list): list of amplitudes resulting from wf processing
        sum_waveform (np.ndarray): array with the sum waveform PEs
            over samples
    """

    if figax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig, ax = figax

    ax.plot(sum_waveform, label='Sum waveform')

    for _lenght, _position in zip(lengths, positions):
        ax.fill_betweenx(y=[0, max(amplitudes)], x1=_position, x2=_position + _lenght,
                         color='C4', alpha=0.3, linestyle='--')

    ax.set_ylabel('PE/ns')
    ax.set_xlabel('# Sample')
    if x_unit == 'time':
        ax.set_xticks(ax.get_xticks(), ax.get_xticks() * 10 / 1000)
        ax.set_xlabel('Time [us]')
        ax.set_xlim(sum_waveform[0], sum_waveform[-1])

    if figax is None:
        plt.show()
    else:
        return fig, ax


def plot_identified_peaks_each_channel(areas_individual_channels: list,
                                       positions: list,
                                       lengths: list,
                                       marker: str = '.',
                                       x_unit: str = 'sample',
                                       figax: Optional[tuple] = None):
    """Plot the identified peaks in each channel by balls.

    Args:
        areas_individual_channels (list): _description_
        positions (list): _description_
        lengths (list): _description_
        figax (Optional[tuple], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    chn_colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
                  7: 'C0', 8: 'C1', 9: 'C2', 10: 'C3', 11: 'C4'}

    if figax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig, ax = figax

    for _peak_id, areas_individual_channels_peak in enumerate(
            areas_individual_channels):
        _middle_of_peak = positions[_peak_id] + lengths[_peak_id] // 2
        for _ch_n in range(areas_individual_channels_peak.shape[0]):
            _ch_ycoord = areas_individual_channels_peak.shape[0] - _ch_n
            _area_of_peak = areas_individual_channels_peak[_ch_n]
            if _area_of_peak > 0:
                ax.scatter(x=_middle_of_peak,
                           y=_ch_ycoord,
                           s=_area_of_peak**0.7,
                           color=chn_colors[_ch_n],
                           alpha=0.8,
                           marker=marker)  # type: ignore
    ax.set_yticks(range(1, 13),
                  ['M', 'L', 'K', 'J', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'])

    if x_unit == 'time':
        ax.set_xticks(ax.get_xticks(), ax.get_xticks() * 10 / 1000)
        ax.set_xlabel('Time [µs]')
        #ax.set_xlim(_start, _end)
    if figax is None:
        plt.show()
    else:
        return fig, ax


def plot_identified_peaks_each_channel_waveforms(
        waveforms_pe_single_event: np.ndarray,
        x_unit: str = 'sample',
        figax: Optional[tuple] = None):
    """Plot the identified peaks in each channel by waveforms.

    Args:
        areas_individual_channels (list): _description_
        positions (list): _description_
        lengths (list): _description_
        figax (Optional[tuple], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    chn_colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6',
                  7: 'C0', 8: 'C1', 9: 'C2', 10: 'C3', 11: 'C4'}

    if figax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig, ax = figax

    max_value_y = np.max(waveforms_pe_single_event)
    for _ch_i in range(waveforms_pe_single_event.shape[0]):
        _ch_ycoord = waveforms_pe_single_event.shape[0] - _ch_i

        ax.fill_between(x=np.arange(waveforms_pe_single_event.shape[1]),
                        y1=_ch_ycoord,
                        y2=waveforms_pe_single_event[_ch_i,
                                                     :] / max_value_y * 1.5 + _ch_ycoord,
                        color=chn_colors[_ch_i],
                        alpha=0.6)

    ax.set_yticks(range(1, 13),
                  ['M', 'L', 'K', 'J', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'])

    ax.set_ylabel('Channel amplitude')
    if x_unit == 'time':
        ax.set_xticks(ax.get_xticks(), ax.get_xticks() * 10 / 1000)
        ax.set_xlabel('Time [µs]')
        #ax.set_xlim(_start, _end)
    if figax is None:
        plt.show()
    else:
        return fig, ax


def plot_full_waveform_peaks(areas: list,
                             lengths: list,
                             positions: list,
                             amplitudes: list,
                             sum_waveform_single: np.ndarray,
                             areas_individual_channels: list,
                             array_layout: np.ndarray,
                             array_labels: list,):
    """Make plot of the full waveform with the identified peaks.

    Args:
        areas (list): area of the identified peaks in the waveform
        lengths (list): length of the identified peaks in the waveform
        positions (list): position of the identified peaks in the waveform
        amplitudes (list): amplitude of the identified peaks in the waveform
        sum_waveform_single (np.ndarray): sum waveform of the event
        areas_individual_channels (list): areas of the peak seen in
            each channel
        array_layout (np.ndarray): layout of the array
        array_labels (list): labels of the array
    """

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax_sumwf = fig.add_subplot(gs[0, 0])
    ax_individual_ch = fig.add_subplot(gs[1, 0], sharex=ax_sumwf)
    ax_hitp = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(hspace=0, wspace=0.1)

    ax_sumwf = plot_identified_peaks_individual_all(
        lengths,
        positions,
        amplitudes,
        sum_waveform_single,
        x_unit='time',
        figax=(fig, ax_sumwf))

    ax_individual_ch = plot_identified_peaks_each_channel(
        areas_individual_channels,
        positions,
        lengths,
        x_unit='time',
        figax=(fig, ax_individual_ch))

    biggest_peak_id = np.where(areas == max(areas))[0][0]
    ax_hitp, _map = plot_hitpattern(
        hitpattern=areas_individual_channels[biggest_peak_id],
        layout=array_layout,
        labels=array_labels,
        r_tpc=160 / 2,
        cmap='coolwarm',
        log=False,
        ax=ax_hitp,)
    ax_hitp.set_xlim(-100, 100)
    ax_hitp.set_ylim(-100, 100)
    ax_hitp.set_aspect('equal')
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('y [mm]')

    fig.colorbar(_map, label='Peak Area [PE]', ax=ax_hitp)

    text_for_box = (f'Area: {areas[biggest_peak_id]:.2f} PE\n'
                    # type: ignore
                    f'Length: {lengths[biggest_peak_id]/100:.2f} µs\n'
                    # type: ignore
                    f'Position: {positions[biggest_peak_id]/100:.2f} µs\n'
                    f'Amplitude: {amplitudes[biggest_peak_id]:.2f} PE/ns')

    ax_text.text(
        0.5,
        0.4,
        s=text_for_box,
        ha='center',
        va='center',
        fontsize=12)
    ax_text.axis('off')

    plt.show()


def plot_peak_info(peak_id: int, areas: list, lengths: list,
                   positions: list, amplitudes: list,
                   sum_waveform_single: np.ndarray,
                   areas_individual_channels: list,
                   array_layout: np.ndarray, array_labels: list,
                   waveforms_pe_single_event: Optional[np.ndarray] = None,
                   mucoin: Optional[int] = None,
                   pos_rec: Optional[np.ndarray] = None,
                   save_fig: Optional[str] = None):
    """Fancy plot of the peak information, i.e., the sum waveform, the
    hitpattern, the individual channel waveforms and the peak information.

    Args:
        peak_id (int): number of the peak in the waveform
        areas (list): area of the identified peaks in the waveform
        lengths (list): length of the identified peaks in the waveform
        positions (list): position of the identified peaks in the waveform
        amplitudes (list): amplitude of the identified peaks in the waveform
        sum_waveform_single (np.ndarray): sum waveform of the event
        areas_individual_channels (list): areas of the peak seen in
            each channel
        array_layout (np.ndarray): layout of the array
        array_labels (list): labels of the array
        waveforms_pe_single_event (Optional[np.ndarray], optional): waveforms
            of each channel to make an even fancier plot. Defaults to None.
        save_fig (Optional[str], optional): path to save the figure (include
            extension). Defaults to None.
    """

    fig = plt.figure(figsize=(10, 5), dpi=120)
    gs = GridSpec(2, 2, width_ratios=[1, 0.5], height_ratios=[1, 0.8])

    ax_sumwf = fig.add_subplot(gs[0, 0])
    ax_individual_ch = fig.add_subplot(gs[1, 0], sharex=ax_sumwf)
    ax_hitp = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(hspace=0, wspace=0.25)

    _area = areas[peak_id]
    _length = lengths[peak_id]
    _position = positions[peak_id]
    _amplitude = amplitudes[peak_id]
    fig, ax_sumwf = plot_identified_peaks_individual_single(_area,  # type: ignore
                                                            _length,
                                                            _position,
                                                            _amplitude,
                                                            sum_waveform_single,
                                                            x_unit='time',
                                                            detail_text=False,
                                                            figax=(fig, ax_sumwf))

    _start = _position - 200 if _position - 200 > 0 else 0
    _end = _position + _length + 200 if _position + _length + \
        200 < len(sum_waveform_single) else len(sum_waveform_single)

    if mucoin is not None:
        ax_sumwf.axvline(x=mucoin, color='C1', linestyle='--',
                         label = 'Muon coin trigger', 
                         zorder = -10)
        ax_sumwf.legend()

    if waveforms_pe_single_event is not None:
        fig, ax_individual_ch = plot_identified_peaks_each_channel_waveforms(
            waveforms_pe_single_event,
            x_unit='time',
            figax=(fig, ax_individual_ch))  # type: ignore

    else:
        fig, ax_individual_ch = plot_identified_peaks_each_channel(  # type: ignore
            areas_individual_channels,
            positions,
            lengths,
            x_unit='time',
            figax=(fig, ax_individual_ch))

    ax_sumwf.set_xlim(_start, _end)
    ax_individual_ch.set_xlim(_start, _end)

    # ax_sumwf.set_xticklabels([])

    ax_hitp, _map = plot_hitpattern(
        hitpattern=np.log10(areas_individual_channels[peak_id]),  # /1e5,
        layout=array_layout,
        labels=array_labels,
        r_tpc=160 / 2,
        cmap='coolwarm',
        log=False,
        ax=ax_hitp,)
    
    if pos_rec is not None:
        ax_hitp.scatter(pos_rec[0], pos_rec[1], 
                        c='C2', marker='X', s=100, 
                        lw = 1,edgecolor='k', zorder = 10,
                        label = 'Rec. pos.', alpha = 0.8)
        ax_hitp.legend(fontsize=8, ncol=2)
        
    ax_hitp.set_xlim(-100, 100)
    ax_hitp.set_ylim(-100, 100)
    ax_hitp.set_aspect('equal')
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('y [mm]')

    cbar = fig.colorbar(_map,
                        label='Peak Area log10 [PE]',
                        ax=ax_hitp)

    #new_ticks = [3.5, 4 , 4.5, 5 ]
    #new_ticklabels = ['$10^{3.5}$', '$10^4$', '$10^{4.5}$', '$10^5$']

    # cbar.set_ticks(new_ticks)
    # cbar.set_ticklabels(new_ticklabels) # type: ignore

    text_for_box = (f'Area: {areas[peak_id]:.2e} PE\n'
                    f'Length: {lengths[peak_id]/100:.2f} µs\n'
                    f'Position: {positions[peak_id]/100:.2f} µs\n'
                    f'Amplitude: {amplitudes[peak_id]:.2f} PE/ns')

    ax_text.text(
        0.4,
        0.4,
        s=text_for_box,
        ha='center',
        va='center',
        fontsize=12)
    ax_text.axis('off')
    if save_fig:
        plt.savefig(save_fig, tight_layout=True, bbox_inches='tight')
    plt.show()
