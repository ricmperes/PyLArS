from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
