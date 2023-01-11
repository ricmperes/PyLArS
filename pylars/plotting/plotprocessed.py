import matplotlib.pyplot as plt


def plot_hist_area(df_results, bins=800, ax=None, color=None):
    """Histogram of area values."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['area'], bins=bins, histtype='step', color=color)
    ax.set_xlabel('Area [integrated ADC counts]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_length(df_results, bins=50, ax=None, color=None):
    """Histogram of length values."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['length'], bins=bins, color=color)
    ax.set_xlabel('Length [# samples]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_position(df_results, bins=200, ax=None, color=None):
    """Histogram of position values."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['position'], bins=bins, color=color)
    ax.set_xlabel('Position [sample #]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_amplitude(df_results, bins=800, ax=None, color=None):
    """Histogram of amplitude values."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['amplitude'], bins=bins, histtype='step', color=color)
    ax.set_xlabel('Amplitude [ADC counts]')
    ax.set_ylabel('# events')

    return ax


def plot_3hists(df_results, axs=None, color=None):
    """Histograms of area, length and position values."""
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0] = plot_hist_area(df_results, ax=axs[0], color=color)
    axs[1] = plot_hist_length(df_results, ax=axs[1], color=color)
    axs[2] = plot_hist_position(df_results, ax=axs[2], color=color)

    return axs


def plot_4hists(df_results, axs=None, color=None):
    """Histograms of area, amplitude, length and position values."""
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    axs[0] = plot_hist_area(df_results, ax=axs[0], color=color)
    axs[1] = plot_hist_amplitude(df_results, ax=axs[1], color=color)
    axs[2] = plot_hist_length(df_results, ax=axs[2], color=color)
    axs[3] = plot_hist_position(df_results, ax=axs[3], color=color)

    return axs
