import numpy as np
import matplotlib.pyplot as plt


def plot_hist_area(df_results, ax=None, color = None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['area'], bins=800, histtype='step', color = color)
    ax.set_xlabel('Area [integrated ADC counts]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_length(df_results, ax=None, color = None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['length'], bins=50, color = color)
    ax.set_xlabel('Length [# samples]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_position(df_results, ax=None, color = None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['position'], bins=200, color = color)
    ax.set_xlabel('Position [sample #]')
    ax.set_ylabel('# events')

    return ax


def plot_3hists(df_results, axs=None, color = None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0] = plot_hist_area(df_results, ax=axs[0], color = color)
    axs[1] = plot_hist_length(df_results, ax=axs[1], color = color)
    axs[2] = plot_hist_position(df_results, ax=axs[2], color = color)

    return axs
