import numpy as np
import matplotlib.pyplot as plt


def plot_hist_area(df_results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['area'], bins=800, histtype='step')
    ax.set_xlabel('Area [integrated ADC counts]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_length(df_results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['length'], bins=50)
    ax.set_xlabel('Length [# samples]')
    ax.set_ylabel('# events')

    return ax


def plot_hist_position(df_results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(df_results['position'], bins=200)
    ax.set_xlabel('Position [sample #]')
    ax.set_ylabel('# events')

    return ax


def plot_3hists(df_results, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0] = plot_hist_area(df_results, ax=axs[0])
    axs[1] = plot_hist_length(df_results, ax=axs[1])
    axs[2] = plot_hist_position(df_results, ax=axs[2])

    return axs
