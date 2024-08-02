import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylars.utils.gsheets_db import xenoscope_db


def plot_run_type_livetimes(style='pie', figax=None, df_run_db=None):
    """Make a piechart of all the available run types by collected livetime.
    """
    if df_run_db is None:
        db = xenoscope_db()

        run_types = db.get_run_types()
        df_run_db = db.get_run_db_df()

    else:
        run_types = np.unique(df_run_db['Run type'])

    livetimes = (pd.to_datetime(df_run_db['End\n(yyyy-mm-dd \nhh:mm:00)']) -
                 pd.to_datetime(df_run_db['Start \n(yyyy-mm-dd \nhh:mm:ss)']))

    sum_livetimes = [
        np.sum(
            livetimes[df_run_db['Run type'] == _run_type]
        ).total_seconds() for _run_type in run_types]

    if figax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = figax

    if style == 'pie':

        ax.pie(sum_livetimes, explode=0.1 * np.ones(len(sum_livetimes)),
               labels=run_types, autopct='%1.2f%%')  # type: ignore
        ax.legend()

        if figax is None:
            plt.show()
        else:
            return fig, ax
    elif style == 'bar':

        ax.bar(
            list(run_types),
            sum_livetimes /
            np.sum(sum_livetimes))  # type: ignore
        ax.set_xlabel('Run type')
        ax.set_ylabel('Livetime [%]')
        ax.legend()

        if figax is None:
            plt.show()
        else:
            return fig, ax

    else:
        raise NotImplementedError('Style not implemented. Please choose '
                                  'between pie and bar.')
