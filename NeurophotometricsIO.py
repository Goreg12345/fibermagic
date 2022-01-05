import os

from functions.martianova import *
from pathlib import Path
import pandas as pd
import numpy as np


def extract_leds(df):
    df['wave_len'] = np.nan
    df.loc[(df.Flags & 1).astype(bool), 'wave_len'] = 410
    df.loc[(df.Flags & 2).astype(bool), 'wave_len'] = 470
    df.loc[(df.Flags & 4).astype(bool), 'wave_len'] = 560
    return df


def perievents(df, logs, event, window, frequency):
    """
    produces a time-frequency plot with each event in one column
    :param df: pd Series
    :param logs: logs df with columns lever and Frame_Bonsai
    :param event: str, name of event, e.g. FD to build the trials of
    :param window: number of SECONDS to cut left and right off
    :return: event-related df with each trial os a column
    """
    # TODO: make a good documentation for this shit
    #pd.concat((df, logs.set_index('FrameCounter', append=True)), axis=1)
    levers = logs.lever.unique()
    channels = df.columns
    logs['FrameCounter'] = logs['Frame_Bonsai']
    logs = logs.set_index('FrameCounter', append=True)
    logs = logs.reset_index().pivot(index=logs.index.dtypes.index, columns='lever', values='Frame_Bonsai')
    df = pd.concat((df, logs), axis=1)
    df = df.fillna(method='ffill', limit=window*frequency)
    df = df.fillna(method='bfill', limit=window*frequency)

    #df = df.reset_index().pivot(index=('Analysis', 'Mouse', 'FD'), columns='FrameCounter', values='560').dropna(how='all')
    #df = df[df['FD'].notnull()].reset_index()
    df = df[df[levers].notnull().any(1)]
    df = df.reset_index().melt(id_vars=['Analysis', 'Mouse', 'FrameCounter', *levers], value_vars=channels).pivot(index=('Analysis', 'Mouse', 'FD'), columns=('variable', 'FrameCounter'), values='value').reset_index(level=2)
        #.pivot(index=('Analysis', 'Mouse', 'FD'), columns='FrameCounter', values='560')
    #df = df.reset_index(level=2)
    #df = df[df['FD']]






    v = df.values
    # TODO: replace with numpy
    a = [[n]*v.shape[1] for n in range(v.shape[0])]
    b = pd.isnull(v).argsort(axis=1, kind = 'mergesort')
    df.values[:] = v[a, b]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
    df = df.rename(columns = {a: b for (a, b) in zip(
        df.columns.values, np.arange(-window, window + 1e-6, 1/frequency))})

    return df

    # TODO: calculate average
    # TODO: make it general for every lever and every channel
    # TODO: delete code below but make sure that code above works with everything
    time_locked = pd.DataFrame()
    i = 1
    dist = window * frequency
    for index, row in logs[logs['lever'] == event].iterrows():
        t = row.Frame_Bonsai
        if df.index[0] > t - dist or df.index[-1] < t + dist:
            continue  # reject events if data don't cover the full window
        time_locked['Trial {n}'.format(n=i)] = \
            df.loc[np.arange(t - dist, t + dist)].reset_index(drop=True)
        i += 1
    time_locked['average'] = time_locked.mean(axis=1)

    time_locked.index = (time_locked.index - dist) / frequency
    return time_locked


def reference(df, region, wave_len, lambd=1e4, smooth_win=10):
    """
    Extracts streams for chosen LED and calculates zdFF
    :param df: raw data with columns FrameCounter, Flags and region
    :param region: column name to reference, e.g. Region6R
    :param wave_len: wave length of signal, e.g. 470 or 560
    :return: df with signal, reference and zdFF
    """
    # dirty hack to come around dropped frames until we find better solution - it makes about 0.16 s difference
    df.FrameCounter = df.index
    df = extract_leds(df).dropna()

    df.FrameCounter //= 3
    df = df.pivot('FrameCounter', 'wave_len', region).dropna()
    return get_zdFF(df, wave_len=wave_len, lambd=lambd, smooth_win=smooth_win)


def synchronize(logs, sync_signals, timestamps):
    """
    attatches Bonsai frame numbers to the the logs
    :param logs: log df with column SI@0.0
    :param sync_signals: df with timestamps of recorded sync signals from FP, columns Item1 and Item2
    :param timestamps: df with timestamp for each FP frame, columns Item1 and Item2
    :return: log file with new column Frame_Bonsai with FP frame number for each event
    """
    logs['lever'] = logs['SI@0.0'].str.split('@').str.get(0)
    logs['timestamp'] = logs['SI@0.0'].str.split('@').str.get(1).astype(float)

    # join FP SI with logs
    sync_signals = sync_signals.drop('Item1', axis=1)
    logs['Timestamp_Bonsai'] = sync_signals.loc[(logs.timestamp // 1).astype(int)].reset_index(drop=True)

    # convert Bonsai Timestamps to Frame number
    logs['Frame_Bonsai'] = timestamps.Item2.searchsorted(logs.Timestamp_Bonsai) // 3
    return logs


def create_giant_logs(project_path):
    """
    Merges all log files from a project into one
    :param project_path: project root path
    :return: Dataframe with logs in the following form:
                        lever  timestamp    Frame_Bonsai
    Trial 1 mouse A     LL         0.5      242
                        SI        0.56      278
                        FD        0.57      360
            mouse B     ...         ...
    Trial 2 mouse A     ...
            mouse B
    ...     ...
    """
    overview = pd.read_csv(project_path / 'meta' / 'overview.csv', delimiter=';').set_index("Mouse")
    giant = pd.DataFrame()

    for analysis in os.listdir(project_path):
        if analysis == 'meta': continue
        sync_signals = pd.read_csv(project_path / analysis / 'input1.csv')
        timestamps = pd.read_csv(project_path / analysis / 'time.csv')

        # find log file for each mouse
        for mouse in overview.index:
            for file in os.listdir(project_path / analysis):
                if '.log' in file and mouse in file:
                    logs = pd.read_csv(project_path / analysis / file)
                    logs = synchronize(logs, sync_signals, timestamps)
                    logs = logs[['lever', 'timestamp', 'Frame_Bonsai']]
                    logs['Mouse'] = mouse
                    logs['Analysis'] = analysis
                    logs = logs.set_index(['Analysis', 'Mouse'])
                    giant = giant.append(logs)
    return giant


def create_giant_dataframe(project_path, data_file):
    """
    Runs standard zdFF processing pipeline on every trial, mouse and sensor and puts data together in a single data frame
    :param project_path: root path of the project
    :param data_file: name of each file containing raw data
    :return: dataframe with the following multicolumn/index structure
                        sensor 470  sensor 560  ...
    Trial 1 mouse A     0.1         0.5
                        0.25        0.56
                        0.13        0.57
            mouse B     ...         ...
    Trial 2 mouse A     ...
            mouse B
    ...     ...
    """
    overview = pd.read_csv(project_path / 'meta' / 'overview.csv', delimiter=';').set_index("Mouse")
    giant = pd.DataFrame()
    for analysis in os.listdir(project_path):
        if analysis == 'meta': continue
        df = pd.read_csv(project_path / analysis / data_file)
        for mouse in overview.index:
            sdf = pd.DataFrame()
            for wave_len in overview.columns:
                sdf[wave_len] = reference(df, overview.loc[mouse, wave_len], int(wave_len)).zdFF
            sdf['Mouse'] = mouse
            sdf['Analysis'] = analysis
            sdf = sdf.set_index(['Analysis', 'Mouse', sdf.index])
            giant = giant.append(sdf)
    return giant

#def create_giant_peri_event(df, logs):

if __name__ == '__main__':
    DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
    df = pd.read_csv('data.csv')
    df = df.set_index(['Analysis', 'Mouse', 'FrameCounter'])
    logs = pd.read_csv('logs.csv')
    logs = logs.set_index(['Analysis', 'Mouse'])
    perievent = perievents(df, logs, 'FD', 10, 25)
    logs = create_giant_logs(DATA_DIR)
    df = create_giant_dataframe(DATA_DIR, 'FED3.csv')
    df
