from functions.martianova import *
from pathlib import Path
import pandas as pd
import numpy as np

from functions.plot import plot_seperate, plot_single, heatmap


def extract_leds(df):
    df['wave_len'] = np.nan
    df.loc[(df.Flags & 1).astype(bool), 'wave_len'] = 410
    df.loc[(df.Flags & 2).astype(bool), 'wave_len'] = 470
    df.loc[(df.Flags & 4).astype(bool), 'wave_len'] = 560
    return df


def lock_time_to_event(df, logs, event, window):
    """
    produces a time-frequency plot with each event in one column
    :param df: zdFF df with zdFF column
    :param logs: logs df with columns lever and Frame_Bonsai
    :param event: str, name of event, e.g. FD to build the trials of
    :param window: number of FP frames to cut left and right off
    :return: event-related df with each trial os a column
    """
    time_locked = pd.DataFrame()
    i = 1
    for index, row in logs[logs['lever'] == event].iterrows():
        t = row.Frame_Bonsai
        time_locked['Trial {n}'.format(n=i)] = df.loc[np.arange(t - window, t + window)].reset_index().zdFF
        i += 1
    time_locked['average'] = time_locked.mean(axis=1)
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


if __name__ == '__main__':
    DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
    ANALYSIS = 'PR8'
    DATA_FILE = 'FED3.csv'
    LOG_FILE = 'FRfD2AdoraCrepilotPR8B8624-1.log'
    MOUSE_NAME = 'B8624'
    LED = '7'
    FREQUENCY = 25

    df = pd.read_csv(DATA_DIR / ANALYSIS / DATA_FILE)
    df = reference(df, 'Region6R', 560)
    plot_seperate(df[470], df[410])
    plot_single(df.zdFF)

    sync_signals = pd.read_csv(DATA_DIR / ANALYSIS / 'input1.csv')
    timestamps = pd.read_csv(DATA_DIR / ANALYSIS / 'time.csv')
    logs = pd.read_csv(DATA_DIR / ANALYSIS / LOG_FILE)
    logs = synchronize(logs, sync_signals, timestamps)

    time_locked = lock_time_to_event(df, logs, 'FD', 15 * FREQUENCY)
    heatmap(time_locked)

