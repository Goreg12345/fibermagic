import os

from functions.martianova import *
from pathlib import Path
import pandas as pd
import numpy as np

from functions.perievents import perievents, perievents_to_columns, enumerate_trials


def extract_leds(df):
    """
    inserts new column 'wave_len' with wave length of LED that was on during the measurement
    based on the flags in the raw data file
    :param df: raw data
    :return: df with 'wave_len' as additional column
    """
    df['wave_len'] = np.nan
    df.loc[(df.LedState & 1).astype(bool), 'wave_len'] = 410
    df.loc[(df.LedState & 2).astype(bool), 'wave_len'] = 470
    df.loc[(df.LedState & 4).astype(bool), 'wave_len'] = 560
    return df


def reference(df, region, wave_len, lambd=1e4, smooth_win=10):
    """
    Extracts streams for chosen LED and calculates zdFF
    :param df: raw data with columns FrameCounter, Flags and region
    :param region: column name to reference, e.g. Region6R
    :param wave_len: wave length of signal, e.g. 470 or 560
    :return: df with signal, reference and zdFF
    """
    # dirty hack to come around dropped frames until we find better solution - it makes about 0.16 s difference
    # TODO: it's not 3 for every experiment, only if isobestic + 2 signals
    df.FrameCounter = df.index // 3
    df = extract_leds(df).dropna()
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
    logs['event'] = logs['SI@0.0'].str.split('@').str.get(0)
    logs['timestamp'] = logs['SI@0.0'].str.split('@').str.get(1).astype(float)

    # join FP SI with logs
    sync_signals = sync_signals.drop('Item1', axis=1)
    logs['Timestamp_Bonsai'] = sync_signals.loc[(logs.timestamp // 1).astype(int)].reset_index(drop=True)

    # convert Bonsai Timestamps to Frame number
    logs['FrameCounter'] = timestamps.Item2.searchsorted(logs.Timestamp_Bonsai) // 3
    return logs.set_index('FrameCounter')


def create_giant_logs(project_path):
    """
    Merges all log files from a project into one
    :param project_path: project root path
    :return: Dataframe with logs in the following form:
                        event  timestamp    Frame_Bonsai
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
                    logs = logs[['event', 'timestamp']]
                    logs['Mouse'] = mouse
                    logs['Analysis'] = analysis
                    logs = logs.reset_index()
                    logs = logs.set_index(['Analysis', 'Mouse', 'FrameCounter'])
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
    giant = list()
    for analysis in os.listdir(project_path):
        if analysis == 'meta': continue
        df = pd.read_csv(project_path / analysis / data_file)
        if 'Flags' in df.columns:  # legacy fix: Flags were renamed to LedState
            df = df.rename(columns={'Flags': 'LedState'})
        for mouse in overview.index:
            for wave_len in overview.columns:
                giant.append(pd.DataFrame(data={
                    'Channel': wave_len,
                    'zdFF': reference(df, overview.loc[mouse, wave_len], int(wave_len)).zdFF,
                    'Mouse': mouse,
                    'Analysis': analysis
                }))
    giant = pd.concat(giant)
    giant = giant.reset_index().set_index(['Analysis', 'Mouse', 'Channel', 'FrameCounter'])
    return giant


if __name__ == '__main__':
    use_debug_data = False
    if use_debug_data:
        df = pd.read_csv('data.csv')
        df = df.head(100000)
        df = df.set_index(['Analysis', 'Mouse', 'FrameCounter'])
        logs = pd.read_csv('logs.csv')
        logs = logs.set_index(['Analysis', 'Mouse'])
    else:
        DATA_DIR = Path(r'C:\Users\glange\OneDrive\data\cDATxAdoraPilotRecording2')
        logs = create_giant_logs(DATA_DIR)
        df = create_giant_dataframe(DATA_DIR, 'FED3.csv')
    logs = logs[logs['event'] == 'FD']
    perievent = perievents(df, logs, 10, 25)
    perievent.to_csv('debug-02-09-22.csv')
