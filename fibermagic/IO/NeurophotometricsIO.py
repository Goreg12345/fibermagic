import os

from pathlib import Path
import pandas as pd
import numpy as np

from fibermagic.core.perievents import perievents
from fibermagic.core.demodulate import zdFF_airPLS, add_zdFF

NPM_RED = 560
NPM_GREEN = 470
NPM_ISO = 410


def extract_leds(df):
    """
    inserts new column 'wave_len' with wave length of LED that was on during the measurement
    based on the flags in the raw data file
    :param df: raw data
    :return: df with 'wave_len' as additional column
    """
    df['wave_len'] = np.nan
    df.loc[(df.LedState & 1).astype(bool), 'wave_len'] = NPM_ISO
    df.loc[(df.LedState & 2).astype(bool), 'wave_len'] = NPM_GREEN
    df.loc[(df.LedState & 4).astype(bool), 'wave_len'] = NPM_RED
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
    return zdFF_airPLS(df, wave_len=wave_len, lambd=lambd, smooth_win=smooth_win)


def sync_from_TTL_gen(logs, path):
    """
    attatches Bonsai frame numbers to the the logs
    :param logs: log df with column SI@0.0
    :param sync_signals: df with timestamps of recorded sync signals from FP, columns Item1 and Item2
    :param timestamps: df with timestamp for each FP frame, columns Item1 and Item2
    :return: log file with new column Frame_Bonsai with FP frame number for each event
    """
    sync_signals = pd.read_csv(path / 'input1.csv')
    timestamps = pd.read_csv(path / 'time.csv')

    logs['Event'] = logs['SI@0.0'].str.split('@').str.get(0)
    logs['Timestamp'] = logs['SI@0.0'].str.split('@').str.get(1).astype(float)

    # join FP SI with logs
    sync_signals = sync_signals.drop('Item1', axis=1)
    logs['Timestamp_Bonsai'] = sync_signals.loc[(logs.Timestamp // 1).astype(int)].reset_index(drop=True)

    # convert Bonsai Timestamps to Frame number
    logs['FrameCounter'] = timestamps.Item2.searchsorted(logs.Timestamp_Bonsai) // 3
    logs = logs[['FrameCounter', 'Event', 'Timestamp']]
    return logs.set_index('FrameCounter')


def read_project_logs(project_path, subdirs, sync_fun=sync_from_TTL_gen, ignore_dirs=['meta']):
    """
    Merges all log files from a project into one
    :param ignore_dirs: list of directories to exclude from reading
    :param sync_fun: function to sync custom log files to NPM's FrameCounter
                     should return a df with 'Event' and 'Timestamp as columns
                     and 'FrameCounter' as Index
    :param subdirs:  name of subsequent directory levels to be included as columns
    :param project_path: project root path
    :return: Dataframe with logs in the following form:
                                  Event  Timestamp
    subdir[0] subdir[1] FrameCounter
    Trial 1   mouse A   345       LL         0.5
                        456       SI        0.56
                        8765      FD        0.57
              mouse B   567       ...         ...
    Trial 2   mouse A   765       ...
              mouse B   456
    ...       ...
    """
    dfs = list()

    def recursive_listdir(path, levels):
        if levels:
            for dir in os.listdir(path):
                if dir in ignore_dirs:
                    continue
                recursive_listdir(path / dir, levels - 1)
        else:
            region_to_mouse = pd.read_csv(path / 'region_to_mouse.csv')
            for mouse in region_to_mouse.mouse.unique():
                for file in os.listdir(path):
                    if '.log' in file and mouse in file:
                        logs = pd.read_csv(path / file)
                        logs = sync_fun(logs, path)
                        for i in range(len(subdirs)):
                            logs[subdirs[i]] = path.parts[- (len(subdirs) - i)]
                        logs['Mouse'] = mouse
                        dfs.append(logs)
    recursive_listdir(Path(project_path), len(subdirs))
    df = pd.concat(dfs)
    df = df.reset_index().set_index([*subdirs, 'Mouse', 'FrameCounter'])
    return df


# TODO: read_recording_rawdata
# TODO: read_mouse_rawdata
def read_project_rawdata(project_path, subdirs, data_file, ignore_dirs=['meta']):
    """
    Runs standard zdFF processing pipeline on every trial, mouse and sensor and puts data together in a single df
    :param ignore_dirs: array of directories to exclude from reading
    :param subdirs: array with description of subdirectories to be a column
    :param project_path: root path of the project
    :param data_file: name of each file containing raw data
    :return: dataframe with the following multicolumn/index structure
                            Signal      Reference
    Trial 1 mouse A 560     0.1         0.5
                    560     0.25        0.56
                    470     0.13        0.57
            mouse B 560    ...         ...
    Trial 2 mouse A 470    ...
            mouse B 560
    ...     ...
    """
    # walk through all specified subdirectories
    dfs = list()

    def recursive_listdir(path, levels):
        if levels:
            for dir in os.listdir(path):
                if dir in ignore_dirs:
                    continue
                recursive_listdir(path / dir, levels - 1)
        else:
            print(path / data_file)
            df = pd.read_csv(path / data_file)
            region_to_mouse = pd.read_csv(path / 'region_to_mouse.csv')
            if 'Flags' in df.columns:  # legacy fix: Flags were renamed to LedState
                df = df.rename(columns={'Flags': 'LedState'})

            df = extract_leds(df).dropna()
            # dirty hack to come around dropped frames until we find better solution -
            # it makes about 0.16 s difference
            df.FrameCounter = df.index // len(df.wave_len.unique())
            df = df.set_index('FrameCounter')
            regions = [column for column in df.columns if 'Region' in column]
            for region in regions:
                channel = NPM_GREEN if 'G' in region else NPM_RED
                sdf = pd.DataFrame(data={
                    **{subdirs[i]: path.parts[- (len(subdirs) - i)] for i in range(len(subdirs))},
                    'Mouse': region_to_mouse[region_to_mouse.region == region].mouse.values[0],
                    'Channel': channel,
                    'Signal': df[region][df.wave_len == channel],
                    'Reference': df[region][df.wave_len == NPM_ISO]
                }
                )
                dfs.append(sdf)
    recursive_listdir(Path(project_path), len(subdirs))
    df = pd.concat(dfs)
    df = df.reset_index().set_index([*subdirs, 'Mouse', 'Channel', 'FrameCounter'])
    return df


if __name__ == '__main__':
    load_debug_from_disc = False
    if load_debug_from_disc:
        df = pd.read_csv('../debug_df.csv').set_index(['Group', 'Paradigm', 'Mouse', 'Channel', 'FrameCounter'])
        logs = pd.read_csv('../debug_logs.csv').set_index(['Group', 'Paradigm', 'Mouse', 'FrameCounter'])
    else:
        logs = read_project_logs(r'C:\Users\Georg\OneDrive - UvA\0 Research\data\fdrd2xadora_PR_NAcc',
                                 ['Group', 'Paradigm'])
        df = read_project_rawdata(r'C:\Users\Georg\OneDrive - UvA\0 Research\data\fdrd2xadora_PR_NAcc',
                                  ['Group', 'Paradigm'], 'FED3.csv')
        df = add_zdFF(df, smooth_win=10, remove=200).set_index('FrameCounter', append=True)
    peri = perievents(df, logs[logs.Event == 'FD'], 5, 25)
    peri.to_csv('debug/peri.csv')
