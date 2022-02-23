import numpy as np
import pandas as pd


def single_perievent(df, logs, event, window, frequency):
    """
    produces a time-frequency plot with each event in one column
    :param df: pd Series
    :param logs: logs df with columns event and Frame_Bonsai
    :param event: str, name of event, e.g. FD to build the trials of
    :param window: number of SECONDS to cut left and right off
    :return: event-related df with each trial os a column
    """
    time_locked = pd.DataFrame()
    i = 1
    dist = window * frequency
    for t, row in logs[logs['event'] == event].iterrows():
        if df.index[0] > t - dist or df.index[-1] < t + dist:
            continue  # reject events if data don't cover the full window
        time_locked['Trial {n}'.format(n=i)] = \
            df.loc[np.arange(t - dist, t + dist)].reset_index(drop=True)
        i += 1
    time_locked['average'] = time_locked.mean(axis=1)

    time_locked.index = (time_locked.index - dist) / frequency
    return time_locked


def perievents(df, logs, window, frequency):
    """
    produces perievent slices for each event in logs
    :param df: df with 'Channel' as index and value columns
    :param logs: logs df with columns event and same index as df
    :param window: number of SECONDS to cut left and right off
    :param frequency: int, frequency of recording in Hz
    :return: perievent dataframe with additional indices event, timestamp and Trial
    """
    channels = df.index.unique(level='Channel')

    if 'Channel' not in logs.index.names:  # make indices the same to intersect
        logs['Channel'] = [list(channels)] * len(logs)
        logs = logs.explode('Channel').set_index('Channel', append=True)
        logs = logs.swaplevel(-1, -2)
    logs = logs.loc[df.index.intersection(logs.index)]  # remove events that are not recorded

    df = df.sort_index()  # to slice it in frame ranges
    logs['Trial'] = logs.groupby(logs.index.names[:-1]).cumcount()
    peri = list()
    timestamps = np.arange(-window, window + 1e-9, 1 / frequency)

    # extract slice for each event and concat
    for index, row in logs.iterrows():
        start = index[:-1] + (index[-1] - window * frequency,)
        end = index[:-1] + (index[-1] + window * frequency,)

        single_event = df.loc[start:end]
        single_event[row.index] = row
        single_event['Timestamp'] = timestamps
        peri.append(single_event)  # Set on copy warning can be ignored because it is a copy anyways
    peri = pd.concat(peri)

    peri = peri.set_index(list(logs.columns), append=True)
    peri = peri.reset_index(['FrameCounter'], drop=True)
    return peri


def perievents_2D(df, logs, window, frequency):
    """
    produces a time-frequency plot with each event in one column
    :param df: pd Series
    :param logs: logs df with columns event and Frame_Bonsai
    :param event: str, name of event, e.g. FD to build the trials of
    :param window: number of SECONDS to cut left and right off
    :return: event-related df with each trial os a column
    """
    # TODO: make a good documentation for this shit
    channels = df.columns

    # stack all idx but not the FrameCounter to be able to select and slice
    logs['channel'] = [list(channels)] * len(logs)
    logs = logs.explode('channel')
    logs = logs.loc[df.index.intersection(logs.index)]
    logs = logs.reset_index(level=-1).set_index(['channel', 'FrameCounter'], append=True)

    df_stacked = df.stack().unstack(level=(*df.index.names[:-1], -1)).sort_index()

    def f(row):
        idx = row.name[:-1]
        frame = row.name[-1]
        s_df = df_stacked[idx].dropna().sort_index()
        s_df = s_df.loc[frame - window * frequency:frame + window * frequency]
        return s_df

    def shift_left(df):
        v = df.values
        # TODO: replace with numpy
        a = [[n] * v.shape[1] for n in range(v.shape[0])]
        b = pd.isnull(v).argsort(axis=1, kind='mergesort')
        df.values[:] = v[a, b]
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
        df = df.rename(columns={a: b for (a, b) in zip(
            df.columns.values, np.arange(-window, window + 1e-6, 1 / frequency))})
        df = df.stack().unstack(level=('channel', -1))
        return df

    peri = logs.apply(f, axis=1)
    peri['event'] = logs.event
    peri = peri.set_index('event', append=True)
    peri = shift_left(peri)
    return enumerate_trials(peri)


def enumerate_trials(perievents):
    """
    adds an index to perievents_2D that counts the number of trials per session and event
    starting with 1, removes FrameCounter index
    :param perievents: perievents df, non-column based format
    :return: perievents df with additional index Trial
    """
    # unstack indices to make several counts for each event and session
    perievents = perievents.reset_index('FrameCounter', drop=True)
    idx = list(perievents.index.names)
    perievents['Trial'] = perievents.groupby(idx).cumcount() + 1
    return perievents.set_index('Trial', append=True)


def perievents_to_columns(perievents):
    """
    rearranges perievents to an column-based format
    that makes it easier to use for plotting frameworks
    :param perievents: perievent df
    :return: perievent df with one column zdFF and other dimensions as index
    """
    # restack to make a column-based format
    perievents = perievents.stack(level=['channel', 'FrameCounter'])
    return perievents.to_frame('zdFF')
