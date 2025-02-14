import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def perievents(df, time_column, event_column, data_columns, window, by=None, baseline=False, only_average=True):
    """
    Calculate peri-event time series for multiple events and data columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing time series data and event markers
    time_column : str
        Name of column containing timestamps
    event_column : str
        Name of column containing event markers
    data_columns : str or list of str
        Name(s) of column(s) containing the data to analyze
    window : float
        Time window (in seconds) before and after each event to include
    by : str or list of str, optional
        Column name(s) to group data by before processing
    baseline : str or bool, optional
        If 'mean', subtract mean of entire window
        If True, subtract mean of baseline period (from -2*window to -window)
        If False, no baseline correction
    only_average : bool, optional
        If True, return trial-averaged data
        If False, return data from individual trials

    Returns
    -------
    pd.DataFrame
        DataFrame containing peri-event time series data with columns:
        - Event: event marker value
        - Trial: trial number (if only_average=False)
        - Relative Time: time relative to event onset
        - Data columns: time-locked data values
        Additional grouping columns included if 'by' parameter specified
    """
    if isinstance(by, str):
        by = [by]
    if isinstance(data_columns, str):
        data_columns = [data_columns]

    rel_time = np.arange(-window, window, 1 / 30)
    rel_time_s = pd.Series(rel_time, name="Relative Time")

    def process_group(group):
        time_locked = []
        unique_events = group[event_column].unique()
        unique_events = unique_events[unique_events != 0]  # exclude 0

        for event in unique_events:
            onset_times = group.loc[group[event_column] == event, time_column].values
            for i, t in enumerate(onset_times):
                # skip if there's not enough data (optional, same logic)
                if (t - window - 1 < group[time_column].min()) or (t + window + 1 > group[time_column].max()):
                    continue

                # subset
                sub = group[(group[time_column] >= t - window) & (group[time_column] < t + window)]
                sub = sub[[time_column] + data_columns].copy()

                # shift time to relative
                sub["Relative Time"] = sub[time_column] - t

                # merge_asof
                merged = pd.merge_asof(rel_time_s, sub, on="Relative Time", direction="nearest")

                # add event/trial
                merged["Event"] = event
                merged["Trial"] = i

                # baseline if needed
                if baseline:
                    if baseline == "mean":
                        merged[data_columns] = merged[data_columns] - merged[data_columns].mean()
                    else:
                        # e.g. baseline from [t - 2*window, t - window]
                        base = group[(group[time_column] >= t - 2 * window) & (group[time_column] < t - window)]
                        for col in data_columns:
                            merged[col] = merged[col] - base[col].mean()

                merged = merged.set_index(["Event", "Trial", "Relative Time"])
                time_locked.append(merged)

        if not time_locked:
            return pd.DataFrame()  # empty if no events

        time_locked = pd.concat(time_locked, axis=0)

        if only_average:
            # average over trials
            time_locked = time_locked.groupby(["Event", "Relative Time"])[data_columns].mean()

        return time_locked

    if by is None:
        return process_group(df).reset_index()
    else:
        return df.groupby(by, sort=False).progress_apply(process_group).reset_index()
