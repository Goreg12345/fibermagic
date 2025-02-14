import pandas as pd
import numpy as np
import warnings

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
    df["wave_len"] = np.nan
    df.loc[(df.LedState & 1).astype(bool), "wave_len"] = NPM_ISO
    df.loc[(df.LedState & 2).astype(bool), "wave_len"] = NPM_GREEN
    df.loc[(df.LedState & 4).astype(bool), "wave_len"] = NPM_RED
    df.loc[(df.LedState == 7), "wave_len"] = np.nan  # start of recording, remove this row
    return df


def read_npm(
    photometry_path, region_to_mouse_path=None, region_to_mouse=None, extra_timestamps_path=None, datetime_col=None
):
    """
    Reads the NPM file and returns a pandas DataFrame with processed photometry data.

    Parameters
    ----------
    photometry_path : str
        Path to the NPM photometry CSV file containing the raw data
    region_to_mouse_path : str, optional
        Path to a CSV file containing the mapping between brain regions and mice.
        Must contain columns 'mouse', 'region', and 'wave_len'
    region_to_mouse : dict or pd.DataFrame, optional
        Direct mapping between brain regions and mice, as alternative to region_to_mouse_path.
        Must contain columns 'mouse', 'region', and 'wave_len'
    extra_timestamps_path : str, optional
        Path to a CSV file containing timestamps if they are not in the NPM file.
        Must contain columns 'Item1' (frame counter) and 'Item2' (timestamp)
    datetime_col : str, optional
        Name of the datetime column in the NPM file, if timestamps are included there

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed photometry data with columns:
        - datetime_col: timestamp of measurement
        - Time (sec): seconds from start of recording
        - mouse: mouse identifier
        - red_signal: signal from red channel (560nm)
        - green_signal: signal from green channel (470nm)
        - red_reference: reference signal for red channel (410nm)
        - green_reference: reference signal for green channel (410nm)

    Raises
    ------
    ValueError
        If required files cannot be read or contain invalid data
        If required columns are missing
        If wavelength values are invalid
        If datetime values cannot be parsed
        If region to mouse mapping is invalid or missing
    """
    # READ PHOTOMETRY FILE
    try:
        df = pd.read_csv(photometry_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Could not read NPM file at {photometry_path}. Error: {str(e)}")
    except pd.errors.ParserError as e:
        raise ValueError(f"File at {photometry_path} is not a valid CSV file. Error: {str(e)}")

    # READ EXTRA TIMESTAMPS FILE
    if extra_timestamps_path is not None:
        # if, for some reason, the timestamps are not in the NPM file, we can use an extra file
        extra_timestamps = pd.read_csv(extra_timestamps_path)
        df = df.merge(extra_timestamps, left_on="FrameCounter", right_on="Item1")

        # Convert timestamp to datetime and rename
        df["Item2"] = pd.to_datetime(df["Item2"])
        df = df.rename(columns={"Item2": "datetimestamp"})

        # Drop the Item1 column since we don't need it anymore
        df = df.drop("Item1", axis=1)
        datetime_col = "datetimestamp"
    elif datetime_col is not None:
        # check if the col exists
        if datetime_col not in df.columns:
            raise ValueError(f"The column {datetime_col} does not exist in the DataFrame.")
    else:
        raise ValueError(
            "Either provide a datetime column (datetime_col) or an extra timestamps file (extra_timestamps_file)."
        )

    # CONVERT DATETIME COLUMN TO DATETIME IF IT'S A STRING
    if pd.api.types.is_string_dtype(df[datetime_col]):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except ValueError:
            raise ValueError(f"Column {datetime_col} contains values that cannot be converted to datetime.")

    # EXTRACT LED WAVELENGTHS
    df = extract_leds(df)

    # READ REGION TO MOUSE FILE
    if region_to_mouse is None and region_to_mouse_path is None:
        raise ValueError(
            "Either provide a region to mouse mapping (region_to_mouse) or a path to a "
            "CSV file containing the mapping (region_to_mouse_path)."
        )
    if region_to_mouse_path is not None:
        try:
            region_to_mouse = pd.read_csv(region_to_mouse_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise ValueError(f"Could not read region to mouse file at {region_to_mouse_path}. Error: {str(e)}")
        except pd.errors.ParserError as e:
            raise ValueError(f"File at {region_to_mouse_path} is not a valid CSV file. Error: {str(e)}")
    if type(region_to_mouse) == dict:
        region_to_mouse = pd.DataFrame(region_to_mouse)
    # Sanity check
    if type(region_to_mouse) != pd.DataFrame:
        raise ValueError(
            "The region_to_mouse variable must be a pandas DataFrame or a python dictionary. "
            "Use the region_to_mouse_path parameter to load a CSV file."
        )
    # Check if region_to_mouse is not empty
    if len(region_to_mouse) == 0:
        raise ValueError("The region to mouse mapping file is empty.")

    # Check required columns exist
    required_cols = ["mouse", "region", "wave_len"]
    missing_cols = [col for col in required_cols if col not in region_to_mouse.columns]
    if missing_cols:
        raise ValueError(
            f"The region to mouse mapping file is missing required columns: {', '.join(missing_cols)}"
        )

    # Convert wave_len to numeric if string, handling any conversion errors
    if pd.api.types.is_string_dtype(region_to_mouse["wave_len"]):
        try:
            region_to_mouse["wave_len"] = pd.to_numeric(region_to_mouse["wave_len"])
        except ValueError:
            raise ValueError("Column 'wave_len' contains values that cannot be converted to numbers.")

    # Check wave_len values are valid
    invalid_wavelengths = region_to_mouse[~region_to_mouse["wave_len"].isin([560, 470])]
    if not invalid_wavelengths.empty:
        invalid_values = invalid_wavelengths["wave_len"].unique()
        raise ValueError(
            f"Invalid wavelength values found in region to mouse mapping: {invalid_values}. "
            f"Only 560 and 470 are allowed."
        )

    # Check if regions exist as columns in dataframe
    invalid_regions = [region for region in region_to_mouse["region"].unique() if region not in df.columns]
    if invalid_regions:
        raise ValueError(
            f"The following regions from the mapping file are not present as columns "
            f"in the data: {', '.join(invalid_regions)}"
        )

    # EXTRACT REGIONS AND PIVOT FOR EVERY MOUSE
    df = df.reset_index().set_index(datetime_col)

    dfs = []
    for mouse in region_to_mouse.mouse.unique():
        red_region = region_to_mouse[(region_to_mouse["mouse"] == mouse) & (region_to_mouse["wave_len"] == 560)][
            "region"
        ].unique()[0]
        green_region = region_to_mouse[(region_to_mouse["mouse"] == mouse) & (region_to_mouse["wave_len"] == 470)][
            "region"
        ].unique()[0]

        red_signal = df.loc[df.wave_len == NPM_RED, red_region]
        green_signal = df.loc[df.wave_len == NPM_GREEN, green_region]
        red_reference = df.loc[df.wave_len == NPM_ISO, red_region]
        green_reference = df.loc[df.wave_len == NPM_ISO, green_region]

        if pd.api.types.is_datetime64_any_dtype(df.index):
            # Merge signals and references with a 0.05 second tolerance
            mouse_data = pd.merge_asof(
                pd.merge_asof(
                    pd.merge_asof(
                        red_signal.reset_index().rename(columns={red_region: "red_signal"}),
                        green_signal.reset_index().rename(columns={green_region: "green_signal"}),
                        on=datetime_col,
                        tolerance=pd.Timedelta("0.05S"),
                    ),
                    red_reference.reset_index().rename(columns={red_region: "red_reference"}),
                    on=datetime_col,
                    tolerance=pd.Timedelta("0.05S"),
                ),
                green_reference.reset_index().rename(columns={green_region: "green_reference"}),
                on=datetime_col,
                tolerance=pd.Timedelta("0.05S"),
            )
            mouse_data["mouse"] = mouse
        else:
            # Merge signals and references with a 0.05 second tolerance, using float timestamps
            mouse_data = pd.merge_asof(
                pd.merge_asof(
                    pd.merge_asof(
                        red_signal.reset_index().rename(columns={red_region: "red_signal"}),
                        green_signal.reset_index().rename(columns={green_region: "green_signal"}),
                        on=datetime_col,
                        tolerance=0.05,
                    ),
                    red_reference.reset_index().rename(columns={red_region: "red_reference"}),
                    on=datetime_col,
                    tolerance=0.05,
                ),
                green_reference.reset_index().rename(columns={green_region: "green_reference"}),
                on=datetime_col,
                tolerance=0.05,
            )
            mouse_data["mouse"] = mouse

        mouse_data["Time (sec)"] = (mouse_data[datetime_col] - df.index[0]).dt.total_seconds()

        dfs.append(mouse_data)
    df = pd.concat(dfs, ignore_index=True)

    # Fill NaN values with 0 where event is not NaN
    signal_cols = ["red_signal", "green_signal", "red_reference", "green_reference"]
    # Drop remaining rows where any signal is NaN
    df = df.dropna(subset=signal_cols)
    return df


def merge_events(
    df,
    events=None,
    events_path=None,
    event_col="event",
    datetime_col="datetimestamp",
    by=None,
    start_event=None,
    stop_event=None,
):
    """
    Merge events data with photometry data and optionally filter based on start/stop events.

    Parameters
    ----------
    df : pd.DataFrame
        Photometry data DataFrame containing signal measurements
    events : pd.DataFrame, optional
        DataFrame containing event data to merge. Must contain datetime_col and event_col columns
    events_path : str, optional
        Path to CSV file containing event data, alternative to providing events DataFrame
    event_col : str, default='event'
        Name of column containing event labels in events DataFrame
    datetime_col : str, default='datetimestamp'
        Name of datetime column in both DataFrames used for merging
    by : str or list of str, optional
        Column name(s) used to separate individual recordings before merging. If None, no grouping is applied.
    start_event : str, optional
        Event label indicating start of analysis period. Data before this event will be filtered out
    stop_event : str, optional
        Event label indicating end of analysis period. Data after this event will be filtered out

    Returns
    -------
    pd.DataFrame
        DataFrame containing merged photometry and event data, optionally filtered by start/stop events.
        Time values are recalculated relative to start_event if specified.

    Notes
    -----
    - Either events DataFrame or events_path must be provided
    - Datetime columns are automatically converted between string/datetime/float types as needed
    - Uses pd.merge_asof with 0.05 second tolerance for merging
    - Warns if start_event or stop_event not found for a recording
    """
    if events is None and events_path is None:
        raise ValueError("Either provide a events (events) or a path to a events file (events_path).")
    if events_path is not None:
        events = pd.read_csv(events_path)

    # Convert string by to list
    by_cols = [by] if isinstance(by, str) else by

    # Convert events datetime column to match df datetime column type
    if pd.api.types.is_string_dtype(events[datetime_col]):
        if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                events[datetime_col] = pd.to_datetime(events[datetime_col])
            except Exception as e:
                raise ValueError(f"Could not convert events {datetime_col} to datetime: {str(e)}")
        else:
            try:
                events[datetime_col] = pd.to_numeric(events[datetime_col])
            except Exception as e:
                raise ValueError(f"Could not convert events {datetime_col} to float: {str(e)}")

    # merge events with photometry data
    tolerance = pd.Timedelta("0.1S") if pd.api.types.is_datetime64_any_dtype(df[datetime_col]) else 0.1
    events = events.sort_values(by=[datetime_col])
    df = df.sort_values(by=[datetime_col])
    df["row_id"] = np.arange(len(df))

    merge_cols = ["row_id", datetime_col]
    if by_cols is not None:
        merge_cols = by_cols + merge_cols

    events = pd.merge_asof(
        events,
        df[merge_cols],
        on=datetime_col,
        by=by_cols,
        tolerance=tolerance,
        direction="nearest",  # Ensures each event maps to exactly one row
    )
    # merge back onto df using row_id
    drop_cols = [datetime_col]
    if by_cols is not None:
        drop_cols.extend(by_cols)
    events = events.drop(columns=drop_cols)
    df = pd.merge(df, events, on="row_id", how="left")
    df.drop(columns=["row_id"], inplace=True)

    if start_event is not None:

        def process_recording_data(recording_df):
            # Find the start event time for this recording
            start_times = recording_df.loc[recording_df[event_col] == start_event, datetime_col]
            if len(start_times) == 0:
                if by_cols is not None:
                    by_values = tuple(recording_df[col].iloc[0] for col in by_cols)
                    warnings.warn(f"start_event '{start_event}' not found for recording {by_values}")
                else:
                    warnings.warn(f"start_event '{start_event}' not found")
                return recording_df
            start_time = start_times.iloc[0]

            # Filter data to keep only rows after start event
            recording_df = recording_df[recording_df[datetime_col] >= start_time]

            # Recalculate Time(sec) relative to start event
            if pd.api.types.is_datetime64_any_dtype(recording_df[datetime_col]):
                recording_df["Time (sec)"] = (recording_df[datetime_col] - start_time).dt.total_seconds()
            else:
                recording_df["Time (sec)"] = recording_df[datetime_col] - start_time

            return recording_df

        # Apply the processing function to each recording's data while preserving index
        if by_cols is not None:
            df = df.groupby(by_cols, group_keys=False).apply(process_recording_data).reset_index(drop=True)
        else:
            df = process_recording_data(df)
        # Ensure original index is preserved
        df.index = range(len(df))

    if stop_event is not None:

        def process_recording_data(recording_df):
            # Find the stop event time for this recording
            stop_times = recording_df.loc[recording_df[event_col] == stop_event, datetime_col]
            if len(stop_times) == 0:
                if by_cols is not None:
                    by_values = tuple(recording_df[col].iloc[0] for col in by_cols)
                    warnings.warn(f"stop_event '{stop_event}' not found for recording {by_values}")
                else:
                    warnings.warn(f"stop_event '{stop_event}' not found")
                return recording_df

            stop_time = stop_times.iloc[0]

            # Filter data to keep only rows before stop event
            recording_df = recording_df[recording_df[datetime_col] <= stop_time]

            return recording_df

        # Apply the processing function to each recording's data while preserving index
        if by_cols is not None:
            df = df.groupby(by_cols, group_keys=False).apply(process_recording_data).reset_index(drop=True)
        else:
            df = process_recording_data(df)
        # Ensure original index is preserved
        df.index = range(len(df))

    return df
