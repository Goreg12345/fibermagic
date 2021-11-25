from functions.martianova import *
from pathlib import Path
import pandas as pd

from functions.plot import plot_seperate, plot_single

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR5'
DATA_FILE = 'FED3.csv'
LOG_FILE = 'FRfD2AdoraCrepilotPR5B8624-1.log'
MOUSE_NAME = 'B8624'
LED = '6'


def extract_leds(df):
    def freq(dual_num):
        # this is how the FP device encodes the frequency in the Flags
        dual_num = str(format(int(dual_num), '04b'))
        if dual_num[-1] == '1':
            return 410
        if dual_num[-2] == '1':
            return 470
        if dual_num[-3] == '1':
            return 560
        return 0
    df['wave_len'] = df['Flags'].apply(freq)
    return df


def main():
    df = pd.read_csv(DATA_DIR / ANALYSIS / DATA_FILE)
    df = extract_leds(df)
    # extract Dopamine reference and signal (Red)
    raw_reference = df[df['wave_len'] == 410]['Region{l}R'.format(l=LED)]
    raw_signal = df[df['wave_len'] == 560]['Region{l}R'.format(l=LED)]
    if len(raw_reference) == len(raw_signal) + 1:
        raw_reference = raw_reference[1:]
    if len(raw_reference) == len(raw_signal) - 1:
        raw_signal = raw_signal[1:]
    assert len(raw_signal) == len(raw_reference)

    plot_seperate(raw_signal, raw_reference)

    zdFF = get_zdFF(raw_reference, raw_signal)
    plot_single(zdFF)
    result = pd.DataFrame()
    result['zdFF'] = zdFF
    # martianova removes first 200 samples for whatever reason
    result['Timestamp'] = np.array(df[df['wave_len'] == 410]['Timestamp'][200:])
    result['Timestamp'] -= df['Timestamp'][0]
    result.to_csv(DATA_DIR / ANALYSIS / 'result_{n}.csv'.format(n=MOUSE_NAME))

    input1 = pd.read_csv(DATA_DIR / ANALYSIS / 'input1.csv')
    time = pd.read_csv(DATA_DIR / ANALYSIS / 'time.csv')
    logs = pd.read_csv(DATA_DIR / ANALYSIS / LOG_FILE)

    logs['lever'] = logs['SI@0.0'].apply(lambda val: val.split("@")[0])
    logs['Timestamp'] = logs['SI@0.0'].apply(lambda val: float(val.split("@")[1]) / 10)
    logs.to_csv(DATA_DIR / ANALYSIS / 'logs_{n}.csv'.format(n=MOUSE_NAME))

    time_locked = pd.DataFrame()
    for index, row in logs[logs['lever']=='FB'].iterrows():
        t = row['Timestamp']
        time_locked[index] = result[result['Timestamp'].between(t - 15, t + 15)]['zdFF'].reset_index()['zdFF']
    time_locked['average'] = time_locked.mean(axis=1)
    time_locked.to_csv(DATA_DIR / ANALYSIS / 'time_locked_reward_{n}.csv'.format(n=MOUSE_NAME))
    df


if __name__ == '__main__':
    main()
