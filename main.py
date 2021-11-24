from functions.martianova import *
from pathlib import Path
import pandas as pd

from functions.plot import plot_seperate, plot_single

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR2'
DATA_FILE = 'FED3.csv'


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
    raw_reference = df[df['wave_len'] == 410]['Region2R']
    raw_signal = df[df['wave_len'] == 560]['Region2R']
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
    result['Timestamp'] = df[df['wave_len'] == 410]['Timestamp']
    result.to_csv(DATA_DIR / ANALYSIS / 'result.csv')


if __name__ == '__main__':
    main()
