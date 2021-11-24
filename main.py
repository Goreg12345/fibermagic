from functions.martianova import *
from pathlib import Path
import pandas as pd


DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR2'
DATA_FILE = 'FED3.csv'


def main():
    df = pd.read_csv(DATA_DIR / ANALYSIS / DATA_FILE)
    df


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
