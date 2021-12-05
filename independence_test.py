from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR5'
DATA_FILE = 'FED3.csv'
LOG_FILE = 'FRfD2AdoraCrepilotPR5B8624-1.log'
columns = ['Region0R', 'Region2R', 'Region4R', 'Region6R', 'Region8R', 'Region10R', 'Region12R']

if __name__ == '__main__':
    df = pd.read_csv('red')
    df = df[columns]
    df = df.diff()
    for column in columns:
        df[column] = df[df[column] > df[column].mean()][column]
    df = df.fillna(0)
    df.plot()
    plt.show()
    df.plot(ylim=(0.0, 0.05))
    plt.show()
    df
